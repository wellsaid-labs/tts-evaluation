from functools import reduce

import logging
import os
import random

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import BCELoss
from torchnlp.utils import pad_batch
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from tqdm import tqdm

import torch

from src.utils import split_dataset
from src.utils import get_total_parameters
from src.datasets import lj_speech_dataset
from src.spectrogram import wav_to_log_mel_spectrograms
from src.experiment_context_manager import ExperimentContextManager
from src.feature_model import FeatureModel
from src.configurable import configurable
from src.optimizer import Optimizer
from src.lr_schedulers import DelayedExponentialLR

Adam.__init__ = configurable(Adam.__init__)

logger = logging.getLogger(__name__)


def load_data(context, cache):
    """ Load the Linda Johnson (LJ) Speech dataset with spectrograms and encoded text.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        cache (str): Path to cache the processed dataset

    Returns:
        (list): Linda Johnson (LJ) Speech dataset with ``log_mel_spectrograms`` and ``text``
        (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
    """
    if not os.path.isfile(cache):
        data = lj_speech_dataset()
        text_encoder = CharacterEncoder([r['text'] for r in data])
        logger.info('Data loaded, creating spectrograms and encoding text...')
        for row in tqdm(data):
            row['log_mel_spectrograms'] = torch.FloatTensor(wav_to_log_mel_spectrograms(row['wav']))
            row['text'] = text_encoder.encode(row['text'])
            row['stop_token'] = torch.LongTensor(
                [0 for _ in range(row['log_mel_spectrograms'].shape[0])])
            row['stop_token'][row['log_mel_spectrograms'].shape[0]] = 1
            row['mask'] = torch.LongTensor([1 for _ in range(row['log_mel_spectrograms'].shape[0])])
        logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)
        to_save = (data, text_encoder)
        context.save(to_save, cache)
        return to_save

    return context.load(cache)


def make_splits(data, splits=(0.8, 0.2)):
    """ Split a dataset at 80% for train and 20% for development.

    Args:
        (list): Data to split

    Returns:
        train (list): Train split.
        dev (list): Development split.
        test (list): Test split.
    """
    random.shuffle(data)
    train, dev = split_dataset(data, splits)
    logger.info('Number Training Rows: %d', len(train))
    logger.info('Number Dev Rows: %d', len(dev))
    logger.info('Sample Data:\n%s', train[:5])
    return train, dev


def get_iterator(context, dataset, batch_size, train=True, sort_key=lambda r: r['text'].size()[0]):
    """ Get a batch iterator over the ``dataset``.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        train (bool): If ``True``, the batch will store gradients.
        sort_key (callable): Sort key used to group similar length data used to minimize padding.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Iterator includes variables:
            text_batch (torch.LongTensor [batch_size, num_tokens])
            frames_batch (torch.LongTensor [num_frames, batch_size, frame_channels])
            stop_token_batch (torch.LongTensor [num_frames, batch_size])
            mask_batch (torch.LongTensor [num_frames, batch_size, 1])
    """

    def to_variable(padded_batch):
        """ Given a ``padded_batch`` (list of Tensors), convert into a ``maybe_cuda`` Variable. """
        return context.maybe_cuda(
            Variable(torch.stack(padded_batch, dim=0).t_(0, 1).contiguous(), volatile=not train),
            async=True)

    def collate_fn(batch):
        """ List of tensors to a batch variable """
        text_batch, _ = pad_batch([row['text'] for row in batch])
        frames_batch, _ = pad_batch([row['log_mel_spectrograms'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        mask_batch, _ = pad_batch([row['mask'] for row in batch])
        return (to_variable(text_batch).t_(0, 1), to_variable(frames_batch),
                to_variable(stop_token_batch), to_variable(mask_batch))

    # Use bucket sampling to group similar sized text but with noise + random
    batch_sampler = BucketBatchSampler(dataset, sort_key, batch_size, sort_key_noise=0.5)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        pin_memory=context.is_cuda,
        num_workers=0)


def init_model(vocab_size):
    """ Intitiate the ``FeatureModel`` with random weights.

    Args:
        vocab_size (int): Size of the vocab used for model embeddings.

    Returns:
        (FeatureModel): Feature model intialized.
    """
    model = FeatureModel(vocab_size)
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)
    return model


def get_loss_frames(criterion, frames_batch, predicted_frames, mask_batch, size_average=True):
    """ Compute the loss for frames.

    Args:
        criterion (torch.nn.modules.loss._Loss): Torch loss module
        frames_batch (torch.FloatTensor [num_frames, batch_size, frame_channels]): Ground truth
            frames.
        predicted_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Predicted
            frames.
        mask_batch (torch.LongTensor [num_frames, batch_size]): Mask of zero's and one's to apply.
        size_average (bool, optional): By default, the losses are averaged over observations for
            each minibatch;However, if the field size_average is set to ``False``, the losses are
            instead summed for each minibatch.

    Returns:
        (torch.Tensor) scalar loss value
    """
    # loss [num_frames, batch_size, frame_channels]
    loss = criterion(frames_batch, predicted_frames, reduce=False)
    loss = loss * mask_batch.unsqueeze(-1)
    return torch.mean(loss) if size_average else torch.sum(loss)


def get_loss_stop_token(criterion,
                        stop_token_batch,
                        predicted_stop_token,
                        mask_batch,
                        size_average=True):
    """ Compute the loss for stop tokens.

    Args:
        criterion (torch.nn.modules.loss._Loss): Torch loss module
        stop_token_batch (torch.FloatTensor [num_frames, batch_size]): Ground truth stop tokens.
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Predicted stop tokens.
        mask_batch (torch.LongTensor [num_frames, batch_size]): Mask of zero's and one's to apply.
        size_average (bool, optional): By default, the losses are averaged over observations for
            each minibatch;However, if the field size_average is set to ``False``, the losses are
            instead summed for each minibatch.

    Returns:
        (torch.Tensor) scalar loss value
    """
    loss = criterion(stop_token_batch, predicted_stop_token, reduce=False)
    loss = loss * mask_batch
    return torch.mean(loss) if size_average else torch.sum(loss)


with ExperimentContextManager(label='feature_model') as context:
    cache = os.path.join(context.root_path, '/data/lj_speech.pt')
    data, text_encoder = load_data(context, cache)
    train, dev = make_splits(data)
    model = init_model(text_encoder.vocab_size)
    # LEARN MORE: https://github.com/pytorch/pytorch/issues/679
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))
    scheduler = DelayedExponentialLR(optimizer)
    criterion_frames = context.maybe_cuda(MSELoss())
    criterion_stop_token = context.maybe_cuda(BCELoss())

    train_batch_size = 64
    dev_batch_size = 256
    logger.info('Train Batch Size: %d', train_batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)
    epoch = -1

    while True:
        epoch += 1
        logger.info('Epoch: %d', epoch)

        # Iterate over the training data
        logger.info('Training...')
        model.train(mode=True)
        train_iterator = get_iterator(context, train, train_batch_size, train=True)
        for text_batch, frames_batch, stop_token_batch, mask_batch in tqdm(train_iterator):
            optimizer.zero_grad()
            predicted_frames, predicted_frames_with_residual, predicted_stop_token = model(
                text_batch, frames_batch)
            loss_frames = get_loss_frames(criterion_frames, frames_batch, predicted_frames,
                                          mask_batch)
            loss_stop_token = get_loss_stop_token(criterion_stop_token, stop_token_batch,
                                                  predicted_stop_token, mask_batch)
            loss_frames.backward()
            loss_stop_token.backward()
            optimizer.step()
            scheduler.step()

        checkpoint_path = context.save(
            '%d.pt' % epoch, {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'text_encoder': text_encoder,
                'train_batch_size': train_batch_size
            })
        logger.info('Checkpoint created at %s', checkpoint_path)

        model.train(mode=False)
        dev_iterator = get_iterator(context, dev, dev_batch_size, train=False)
        loss_frames = 0
        loss_stop_token = 0
        num_elements_stop_token = 0
        num_elements_frames = 0
        for text_batch, frames_batch, stop_token_batch, mask_batch in tqdm(dev_iterator):
            predicted_frames, predicted_frames_with_residual, predicted_stop_token = model(
                text_batch, frames_batch)
            loss_frames += get_loss_frames(
                criterion_frames, frames_batch, predicted_frames, mask_batch,
                size_average=False).data[0]
            num_elements_frames += reduce(lambda x, y: x * y, frames_batch.shape)
            loss_stop_token += get_loss_stop_token(
                criterion_stop_token,
                stop_token_batch,
                predicted_stop_token,
                mask_batch,
                size_average=False).data[0]
            num_elements_stop_token += reduce(lambda x, y: x * y, stop_token_batch.shape)

        logger.info('Frame loss: %f', loss_frames / num_elements_frames)
        logger.info('Stop token loss: %f', loss_stop_token / num_elements_stop_token)

        print('–' * 100)
