# TODO: Plot attention alignment
# TODO: Write generation code

from functools import reduce

import argparse
import gc
import logging
import math
import os
import random

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import BCELoss
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from tqdm import tqdm

import torch
import tensorflow as tf

tf.enable_eager_execution()

from src.datasets import lj_speech_dataset
from src.experiment_context_manager import ExperimentContextManager
from src.feature_model import FeatureModel
from src.lr_schedulers import DelayedExponentialLR
from src.optimizer import Optimizer
from src.spectrogram import plot_spectrogram
from src.spectrogram import log_mel_spectrogram_to_wav
from src.spectrogram import wav_to_log_mel_spectrogram
from src.utils import get_total_parameters
from src.utils import pad_batch
from src.utils import split_dataset
from src.utils import Average

logger = logging.getLogger(__name__)

# TODO: Consider an integration test run on this with a mock over load_data to have 10 items


def load_data(context, cache, text_encoder=None):
    """ Load the Linda Johnson (LJ) Speech dataset with spectrograms and encoded text.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        cache (str): Path to cache the processed dataset
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.

    Returns:
        (list): Linda Johnson (LJ) Speech dataset with ``log_mel_spectrograms`` and ``text``
        (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
    """
    if not os.path.isfile(cache):
        data = lj_speech_dataset()
        logger.info('Sample Data:\n%s', data[:5])
        if text_encoder is None:
            text_encoder = CharacterEncoder([r['text'] for r in data])
        logger.info('Data loaded, creating spectrograms and encoding text...')
        # ``requires_grad`` Neither the inputs nor outputs are to change with the gradient
        tf_device = '/gpu:%d' % context.device if context.is_cuda else '/cpu'
        with tf.device(tf_device):
            for row in tqdm(data):
                row['log_mel_spectrograms'] = torch.tensor(wav_to_log_mel_spectrogram(row['wav']))
                row['text'] = torch.tensor(text_encoder.encode(row['text']).data)
                row['stop_token'] = torch.FloatTensor(
                    [0 for _ in range(row['log_mel_spectrograms'].shape[0])])
                row['stop_token'][row['log_mel_spectrograms'].shape[0] - 1] = 1
                row['mask'] = torch.FloatTensor(
                    [1 for _ in range(row['log_mel_spectrograms'].shape[0])])
        logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)
        to_save = (data, text_encoder)
        context.save(cache, to_save)
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
    return train, dev


class DataIterator(object):
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

    def __init__(self,
                 context,
                 dataset,
                 batch_size,
                 train=True,
                 sort_key=lambda r: r['log_mel_spectrograms'].shape[0]):
        batch_sampler = BucketBatchSampler(
            dataset, batch_size, False, sort_key=sort_key, biggest_batches_first=False)
        self.context = context
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=DataIterator._collate_fn,
            pin_memory=True,
            num_workers=1)

    @staticmethod
    def _collate_fn(batch):
        """ List of tensors to a batch variable """
        text_batch, _ = pad_batch([row['text'] for row in batch])
        frames_batch, _ = pad_batch([row['log_mel_spectrograms'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        mask_batch, _ = pad_batch([row['mask'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        return (text_batch, transpose(frames_batch), transpose(stop_token_batch),
                transpose(mask_batch))

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple([self.context.maybe_cuda(t, non_blocking=True) for t in batch])


def init_model(context, vocab_size):
    """ Intitiate the ``FeatureModel`` with random weights.

    Args:
        vocab_size (int): Size of the vocab used for model embeddings.

    Returns:
        (FeatureModel): Feature model intialized.
    """
    model = FeatureModel(vocab_size)
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)
    model = context.maybe_cuda(model)
    return model


def get_loss(criterion_frames,
             criterion_stop_token,
             frames_batch,
             stop_token_batch,
             predicted_frames,
             predicted_frames_with_residual,
             predicted_stop_token,
             mask_batch,
             size_average=True):
    """ Compute the losses for Tacotron.

    Args:
        criterion_frames (torch.nn.modules.loss._Loss): Torch loss module instantiated for frames.
        criterion_stop_token (torch.nn.modules.loss._Loss): Torch loss module instantiated for
           stop tokens.
        frames_batch (torch.FloatTensor [num_frames, batch_size, frame_channels]): Ground truth
            frames.
        stop_token_batch (torch.FloatTensor [num_frames, batch_size]): Ground truth stop tokens.
        predicted_frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Predicted
            frames.
        predicted_frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Predicted frames with residual
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Predicted stop tokens.
        mask_batch (torch.LongTensor [num_frames, batch_size]): Mask of zero's and one's to apply.
        size_average (bool, optional): By default, the losses are averaged over observations for
            each minibatch;However, if the field size_average is set to ``False``, the losses are
            instead summed for each minibatch.

    Returns:
        (torch.Tensor) scalar loss value
    """

    def get_loss_frames(predicted_frames):
        loss = criterion_frames(predicted_frames, frames_batch)
        loss = loss * mask_batch.unsqueeze(-1)
        return torch.mean(loss) if size_average else torch.sum(loss)

    loss_frames = get_loss_frames(predicted_frames)
    loss_frames_with_residual = get_loss_frames(predicted_frames_with_residual)

    loss_stop_token = criterion_stop_token(predicted_stop_token, stop_token_batch)
    loss_stop_token = loss_stop_token * mask_batch
    loss_stop_token = torch.mean(loss_stop_token) if size_average else torch.sum(loss_stop_token)

    return loss_frames, loss_frames_with_residual, loss_stop_token


best_post_frames_loss = math.inf
best_stop_token_loss = math.inf


def get_model_iterator(context,
                       dataset,
                       batch_size,
                       model,
                       criterion_frames,
                       criterion_stop_token,
                       train=True,
                       label=''):
    """ Iterate over a dataset with the model, computing the loss function every iteration.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        model (FeatureModel): Model used to predict frames and a stop token.
        criterion_frames (torch.nn.modules.loss._Loss): Torch loss module instantiated for frames.
        criterion_stop_token (torch.nn.modules.loss._Loss): Torch loss module instantiated for
           stop tokens.
        train (bool): If ``True``, the batch will store gradients.
        label (str): Label to add to progress bar and logs.

    Returns:
        (torch.Tensor) Loss at every iteration
    """
    global best_stop_token_loss, best_post_frames_loss
    avg_post_frames_loss, avg_stop_token_loss, avg_pre_frames_loss = Average(), Average(), Average()
    torch.set_grad_enabled(train)
    model.train(mode=train)
    data_iterator = DataIterator(context, dataset, batch_size, train=train)
    with tqdm(data_iterator) as iterator:
        iterator.set_description(label)

        for text_batch, frames_batch, stop_token_batch, mask_batch in iterator:
            predicted = model(text_batch, frames_batch)
            losses = get_loss(criterion_frames, criterion_stop_token, frames_batch,
                              stop_token_batch, *predicted, mask_batch, True)
            yield sum(losses)

            # Clear Memory
            predicted_pre_frames, predicted_post_frames = tuple(t.detach() for t in predicted[:-1])
            pre_frames_loss, post_frames_loss, stop_token_loss = tuple(t.item() for t in losses)

            # Compute metrics
            num_frame_values = reduce(lambda x, y: x * y, frames_batch.shape)
            num_stop_token_values = reduce(lambda x, y: x * y, stop_token_batch.shape)
            avg_post_frames_loss.add(post_frames_loss * num_frame_values, num_frame_values)
            avg_pre_frames_loss.add(pre_frames_loss * num_frame_values, num_frame_values)
            avg_stop_token_loss.add(stop_token_loss * num_stop_token_values, num_stop_token_values)

            # Postfix will be displayed on the right of progress bar
            iterator.set_postfix(
                pre_frames_loss=avg_pre_frames_loss.get(),
                post_frames_loss=avg_post_frames_loss.get(),
                stop_token_loss=avg_stop_token_loss.get())

    if not train and avg_post_frames_loss.get() < best_post_frames_loss:
        logger.info('[%s] Best Post Frame Loss', label.upper())
        best_post_frames_loss = avg_post_frames_loss.get()

    if not train and avg_stop_token_loss.get() < best_stop_token_loss:
        logger.info('[%s] Best Stop Token Loss', label.upper())
        best_stop_token_loss = avg_stop_token_loss.get()

    logger.info('[%s] Pre Frame Loss: %f', label.upper(), avg_pre_frames_loss.get())
    logger.info('[%s] Post Frame Loss: %f', label.upper(), avg_post_frames_loss.get())
    logger.info('[%s] Stop Token Loss: %f', label.upper(), avg_stop_token_loss.get())

    def sample_spectrogram(batch, name):
        spectrogram = batch.transpose_(0, 1)[random.randint(0, batch_size - 1)].cpu().numpy()
        name = os.path.join(context.epoch_directory, label.lower() + '_' + name)
        plot_spectrogram(spectrogram, name + '.png')
        with tf.device('/cpu'):
            log_mel_spectrogram_to_wav(spectrogram, name + '.wav')

    if not train:
        sample_spectrogram(frames_batch, 'sample_spectrogram')
        sample_spectrogram(predicted_pre_frames, 'sample_predicted_pre_spectrogram')
        sample_spectrogram(predicted_post_frames, 'sample_predicted_post_spectrogram')

    # Clear any extra memory
    gc.collect()


def main():
    """ Main module if this file is invoked directly """
    with ExperimentContextManager(label='feature_model') as context:
        # Load checkpoint
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--checkpoint", type=str, default=None, help="load a checkpoint")
        args = parser.parse_args()
        checkpoint, text_encoder = None, None
        if args.checkpoint is not None:
            checkpoint = context.load(os.path.join(context.root_path, args.checkpoint))
            logger.info('Loaded checkpoint: %s' % (args.checkpoint,))
            text_encoder = checkpoint['text_encoder']

        # Load data
        cache = os.path.join(context.root_path, 'data/lj_speech.pt')
        data, text_encoder = load_data(context, cache, text_encoder=text_encoder)
        train, dev = make_splits(data)

        # Initialize deep learning components
        if checkpoint is not None:
            model = checkpoint['model']
            model.apply(
                lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
            optimizer = checkpoint['optimizer']
            scheduler = checkpoint['scheduler']
            # ISSUE: https://github.com/pytorch/pytorch/issues/7255
            scheduler.optimizer = optimizer.optimizer
        else:
            model = init_model(context, text_encoder.vocab_size)
            optimizer = Optimizer(
                Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
            scheduler = DelayedExponentialLR(optimizer.optimizer)
        criterion_frames = context.maybe_cuda(MSELoss(reduce=False))
        criterion_stop_token = context.maybe_cuda(BCELoss(reduce=False))

        train_batch_size = 36
        dev_batch_size = 72
        logger.info('Train Batch Size: %d', train_batch_size)
        logger.info('Dev Batch Size: %d', dev_batch_size)
        logger.info('Total Parameters: %d', get_total_parameters(model))
        logger.info('Model:\n%s' % model)
        epoch = 0
        step = 0

        while True:
            logger.info('Starting Epoch %d, Step %d', epoch, step)
            context.epoch(epoch)

            # Iterate over the training data
            logger.info('Training...')
            iterator = get_model_iterator(context, train, train_batch_size, model, criterion_frames,
                                          criterion_stop_token, True, 'TRAIN')
            for loss in iterator:
                loss.backward()

                optimizer.step()
                scheduler.step()
                step += 1

                # Clean up
                optimizer.zero_grad()
                del loss

            checkpoint = {
                'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'text_encoder': text_encoder
            }
            checkpoint_path = context.save(
                os.path.join(context.epoch_directory, 'checkpoint.pt'), checkpoint)
            logger.info('Saved Checkpoint: %s', checkpoint_path)

            logger.info('Evaluating...')
            list(
                get_model_iterator(context, dev, dev_batch_size, model, criterion_frames,
                                   criterion_stop_token, False, 'DEV'))

            epoch += 1
            print('–' * 100)


if __name__ == '__main__':
    main()
