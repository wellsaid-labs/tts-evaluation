import logging
import os
import random

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam
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
        logger.info('Text encoder vocab size: %d' % text_encoder.vocab_size)
        to_save = (data, text_encoder)
        context.save(to_save, cache)
        return to_save

    return context.load(cache)


def make_splits(data, splits=(0.7, 0.15, 0.15)):
    """ Split a dataset at 70% for train, 15% for development and 15% for test.

    Args:
        (list): Data to split

    Returns:
        train (list): Train split.
        dev (list): Development split.
        test (list): Test split.
    """
    random.shuffle(data)
    train, dev, test = split_dataset(data, splits)
    logger.info('Number Training Rows: %d', len(train))
    logger.info('Number Development Rows: %d', len(dev))
    logger.info('Number Test Rows: %d', len(test))
    logger.info('Sample Data:\n%s', train[:5])
    return train, dev, test


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
    """

    def collate_fn(batch):
        """ List of tensors to a batch variable """
        output_batch, _ = pad_batch([row['log_mel_spectrograms'] for row in batch])
        input_batch, _ = pad_batch([row['text'] for row in batch])
        to_variable = (lambda b: Variable(torch.stack(b).contiguous(), volatile=not train))
        return (to_variable(input_batch), to_variable(output_batch).t_(0, 1))

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


with ExperimentContextManager(label='feature_model') as context:

    cache = os.path.join(context.root_path, '/data/lj_speech.pt')
    data, text_encoder = load_data(context, cache)
    train, dev, test = make_splits(data)
    model = init_model(text_encoder.vocab_size)
    # LEARN MORE: https://github.com/pytorch/pytorch/issues/679
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Optimizer(Adam(params=params))


def train(resources=30, checkpoint=None, **kwargs):

    if isinstance(checkpoint, str):
        checkpoint = Checkpoint(checkpoint)
        model = checkpoint.model
        train_batch_size = checkpoint.train_batch_size
        optimizer = checkpoint.optimizer
        n_bad_epochs = checkpoint.n_bad_epochs
        max_score = checkpoint.max_score
    else:
        model = make_model()
        train_batch_size = 32
        # NOTE: https://github.com/pytorch/pytorch/issues/679
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Optimizer(Adam(params=params))
        n_bad_epochs = 0
        max_score = 0

    # NOTE: Because the training dataset was used to train the subject recongition, its better; therefore,
    # we cannot mix them
    epochs = max(round(resources), 1)
    train_max_batch_size = 1024
    patience = 3
    criterion = cuda(NLLLoss())
    logger.info('Epochs: %d', epochs)
    logger.info('Train Dataset Size: %d', len(train_dataset))
    logger.info('Dev Dataset Size: %d', len(dev_dataset))
    logger.info('Train Batch Size: %d', train_batch_size)
    logger.info('Train Max Batch Size: %d', train_max_batch_size)
    logger.info('Total Parameters: %d', get_total_parameters(model))
    logger.info('Model:\n%s' % model)

    # Train!
    for epoch in range(epochs):
        logger.info('Epoch: %d', epoch)

        # Iterate over the training data
        logger.info('Training...')
        model.train(mode=True)
        train_iterator = get_iterator(train_dataset, train_batch_size, train=True)
        for text, relation, mask in tqdm_notebook(train_iterator):
            optimizer.zero_grad()
            output = model(cuda_async(text), cuda_async(mask))
            loss = criterion(output, cuda_async(relation))

            # Backward propagation
            loss.backward()
            optimizer.step()

        # Save checkpoint
        checkpoint_path = Checkpoint.save(
            experiment_folder, {
                'model': model,
                'optimizer': optimizer,
                'relation_encoder': relation_encoder,
                'text_encoder': text_encoder,
                'train_batch_size': train_batch_size,
                'n_bad_epochs': n_bad_epochs,
                'max_score': max_score
            },
            device=device)

        # Evaluate
        score = evaluate_softmax(dev_dataset, model, 4096)

        # Scheduler for increasing batch_size inspired by this paper:
        # https://openreview.net/forum?id=B1Yy1BxCZ
        if max_score > score:
            n_bad_epochs += 1
        else:
            n_bad_epochs = 0
            max_score = score

        if n_bad_epochs >= patience:
            train_batch_size = min(train_max_batch_size, train_batch_size * 2)
            logger.info('Ran out of patience, increasing train batch size to: %d', train_batch_size)

        print('–' * 100)

    return -max_score, checkpoint_path
