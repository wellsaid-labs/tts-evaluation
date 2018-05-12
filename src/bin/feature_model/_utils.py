import matplotlib
matplotlib.use('Agg')

import logging
import os
import random

from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch
import tensorflow as tf

from src.datasets import lj_speech_dataset
from src.spectrogram import log_mel_spectrogram_to_wav
from src.spectrogram import plot_spectrogram
from src.spectrogram import wav_to_log_mel_spectrogram
from src.utils import plot_attention
from src.utils import split_dataset
from src.utils import get_root_path
from src.utils.experiment_context_manager import load

logger = logging.getLogger(__name__)


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
    """

    def __init__(self,
                 context,
                 dataset,
                 batch_size,
                 train=True,
                 sort_key=lambda r: r['log_mel_spectrogram'].shape[0],
                 trial_run=False):
        batch_sampler = BucketBatchSampler(dataset, batch_size, False, sort_key=sort_key)
        self.context = context
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=DataIterator._collate_fn,
            pin_memory=True,
            num_workers=0)
        self.trial_run = trial_run

    @staticmethod
    def _collate_fn(batch):
        """ List of tensors to a batch variable """
        text_batch, _ = pad_batch([row['text'] for row in batch])
        frame_batch, frame_length_batch = pad_batch([row['log_mel_spectrogram'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        return (text_batch, transpose(frame_batch), frame_length_batch, transpose(stop_token_batch))

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple([
                self.context.maybe_cuda(t, non_blocking=True) if torch.is_tensor(t) else t
                for t in batch
            ])

            if self.trial_run:
                break


def load_data(context, cache, text_encoder=None, splits=(0.8, 0.2)):
    """ Load the Linda Johnson (LJ) Speech dataset with spectrograms and encoded text.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        cache (str): Path to cache the processed dataset
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        splits (tuple): Train and dev splits to use with the dataset

    Returns:
        (list): Linda Johnson (LJ) Speech dataset with ``log_mel_spectrogram`` and ``text``
        (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
    """
    assert len(splits) == 2

    cache = os.path.join(context.root_path, cache)
    if not os.path.isfile(cache):
        data = lj_speech_dataset()
        random.shuffle(data)
        logger.info('Sample Data:\n%s', data[:5])

        if text_encoder is None:
            text_encoder = CharacterEncoder([r['text'] for r in data])

        with tf.device('/gpu:%d' % context.device if context.is_cuda else '/cpu'):
            for row in tqdm(data):
                log_mel_spectrogram, signal, sample_rate = wav_to_log_mel_spectrogram(row['wav'])
                row['log_mel_spectrogram'] = torch.tensor(log_mel_spectrogram)
                row['signal'] = torch.tensor(signal)
                row['sample_rate'] = sample_rate
                row['text'] = torch.tensor(text_encoder.encode(row['text']).data)
                row['stop_token'] = torch.FloatTensor(row['log_mel_spectrogram'].shape[0]).zero_()
                row['stop_token'][row['log_mel_spectrogram'].shape[0] - 1] = 1

        train, dev = split_dataset(data, splits)
        to_save = (train, dev, text_encoder)
        context.save(cache, to_save)
        return to_save

    return context.load(cache)


def load_checkpoint(checkpoint=None):
    """ Load a checkpoint.

    Args:
        checkpoint (str or None): Path to a checkpoint to load.

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None
    """
    # Load checkpoint
    if checkpoint is not None:
        checkpoint = load(os.path.join(get_root_path(), checkpoint))
        if 'model' in checkpoint:
            checkpoint['model'].apply(
                lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
        if 'scheduler' in checkpoint and 'optimizer' in checkpoint:
            # ISSUE: https://github.com/pytorch/pytorch/issues/7255
            checkpoint['scheduler'].optimizer = checkpoint['optimizer'].optimizer
    return checkpoint


def save_checkpoint(context,
                    model=None,
                    optimizer=None,
                    scheduler=None,
                    text_encoder=None,
                    epoch=None,
                    step=None,
                    filename=None):
    """ Save a checkpoint.

    Args:
        context (ExperimentContextManager): Context manager for the experiment
        model (torch.nn.Module, optional): Model to train and evaluate.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler used to adjust learning rate.
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        filename (str, optional): Filename to save the checkpoint too, by default the checkpoint
            is saved in ``os.path.join(context.epoch_directory, 'checkpoint.pt')``

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None
    """
    if filename is None:
        filename = os.path.join(context.epoch_directory, 'checkpoint.pt')

    context.save(
        filename, {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'text_encoder': text_encoder,
            'epoch': epoch,
            'step': step,
        })

    return filename


def sample_spectrogram(batch, filename, item=0):
    """ Sample a spectrogram from a batch and save a visualization.

    Args:
        batch (torch.FloatTensor [num_frames, batch_size, frame_channels]): Batch of frames.
        item (int): Item from the batch to sample.
        filename (str): Filename to use for sample without an extension
    """
    _, batch_size, _ = batch.shape
    spectrogram = batch.detach().transpose_(0, 1)[item].cpu().numpy()
    plot_spectrogram(spectrogram, filename + '.png')
    log_mel_spectrogram_to_wav(spectrogram, filename + '.wav')


def sample_attention(batch, filename, item=0):
    """ Sample an alignment from a batch and save a visualization.

    Args:
        batch (torch.FloatTensor [num_frames, batch_size, num_tokens]): Batch of alignments.
        item (int): Item from the batch to sample.
        filename (str): Filename to use for sample without an extension
    """
    _, batch_size, _ = batch.shape
    alignment = batch.detach().transpose_(0, 1)[item].cpu().numpy()
    plot_attention(alignment, filename + '.png')
