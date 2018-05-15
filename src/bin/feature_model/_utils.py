import matplotlib
matplotlib.use('Agg')

import logging
import os
import random

from multiprocessing import Pool
from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch
import numpy as np

from src.datasets import lj_speech_dataset
from src.preprocess import find_silence
from src.preprocess import log_mel_spectrogram_to_wav
from src.preprocess import mu_law_quantize
from src.preprocess import plot_spectrogram
from src.preprocess import read_audio
from src.preprocess import wav_to_log_mel_spectrogram
from src.utils import get_root_path
from src.utils import load
from src.utils import plot_attention
from src.utils import save
from src.utils import split_dataset

logger = logging.getLogger(__name__)


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (int, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        train (bool): If ``True``, the batch will store gradients.
        sort_key (callable): Sort key used to group similar length data used to minimize padding.
        load_signal (bool, optional): If `True`, return signal during iteration.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Iterator includes variables:
            text_batch (torch.LongTensor [batch_size, num_tokens])
            frames_batch (torch.LongTensor [num_frames, batch_size, frame_channels])
            frame_length_batch (list): List of lengths for each spectrogram.
            stop_token_batch (torch.LongTensor [num_frames, batch_size])
            signal_batch (list): List of signals.
    """

    def __init__(self,
                 device,
                 dataset,
                 batch_size,
                 train=True,
                 sort_key=lambda r: r['log_mel_spectrogram'].shape[0],
                 trial_run=False,
                 load_signal=False):
        batch_sampler = BucketBatchSampler(dataset, batch_size, False, sort_key=sort_key)
        self.maybe_cuda = lambda t, **kwargs: t.cuda(device=device, **kwargs) if device > -1 else t
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=0)
        self.trial_run = trial_run
        self.load_signal = load_signal

    def _collate_fn(self, batch):
        """ List of tensors to a batch variable """
        text_batch, _ = pad_batch([row['text'] for row in batch])
        frame_batch, frame_length_batch = pad_batch([row['log_mel_spectrogram'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        ret = [text_batch, transpose(frame_batch), frame_length_batch, transpose(stop_token_batch)]
        if self.load_signal:
            signal_batch = [row['signal'] for row in batch]
            ret.append(signal_batch)
        return ret

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple(
                [self.maybe_cuda(t, non_blocking=True) if torch.is_tensor(t) else t for t in batch])

            if self.trial_run:
                break


def _preprocess_audio(row):
    """ Preprocess a speech dataset: computing a spectrogram, encoding text and quantizing.

    Args:
        row (dict {'wav', 'text'}): Example with a corresponding wav filename and text snippet.
    """
    signal, sample_rate = read_audio(row['wav'])
    quantized_signal = mu_law_quantize(signal)

    # Trim silence
    end_silence, start_silence = find_silence(quantized_signal)
    signal = signal[end_silence:start_silence]
    quantized_signal = quantized_signal[end_silence:start_silence]

    log_mel_spectrogram, right_pad = wav_to_log_mel_spectrogram(signal, sample_rate)
    log_mel_spectrogram = torch.tensor(log_mel_spectrogram)
    stop_token = torch.FloatTensor(log_mel_spectrogram.shape[0]).zero_()
    stop_token[-1] = 1

    # Right pad so: ``log_mel_spectrogram.shape[0] % quantized_signal.shape[0] == frame_hop``
    # We property is required for Wavenet.
    quantized_signal = np.concatenate((quantized_signal, np.zeros((right_pad))))
    row.update({
        'log_mel_spectrogram': log_mel_spectrogram,
        'stop_token': stop_token,
        'signal': torch.tensor(quantized_signal),
    })
    return row


def load_data(device=-1,
              cache='data/cache.pt',
              signal_cache='data/cache_signals.pt',
              load_signal=False,
              text_encoder=None,
              splits=(0.8, 0.2),
              use_multiprocessing=True):
    """ Load the Linda Johnson (LJ) Speech dataset with spectrograms and encoded text.

    Notes:
        * We use a seperate cache for signals due them being the majority of the dataset memory
              footprint.

    Args:
        device (int, optional): Device onto which to load data.
        cache (str, optional): Path to cache the processed dataset.
        signal_cache (str, optional): Path to cache signal in the processed dataset.
        load_signal (bool, optional): If `True`, load signal files.
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
        splits (tuple, optional): Train and dev splits to use with the dataset.
        use_multiprocessing (bool, optional): If `True`, use multiple processes to preprocess data.

    Returns:
        (list): Linda Johnson (LJ) Speech dataset with ``log_mel_spectrogram`` and ``text``
        (torchnlp.TextEncoder): Text encoder used to encode and decode the text.
    """
    assert len(splits) == 2

    cache = os.path.join(get_root_path(), cache)
    signal_cache = os.path.join(get_root_path(), signal_cache)

    if os.path.isfile(cache) and os.path.isfile(signal_cache):  # Load cache
        train, dev, text_encoder = load(cache, device)
        if load_signal:
            train_signals, dev_signals = load(signal_cache, device)
    else:  # Otherwise, preprocess dataset
        data = lj_speech_dataset()
        random.shuffle(data)
        logger.info('Sample Data:\n%s', data[:5])

        # Preprocess text
        text_encoder = CharacterEncoder(
            [r['text'] for r in data]) if text_encoder is None else text_encoder
        for row in data:
            row['text'] = text_encoder.encode(row['text'])
        logger.info('Done ... Processing Text')

        if use_multiprocessing:
            # Preprocess audio with multi-threading
            pool = Pool()
            logger.info('Created process pool.')
            # LEARN MORE (multiprocessing and tqdm integration):
            # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
            data = list(tqdm(pool.imap(_preprocess_audio, data), total=len(data)))
        else:
            data = [_preprocess_audio(r) for r in data]
        logger.info('Done ... Processing Audio')

        train, dev = split_dataset(data, splits)
        # Save cache
        train_signals = [r.pop('signal') for r in train]
        dev_signals = [r.pop('signal') for r in dev]
        save(cache, (train, dev, text_encoder))
        save(signal_cache, (train_signals, dev_signals))

    if load_signal:
        # Combine signals and dataset together
        for row, signal in zip(train, train_signals):
            row['signal'] = signal

        for row, signal in zip(dev, dev_signals):
            row['signal'] = signal

    return train, dev, text_encoder


def load_checkpoint(checkpoint=None, device=-1):
    """ Load a checkpoint.

    Args:
        checkpoint (str or None): Path to a checkpoint to load.
        device (int): Device to load checkpoint onto where -1 is the CPU while 0+ is a GPU.

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None
    """
    # Load checkpoint
    if checkpoint is not None:
        checkpoint = load(os.path.join(get_root_path(), checkpoint), device=device)
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


def sample_spectrogram(batch, filename, item=0, synthesize=False):
    """ Sample a spectrogram from a batch and save a visualization.

    Args:
        batch (torch.FloatTensor [num_frames, batch_size, frame_channels]): Batch of frames.
        filename (str): Filename to use for sample without an extension.
        item (int, optional): Item from the batch to sample.
        synthesize (bool, optional): Use griffin-lim to synthesize spectrogram.
    """
    _, batch_size, _ = batch.shape
    spectrogram = batch.detach().transpose_(0, 1)[item].cpu().numpy()
    plot_spectrogram(spectrogram, filename + '.png')
    if synthesize:
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
