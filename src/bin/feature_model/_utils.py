import matplotlib
matplotlib.use('Agg')

import logging
import os
import random

from torch import multiprocessing
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader
from torchnlp.samplers import BucketBatchSampler
from torchnlp.text_encoders import CharacterEncoder
from torchnlp.utils import pad_batch
from tqdm import tqdm

import torch
import numpy as np

from src.audio import find_silence
from src.audio import mu_law_quantize
from src.audio import read_audio
from src.audio import wav_to_log_mel_spectrogram
from src.datasets import lj_speech_dataset
from src.utils import ROOT_PATH
from src.utils import split_dataset
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config

logger = logging.getLogger(__name__)


def set_hparams():
    """ Set auxillary hyperparameters specific to the signal model. """

    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'lr': 10**-3,
            'weight_decay': 10**-6,
        }
    })


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (torch.device, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        sort_key (callable): Sort key used to group similar length data used to minimize padding.
        trial_run (bool or int): If ``True``, iterates over one batch.
        load_signal (bool, optional): If `True`, return signal during iteration.
        num_workers (int, optional): Number of workers for data loading.

    Returns:
        (torch.utils.data.DataLoader) Single-process or multi-process iterators over the dataset.
        Iterator includes variables:
            text_batch (torch.LongTensor [batch_size, num_tokens])
            text_length_batch (list): List of lengths for each sentence.
            frames_batch (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frame_length_batch (list): List of lengths for each spectrogram.
            stop_token_batch (torch.FloatTensor [num_frames, batch_size])
            signal_batch (list): List of signals.
    """

    def __init__(self,
                 device,
                 dataset,
                 batch_size,
                 sort_key=lambda r: r['log_mel_spectrogram'].shape[0],
                 trial_run=False,
                 load_signal=False,
                 num_workers=0):
        batch_sampler = BucketBatchSampler(dataset, batch_size, False, sort_key=sort_key)
        self.device = device
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers)
        self.trial_run = trial_run
        self.load_signal = load_signal

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ List of tensors to a batch variable """
        text_batch, text_length_batch = pad_batch([row['text'] for row in batch])
        frame_batch, frame_length_batch = pad_batch([row['log_mel_spectrogram'] for row in batch])
        stop_token_batch, _ = pad_batch([row['stop_token'] for row in batch])
        transpose = lambda b: b.transpose_(0, 1).contiguous()
        ret = [
            text_batch, text_length_batch,
            transpose(frame_batch), frame_length_batch,
            transpose(stop_token_batch)
        ]
        if self.load_signal:
            signal_batch = [row['quantized_signal'] for row in batch]
            ret.append(signal_batch)
        return ret

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple([
                self._maybe_cuda(t, non_blocking=True) if torch.is_tensor(t) else t for t in batch
            ])

            if self.trial_run:
                break


def _preprocess_audio(row):
    """ Preprocess a speech dataset: computing a spectrogram, encoding text and quantizing.

    Args:
        row (dict {'wav', 'text'}): Example with a corresponding wav filename and text snippet.

    Returns:
        row (dict {'log_mel_spectrogram', 'stop_token', 'quantized_signal', 'text', 'wav'}): Updated
            row with a ``log_mel_spectrogram``, ``stop_token``, and ``quantized_signal`` features.
    """
    signal, _ = read_audio(row['wav'])
    quantized_signal = mu_law_quantize(signal)

    # Trim silence
    end_silence, start_silence = find_silence(quantized_signal)
    signal = signal[end_silence:start_silence]
    quantized_signal = quantized_signal[end_silence:start_silence]

    log_mel_spectrogram, right_pad = wav_to_log_mel_spectrogram(signal)
    log_mel_spectrogram = torch.tensor(log_mel_spectrogram)
    stop_token = torch.FloatTensor(log_mel_spectrogram.shape[0]).zero_()
    stop_token[-1] = 1

    # Right pad so: ``log_mel_spectrogram.shape[0] % quantized_signal.shape[0] == frame_hop``
    # We property is required for Wavenet.
    padding_index = mu_law_quantize(0)
    assert quantized_signal.shape == signal.shape
    quantized_signal = np.concatenate((quantized_signal, np.full((right_pad), padding_index)))
    row.update({
        'log_mel_spectrogram': log_mel_spectrogram,
        'stop_token': stop_token,
        'quantized_signal': torch.tensor(quantized_signal)
    })
    return row


def load_data(
        device=torch.device('cpu'),
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
        device (torch.device, optional): Device onto which to load data.
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

    cache = os.path.join(ROOT_PATH, cache)
    signal_cache = os.path.join(ROOT_PATH, signal_cache)

    if os.path.isfile(cache) and os.path.isfile(signal_cache):  # Load cache
        train, dev, text_encoder = torch_load(cache, device)
        if load_signal:
            train_signals, dev_signals = torch_load(signal_cache, device)
    else:  # Otherwise, preprocess dataset
        data = lj_speech_dataset()
        random.shuffle(data)
        logger.info('Sample Data:\n%s', data[:5])

        # Preprocess text
        text_encoder = CharacterEncoder(
            [r['text'] for r in data]) if text_encoder is None else text_encoder
        for row in data:
            row['text'] = text_encoder.encode(row['text'])

        if use_multiprocessing:
            # Preprocess audio with multi-threading
            pool = Pool(processes=min(len(data), multiprocessing.cpu_count()))
            # LEARN MORE (multiprocessing and tqdm integration):
            # https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
            data = list(tqdm(pool.imap(_preprocess_audio, data), total=len(data)))
            pool.close()
        else:
            data = [_preprocess_audio(r) for r in tqdm(data)]

        train, dev = split_dataset(data, splits)
        # Save cache
        train_signals = [r.pop('quantized_signal') for r in train]
        dev_signals = [r.pop('quantized_signal') for r in dev]
        torch_save(cache, (train, dev, text_encoder))
        torch_save(signal_cache, (train_signals, dev_signals))

    if load_signal:
        # Combine signals and dataset together
        for row, signal in zip(train, train_signals):
            row['quantized_signal'] = signal

        for row, signal in zip(dev, dev_signals):
            row['quantized_signal'] = signal

    return train, dev, text_encoder


def load_checkpoint(checkpoint=None, device=torch.device('cpu')):
    """ Load a checkpoint.

    Args:
        checkpoint (str or None): Path to a checkpoint to load.
        device (int): Device to load checkpoint onto where -1 is the CPU while 0+ is a GPU.

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None
    """
    # Load checkpoint
    if checkpoint is not None:
        checkpoint = torch_load(os.path.join(ROOT_PATH, checkpoint), device=device)
        if 'model' in checkpoint:
            checkpoint['model'].apply(
                lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
        if 'scheduler' in checkpoint and 'optimizer' in checkpoint:
            # ISSUE: https://github.com/pytorch/pytorch/issues/7255
            checkpoint['scheduler'].optimizer = checkpoint['optimizer'].optimizer
    return checkpoint


def save_checkpoint(directory,
                    model=None,
                    optimizer=None,
                    scheduler=None,
                    text_encoder=None,
                    epoch=None,
                    step=None,
                    filename=None):
    """ Save a checkpoint.

    Args:
        directory (str): Directory where to save the checkpoint.
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
        name = 'step_%d.pt' % (step,) if step is not None else 'checkpoint.pt'
        filename = os.path.join(directory, name)

    torch_save(
        filename, {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'text_encoder': text_encoder,
            'epoch': epoch,
            'step': step,
        })

    return filename
