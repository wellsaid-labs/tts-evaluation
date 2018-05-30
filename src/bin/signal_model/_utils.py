import matplotlib
matplotlib.use('Agg')

import logging
import math
import os
import random

from torch.utils.data import DataLoader
from torchnlp.utils import pad_batch
from torch.utils import data
from torch import nn

import torch
import numpy as np

from src.audio import mu_law_quantize
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


class SignalDataset(data.Dataset):
    """ Signal dataset loads and preprocesses a spectrogram and signal for training.

    Args:
        source (str): Directory with data.
        log_mel_spectrogram_prefix (str): Prefix of log mel spectrogram files.
        quantized_signal_prefix (str): Prefix of quantized signal files.
        extension (str): Filename extension to load.
        slice_size (int): Size of slices to load for training data.
        receptive_field_size (int): Context added to slice; to compute target signal.

    References:
        * Parallel WaveNet https://arxiv.org/pdf/1711.10433.pdf
          "each containing 7,680 timesteps (roughly 320ms)."
    """

    @configurable
    def __init__(self,
                 source,
                 log_mel_spectrogram_prefix='log_mel_spectrogram',
                 quantized_signal_prefix='quantized_signal',
                 extension='.npy',
                 slice_size=7000,
                 receptive_field_size=1):
        # Invariant: Must be at least one. ``receptive_field_size`` includes the current timestep
        # that must be taken into consideration at very least to predict the next timestep.
        assert receptive_field_size >= 1
        prefixes = [log_mel_spectrogram_prefix, quantized_signal_prefix]
        self.rows = _get_filename_table(source, prefixes=prefixes, extension=extension)
        self.slice_samples = slice_size
        self.set_receptive_field_size(receptive_field_size)
        self.log_mel_spectrogram_prefix = log_mel_spectrogram_prefix
        self.quantized_signal_prefix = quantized_signal_prefix

    def __len__(self):
        return len(self.rows)

    def set_receptive_field_size(self, receptive_field_size):
        """
        Args:
            receptive_field_size (int): Context added to slice; to compute target signal.
        """
        # Remove one, because the current sample is not tallied as context
        self.context_samples = receptive_field_size - 1

    def _preprocess(self, log_mel_spectrogram, quantized_signal):
        """ Slice the data into bite sized chunks that fit onto GPU memory for training.

        Notes:
            * Frames batch needs to line up with the target signal. Each frame, is used to predict
              the target. While for the source singal, we use the last output to predict the
              target; therefore, the source signal is one timestep behind.
            * Source signal batch is one time step behind the target batch. They have the same
              signal lengths.
            * With a large batch size and 1s+ clips, its probable that every batch will have at
              least one sample with full context; therefore, rather than aligning source signal
              with the target signal later adding more computation we align them now with left
              padding now by ensuring the context size is the same.

        Args:
            log_mel_spectrogram (torch.Tensor [num_frames, channels])
            quantized_signal (torch.Tensor [signal_length])

        Returns:
            (dict): Dictionary with slices up to ``max_samples`` appropriate size for training.
        """
        samples, num_frames = quantized_signal.shape[0], log_mel_spectrogram.shape[0]
        samples_per_frame = int(samples / num_frames)
        slice_frames = int(self.slice_samples / samples_per_frame)
        context_frames = int(math.ceil(self.context_samples / samples_per_frame))

        # Invariants
        assert self.slice_samples % samples_per_frame == 0
        # Signal model requires that there is a scaling factor between the signal and frames
        assert samples % num_frames == 0

        # Get a frame slice
        # ``-slice_frames + 1, num_frames - 1`` to ensure there is an equal chance to that a
        # sample will be included inside the slice.
        # For example, with signal ``[1, 2, 3]`` and a ``slice_samples`` of 2 you'd get slices of:
        # (1), (1, 2), (2, 3), (3).
        # With each number represented at twice.
        start_frame = max(random.randint(-slice_frames + 1, num_frames - 1), 0)
        end_frame = min(start_frame + slice_frames, num_frames)
        start_context_frame = max(start_frame - context_frames, 0)
        frames_slice = log_mel_spectrogram[start_context_frame:end_frame]

        # Get a source sample slice shifted back one and target sample
        start_context_sample = start_context_frame * samples_per_frame
        end_sample = end_frame * samples_per_frame
        start_sample = start_frame * samples_per_frame
        source_signal_slice = quantized_signal[max(start_context_sample - 1, 0):end_sample - 1]
        target_signal_slice = quantized_signal[start_sample:end_sample]

        # EDGE CASE: Pad context incase it's cut off and add a go sample for source
        if start_context_frame == 0:
            go_sample = quantized_signal.new_tensor([mu_law_quantize(0)])
            source_signal_slice = torch.cat((go_sample, source_signal_slice), dim=0)

            context_frame_pad = context_frames - start_frame
            frames_slice = nn.functional.pad(frames_slice, (0, 0, context_frame_pad, 0))

            context_sample_pad = context_frame_pad * samples_per_frame
            source_signal_slice = nn.functional.pad(source_signal_slice, (context_sample_pad, 0))

        return {
            self.log_mel_spectrogram_prefix: log_mel_spectrogram,  # [num_frames, channels]
            self.quantized_signal_prefix: quantized_signal,  # [signal_length]
            'source_signal_slice': source_signal_slice,  # [slice_size + receptive_field_size]
            'target_signal_slice': target_signal_slice,  # [slice_size]
            # [(slice_size + receptive_field_size) / samples_per_frame]
            'frames_slice': frames_slice,
        }

    def __getitem__(self, index):
        # Load data
        # log_mel_spectrogram [num_frames, channels]
        log_mel_spectrogram = torch.from_numpy(np.load(
            self.rows[index]['log_mel_spectrogram'])).contiguous()
        # quantized_signal [signal_length]
        quantized_signal = torch.from_numpy(np.load(
            self.rows[index]['quantized_signal'])).contiguous()

        return self._preprocess(log_mel_spectrogram, quantized_signal)


def _get_filename_table(directory, prefixes=[], extension=''):
    """ Get a table of filenames; such that every row has multiple filenames of different prefixes.

    Notes:
        * Filenames are aligned via string sorting.
        * The table must be full; therefore, all filenames associated with a prefix must have an
          equal number of files as every other prefix.

    Args:
        directory (str): Path to a directory.
        prefixes (str): Prefixes to load.
        extension (str): Filename extensions to load.

    Returns:
        (list of dict): List of dictionaries where prefixes are the key names.
    """
    rows = []
    for prefix in prefixes:
        # Get filenames with associated prefixes
        filenames = []
        for filename in os.listdir(directory):
            if filename.endswith(extension) and prefix in filename:
                filenames.append(os.path.join(directory, filename))

        # Sorted to align with other prefixes
        filenames = sorted(filenames)

        # Add to rows
        if len(rows) == 0:
            rows = [{prefix: filename} for filename in filenames]
        else:
            assert len(filenames) == len(rows), "Each row must have an associated filename."
            for i, filename in enumerate(filenames):
                rows[i][prefix] = filename

    return rows


def load_data(source_train='data/signal_dataset/train',
              source_dev='data/signal_dataset/dev',
              log_mel_spectrogram_prefix='log_mel_spectrogram',
              quantized_signal_prefix='quantized_signal',
              extension='.npy'):
    """ Load train and dev datasets as ``SignalDataset``s.

    Args:
        source_train (str): Directory with training examples.
        source_dev (str): Directory with dev examples.
        log_mel_spectrogram_prefix (str): Prefix of log mel spectrogram files.
        quantized_signal_prefix (str): Prefix of quantized signal files.
        extension (str): Filename extension to load.

    Returns:
        train (SignalDataset)
        dev (SignalDataset)
    """
    kwargs = {
        'log_mel_spectrogram_prefix': log_mel_spectrogram_prefix,
        'quantized_signal_prefix': quantized_signal_prefix,
        'extension': extension,
    }
    return SignalDataset(source_train, **kwargs), SignalDataset(source_dev, **kwargs)


def set_hparams():
    """ Set auxillary hyperparameters specific to the signal model. """

    add_config({
        # SOURCE (Tacotron 2):
        # We train with a batch size of 128 distributed across 32 GPUs with synchronous updates,
        # using the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−8 and a fixed learning rate
        # of 10−4
        # SOURCE (Deep Voice):
        # For training, we use the Adam optimization algorithm with β1 = 0.9, β2 = 0.999, ε = 10−8,
        # a batch size of 8, a learning rate of 10−3
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-8,
            # NOTE: assuming a batch size of 36 with 7900 samples per element is around 284400
            # samples while DeepVoice had a batch size of 8 with 19840  samples per element is
            # around 158720 samples; Therefore, with a larger number of samples than DeepVoice,
            # we increase our learning rate.
            'lr': 10**-3,
            'weight_decay': 0
        }
    })


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (torch.device, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        trial_run (bool): If ``True``, the data iterator runs only one batch.
        num_workers (int, optional): Number of workers for data loading.
    """

    def __init__(self, device, dataset, batch_size, trial_run=False, num_workers=0):
        # ``drop_last`` to ensure full utilization of mutliple GPUs
        self.device = device
        self.iterator = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers,
            drop_last=True)
        self.trial_run = trial_run

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ Collage function to turn a list of tensors into one batch tensor.

        Returns: (dict) with:
            * source_signal (torch.FloatTensor [batch_size, signal_length])
            * target_signal (torch.FloatTensor [batch_size, signal_length])
            * signal_lengths (list): List of lengths for each signal.
            * frames (torch.FloatTensor [batch_size, num_frames, frame_channels])
            * spectrograms (list): List of spectrograms to be used for sampling.
        """
        source_signals, source_signal_lengths = pad_batch([r['source_signal_slice'] for r in batch])
        target_signals, target_signal_lengths = pad_batch([r['target_signal_slice'] for r in batch])
        frames, frames_lengths = pad_batch([r['frames_slice'] for r in batch])
        spectrograms = [r['log_mel_spectrogram'] for r in batch]
        signals = [r['quantized_signal'] for r in batch]
        length_diff = [s - t for s, t in zip(source_signal_lengths, target_signal_lengths)]
        assert length_diff.count(length_diff[0]) == len(length_diff), (
            "Source must be a constant amount longer than target; "
            "otherwise, they wont be aligned after padding.")
        return {
            'source_signals': source_signals,
            'target_signals': target_signals,
            'target_signal_lengths': target_signal_lengths,
            'frames': frames,
            'spectrograms': spectrograms,
            'frames_lengths': frames_lengths,
            'signals': signals,
        }

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            batch['source_signals'] = self._maybe_cuda(batch['source_signals'], non_blocking=True)
            batch['target_signals'] = self._maybe_cuda(batch['target_signals'], non_blocking=True)
            batch['frames'] = self._maybe_cuda(batch['frames'], non_blocking=True)
            batch['spectrograms'] = [
                self._maybe_cuda(s, non_blocking=True) for s in batch['spectrograms']
            ]

            yield batch

            if self.trial_run:
                break


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
    return checkpoint


def save_checkpoint(directory, model=None, optimizer=None, epoch=None, step=None, filename=None):
    """ Save a checkpoint.

    Args:
        directory (str): Directory where to save the checkpoint.
        model (torch.nn.Module, optional): Model to train and evaluate.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
        })

    return filename
