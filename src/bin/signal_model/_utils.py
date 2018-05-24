import matplotlib
matplotlib.use('Agg')

import logging
import os
import random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchnlp.samplers import ShuffleBatchSampler
from torchnlp.utils import pad_batch

import torch

from src.audio import mu_law_quantize
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


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
            'eps': 10e-8,
            'lr': 10e-3,
            'weight_decay': 0
        }
    })


class ShuffleSampler(Sampler):
    """Samples elements randomly.

    Args:
        data (iterable): Iterable data.

    Example:
        >>> list(ShuffleSampler(range(10)))
        [5, 6, 2, 0, 3, 4, 7, 9, 1, 8]
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.shuffled_indicies = list(range(len(self.data)))
        random.shuffle(self.shuffled_indicies)

    def __iter__(self):
        return iter(self.shuffled_indicies)

    def __len__(self):
        return len(self.data)


class DataIterator(object):
    """ Get a batch iterator over the ``dataset``.

    Args:
        device (torch.device, optional): Device onto which to load data.
        dataset (list): Dataset to iterate over.
        batch_size (int): Size of the batch for iteration.
        train (bool): If ``True``, the batch will store gradients.
        num_workers (int, optional): Number of workers for data loading.
    """

    @configurable
    def __init__(self,
                 device,
                 dataset,
                 batch_size,
                 trial_run=False,
                 num_workers=0,
                 max_samples=7800):
        # ``drop_last`` to ensure full utilization of mutliple GPUs
        batch_sampler = ShuffleBatchSampler(ShuffleSampler(dataset), batch_size, drop_last=True)
        self.device = device
        self.iterator = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._collate_fn,
            pin_memory=True,
            num_workers=num_workers)
        self.trial_run = trial_run
        self.max_samples = max_samples

    def _maybe_cuda(self, tensor, **kwargs):
        return tensor.cuda(device=self.device, **kwargs) if self.device.type == 'cuda' else tensor

    def _collate_fn(self, batch):
        """ Collage function to turn a list of tensors into one batch tensor.

        Notes:
            * Frames batch needs to line up with the target signal. Each frame, is used to predict
              the target. While for the source singal, we use the last output to predict the
              target; therefore, the source signal is one timestep behind.
            * Source signal batch is one time step behind the target batch. They have the same
              signal lengths.

        Returns:
            source_signal_batch (torch.FloatTensor [signal_length, batch_size])
            target_signal_batch (torch.FloatTensor [signal_length, batch_size])
            signal_length_batch (list): List of lengths for each signal.
            frames_batch (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frame_length_batch (list): List of lengths for each spectrogram.
        """
        source_signal_batch = []
        target_signal_batch = []
        frames_batch = []
        zero_index = mu_law_quantize(0)
        for row in batch:
            assert len(row['quantized_signal']) % len(row['log_mel_spectrogram']) == 0
            factor = int(len(row['quantized_signal']) / len(row['log_mel_spectrogram']))

            if len(row['quantized_signal']) < self.max_samples:
                target_signal_batch.append(row['quantized_signal'])
                zero_point = row['quantized_signal'].new_full((1,), zero_index)
                source_signal_batch.append(
                    torch.cat((zero_point, target_signal_batch[-1][:1]), dim=0))
                frames_batch.append(row['log_mel_spectrogram'])
            else:
                assert self.max_samples % factor == 0

                # Get a frame slice
                max_frames = int(self.max_samples / factor)
                start_frame = random.randint(0, len(row['log_mel_spectrogram']) - max_frames)
                frames_slice = row['log_mel_spectrogram'][start_frame:start_frame + max_frames]

                # Get a signal slice
                start_sample = start_frame * factor
                target_signal_slice = row['quantized_signal'][start_sample:
                                                              start_sample + self.max_samples]
                last_index = zero_index if start_frame == 0 else row['quantized_signal'][
                    start_sample - 1]
                last_index = target_signal_slice.new_full((1,), last_index)

                frames_batch.append(frames_slice)
                source_signal_batch.append(torch.cat((last_index, target_signal_slice[:1]), dim=0))
                target_signal_batch.append(target_signal_slice)

        target_signal_batch, signal_length_batch = pad_batch(
            target_signal_batch, padding_index=zero_index)
        source_signal_batch, _ = pad_batch(source_signal_batch, padding_index=zero_index)
        frames_batch, frame_length_batch = pad_batch(frames_batch)
        return (source_signal_batch, target_signal_batch, signal_length_batch, frames_batch,
                frame_length_batch)

    def __len__(self):
        return 1 if self.trial_run else len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            yield tuple([
                self._maybe_cuda(t, non_blocking=True) if torch.is_tensor(t) else t for t in batch
            ])

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
        if 'model' in checkpoint:
            checkpoint['model'].apply(
                lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
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

    torch_save(filename, {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'step': step,
    })

    return filename
