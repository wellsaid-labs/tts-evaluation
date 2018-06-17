import logging
import os

import torch

from src.bin.signal_model._dataset import SignalDataset
from src.hparams import set_hparams as set_base_hparams
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config

logger = logging.getLogger(__name__)


def load_data(source_train='data/.signal_dataset/train',
              source_dev='data/.signal_dataset/dev',
              log_mel_spectrogram_prefix='log_mel_spectrogram',
              signal_prefix='signal',
              extension='.npy'):
    """ Load train and dev datasets as ``SignalDataset``s.

    Args:
        source_train (str): Directory with training examples.
        source_dev (str): Directory with dev examples.
        log_mel_spectrogram_prefix (str): Prefix of log mel spectrogram files.
        signal_prefix (str): Prefix of signal files.
        extension (str): Filename extension to load.

    Returns:
        train (SignalDataset)
        dev (SignalDataset)
    """
    if not os.path.isdir(source_dev) or not os.path.isdir(source_train):
        raise ValueError('Data files not found. '
                         'Did you run ``src/bin/feature_model/generate.py``?')

    kwargs = {
        'log_mel_spectrogram_prefix': log_mel_spectrogram_prefix,
        'signal_prefix': signal_prefix,
        'extension': extension,
    }
    return SignalDataset(source_train, **kwargs), SignalDataset(source_dev, **kwargs)


def set_hparams():
    """ Set auxillary hyperparameters specific to the signal model. """
    set_base_hparams()
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
            'weight_decay': 0,
        },
        'src.bin.signal_model._dataset.SignalDataset.__init__': {
            # SOURCE (Parallel WaveNet):
            # minibatch size of 32 audio clips, each containing 7,680 timesteps
            # (roughly 320ms).
            # SOURCE (DeepVoice):
            # We divide the utterances in our audio dataset into one second chunks with
            # a quarter second of context for each chunk, padding each utterance with a
            # quarter second of silence at the beginning. We filter out chunks that are
            # predominantly silence and end up with 74,348 total chunks.
            'slice_size': 7800,
        }
    })


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

    torch_save(filename, {
        'model': model,
        'optimizer': optimizer,
        'epoch': epoch,
        'step': step,
    })

    return filename
