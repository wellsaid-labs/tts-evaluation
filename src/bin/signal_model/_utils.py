import logging
import os

import torch

from src.bin.signal_model._dataset import SignalDataset
from src.hparams import set_hparams as set_base_hparams
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


@configurable
def load_data(generated_train='data/.signal_dataset/train',
              generated_dev='data/.signal_dataset/dev',
              real_train='data/.feature_dataset/train',
              real_dev='data/.feature_dataset/dev',
              log_mel_spectrogram_prefix='log_mel_spectrogram',
              signal_prefix='signal',
              extension='.npy',
              generated=True):
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
    if generated:
        source_train, source_dev = generated_train, generated_dev
    else:
        source_train, source_dev = real_train, real_dev

    source_train = os.path.join(ROOT_PATH, source_train)
    source_dev = os.path.join(ROOT_PATH, source_dev)

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
            'lr': 10**-3
        },
        'src.optimizer.Optimizer.__init__': {
            'max_grad_norm': 15.0
        }
    })


def load_checkpoint(checkpoint_path=None, device=torch.device('cpu')):
    """ Load a checkpoint.

    Args:
        checkpoint_path (str or None): Path to a checkpoint to load.
        device (int): Device to load checkpoint onto where -1 is the CPU while 0+ is a GPU.

    Returns:
        checkpoint (dict or None): Loaded checkpoint or None.
        checkpoint_path (str or None): Path of loaded checkpoint.
    """
    if checkpoint_path is None:
        return None, None

    checkpoint_path = os.path.join(ROOT_PATH, checkpoint_path)
    checkpoint = torch_load(checkpoint_path, device=device)
    if 'model' in checkpoint:
        checkpoint['model'].apply(
            lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
    return checkpoint


def save_checkpoint(directory,
                    model=None,
                    optimizer=None,
                    epoch=None,
                    step=None,
                    filename=None,
                    experiment_directory=None):
    """ Save a checkpoint.

    Args:
        directory (str): Directory where to save the checkpoint.
        model (torch.nn.Module, optional): Model to train and evaluate.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
        epoch (int, optional): Starting epoch, useful warm starts (i.e. checkpoints).
        step (int, optional): Starting step, useful warm starts (i.e. checkpoints).
        filename (str, optional): Filename to save the checkpoint too, by default the checkpoint
            is saved in ``os.path.join(context.epoch_directory, 'checkpoint.pt')``
        experiment_directory (str, optional): Directory experiment logs are saved in.

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
            'epoch': epoch,
            'step': step,
            'experiment_directory': experiment_directory
        })

    return filename
