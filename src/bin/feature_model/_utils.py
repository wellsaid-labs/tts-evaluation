import logging
import os

import torch

from src.bin.feature_model._dataset import FeatureDataset
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config
from src.hparams import set_hparams as set_base_hparams

logger = logging.getLogger(__name__)


def set_hparams():
    """ Set hyperparameters specific to the signal model. """
    set_base_hparams()
    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'weight_decay': 10**-6,
        }
    })


def load_data(source_train='data/.feature_dataset/train',
              source_dev='data/.feature_dataset/dev',
              text_encoder=None,
              load_signal=False):
    """ Load train and dev datasets as ``SignalDataset``s.

    Args:
        source_train (str): Directory with training examples.
        source_dev (str): Directory with dev examples.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the
            text.
        load_signal (bool): If ``True`` the FeatureDataset, loads the signal.

    Returns:
        train (FeatureDataset)
        dev (FeatureDataset)
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the
            text.
    """
    if not os.path.isdir(source_dev) or not os.path.isdir(source_train):
        raise ValueError('Data files not found. '
                         'Did you run ``src.bin.feature_model.preprocess``?')

    train = FeatureDataset(source_train, text_encoder=text_encoder, load_signal=load_signal)
    dev = FeatureDataset(source_dev, text_encoder=train.text_encoder, load_signal=load_signal)
    return train, dev, train.text_encoder


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


def save_checkpoint(directory,
                    model=None,
                    optimizer=None,
                    text_encoder=None,
                    epoch=None,
                    step=None,
                    filename=None,
                    experiment_directory=None):
    """ Save a checkpoint.

    Args:
        directory (str): Directory where to save the checkpoint.
        model (torch.nn.Module, optional): Model to train and evaluate.
        optimizer (torch.optim.Optimizer, optional): Optimizer used for gradient descent.
        text_encoder (torchnlp.TextEncoder, optional): Text encoder used to encode and decode the
            text.
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
            'text_encoder': text_encoder,
            'epoch': epoch,
            'step': step,
            'experiment_directory': experiment_directory
        })

    return filename
