import logging
import os

from torchnlp.text_encoders import CharacterEncoder

import torch

from src.bin.feature_model._dataset import FeatureDataset
from src.datasets import lj_speech_dataset
from src.utils import ROOT_PATH
from src.utils import torch_load
from src.utils import torch_save
from src.utils.configurable import add_config
from src.utils.configurable import configurable
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
        },
        'src.optimizer.Optimizer.__init__': {
            # NOTE: Tacotron authors did not mention using this; but this is fairly common
            # practice. Used in both the NVIDIA/tacotron2, Rayhane-mamah/Tacotron-2, and
            # mozilla/TTS implementations.
            # 'max_grad_norm': 1.0
        }
    })


@configurable
def load_data(sample_rate=24000, text_encoder=None):
    """ Load train and dev datasets as ``SignalDataset``s.

    Args:
        sample_rate (int): Sample rate of the signal.
        text_encoder (torchnlp.TextEncoder): Text encoder used to encode and decode the
            text.

    Returns:
        train (FeatureDataset)
        dev (FeatureDataset)
    """
    train, dev = lj_speech_dataset(resample=sample_rate)
    logger.info('Sample Data:\n%s', train[:5])
    text_encoder = CharacterEncoder(
        [r['text'] for r in train]) if text_encoder is None else text_encoder
    kwargs = {'sample_rate': sample_rate, 'text_encoder': text_encoder}
    return FeatureDataset(train, **kwargs), FeatureDataset(dev, **kwargs), text_encoder


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
