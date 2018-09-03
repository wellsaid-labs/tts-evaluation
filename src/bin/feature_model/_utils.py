from pathlib import Path

import logging

from src.bin.feature_model._dataset import FeatureDataset
from src.utils.configurable import add_config
from src.hparams import set_hparams as set_base_hparams

logger = logging.getLogger(__name__)


def set_hparams():
    """ Set hyperparameters specific to the signal model. """
    set_base_hparams()

    # SOURCE: Tacotron 2
    # To train the feature prediction network, we apply the standard maximum-likelihood training
    # procedure (feeding in the correct output instead of the predicted output on the decoder side,
    # also referred to as teacher-forcing) with a batch size of 64 on a single GPU.

    # NOTE: Parameters set after experimentation on a 1 Px100 GPU.
    dev_batch_size = 256
    train_batch_size = 56
    num_workers = 12

    add_config({
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999, eps = 10−6
        # learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        'torch.optim.adam.Adam.__init__': {
            'eps': 10**-6,
            'weight_decay': 10**-6,
        },
        'src.bin.feature_model': {
            'train.Trainer.__init__': {
                'train_batch_size': train_batch_size,
                'dev_batch_size': dev_batch_size,
                'num_workers': num_workers,
            },
            'generate.main': {
                'num_workers': num_workers,
                'max_batch_size': dev_batch_size,
            }
        },
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
    source_dev = Path(source_dev)
    source_train = Path(source_train)

    if not source_dev.is_dir() or not source_train.is_dir():
        raise ValueError('Data files not found. '
                         'Did you run ``src.bin.feature_model.preprocess``?')

    train = FeatureDataset(source_train, text_encoder=text_encoder, load_signal=load_signal)
    dev = FeatureDataset(source_dev, text_encoder=train.text_encoder, load_signal=load_signal)
    return train, dev, train.text_encoder
