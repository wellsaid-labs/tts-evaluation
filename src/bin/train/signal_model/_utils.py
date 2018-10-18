from pathlib import Path

import logging

from src.bin.train.signal_model._dataset import SignalDataset
from src.hparams import set_hparams as set_base_hparams
from src.utils.configurable import add_config
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


@configurable
def load_data(predicted_source='data/.signal_dataset',
              real_source='data/.feature_dataset',
              source_train='train',
              source_dev='dev',
              log_mel_spectrogram_prefix='log_mel_spectrogram',
              signal_prefix='signal',
              extension='.npy',
              predicted=True):
    """ Load train and dev datasets as ``SignalDataset``s.

    Args:
        predicted_source (str): Directory with all predicted examples.
        real_source (str): Directory with all real examples.
        source_train (str): Directory name with training examples.
        source_dev (str): Directory name with dev examples.
        log_mel_spectrogram_prefix (str, optional): Prefix of log mel spectrogram files.
        signal_prefix (str, optional): Prefix of signal files.
        extension (str, optional): Filename extension to load.
        predicted (bool, optional): Load predicted examples instead of ground truth.

    Returns:
        train (SignalDataset)
        dev (SignalDataset)
    """
    if predicted:
        source = Path(predicted_source)
    else:
        source = Path(real_source)

    source_dev = source / source_dev
    source_train = source / source_train

    if not source_dev.is_dir() or not source_train.is_dir():
        raise ValueError('Data files not found. '
                         'Did you run ``src.bin.train.feature_model.generate``?')

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
        'src.bin.train.signal_model.__main__.Trainer.__init__': {
            # Optimized for 4x P100 GPU
            'train_batch_size': 64,
            'dev_batch_size': 256,
            'num_workers': 12,
        },
    })
