import logging
import logging.config
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def get_root_path():
    """ Get the path to the root directory

    Returns (str):
        Root directory path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(dataset, splits):
    """
    Args:
        dataset (list): Dataset to split.
        splits (tuple): Tuple of percentages determining dataset splits.

    Returns:
        (list): splits of the dataset

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> splits = (.6, .2, .2)
        >>> split_dataset(dataset, splits)
        [[1, 2, 3], [4], [5]]
    """
    assert sum(splits) == 1, 'Splits must sum to 100%'
    splits = [round(s * len(dataset)) for s in splits]
    datasets = []
    for split in splits:
        datasets.append(dataset[:split])
        dataset = dataset[split:]
    return datasets


def plot_attention(alignment, filename, title='Attention Alignment'):
    """ Plot alignment of attention.

    Args:
        alignment (numpy.array([decoder_timestep, encoder_timestep])): Attention alignment weights
            computed at every timestep of the decoder.
        filename (str): Location to save the file.
        title (str, optional): Title of the plot.

    Returns:
        None
    """
    assert '.png' in filename.lower(), "Filename saves in PNG format"
    alignment = np.transpose(alignment)
    plt.style.use('ggplot')
    figure, axis = plt.subplots()
    im = axis.imshow(alignment, aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=1)
    figure.colorbar(im, ax=axis, orientation='horizontal')
    plt.xlabel('Decoder timestep')
    plt.title(title, y=1.1)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()
