import logging
import logging.config
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)

# TODO: Plot stop token

# Repository root path
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')


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


def plot_stop_token(stop_token, filename, title='Stop Token'):
    """ Plot alignment of attention.

    TODO: test this.

    Args:
        alignment (numpy.array([decoder_timestep])): Stop token probablity per decoder timestep.
        filename (str): Location to save the file.
        title (str, optional): Title of the plot.

    Returns:
        None
    """
    plt.style.use('ggplot')
    plt.plot(list(range(len(stop_token))), stop_token, marker='.', linestyle='solid')
    plt.title(title, y=1.1)
    plt.ylabel('Probability')
    plt.xlabel('Timestep')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()


def torch_load(path, device=torch.device('cpu')):
    """ Using ``torch.load`` and ``dill`` load an object from ``path`` onto ``self.device``.

    Args:
        path (str): Filename to load.

    Returns:
        (any): Object loaded.
    """
    logger.info('Loading: %s' % (path,))

    def remap(storage, loc):
        if 'cuda' in loc and device.type == 'cuda':
            return storage.to(device=device)
        return storage

    torch.load(path, map_location=remap, pickle_module=dill)


def torch_save(path, data, device=torch.device('cpu')):
    """ Using ``torch.save`` and ``dill`` save an object to ``path``.

    Args:
        path (str): Filename to save to.
        data (any): Data to save into file.
    """
    logger.info('Saving: %s' % (path,))
    torch.save(data, path, pickle_module=dill)
