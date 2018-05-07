import logging
import logging.config
import os

import matplotlib.pyplot as plt
import torch

from torchnlp.text_encoders import PADDING_INDEX

logger = logging.getLogger(__name__)


def pad_tensor(tensor, length, padding_index=PADDING_INDEX):
    """ Pad a ``tensor`` to ``length`` with ``padding_index``.

    Args:
        tensor (torch.Tensor [n, *]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, *]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    assert n_padding >= 0
    if n_padding == 0:
        return tensor
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return torch.cat((tensor, padding), dim=0)


def pad_batch(batch, padding_index=PADDING_INDEX):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.

    Returns
        torch.Tensor, list of int: Padded tensors and original lengths of tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    padded = torch.stack(padded, dim=0).contiguous()
    return padded, lengths


def get_root_path():
    """ Get the path to the root directory

    Returns (str):
        Root directory path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


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
        title (str): Title of the plot.

    Returns:
        None
    """
    assert '.png' in filename.lower(), "Filename saves in PNG format"

    plt.style.use('ggplot')
    figure, axis = plt.subplots()
    im = axis.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
    figure.colorbar(im, ax=axis, orientation='horizontal')
    plt.xlabel('Encoder timestep')
    plt.title(title, y=1.1)
    plt.ylabel('Decoder timestep')
    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close()


class Average(object):
    """ Average metric helps track and compute an average """

    def __init__(self):
        self.total = 0
        self.num_values = 0

    def add(self, value, num_values=1):
        """ Add values to average metric to track.

        Args:
            value (int): Value to add to the total
            num_values (int): Number of values considered
        """
        self.total += value
        self.num_values += num_values

    def get(self):
        """ Get the average.

        Returns:
            (float): average value
        """
        return self.total / self.num_values
