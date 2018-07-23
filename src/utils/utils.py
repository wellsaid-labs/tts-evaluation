import matplotlib

matplotlib.use('Agg', warn=False)

import ast
import glob
import logging
import logging.config
import os

from torchnlp.utils import shuffle as do_deterministic_shuffle

import dill
import torch

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)

# Repository root path
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(dataset, splits, deterministic_shuffle=True, random_seed=123):
    """
    Args:
        dataset (list): Dataset to split.
        splits (tuple): Tuple of percentages determining dataset splits.
        shuffle (bool, optional): If ``True`` determinisitically shuffle the dataset before
            splitting.

    Returns:
        (list): splits of the dataset

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> splits = (.6, .2, .2)
        >>> split_dataset(dataset, splits)
        [[1, 2, 3], [4], [5]]
    """
    if deterministic_shuffle:
        do_deterministic_shuffle(dataset, random_seed=random_seed)
    assert sum(splits) == 1, 'Splits must sum to 100%'
    splits = [round(s * len(dataset)) for s in splits]
    datasets = []
    for split in splits:
        datasets.append(dataset[:split])
        dataset = dataset[split:]
    return datasets


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
            return storage.cuda(device=device.index)
        return storage

    return torch.load(path, map_location=remap, pickle_module=dill)


def torch_save(path, data):
    """ Using ``torch.save`` and ``dill`` save an object to ``path``.

    Args:
        path (str): Filename to save to.
        data (any): Data to save into file.
    """
    torch.save(data, path, pickle_module=dill)
    logger.info('Saved: %s' % (path,))


def get_filename_table(directory, prefixes=[], extension=''):
    """ Get a table of filenames; such that every row has multiple filenames of different prefixes.

    Notes:
        * Filenames are aligned via string sorting.
        * The table must be full; therefore, all filenames associated with a prefix must have an
          equal number of files as every other prefix.

    Args:
        directory (str): Path to a directory.
        prefixes (str): Prefixes to load.
        extension (str): Filename extensions to load.

    Returns:
        (list of dict): List of dictionaries where prefixes are the key names.
    """
    rows = []
    for prefix in prefixes:
        # Get filenames with associated prefixes
        filenames = []
        for filename in os.listdir(directory):
            # TODO: Rename prefix because it does not look at directly the beginning of the filename
            if filename.endswith(extension) and prefix in filename:
                filenames.append(os.path.join(directory, filename))

        # Sorted to align with other prefixes
        filenames = sorted(filenames)

        # Add to rows
        if len(rows) == 0:
            rows = [{prefix: filename} for filename in filenames]
        else:
            assert len(filenames) == len(rows), "Each row must have an associated filename."
            for i, filename in enumerate(filenames):
                rows[i][prefix] = filename

    return rows


def parse_hparam_args(hparam_args):
    """ Parse CLI arguments like ``['--torch.optim.adam.Adam.__init__.lr 0.1',]`` to :class:`dict`.

    Args:
        hparams_args (list of str): List of CLI arguments

    Returns
        (dict): Dictionary of arguments.
    """

    def to_literal(value):
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        return value

    return_ = {}

    for hparam in hparam_args:
        assert '--' in hparam, 'Hparam argument (%s) must have a double flag' % hparam
        split = hparam.replace('=', ' ').split()
        assert len(split) == 2, 'Hparam %s must be equal to one value' % split
        key, value = tuple(split)
        assert key[:2] == '--', 'Hparam argument (%s) must have a double flag' % hparam
        key = key[2:]  # Remove flag
        value = to_literal(value)
        return_[key] = value

    return return_


@configurable
def split_signal(signal, bits=16):
    """ Compute the coarse and fine components of the signal.

    Args:
        signal (torch.FloatTensor [signal_length]): Signal with values ranging from [-1, 1]
        bits (int): Number of bits to encode signal in.

    Returns:
        coarse (torch.FloatTensor [signal_length]): Top bits of the signal.
        fine (torch.FloatTensor [signal_length]): Bottom bits of the signal.
    """
    assert torch.min(signal) >= -1.0 and torch.max(signal) <= 1.0
    assert (bits %
            2 == 0), 'To support an even split between coarse and fine, use an even number of bits'
    range_ = int((2**(bits - 1)))
    signal = torch.round(signal * range_)
    signal = torch.clamp(signal, -1 * range_, range_ - 1)
    unsigned = signal + range_  # Move range minimum to 0
    bins = int(2**(bits / 2))
    coarse = torch.floor(unsigned / bins)
    fine = unsigned % bins
    return coarse, fine


@configurable
def combine_signal(coarse, fine, bits=16):
    """ Compute the coarse and fine components of the signal.

    Args:
        coarse (torch.FloatTensor [signal_length]): Top bits of the signal.
        fine (torch.FloatTensor [signal_length]): Bottom bits of the signal.
        bits (int): Number of bits to encode signal in.

    Returns:
        signal (torch.FloatTensor [signal_length]): Signal with values ranging from [-1, 1]
    """
    bins = int(2**(bits / 2))
    assert torch.min(coarse) >= 0 and torch.max(coarse) < bins
    assert torch.min(fine) >= 0 and torch.max(fine) < bins
    signal = coarse * bins + fine - 2**(bits - 1)
    return signal.float() / 2**(bits - 1)


def load_most_recent_checkpoint(pattern, load_checkpoint=torch_load):
    """ Load the most recent checkpoint from ``root``.

    Args:
        pattern (str): Pattern to glob recursively for checkpoints.
        load_checkpoint (callable): Callable to load checkpoint.

    Returns:
        (any): Return value of ``load_checkpoint`` callable or None.
        (str): Path of loaded checkpoint or None.
    """
    checkpoints = list(glob.iglob(pattern, recursive=True))
    if len(checkpoints) == 0:
        # NOTE: Using print because this runs before logger is setup typically
        print('No checkpoints found in %s' % pattern)
        return None, None

    checkpoints = sorted(list(checkpoints), key=os.path.getctime, reverse=True)
    for checkpoint in checkpoints:
        try:
            return load_checkpoint(checkpoint), checkpoint
        except (EOFError, RuntimeError):
            print('Failed to load checkpoint %s' % checkpoint)
            pass
