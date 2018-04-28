import logging
import logging.config
import os
import sys

logger = logging.getLogger(__name__)


def config_logging():
    """ Configure the root logger with basic settings.
    """
    logging.basicConfig(
        format='[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] %(message)s',
        level=logging.INFO,
        stream=sys.stdout)


# http://pytorch.org/docs/master/torch.html?highlight=torch%20load#torch.load
def remap(storage, loc, device):
    """ Remap function to use with ``torch.load`` enabling you to load onto any device.

    Example:
        >>> from functools import partial
        >>> device = 0
        >>> torch.load(path, remap=partial(remap, device=device))
    """
    if 'cuda' in loc and device >= 0:
        return storage.cuda(device=device)
    return storage


def get_root_path():
    """ Get the path to the root directory
    Returns (str):
        Root directory path
    """
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


# Reference:
# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batch(iterable, n=1):
    if not hasattr(iterable, '__len__'):
        # Slow version if len is not defined
        current_batch = []
        for item in iterable:
            current_batch.append(item)
            if len(current_batch) == n:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch
    else:
        # Fast version is len is defined
        for ndx in range(0, len(iterable), n):
            yield iterable[ndx:min(ndx + n, len(iterable))]


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
