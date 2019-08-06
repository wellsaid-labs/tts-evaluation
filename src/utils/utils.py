from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from math import isclose
from pathlib import Path
from threading import Lock
from threading import Timer

import itertools
import logging
import logging.config
import pprint
import time

from torch import multiprocessing
from torchnlp.utils import shuffle as do_deterministic_shuffle

import torch
import torch.utils.data

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)


def get_chunks(list_, n):
    """ Yield successive `n`-sized chunks from `list_`. """
    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def dict_collapse(dict_, keys=[], delimitator='.'):
    """ Recursive ``dict`` collapse a nested dictionary.

    Collapses a multi-level ``dict`` into a single level dict by merging the strings with a
    delimitator.

    Args:
        dict_ (dict)
        keys (list, optional): Base keys.
        delimitator (str, optional): Delimitator used to join keys.

    Returns:
        (dict): Collapsed ``dict``.
    """
    ret_ = {}
    for key in dict_:
        if isinstance(dict_[key], dict):
            ret_.update(dict_collapse(dict_[key], keys + [key]))
        else:
            ret_[delimitator.join(keys + [key])] = dict_[key]
    return ret_


def get_weighted_stdev(tensor, dim=0, mask=None):
    """ Computed the average weighted standard deviation accross some dimesnion.

    This assumes the weights are normalized between zero and one summing up to one on ``dim``.

    Learn more:
        - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
        - https://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean # noqa: E501

    Args:
        tensor (torch.FloatTensor): Some tensor along which to compute the standard deviation.
        dim (int, optional): Dimension of ``tensor`` along which to compute the standard deviation.
        mask (torch.FloatTensor, optional)

    Returns:
        (float): Returns the average weighted standard deviation of each row of the input tensor in
            the given dimension ``dim``.
    """
    # Expects normalized weightes total of 0, 1 to ensure correct variance decisions
    assert all([isclose(value, 1, abs_tol=1e-3) for value in tensor.sum(dim=dim).view(-1).tolist()])

    # Create position matrix where the index is the position and the value is the weight
    indicies = torch.arange(0, tensor.shape[dim], dtype=tensor.dtype, device=tensor.device)
    shape = [1] * len(tensor.shape)
    shape[dim] = tensor.shape[dim]
    indicies = indicies.view(*shape).expand_as(tensor).float()

    weighted_mean = (indicies * tensor).sum(dim=dim) / tensor.sum(dim=dim)
    weighted_variance = ((indicies - weighted_mean.unsqueeze(dim=dim))**2 * tensor).sum(dim=dim)
    weighted_standard_deviation = weighted_variance**0.5

    assert not torch.isnan(weighted_standard_deviation.min()), 'NaN detected.'

    if mask is not None:
        weighted_standard_deviation = weighted_standard_deviation.masked_select(mask)

    return weighted_standard_deviation.mean().item()


def get_average_norm(tensor, dim=0, mask=None, norm=2):
    """ The average norm accross some ``dim``.

    Args:
        tensor (torch.FloatTensor)
        dim (int, optional)
        mask (torch.FloatTensor, optional): Mask applied to tensor. The shape is the same as
          ``tensor`` without the norm dimension.
        norm (float, optional): The exponent value in the norm formulation.

    Returns:
        (float): The norm over ``dim``, reduced to a scalar average.
    """
    norm = tensor.norm(norm, dim=dim)

    if mask is not None:
        norm = norm.masked_select(mask)

    return norm.mean().item()


def get_total_parameters(model):
    """ Return the total number of trainable parameters in ``model``.

    Args:
        model (torch.nn.Module)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load(path, device=torch.device('cpu')):
    """ Using ``torch.load`` load an object from ``path`` onto ``self.device``.

    Args:
        path (Path or str): Filename to load.
        device (torch.device, optional): Device to load onto.

    Returns:
        (any): Object loaded.
    """
    logger.info('Loading: %s' % (path,))

    assert Path(path).is_file(), 'Path (%s) must point to a file' % str(path)

    def remap(storage, loc):
        if 'cuda' in loc and device.type == 'cuda':
            return storage.cuda(device=device.index)
        return storage

    return torch.load(str(path), map_location=remap)


def save(path, data):
    """ Using ``torch.save`` to save an object to ``path``.

    Args:
        path (Path or str): Filename to save to.
        data (any): Data to save into file.
    """
    torch.save(data, str(path))
    logger.info('Saved: %s', str(path))


""" Flatten a list of lists into a list. """
flatten = lambda l: [item for sublist in l for item in sublist]


def flatten_parameters(model):
    """ Apply ``flatten_parameters`` to ``model``.

    Args:
        model (torch.nn.Module)
    """
    return model.apply(lambda m: m.flatten_parameters()
                       if hasattr(m, 'flatten_parameters') else None)


def maybe_get_model_devices(model):
    """ Try to get all the devices the model is running on.

    As of April 2019, for this repositories usage, this should work.
    """
    # Learn more:
    # https://github.com/pytorch/pytorch/issues/7460
    # https://discuss.pytorch.org/t/which-device-is-model-tensor-stored-on/4908
    # https://github.com/pytorch/pytorch/issues/12460

    # NOTE: ``torch.nn.Module.to`` changes the parameters and buffers of a module.
    module_devices = [t.device for t in model.parameters()]
    if hasattr(model, 'buffers'):
        module_devices += [t.device for t in model.buffers()]
    return list(set(module_devices))


@contextmanager
def evaluate(*modules, device=None):
    """ Context manager for ``torch.nn.Module`` evaluatation mode.

    Args:
        *modules (torch.nn.Module)
        device (torch.device)

    Returns: None
    """
    assert all(isinstance(m, torch.nn.Module) for m in modules), 'Every argument must be a module.'

    modules_metadata = []
    for module in modules:
        module_device = None
        if device is not None:
            module_devices = maybe_get_model_devices(module)
            if len(module_devices) > 1:
                raise TypeError('Context manager supports models on a single device')
            module_device = module_devices[0]
            module.to(device)

        modules_metadata.append({'is_train': module.training, 'last_device': module_device})
        module.train(mode=False)

    with torch.autograd.no_grad():
        yield

    for module, metadata in zip(modules, modules_metadata):
        module.train(mode=metadata['is_train'])
        if metadata['last_device'] is not None:
            module.to(metadata['last_device'])


def identity(x):
    return x


def seconds_to_string(seconds):
    """ Rewrite seconds as a string.

    Example:
        >>> seconds_to_string(123)
        '2m 3s 0ms'
        >>> seconds_to_string(123.100)
        '2m 3s 100ms'
        >>> seconds_to_string(86400)
        '1d 0h 0m 0s 0ms'
        >>> seconds_to_string(3600)
        '1h 0m 0s 0ms'

    Args:
        seconds (int)

    Returns
        str
    """
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd %dh %dm %ds %dms' % (days, hours, minutes, seconds, milliseconds)
    elif hours > 0:
        return '%dh %dm %ds %dms' % (hours, minutes, seconds, milliseconds)
    elif minutes > 0:
        return '%dm %ds %dms' % (minutes, seconds, milliseconds)
    elif seconds > 0:
        return '%ds %dms' % (seconds, milliseconds)
    else:
        return '%dms' % (milliseconds)


def log_runtime(function):
    """ Decorator for measuring the execution time of a function.
    """

    @wraps(function)
    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        elapsed = seconds_to_string(time.time() - start)
        logger.info('`%s` ran for %s', function.__qualname__, elapsed)
        return result

    return decorator


def sort_together(list_, sort_key):
    """
    Args:
        list_ (list)
        sort_key (list)

    Returns:
        list: ``list_`` sorted with the sort key list ``sort_key``.
    """
    return [x for _, x in sorted(zip(sort_key, list_), key=lambda pair: pair[0])]


def split_list(list_, splits):
    """ Split ``list_`` using the ``splits`` ratio.

    Args:
        list_ (list): List to split.
        splits (tuple): Tuple of decimals determining list splits summing up to 1.0.

    Returns:
        (list): Splits of the list.

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> split_list(dataset, splits=(.6, .2, .2))
        [[1, 2, 3], [4], [5]]
    """
    assert sum(splits) == 1, 'Splits must sum to 1.0'
    splits = [round(s * len(list_)) for s in splits]
    lists = []
    for split in splits[:-1]:
        lists.append(list_[:split])
        list_ = list_[split:]
    lists.append(list_)
    return lists


def slice_by_cumulative_sum(list_, max_total_value, get_value=lambda x: x):
    """ Get slice of a list such that the cumulative sum is less than or equal to ``max_total``.

    Args:
        list_ (iterable)
        max_total_value (float): Maximum cumlative sum of the list slice.
        get_value (callable, optional): Given a list item, determine the value of the list item.

    Returns:
        (iterable): Slice of the list.
    """
    return_ = []
    total = 0
    for item in list_:
        total += get_value(item)
        if total > max_total_value:
            return return_
        else:
            return_.append(item)
    return return_


@log_runtime
def balance_list(list_, get_class=identity, get_weight=lambda x: 1, random_seed=None):
    """ Returns a random subsample of the list such that each class has equal representation.

    Args:
        list_ (iterable)
        get_class (callable, optional): Given a list item, returns a class.
        get_weight (callable, optional): Given a list item, determine the weight of the list item.
        random_seed (int, optiona): If provided the shuffle is deterministic based on the seed
            instead of the global generator state.

    Returns:
        (iterable): Subsample of ``list_`` such that each class has the same number of samples.
    """
    if len(list_) == 0:
        return list_

    split = defaultdict(list)

    # Learn more:
    # https://stackoverflow.com/questions/16270374/how-to-make-a-shallow-copy-of-a-list-in-python
    list_ = list_[:]
    do_deterministic_shuffle(list_, random_seed=random_seed)
    for item in list_:
        split[get_class(item)].append(item)

    partitions = {k: sum([get_weight(i) for i in v]) for k, v in split.items()}
    min_weight = min(partitions.values())

    logger.info('Balanced distribution from\n%s\nto equal partitions of weight %d.',
                pprint.pformat(partitions), min_weight)

    subsample = [
        slice_by_cumulative_sum(l, max_total_value=min_weight, get_value=get_weight)
        for l in split.values()
    ]
    subsample = list(itertools.chain(*subsample))  # Flatten list
    do_deterministic_shuffle(subsample, random_seed=random_seed)
    return subsample


class ResetableTimer(Timer):
    """ Similar to `Timer` but with an included `reset` method.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continue_waiting = True
        self.reset_lock = Lock()

    def run(self):
        while self.continue_waiting:
            with self.reset_lock:  # TODO: Test reset lock with various race conditions.
                self.continue_waiting = False
            self.finished.wait(self.interval)

        if not self.finished.isSet():
            self.function(*self.args, **self.kwargs)
        self.finished.set()

    def reset(self):
        """ Reset the timer.

        NOTE: The `Timer` executes an action only once; therefore, the timer can be reset up until
        the action has executed. Following the action execution the timer thread exits.
        """
        with self.reset_lock:
            self.continue_waiting = True
            self.finished.set()
            self.finished.clear()


@contextmanager
def Pool(*args, **kwargs):
    """ Alternative implementation of a `Pool` context manager. The original `multiprocessing.Pool`
    context manager calls `terminate` rather than `close` followed by `join`.
    """
    # Learn more: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.close()  # Marks the pool as closed.
    pool.join()  # Waits for workers to exit.


def bash_time_label():
    """ Get a bash friendly string representing the time.

    NOTE: This string is optimized for sorting by ordering units of time from largest to smallest.
    NOTE: This string avoids any special bash characters, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments

    Returns:
        (str)
    """
    return str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())).lower()
