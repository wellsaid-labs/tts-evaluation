from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from math import isclose
from pathlib import Path
from threading import Lock
from threading import Timer

import glob
import inspect
import itertools
import logging
import logging.config
import math
import os
import pprint
import random
import re
import statistics
import time
import typing

from torch import multiprocessing

import torch
import torch.utils.data

from src.environment import DISK_CACHE_PATH
from src.utils.disk_cache_ import DiskCache

import src.distributed

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)


def cumulative_split(list_, thresholds, get_value=lambda x: x):
    """ Split list by cumulative sum.

    TODO: Test

    Args:
        list_ (iterable)
        thresholds (list of int)
        get_value (callable, optional): Given a list item, determine the value of the list item.

    Returns:
        (iterable): Slice of the list.
    """
    return_ = []
    threshold = float('inf') if len(thresholds) == 0 else thresholds.pop(0)
    split = []
    total = 0
    for item in list_:
        total += get_value(item)
        if total > threshold:
            return_.append(split)
            split = []
            threshold = float('inf') if len(thresholds) == 0 else thresholds.pop(0)
        split.append(item)
    if len(split) != 0:
        return_.append(split)
    return tuple(return_)


def random_sample(list_, sample_size):
    """ Random sample function like `random.sample` that doesn't error if `list_` is smaller than
        `sample_size`.
    """
    # NOTE: `random.sample` returns an error for a list smaller than `sample_size`
    return random.sample(list_, min(len(list_), sample_size))


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


def mean(list_):
    """ Mean function like `statistics.mean` that does not return an error if `list_` is empty. """
    list_ = list(list_)
    if len(list_) == 0:
        return math.nan
    # NOTE: `statistics.mean` returns an error for an empty list
    return statistics.mean(list_)


def get_chunks(list_, n):
    """ Yield successive `n`-sized chunks from `list_`. """
    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def get_weighted_stdev(tensor, dim=0, mask=None):
    """ Computed the average weighted standard deviation accross some dimesnion.

    This assumes the weights are normalized between zero and one summing up to one on ``dim``.

    Learn more:
        - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
        - https://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean # noqa: E501

    Args:
        tensor (torch.FloatTensor): Some tensor along which to compute the standard deviation.
        dim (int, optional): Dimension of ``tensor`` along which to compute the standard deviation.
        mask (torch.BoolTensor, optional)

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
        mask (torch.BoolTensor, optional): Mask applied to tensor. The shape is the same as
          ``tensor`` without the norm dimension.
        norm (float, optional): The exponent value in the norm formulation.

    Returns:
        (float): The norm over ``dim``, reduced to a scalar average.
    """
    norm = tensor.norm(norm, dim=dim)

    if mask is not None:
        norm = norm.masked_select(mask)

    return norm.mean().item()


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


def save(path, data, overwrite=False):
    """ Using ``torch.save`` to save an object to ``path``.

    Raises:
        (ValueError): If a file already exists at `path`.

    Args:
        path (Path or str): Filename to save to.
        data (any): Data to save into file.
        overwrite (bool, optional): If `True` this allows for `path` to be overwritten.
    """
    if not overwrite and Path(path).exists():
        raise ValueError('A file already exists at %s' % path)

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

    with torch.no_grad():
        yield

    for module, metadata in zip(modules, modules_metadata):
        assert not module.training, 'Invariant failure.'
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
    assert seconds > 0, 'Seconds must be positive.'
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


def sort_together(list_, sort_key, **kwargs):
    """
    Args:
        list_ (list)
        sort_key (list)
        **kwargs: Additional keyword arguments passed to `sorted`.

    Returns:
        list: ``list_`` sorted with the sort key list ``sort_key``.
    """
    return [x for _, x in sorted(zip(sort_key, list_), key=lambda pair: pair[0], **kwargs)]


def cumulative_sum_slice(list_, max_total_value, get_value=lambda x: x):
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


class RepeatTimer(Timer):
    """ Similar to `Timer` but it repeats a function every `self.interval`.
    """

    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


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
    # Learn more about `forkserver` / `spawn` / `fork`:
    # https://github.com/pytorch/pytorch/issues/2245
    # https://codewithoutrules.com/2018/09/04/python-multiprocessing/
    # https://pythontic.com/multiprocessing/multiprocessing/introduction
    pool = multiprocessing.get_context('forkserver').Pool(*args, **kwargs)
    yield pool
    pool.close()  # Marks the pool as closed.
    pool.join()  # Waits for workers to exit.


# TODO: Create a new module for tensor specific operations.


def pad_tensors(input_, pad, dim=0, **kwargs):
    """ Pad a tensor dimension.

    Args:
        input_ (torch.Tensor)
        pad (tuple): The padding to apply.
        dim (int): The dimension to apply padding.
        **kwargs: Keyword arguments passed onto `torch.nn.functional.pad`.

    Returns:
        torch.Tensor
    """
    padding = [(0, 0) for _ in range(input_.dim())]
    padding[dim] = pad
    # NOTE: `torch.nn.functional.pad` accepts the last dimension first.
    padding = flatten(reversed(padding))
    return torch.nn.functional.pad(input_, padding, **kwargs)


def trim_tensors(*args, dim=2):
    """ Trim such that all tensors sizes match on `dim`.

    Args:
        *args (torch.Tensor): A list of tensors.
        dim (int): The dimension to trim.

    Returns:
        *args (torch.Tensor)
    """
    minimum = min(a.shape[dim] for a in args)
    assert all((a.shape[dim] - minimum) % 2 == 0 for a in args), 'Uneven padding'
    return tuple([a.narrow(dim, (a.shape[dim] - minimum) // 2, minimum) for a in args])


class LSTM(torch.nn.LSTM):
    """ LSTM with a trainable initial hidden state.

    Args:
        ... See the documentation for `torch.nn.LSTM`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_directions = 2 if self.bidirectional else 1
        self.initial_hidden_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size))
        self.initial_cell_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size))

    def forward(self, input, hx=None):
        if hx is None:
            batch_size = input.shape[0] if self.batch_first else input.shape[1]
            hx = (self.initial_hidden_state.expand(-1, batch_size, -1).contiguous(),
                  self.initial_cell_state.expand(-1, batch_size, -1).contiguous())
        return super().forward(input, hx=hx)


class LSTMCell(torch.nn.LSTMCell):
    """ LSTMCell with a trainable initial hidden state.

    Args:
        ... See the documentation for `torch.nn.LSTMCell`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_hidden_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))
        self.initial_cell_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))

    def forward(self, input, hx=None):
        if hx is None:
            hx = (self.initial_hidden_state.expand(input.shape[0], -1).contiguous(),
                  self.initial_cell_state.expand(input.shape[0], -1).contiguous())
        return super().forward(input, hx=hx)


class AveragedMetric():
    """ Compute and track the average of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset the metric statistics.

        Returns:
            (float): The average value before reseting.
        """
        if hasattr(self, 'total_value') and hasattr(self, 'total_count') and self.total_count > 0:
            average = self.total_value / self.total_count
        else:
            average = None

        self.last_update_value = 0.0
        self.last_update_count = 0.0
        self.total_value = 0.0
        self.total_count = 0.0

        return average

    def update(self, value, count=1):
        """ Add a new measurement of this metric.

        Args:
            value (number)
            count (int): Number of times to add value / frequency of value.

        Returns:
            AverageMetric: `self`
        """
        if torch.is_tensor(value):
            value = value.item()

        if torch.is_tensor(count):
            count = count.item()

        assert count > 0, '%s count must be a positive number' % count

        self.total_value += value * count
        self.total_count += count
        self.last_update_value = value * count
        self.last_update_count = count

        return self

    def last_update(self):
        """ Get the measurement update.

        Returns:
            (float): The average value of the last measurement.
        """
        if self.last_update_count == 0:
            return None

        return self.last_update_value / self.last_update_count


class DistributedAveragedMetric(AveragedMetric):
    """ Compute and track the average of a metric in a distributed environment.
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()

        if (hasattr(self, 'post_sync_total_value') and hasattr(self, 'post_sync_total_value') and
                self.post_sync_total_count > 0):
            average = self.post_sync_total_value / self.post_sync_total_count
        else:
            average = None

        self.post_sync_total_value = 0
        self.post_sync_total_count = 0

        return average

    def sync(self):
        """ Synchronize measurements from multiple processes.

        Returns:
            AverageMetric: `self`
        """
        last_post_sync_total_value = self.post_sync_total_value
        last_post_sync_total_count = self.post_sync_total_count
        if src.distributed.is_initialized():
            torch_ = torch.cuda if torch.cuda.is_available() else torch
            packed = torch_.FloatTensor([self.total_value, self.total_count])
            torch.distributed.reduce(packed, dst=src.distributed.get_master_rank())
            self.post_sync_total_value, self.post_sync_total_count = tuple(packed.tolist())
        else:
            self.post_sync_total_value = self.total_value
            self.post_sync_total_count = self.total_count
        self.last_update_value = self.post_sync_total_value - last_post_sync_total_value
        self.last_update_count = self.post_sync_total_count - last_post_sync_total_count
        return self


_GetMostRecentReturnType = typing.TypeVar('_GetMostRecentReturnType')


def load_most_recent_file(
        pattern: Path, read: typing.Callable[[Path],
                                             _GetMostRecentReturnType]) -> _GetMostRecentReturnType:
    """ Get the most recently modified file.

    Args:
        pattern: Pattern to search for files.

    Returns:
        The most recent non-corrupted file that is found based on modification time.
    """
    paths = list(glob.iglob(str(pattern), recursive=True))
    if len(paths) == 0:
        raise ValueError(f"No files found with {pattern} pattern.")

    # NOTE: We used modified time for sorting files because creation time is unreliable, learn more:
    # https://stackoverflow.com/questions/237079/how-to-get-file-creation-modification-date-times-in-python/237084
    modified_time = [os.path.getmtime(c) for c in paths]
    assert len(modified_time) == len(
        set(modified_time)), 'Multiple paths found with the same modification time'
    for path in sort_together(paths, modified_time, reverse=True):
        try:
            return read(path)
        except (EOFError, RuntimeError):
            logger.exception(f"Failed to load file at {path}.")
    raise ValueError('Unable to load recent file.')
