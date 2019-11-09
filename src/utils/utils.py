from contextlib import contextmanager
from functools import wraps
from math import isclose
from pathlib import Path
from threading import Lock
from threading import Timer
from unittest.mock import patch

import hashlib
import logging
import logging.config
import os
import pickle
import pprint
import time

from torch import multiprocessing
from torch.utils import cpp_extension

import torch
import torch.utils.data

from src.environment import NINJA_BUILD_PATH

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
    """ Get a bash friendly string representing the time and process.

    NOTE: This string is optimized for sorting by ordering units of time from largest to smallest.
    NOTE: This string avoids any special bash characters, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    NOTE: `os.getpid` is often used by routines that generate unique identifiers, learn more:
    http://manpages.ubuntu.com/manpages/cosmic/man2/getpid.2.html

    Returns:
        (str)
    """
    return str(time.strftime('DATE=%Y-%m-%d_TIME=%H-%M-%S',
                             time.localtime())).lower() + '_PID=%s' % str(os.getpid())


def torch_cpp_extension_load(name, sources, build_directory=None, **kwargs):
    """ Loads a PyTorch C++ extension just-in-time (JIT).

    This function extends the functionality `torch.utils.cpp_extension.load`, like so:
    - The disk cache is now used between processes and process restarts.
    - The `build_directory` default is updated to be based in `src.environment.NINJA_BUILD_PATH`.
    - Additional logging was added.

    Args:
        See `torch.utils.cpp_extension.load`.

    Returns:
        See `torch.utils.cpp_extension.load`.
    """
    build_directory = NINJA_BUILD_PATH / name if build_directory is None else build_directory
    build_directory.mkdir(exist_ok=True, parents=True)
    logger.info('The cpp extension "%s" will be cached here: `%s`', name, build_directory)

    # NOTE: Cache the version between `Python` restarts (this is not multiprocess safe).
    version_filename = build_directory / 'version.pkl'
    if version_filename.exists():
        version = pickle.loads(version_filename.read_bytes())
        cpp_extension.JIT_EXTENSION_VERSIONER.entries[name] = version
        logger.info('Found cached build of cpp extension "%s" on disk versioned `%s`.', name,
                    version)

    # TODO: Submit a PR to `pytorch` to use a process insensitive hash method.
    with patch('torch.utils._cpp_extension_versioner.update_hash') as patched_update_hash:

        def update_hash(seed, value):
            """ Patched hash such that the hash is the same from process to process. """
            hash_ = int(hashlib.md5(str(value).encode('utf-8')).hexdigest(), 16)
            return seed ^ (hash_ + 0x9e3779b9 + (seed << 6) + (seed >> 2))

        patched_update_hash.side_effect = update_hash

        module = cpp_extension.load(
            name=name, sources=sources, build_directory=build_directory, **kwargs)

    version = cpp_extension.JIT_EXTENSION_VERSIONER.entries[name]
    version_filename.write_bytes(pickle.dumps(version))
    return module
