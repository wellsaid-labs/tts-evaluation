from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from functools import wraps
from math import isclose
from pathlib import Path

import ast
import atexit
import itertools
import logging
import logging.config
import os
import pprint
import subprocess
import sys
import time

from tqdm import tqdm

import torch
import torch.utils.data

import src.distributed

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)

ROOT_PATH = Path(__file__).parents[2].resolve()  # Repository root path

TTS_DISK_CACHE_NAME = '.tts_cache'  # Directory name for any disk cache's created by this repository


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


def set_basic_logging_config(device=None, **kwargs):
    """ Set up basic logging handlers.

    Args:
        device (torch.device, optional): Device used to prefix the logs.
        **kwargs: Additional key word arguments passed to ``logging.basicConfig``.
    """
    if device is None:
        device_prefix = ''
    else:
        device_prefix = '[%s]' % device

    logging.basicConfig(
        level=logging.INFO,
        format='\033[90m[%(asctime)s]' + device_prefix +
        '[%(name)s][%(levelname)s]\033[0m %(message)s',
        **kwargs)


def duplicate_stream(from_, to):
    """ Writes any messages to file object ``from_`` in file object ``to`` as well.

    Note:
        With the various references below, we were unable to add C support. Find more details
        here: https://travis-ci.com/AI2Incubator/WellSaidLabs/jobs/152504931

    Learn more:
        - https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
        - https://stackoverflow.com/questions/17942874/stdout-redirection-with-ctypes
        - https://gist.github.com/denilsonsa/9c8f5c44bf2038fd000f
        - https://github.com/IDSIA/sacred/blob/master/sacred/stdout_capturing.py
        - http://stackoverflow.com/a/651718/1388435
        - http://stackoverflow.com/a/22434262/1388435

    Args:
        from_ (file object)
        to (str or Path): Filename to write in.

    Returns:
        callable: Executing the callable stops the duplication.
    """
    from_.flush()

    to = Path(to)
    to.touch()

    # Keep a file descriptor open to the original file object
    original_fileno = os.dup(from_.fileno())
    tee = subprocess.Popen(['tee', '-a', str(to)], stdin=subprocess.PIPE)
    time.sleep(0.01)  # HACK: ``tee`` needs time to open
    os.dup2(tee.stdin.fileno(), from_.fileno())

    def _clean_up():
        """ Clean up called during exit or by user. """
        # (High Level) Ensure ``from_`` flushes before tee is closed
        from_.flush()

        # Tee Flush / close / terminate
        tee.stdin.close()
        tee.terminate()
        tee.wait()

        # Reset ``from_``
        os.dup2(original_fileno, from_.fileno())
        os.close(original_fileno)

    def stop():
        """ Stop duplication early before the program exits. """
        atexit.unregister(_clean_up)
        _clean_up()

    atexit.register(_clean_up)
    return stop


def record_stream(directory, stdout_log_filename='stdout.log', stderr_log_filename='stderr.log'):
    """ Record output ``sys.stdout`` and ``sys.stderr`` to log files.

    Args:
        directory (Path or str): Directory to save log files in.
        stdout_log_filename (str, optional)
        stderr_log_filename (str, optional)
    """
    directory = Path(directory)
    duplicate_stream(sys.stdout, directory / stdout_log_filename)
    duplicate_stream(sys.stderr, directory / stderr_log_filename)


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


""" Flatten a list of lists into a list. """
flatten = lambda l: [item for sublist in l for item in sublist]


def flatten_parameters(model):
    """ Apply ``flatten_parameters`` to ``model``.

    Args:
        model (torch.nn.Module)
    """
    return model.apply(
        lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)


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


def _get_tensor_dim_length(tensor, dim):
    """
    Args:
        tensor (OnDiskTensor or torch.Tensor or np.ndarray)

    Returns:
        (int): Length of ``dim`` in ``tensor``.
    """
    return tensor.shape[dim]


@log_runtime
def get_tensors_dim_length(tensors, dim=0, use_tqdm=False):
    """ Get the length of ``dim`` for each tensor in ``tensors``.

    Args:
        tensors (iterable of torch.Tensor)
        dim (int, optional)
        use_tqdm (bool, optional)

    Returns:
        (list of int): The length of ``dim`` for each tensor.
    """
    with ThreadPoolExecutor() as pool:
        iterator = pool.map(partial(_get_tensor_dim_length, dim=dim), tensors)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(tensors))
        lengths = list(iterator)
    return lengths


def sort_together(list_, sort_key):
    """
    Args:
        list_ (list)
        sort_key (list)

    Returns:
        list: ``list_`` sorted with the sort key list ``sort_key``.
    """
    return [x for _, x in sorted(zip(sort_key, list_), key=lambda pair: pair[0])]


def assert_enough_disk_space(min_space=0.2):
    """ Check if there is enough disk space.

    Args:
        min_space (float): Minimum percentage of free disk space.
    """
    st = os.statvfs(ROOT_PATH)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    available = free / total
    assert available > min_space, 'There is not enough available (%f < %f) disk space.' % (
        available, min_space)


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
    split = defaultdict(list)

    # Learn more:
    # https://stackoverflow.com/questions/16270374/how-to-make-a-shallow-copy-of-a-list-in-python
    list_ = list_[:]
    src.distributed.random_shuffle(list_, random_seed=random_seed)
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
    src.distributed.random_shuffle(subsample, random_seed=random_seed)
    return subsample
