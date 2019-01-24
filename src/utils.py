from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import lru_cache
from functools import wraps
from math import isclose
from pathlib import Path

import ast
import atexit
import glob
import logging
import logging.config
import math
import os
import random
import subprocess
import sys
import time

from third_party.data_loader import DataLoader
from torch.multiprocessing import cpu_count
from torch.utils import data
from torch.utils.data.sampler import Sampler
from torchnlp.text_encoders import PADDING_INDEX
from torchnlp.utils import pad_tensor
from tqdm import tqdm

import torch
import numpy as np

from src.hparams import configurable
from src.hparams import ConfiguredArg

logger = logging.getLogger(__name__)

# Repository root path
ROOT_PATH = Path(__file__).parent.parent.resolve()


def dict_collapse(dict_, keys=[], delimitator='.'):
    """ Recursive `dict` collapse.

    Collapses a multi-level `dict` into a single level dict by merging the strings with a
    delimitator.

    Args:
        dict_ (dict)
        keys (list, optional): Base keys.
        delimitator (str, optional): Delimitator used to join keys.

    Returns:
        (dict): Collapsed `dict`.
    """
    ret_ = {}
    for key in dict_:
        if isinstance(dict_[key], dict):
            ret_.update(dict_collapse(dict_[key], keys + [key]))
        else:
            ret_[delimitator.join(keys + [key])] = dict_[key]
    return ret_


def set_basic_logging_config(device=None):
    """ Set up basic logging handlers. """
    if device is None:
        device_str = ''
    else:
        device_str = '[%s]' % device

    logging.basicConfig(
        level=logging.INFO,
        format='\033[90m[%(asctime)s]' + device_str +
        '[%(name)s][%(levelname)s]\033[0m %(message)s')


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
        (callable): Stop the duplication.
    """
    from_.flush()

    # Keep a file descriptor open to the original file object
    original_fileno = os.dup(from_.fileno())
    tee = subprocess.Popen(['tee', str(to)], stdin=subprocess.PIPE)
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
    """ Record output ``sys.stdout`` and ``sys.stderr`` to log files

    Args:
        directory (Path or str): Directory to save log files in.
        stdout_log_filename (str, optional)
        stderr_log_filename (str, optional)
    """
    directory = Path(directory)

    duplicate_stream(sys.stdout, directory / stdout_log_filename)
    duplicate_stream(sys.stderr, directory / stderr_log_filename)


def chunks(list_, n):
    """ Yield successive n-sized chunks from list. """
    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def get_weighted_stdev(tensor, dim=0, mask=None):
    """ Computed the average weighted standard deviation accross some dimesnion.

    We assume the weights are normalized between zero and one summing up to one on ``dim``.

    Learn more:
        - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
        - https://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean # noqa: E501

    Args:
        tensor (torch.FloatTensor): Some tensor along which to compute the standard deviation.
        dim (int): Dimension of ``tensor`` along which to compute the standard deviation.
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
        dim (int)
        mask (torch.FloatTensor, optional): Tensor minus the norm dimension.
        norm (float, optional): The exponent value in the norm formulation.

    Returns:
        (float): The norm over ``dim``, reduced to a scalar average.
    """
    norm = tensor.norm(norm, dim=dim)

    if mask is not None:
        norm = norm.masked_select(mask)

    return norm.mean().item()


class ExponentiallyWeightedMovingAverage():
    """ Keep track of an exponentially weighted mean and standard deviation every step.

    Args:
       beta (float): Beta used to weight the exponential mean and standard deviation.
    """

    def __init__(self, beta=0.99):

        self._average = 0.0
        self._variance = 0.0
        self.beta = beta
        self.step_counter = 0

    def step(self, value):
        """
        Args:
            value (float): Next value to take into account.

        Returns:
            average (float): Moving average.
            standard_deviation (float): Moving standard deviation.
        """
        self.step_counter += 1

        self._average = self.beta * self._average + (1 - self.beta) * value
        # The initial 0.0 variance and 0.0 average values introduce bias that is corrected.
        # LEARN MORE:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        average_bias_corrected = self._average / (1 - self.beta**(self.step_counter))

        # TODO: Double check the math we might not be debiasing this correctly assuming
        # "Reliability weights"
        # LEARN MORE:
        # http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
        self._variance = self.beta * self._variance + (1 - self.beta) * (
            value - average_bias_corrected)**2
        variance_bias_corrected = self._variance / (1 - self.beta**(self.step_counter))

        return average_bias_corrected, math.sqrt(variance_bias_corrected)


class AnomalyDetector(ExponentiallyWeightedMovingAverage):
    """ Detect anomalies at every step with a moving average and standard deviation.

    Args:
       beta (float): Beta used to weight the exponential mean and standard deviation.
       sigma (float): Number of standard deviations in order to classify as an anomaly.
       type_ (str): Detect anomalies that are too 'high', too 'low', or 'both'.
       eps (float, optional): Minimum difference to be considered an anomaly used for numerical
          stability.
       min_steps (int, optional): Minimum number of steps to wait before detecting anomalies.
    """

    TYPE_HIGH = 'high'
    TYPE_LOW = 'low'
    TYPE_BOTH = 'both'

    # Below 10 samples there can be significant bias in the variance estimation causing it
    # to be underestimated.
    # LEARN MORE: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    @configurable
    def __init__(self,
                 beta=ConfiguredArg(),
                 sigma=ConfiguredArg(),
                 type_=ConfiguredArg(),
                 eps=10**-6,
                 min_steps=10):
        super().__init__(beta=beta)
        self.sigma = sigma
        self.last_standard_deviation = 0.0
        self.last_average = 0.0
        self.min_steps = min_steps
        self.eps = eps
        self.anomaly_counter = 0
        self.type = type_

    @property
    def max_deviation(self):
        """ Maximum value can deviate from ``last_average`` before being considered an anomaly. """
        return self.sigma * self.last_standard_deviation + self.eps

    def _is_anomaly(self, value):
        """ Check if ``value`` is an anomaly.

        Args:
            value (float)

        Returns:
            (bool): If ``value`` is an anomaly.
        """
        if self.step_counter + 1 < self.min_steps:
            return False

        if not np.isfinite(value):
            return True

        if self.type == self.TYPE_HIGH and value - self.last_average > self.max_deviation:
            return True

        if self.type == self.TYPE_LOW and self.last_average - value > self.max_deviation:
            return True

        if self.type == self.TYPE_BOTH and abs(value - self.last_average) > self.max_deviation:
            return True

        return False

    def step(self, value):
        """ Check if ``value`` is an anomaly whilst updating stats for the next step.

        Args:
            value (float)

        Returns:
            (bool): If ``value`` is an anomaly.
        """
        is_anomaly = self._is_anomaly(value)
        if is_anomaly:
            self.anomaly_counter += 1
        else:
            self.last_average, self.last_standard_deviation = super().step(value)
        return is_anomaly


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load(path, device=torch.device('cpu')):
    """ Using ``torch.load`` and ``dill`` load an object from ``path`` onto ``self.device``.

    Args:
        path (Path or str): Filename to load.

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
    """ Using ``torch.save`` and ``dill`` save an object to ``path``.

    Args:
        path (Path or str): Filename to save to.
        data (any): Data to save into file.
    """
    torch.save(data, str(path))
    logger.info('Saved: %s' % (path,))


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


class Checkpoint():
    """ Model checkpoint object
    Args:
        directory (Path or str): Directory where to save the checkpoint.
        model (torch.nn.Module): Model to train and evaluate.
        step (int): Starting step, useful warm starts (i.e. checkpoints).
        **kwargs (dict, optional): Any other checkpoint attributes.
    """

    def __init__(self, directory, step, model=None, **kwargs):
        self.directory = Path(directory)
        self.step = step
        self.model = model

        for key, value in kwargs.items():
            setattr(self, key, value)

    def flatten_parameters(self, model):
        model.apply(lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)

    @classmethod
    def from_path(class_, path, device=torch.device('cpu')):
        """ Load a checkpoint from path.
        Args:
            path (Path or str or None): Path to a checkpoint to load.
            device (torch.device, optional): Device to load checkpoint onto.
        Returns:
            checkpoint (Checkpoint or None): Loaded checkpoint or None.
        """
        if path is None:
            return None

        torch.nn.Module.dump_patches = True  # NOTE: Dump code that's changed since the checkpoint.
        instance = load(str(path), device=device)
        setattr(instance, 'path', Path(path))
        instance.flatten_parameters(instance.model)
        logger.info('Loaded checkpoint at step %d from %s with model:\n%s', instance.step,
                    instance.path, instance.model)
        return instance

    @classmethod
    def most_recent(class_, pattern, **kwargs):
        """ Load the most recent checkpoint from ``root``.

        Args:
            pattern (str): Pattern to glob recursively for checkpoints.
            **kwargs (dict, optional): Any additional parameters to pass to ``class.from_path``

        Returns:
            (Checkpoint or None): The most recent checkpoint found or None if none is found.
        """
        checkpoints = list(glob.iglob(str(pattern), recursive=True))
        if len(checkpoints) == 0:
            logger.warning('No checkpoints found in %s' % pattern)
            return None

        checkpoints = sorted(list(checkpoints), key=os.path.getctime, reverse=True)
        for path in checkpoints:
            try:
                return class_.from_path(path, **kwargs)
            except (EOFError, RuntimeError):
                logger.exception('Failed to load checkpoint %s' % path)
                pass

        return None

    def save(self):
        """ Save a checkpoint. """
        name = 'step_{}.pt'.format(self.step)
        filename = Path(self.directory) / name
        self.path = filename
        save(filename, self)
        return self.path


class RandomSampler(Sampler):
    """ Samples elements randomly, without replacement.

    Args:
        data (list): Data to sample from.
        random (random.Random, optional): Random number generator to sample data.
    """

    def __init__(self, data, random=random):
        self.data = data
        self.random = random

    def __iter__(self):
        indicies = list(range(len(self.data)))
        self.random.shuffle(indicies)
        return iter(indicies)

    def __len__(self):
        return len(self.data)


@contextmanager
def evaluate(*modules, device=None):
    """ Temporarily switch to evaluation mode for a ``torch.nn.Module``.

    Args:
        *modules (torch.nn.Module)
        device (torch.device)

    Returns: None
    """
    assert all(isinstance(m, torch.nn.Module) for m in modules), 'Every argument must be a module.'

    modules_metadata = []
    for module in modules:
        # torch.nn.Module.to changes the parameters and buffers of a module
        module_devices = [t.device for t in module.parameters()]
        if hasattr(module, 'buffers'):
            module_devices += [t.device for t in module.buffers()]
        module_devices = list(set(module_devices))
        # For switching devices, the implementation only supports modules on a single device.
        assert len(module_devices) <= 1
        module_device = module_devices[0] if len(
            module_devices) == 1 and device is not None else None

        modules_metadata.append({'is_train': module.training, 'last_device': module_device})
        module.train(mode=False)
        if device is not None:
            module.to(device)

    with torch.autograd.no_grad():
        yield

    for module, metadata in zip(modules, modules_metadata):
        module.train(mode=metadata['is_train'])
        if metadata['last_device'] is not None:
            module.to(metadata['last_device'])


def is_namedtuple(object_):
    return hasattr(object_, '_asdict') and isinstance(object_, tuple)


def pad_batch(batch, padding_index=PADDING_INDEX, dim=0):
    """ Pad a :class:`list` of ``tensors`` (``batch``) with ``padding_index``.

    Args:
        batch (:class:`list` of :class:`torch.Tensor`): Batch of tensors to pad.
        padding_index (int, optional): Index to pad tensors with.
        dim (int, optional): Dimension on to which to concatenate the batch of tensors.

    Returns
        torch.Tensor, torch.Tensor: Padded tensors and original lengths of tensors.
    """
    lengths = [tensor.shape[0] for tensor in batch]
    max_len = max(lengths)
    padded = [pad_tensor(tensor, max_len, padding_index) for tensor in batch]
    lengths = torch.tensor(lengths)
    padded = torch.stack(padded, dim=dim).contiguous()
    for _ in range(dim):
        lengths = lengths.unsqueeze(0)
    return padded, lengths


def collate_tensors(batch, stack_tensors=torch.stack):
    """ Collate a list of type k (dict, namedtuple, list, etc.) with tensors.

    TODO: Contribute this to PyTorch-NLP as a generic sequence collate function.

    NOTE:
        * For a none-tensors, the batch is returned.

    Args:
        batch (list of k): List of rows of type ``k``
        stack_tensors (callable): Function to stack tensors into a batch.

    Returns:
        k: Collated batch of type ``k``.
    """
    if all([torch.is_tensor(b) for b in batch]):
        return stack_tensors(batch)
    if (all([isinstance(b, dict) for b in batch]) and
            all([b.keys() == batch[0].keys() for b in batch])):
        return {key: collate_tensors([d[key] for d in batch], stack_tensors) for key in batch[0]}
    elif all([is_namedtuple(b) for b in batch]):  # Handle ``namedtuple``
        return batch[0].__class__(**collate_tensors([b._asdict() for b in batch], stack_tensors))
    elif all([isinstance(b, list) for b in batch]):
        # Handle list of lists such each list has some column to be batched, similar to:
        # [['a', 'b'], ['a', 'b']] → [['a', 'a'], ['b', 'b']]
        transposed = zip(*batch)
        return [collate_tensors(samples, stack_tensors) for samples in transposed]
    else:
        return batch


def tensors_to(tensors, **kwargs):
    """ Move a generic data structure of tensors to another device

    TODO: Contribute this to PyTorch-NLP as a generic ``to`` function.

    Args:
        tensors (tensor, dict, list, namedtuple or tuple): Data structure with tensor values to
            move.

    Returns:
        Same type as inputed with all tensor values moved in accordance to ``kwargs``.
    """
    if torch.is_tensor(tensors):
        return tensors.to(**kwargs)
    elif isinstance(tensors, dict):
        return {k: tensors_to(v, **kwargs) for k, v in tensors.items()}
    elif hasattr(tensors, '_asdict') and isinstance(tensors, tuple):  # Handle ``namedtuple``
        return tensors.__class__(**tensors_to(tensors._asdict(), **kwargs))
    elif isinstance(tensors, list):
        return [tensors_to(t, **kwargs) for t in tensors]
    elif isinstance(tensors, tuple):
        return tuple([tensors_to(t, **kwargs) for t in tensors])
    else:
        return tensors


def lengths_to_mask(*lengths, **kwargs):
    """ Given a list of lengths, create a batch mask.

    Example:
        >>> from src.utils import lengths_to_mask
        >>> lengths_to_mask([1, 2, 3])
        tensor([[1, 0, 0],
                [1, 1, 0],
                [1, 1, 1]], dtype=torch.uint8)
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[1, 0],
                 [0, 0]],
        <BLANKLINE>
                [[1, 1],
                 [1, 1]],
        <BLANKLINE>
                [[1, 1],
                 [1, 1]]], dtype=torch.uint8)

    Args:
        lengths (list of int or torch.Tensor)
        **kwargs: Auxiliary arguments passed to ``torch.zeros``.

    Returns:
        torch.Tensor
    """
    # Squeeze to deal with random additional dimensions
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, we need to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.byte()


def identity(x):
    return x


class _DataLoaderDataset(data.Dataset):
    """ Dataset that allows for a callable upon loading a single example.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable): Function to run
    """

    def __init__(self, dataset, load_fn):
        self.dataset = dataset
        self.load_fn = load_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.load_fn(self.dataset[index])


class DataLoader(DataLoader):
    """ PyTorch DataLoader that supports a ``load_fn``.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable, optional): Callable run to load a single row of the dataset.
        post_process_fn (callable, optional): Callable run directly before the batch is returned.
        num_workers (int, optional): Number ofsubprocesses to use for data loading. 0 means that the
            data will be loaded in the main process.
        trial_run (bool, optional):
        **kwargs: Other key word arguments to be passed to ``torch.utils.data.DataLoader``
    """

    def __init__(self,
                 dataset,
                 load_fn=identity,
                 post_processing_fn=identity,
                 num_workers=cpu_count(),
                 trial_run=False,
                 use_tqdm=False,
                 **kwargs):
        super().__init__(
            dataset=_DataLoaderDataset(dataset, load_fn), num_workers=num_workers, **kwargs)
        logger.info('Launching with %d workers', num_workers)
        self.post_processing_fn = post_processing_fn
        self.trial_run = trial_run
        self.use_tqdm = use_tqdm

    def __len__(self):
        return 1 if self.trial_run else super().__len__()

    def __iter__(self):
        start = time.time()
        is_first = True

        iterator = super().__iter__()
        if self.use_tqdm:
            iterator = tqdm(iterator, total=len(self))

        for batch in iterator:
            yield self.post_processing_fn(batch)

            if is_first:
                elapsed = seconds_to_string(time.time() - start)
                logger.info('Time to first batch was %s.', elapsed)
                is_first = False

            if self.trial_run:
                break


class OnDiskTensor():
    """ Tensor that resides on disk.

    Args:
        path (str or Path)
    """

    def __init__(self, path, allow_pickle=False):
        self.path = Path(path)
        self.allow_pickle = allow_pickle

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, OnDiskTensor):
            return self.path == other.path
        return NotImplemented

    @property
    def shape(self):
        return np.load(str(self.path), mmap_mode='r', allow_pickle=self.allow_pickle).shape

    def to_tensor(self):
        """ Convert to a in-memory ``torch.tensor``. """
        loaded = np.load(str(self.path), allow_pickle=self.allow_pickle)
        return torch.from_numpy(loaded).contiguous()

    def does_exist(self):
        """ If ``True``, the tensor exists on disk. """
        return self.path.is_file()

    def unlink(self):
        """ Remove the file this tensor was pointed to """
        self.path.unlink()
        return self

    def from_tensor(self, tensor):
        """ Make a ``OnDiskTensor`` from a tensor

        Args:
            path (str or Path)
            tensor (np.array or torch.tensor)
        """
        if torch.is_tensor(tensor):
            tensor = tensor.cpu().numpy()

        # This storage was picked using this benchmark:
        # https://github.com/mverleg/array_storage_benchmark
        np.save(str(self.path), tensor, allow_pickle=self.allow_pickle)
        return self


@lru_cache(maxsize=None)
def _get_spectrogram_length(spectrogram):
    """ Get length of spectrogram (shape [num_frames, num_channels]).

    Args:
        spectrogram (OnDiskTensor or torch.Tensor or np.ndarray)

    Returns:
        (int) Length of spectrogram
    """
    return spectrogram.shape[0]


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


@log_runtime
def get_spectrogram_lengths(data, use_tqdm=False):
    """ Get the spectrograms lengths for every data row.

    Args:
        data (iterable of SpectrogramTextSpeechRow)
        use_tqdm (bool)

    Returns:
        (list of int): Spectrogram length for every data point.
    """
    logger.info('Computing spectrogram lengths...')
    with ThreadPoolExecutor() as pool:
        iterator = pool.map(_get_spectrogram_length, [r.spectrogram for r in data])
        if use_tqdm:
            iterator = tqdm(iterator, total=len(data))
        lengths = list(iterator)
    return lengths


def sort_by_spectrogram_length(data, **kwargs):
    """ Sort ``SpectrogramTextSpeechRow`` rows by spectrogram lengths.

    Args:
        data (iterable of SpectrogramTextSpeechRow)
        **kwargs: Further key word arguments passed to ``get_spectrogram_lengths``

    Returns:
        (iterable of SpectrogramTextSpeechRow)
    """
    lengths = get_spectrogram_lengths(data, **kwargs)
    _, return_ = zip(*sorted(zip(lengths, data), key=lambda r: r[0]))
    return return_


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
