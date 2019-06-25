from collections import defaultdict
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from functools import wraps
from math import isclose
from pathlib import Path

import ast
import atexit
import glob
import itertools
import logging
import logging.config
import math
import os
import pprint
import random
import subprocess
import sys
import time

from Levenshtein import distance
from torch.multiprocessing import cpu_count
from torch.nn.functional import mse_loss
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import collate_tensors
from torchnlp.utils import lengths_to_mask
from torchnlp.utils import tensors_to
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.data

from src.hparams import configurable
from src.hparams import ConfiguredArg

import src.distributed

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)

ROOT_PATH = Path(__file__).parent.parent.resolve()  # Repository root path


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

        # TODO: Double check the math, this might not be debiasing this correctly assuming
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

        if not np.isfinite(value):  # NOTE: Handles inf and NaN values.
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
        if not is_anomaly:
            self.last_average, self.last_standard_deviation = super().step(value)
        return is_anomaly


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


def flatten_parameters(model):
    """ Apply ``flatten_parameters`` to ``model``.

    Args:
        model (torch.nn.Module)
    """
    return model.apply(
        lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)


class Checkpoint():
    """ Torch model checkpoint object.

    Args:
        directory (Path or str): Directory where to save the checkpoint.
        model (torch.nn.Module): Model to train and evaluate.
        step (int): Starting step, useful warm starts (i.e. checkpoints).
        **kwargs (dict, optional): Any other checkpoint attributes.
    """

    def __init__(self, directory, step, model, **kwargs):
        self.directory = Path(directory)
        self.step = step
        self.model = model
        self.path = Path(self.directory) / 'step_{}.pt'.format(self.step)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_path(class_, path, device=torch.device('cpu'), load_random_state=True):
        """ Load a checkpoint from path.

        NOTE: Given ``path`` is different than the loaded instance, the original path is not
        overwritten.

        Args:
            path (Path or str): Path to a checkpoint to load.
            device (torch.device, optional): Device to load checkpoint onto.
            load_random_state (bool, optional): Load the random state with the checkpoint.

        Returns:
            checkpoint (Checkpoint): Loaded checkpoint.
        """
        from src.datasets import Gender  # NOTE: Prevent circular dependency
        from src.datasets import Speaker

        instance = load(str(path), device=device)
        if hasattr(instance, 'random_generator_state'):
            if load_random_state:
                set_random_generator_state(instance.random_generator_state)
        else:
            logger.warning('Old Checkpoint: unable to load checkpoint random generator state')

        if (hasattr(instance, 'input_encoder') and Speaker(
                'Frank Bonacquisti', Gender.FEMALE) in instance.input_encoder.speaker_encoder.stoi):
            logger.warning('Old Checkpoint: detected and fixed Frank\'s gender')

            old_speaker = Speaker('Frank Bonacquisti', Gender.FEMALE)
            new_speaker = Speaker('Frank Bonacquisti', Gender.MALE)

            index = instance.input_encoder.speaker_encoder.stoi[old_speaker]
            instance.input_encoder.speaker_encoder.itos[index] = new_speaker
            del instance.input_encoder.speaker_encoder.stoi[old_speaker]
            instance.input_encoder.speaker_encoder.stoi[new_speaker] = index

            count = instance.input_encoder.speaker_encoder.tokens[old_speaker]
            del instance.input_encoder.speaker_encoder.tokens[old_speaker]
            instance.input_encoder.speaker_encoder.tokens[old_speaker] = count

        flatten_parameters(instance.model)
        # Backwards compatibility for instances without paths.
        instance.path = instance.path if hasattr(instance, 'path') else path
        logger.info('Loaded checkpoint at step %d from %s with model:\n%s', instance.step,
                    instance.path, instance.model)
        return instance

    @classmethod
    def most_recent(class_, pattern, **kwargs):
        """ Load the most recent checkpoint from ``root``.

        Args:
            pattern (str): Pattern to glob recursively for checkpoints.
            **kwargs (dict, optional): Any additional parameters to pass to ``class_.from_path``.

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

        raise ValueError('Unable to load recent checkpoint.')

    def save(self):
        """ Save a checkpoint. """
        self.random_generator_state = get_random_generator_state()
        save(self.path, self)
        return self.path


RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state():
    """ Get the `torch`, `numpy` and `random` random generator state.

    Returns:
        RandomGeneratorState
    """
    return RandomGeneratorState(
        random.getstate(), torch.random.get_rng_state(), np.random.get_state(),
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_random_generator_state(state):
    """ Set the `torch`, `numpy` and `random` random generator state.

    Args:
        state (RandomGeneratorState)
    """
    logger.info('Setting the random state for `torch`, `numpy` and `random`.')
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if torch.cuda.is_available() and state.torch_cuda is not None and len(
            state.torch_cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state.torch_cuda)


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


class DataLoaderDataset(torch.utils.data.Dataset):
    """ Dataset that allows for a callable upon loading a single example.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable): Function to run on `__getitem__`.
    """

    def __init__(self, dataset, load_fn):
        self.dataset = dataset
        self.load_fn = load_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.load_fn(self.dataset[index])


class DataLoader(torch.utils.data.dataloader.DataLoader):
    """ PyTorch DataLoader that supports a ``load_fn``.

    Args:
        dataset (torch.utils.data. Dataset): Dataset from which to load the data.
        load_fn (callable, optional): Callable run to load a single row of the dataset.
        post_process_fn (callable, optional): Callable run directly before the batch is returned.
        num_workers (int, optional): Number of subprocesses to use for data loading. Given a 0 value
          the data will be loaded in the main process.
        trial_run (bool, optional): If ``True`` iteration stops after the first batch.
        use_tqdm (bool, optional): Log a TQDM progress bar.
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
            dataset=DataLoaderDataset(dataset, load_fn), num_workers=num_workers, **kwargs)
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
        path (str or Path): Path to a tensor saved on disk as an ``.npy`` file.
        allow_pickle (bool, optional): Allow saving object arrays using Python pickles. This
          is not recommended for performance reasons.
    """

    def __init__(self, path, allow_pickle=False):
        assert '.npy' in str(path), 'Path must include ``.npy`` extension.'

        self.path = Path(path)
        self.allow_pickle = allow_pickle
        self._shape = None
        self._shape_path = None

    def __hash__(self):
        return hash(self.path)

    def __eq__(self, other):
        if isinstance(other, OnDiskTensor):
            return self.path == other.path

        # Learn more:
        # https://stackoverflow.com/questions/878943/why-return-notimplemented-instead-of-raising-notimplementederror
        return NotImplemented

    @property
    def shape(self):
        if not self.exists():
            raise RuntimeError('Tensor not found on disk.')

        if self.path == self._shape_path:
            return self._shape

        with open(str(self.path), 'rb') as file_:
            version = np.lib.format.read_magic(file_)
            shape, _, _ = np.lib.format._read_array_header(file_, version)

        self._shape = shape
        self._shape_path = self.path

        return shape

    def to_tensor(self):
        """ Convert to a in-memory ``torch.tensor``. """
        if not self.exists():
            raise RuntimeError('Tensor not found on disk.')

        loaded = np.load(str(self.path), allow_pickle=self.allow_pickle)
        return torch.from_numpy(loaded).contiguous()

    def exists(self):
        """ If ``True``, the tensor exists on disk. """
        return self.path.is_file()

    def unlink(self):
        """ Delete the ``OnDiskTensor`` from disk.

        Returns:
            (Path): The path the ``OnDiskTensor`` used to reside in.
        """
        self.path.unlink()
        return self

    @classmethod
    def from_tensor(class_, path, tensor, allow_pickle=False):
        """ Make a ``OnDiskTensor`` from a tensor.

        Args:
            path (str or Path): Path to a tensor saved on disk as an ``.npy`` file.
            tensor (np.array or torch.tensor)
            allow_pickle (bool, optional): Allow saving object arrays using Python pickles. This
              is not recommended for performance reasons.
        """
        if torch.is_tensor(tensor):
            tensor = tensor.cpu().numpy()

        # This storage was picked using this benchmark:
        # https://github.com/mverleg/array_storage_benchmark
        np.save(str(path), tensor, allow_pickle=allow_pickle)
        return class_(path, allow_pickle)


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


class AccumulatedMetrics():
    """
    Args:
        type_: Default torch tensor type.
    """

    def __init__(self, type_=torch.cuda):
        self.type_ = type_
        self.reset()

    def reset(self):
        self.metrics = {
            'epoch_total': defaultdict(float),
            'epoch_count': defaultdict(float),
            'step_total': defaultdict(float),
            'step_count': defaultdict(float)
        }

    def add_metric(self, name, value, count=1):
        """ Add metric as part of the current step.

        Args:
            name (str)
            value (number)
            count (int): Number of times to add value / frequency of value.
        """
        if torch.is_tensor(value):
            value = value.item()

        if torch.is_tensor(count):
            count = count.item()

        assert count > 0, '%s count, %f, must be a positive number' % (name, count)

        self.metrics['step_total'][name] += value * count
        self.metrics['step_count'][name] += count

    def add_metrics(self, dict_, count=1):
        """ Add multiple metrics as part of the current step.

        Args:
            dict_ (dict): Metrics in the form of key value pairs.
            count (int): Number of times to add value / frequency of value.
        """
        for metric, value in dict_.items():
            self.add_metric(metric, value, count)

    def log_step_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_step_end``.

        Note that in the distributed case, only the master node gets the accurate metric.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        # Temporary until this issue is fixed: https://github.com/pytorch/pytorch/issues/20651
        if len(self.metrics['step_total']) == 0 and len(self.metrics['step_count']) == 0:
            return

        # Accumulate metrics between multiple processes.
        if src.distributed.is_initialized():
            metrics_total_items = sorted(self.metrics['step_total'].items(), key=lambda t: t[0])
            metrics_total_values = [value for _, value in metrics_total_items]

            metrics_count_items = sorted(self.metrics['step_count'].items(), key=lambda t: t[0])
            metrics_count_values = [value for _, value in metrics_count_items]

            packed = self.type_.FloatTensor(metrics_total_values + metrics_count_values)
            torch.distributed.reduce(packed, dst=src.distributed.get_master_rank())
            packed = packed.tolist()

            for (key, _), value in zip(metrics_total_items, packed[:len(metrics_total_items)]):
                self.metrics['step_total'][key] = value

            for (key, _), value in zip(metrics_count_items, packed[len(metrics_total_items):]):
                self.metrics['step_count'][key] = value

        # Log step metrics and update epoch metrics.
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['step_total'].items(), self.metrics['step_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)

            self.metrics['epoch_total'][total_key] += total_value
            self.metrics['epoch_count'][total_key] += count_value

        # Reset step metrics
        self.metrics['step_total'] = defaultdict(float)
        self.metrics['step_count'] = defaultdict(float)

    def get_epoch_metric(self, metric):
        """ Get the current epoch value for a metric.
        """
        return self.metrics['epoch_total'][metric] / self.metrics['epoch_count'][metric]

    def log_epoch_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_epoch_end``.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        self.log_step_end(lambda *args, **kwargs: None)

        # Log epoch metrics
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['epoch_total'].items(), self.metrics['epoch_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)


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


# LEARN MORE: https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy('file_system')


def _batch_predict_spectrogram_load_fn(row, input_encoder, load_spectrogram=False):
    """ Load function for loading a single row.

    Args:
        row (TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        load_spectrogram (bool, optional)

    Returns:
        (TextSpeechRow)
    """
    encoded_text, encoded_speaker = input_encoder.encode((row.text, row.speaker))
    row = row._replace(text=encoded_text, speaker=encoded_speaker)
    if load_spectrogram and isinstance(row.spectrogram, OnDiskTensor):
        row = row._replace(spectrogram=row.spectrogram.to_tensor())
    return row


def batch_predict_spectrograms(data,
                               input_encoder,
                               model,
                               device,
                               batch_size,
                               filenames=None,
                               aligned=True,
                               use_tqdm=True):
    """ Batch predict spectrograms.

    Args:
        data (iterable of TextSpeechRow)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        model (torch.nn.Module): Model used to compute spectrograms.
        batch_size (int)
        device (torch.device): Device to run model on.
        filenames (list, optional): If provided, this saves predictions to these paths.
        aligned (bool, optional): If ``True``, predict a ground truth aligned spectrogram.
        use_tqdm (bool, optional): Write a progress bar to standard streams.

    Returns:
        (iterable of torch.Tensor or OnDiskTensor)
    """
    data = [d._replace(metadata={'index': i}) for i, d in enumerate(data)]
    if filenames is not None:
        assert len(filenames) == len(data)
        iterator = zip(data, filenames)
        data = [d._replace(metadata=dict({'filename': f}, **d.metadata)) for d, f in iterator]
    return_ = [None] * len(data)

    # Sort by sequence length to reduce padding in batches.
    if all([r.spectrogram is not None for r in data]):
        spectrogram_lengths = get_tensors_dim_length([r.spectrogram for r in data])
        data = sort_together(data, spectrogram_lengths)
    else:
        data = sorted(data, key=lambda r: len(r.text))

    load_fn_partial = partial(
        _batch_predict_spectrogram_load_fn, input_encoder=input_encoder, load_spectrogram=aligned)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        load_fn=load_fn_partial,
        post_processing_fn=partial(tensors_to, device=device, non_blocking=True),
        collate_fn=partial(collate_tensors, stack_tensors=partial(stack_and_pad_tensors, dim=1)),
        pin_memory=True,
        use_tqdm=use_tqdm)
    with evaluate(model, device=device):
        metrics = AccumulatedMetrics()
        for batch in loader:
            # Predict spectrogram
            text, text_lengths = batch.text
            speaker = batch.speaker[0]
            if aligned:
                spectrogram, spectrogram_lengths = batch.spectrogram
                _, predictions, _, alignments = model(text, speaker, text_lengths, spectrogram,
                                                      spectrogram_lengths)
            else:
                _, predictions, _, alignments, spectrogram_lengths = model(text, speaker)

            # Compute metrics for logging
            mask = lengths_to_mask(spectrogram_lengths, device=predictions.device).transpose(0, 1)
            metrics.add_metrics({
                'attention_norm': get_average_norm(alignments, norm=math.inf, dim=2, mask=mask),
                'attention_std': get_weighted_stdev(alignments, dim=2, mask=mask),
            }, mask.sum())
            if aligned:
                mask = mask.unsqueeze(2).expand_as(predictions)
                loss = mse_loss(predictions, spectrogram, reduction='none')
                metrics.add_metric('loss', loss.masked_select(mask).mean(), mask.sum())

            # Split batch and store in-memory or on-disk
            spectrogram_lengths = spectrogram_lengths.squeeze(0).tolist()
            predictions = predictions.split(1, dim=1)
            predictions = [p[:l, 0] for p, l in zip(predictions, spectrogram_lengths)]
            for i, prediction in enumerate(predictions):
                return_index = batch.metadata['index'][i]
                if filenames is None:
                    return_[return_index] = prediction
                else:
                    filename = batch.metadata['filename'][i]
                    return_[return_index] = OnDiskTensor.from_tensor(filename, prediction)

        metrics.log_epoch_end(lambda k, v: logger.info('Prediction metric (%s): %s', k, v))

    return return_


def format_alignment(tokens, other_tokens, alignments):
    """ Format strings to be printed of the alignment.

    Example:

        >>> tokens = ['i','n','t','e','n','t','i','o','n']
        >>> other_tokens = ['e','x','e','c','u','t','i','o','n']
        >>> alignment = [(0, 0), (2, 1), (3, 2), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        >>> formatted = format_alignment(tokens, other_tokens, alignment)
        >>> formatted[0]
        'i n t e   n t i o n'
        >>> formatted[1]
        'e   x e c u t i o n'

    Args:
        tokens (list or str): Sequences of strings.
        other_tokens (list or str): Sequences of strings.
        alignment (list): Alignment between the tokens in both sequences. The alignment is a
            sequence of sorted tuples with a `tokens` index and `other_tokens` index.

    Returns:
        formatted_tokens (str): String formatted for printing.
        formatted_other_tokens (str): String formatted for printing.
    """
    tokens_index = 0
    other_tokens_index = 0
    alignments_index = 0
    tokens_string = ''
    other_tokens_string = ''
    while True and len(alignments) != 0:
        alignment = alignments[alignments_index]
        other_token = other_tokens[other_tokens_index]
        token = tokens[tokens_index]
        if tokens_index == alignment[0] and other_tokens_index == alignment[1]:
            padded_format = '{:' + str(max(len(token), len(other_token))) + 's} '
            tokens_string += padded_format.format(token)
            other_tokens_string += padded_format.format(other_token)
            tokens_index += 1
            other_tokens_index += 1
            alignments_index += 1
        elif tokens_index == alignment[0]:
            other_tokens_string += other_token + ' '
            tokens_string += ' ' * (len(other_token) + 1)
            other_tokens_index += 1
        elif other_tokens_index == alignment[1]:
            tokens_string += token + ' '
            other_tokens_string += ' ' * (len(token) + 1)
            tokens_index += 1

        if alignments_index == len(alignments):
            break

    return tokens_string.rstrip(), other_tokens_string.rstrip()


def _is_in_window(value, window):
    """ Check if `value` is in the range [`window[0]`, `window[1]`)
    """
    return value >= window[0] and value < window[1]


""" Flatten a list of lists into a list. """
flatten = lambda l: [item for sublist in l for item in sublist]


def align_tokens(tokens,
                 other_tokens,
                 window_length=None,
                 all_alignments=False,
                 allow_substitution=lambda a, b: True):
    """ Compute the alignment between `tokens` and `other_tokens`.

    Base algorithm implementation: https://en.wikipedia.org/wiki/Levenshtein_distance

    This implements a modified version of the levenshtein distance algorithm. The modifications are
    as follows:

      - We do not assume each token is a character; therefore, the cost of substition is the
        levenshtein distance between tokens. The cost of deletion or insertion is the length
        of the token. In the case that the tokens are characters, this algorithms functions
        equivalently to the original.
      - The user may specify a `window_length`. Given that the window length is the same length as
        the smallest sequence, then this algorithm functions equivalently to the original;
        otherwise, not every token in both sequences is compared to compute the alignment. A user
        would use this parameter if the two sequences are mostly aligned. The runtime of the
        algorithm with a window length is
        O(`window_length` * `max(len(tokens), len(other_tokens))`).

    Args:
        tokens (list or str): Sequences of strings.
        other_tokens (list or str): Sequences of strings.
        window_length (int or None, optional): Approximately the maximum number of consecutive
            insertions or deletions required to align two similar sequences.
        all_alignments (bool, optional): If `True` return all optimal alignments, otherwise, return
            one optimal alignment.
        allow_substitution (callable, optional): Return a `bool` if the substitution is allowed
            between two tokens `a` and `b`.

    Returns:
        (int): The cost of alignment.
        (list): The alignment consisting of a sequence of indicies.
    """
    # For `window_center` to be on a diagonal with an appropriate slope to align both sequences,
    # `tokens` needs to be the longer sequence.
    flipped = len(other_tokens) > len(tokens)
    if flipped:
        other_tokens, tokens = (tokens, other_tokens)

    if window_length is None:
        window_length = len(other_tokens)

    alignment_window_length = min(2 * window_length + 1, len(other_tokens) + 1)
    row_one = [None] * alignment_window_length
    row_two = [None] * alignment_window_length
    # NOTE: This operation copies a reference to the initial list `alignment_window_length` times.
    # Example:
    # >>> list_of_lists = [[]] * 10
    # >>> list_of_lists[0].append(1)
    # >>> list_of_lists
    # [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    row_one_paths = [[[]]] * alignment_window_length
    row_two_paths = [[[]]] * alignment_window_length

    # Number of edits to convert a substring of the `other_tokens` into the empty string.
    row_one[0] = 0
    for i in range(min(window_length, len(other_tokens))):
        row_one[i + 1] = row_one[i] + len(other_tokens[i])  # Deletion of `other_tokens[:i + 1]`.

    # Both `row_one_window` and `row_two_window` are not inclusive of the maximum.
    row_one_window = (0, min(len(other_tokens) + 1, window_length + 1))

    for i in tqdm(range(len(tokens))):
        # TODO: Consider setting the `window_center` at the minimum index in `row_one`. There are
        # a number of considerations to make:
        # 1. The window no longer guarantees that the last window will have completed both
        # sequences.
        # 2. Smaller indicies in `row_one` have not completed as much of the sequences as larger
        # sequences in `row_one`.
        window_center = min(i, len(other_tokens))
        row_two_window = (max(0, window_center - window_length),
                          min(len(other_tokens) + 1, window_center + window_length + 1))
        if ((row_two_window[1] - row_two_window[0]) < len(row_two) and
                row_two_window[1] == len(other_tokens) + 1):
            row_two = row_two[:(row_two_window[1] - row_two_window[0])]
            row_two_paths = row_two_paths[:(row_two_window[1] - row_two_window[0])]

        for j in range(*row_two_window):
            choices = []

            if _is_in_window(j, row_one_window):
                deletion_cost = row_one[j - row_one_window[0]] + len(tokens[i])
                deletion_path = row_one_paths[j - row_one_window[0]]
                choices.append((deletion_cost, deletion_path))
            if _is_in_window(j - 1, row_two_window):
                insertion_cost = row_two[j - 1 - row_two_window[0]] + len(other_tokens[j - 1])
                insertion_path = row_two_paths[j - 1 - row_two_window[0]]
                choices.append((insertion_cost, insertion_path))
            if _is_in_window(j - 1, row_one_window):
                token = tokens[i]
                other_token = other_tokens[j - 1]
                substition_cost = distance(token, other_token)
                if allow_substitution(token, other_token):
                    substition_cost = row_one[j - 1 - row_one_window[0]] + substition_cost
                    alignment = (j - 1, i) if flipped else (i, j - 1)
                    substition_path = row_one_paths[j - 1 - row_one_window[0]]
                    substition_path = [p + [alignment] for p in substition_path]
                    choices.append((substition_cost, substition_path))

            # NOTE: `min` picks the first occurring minimum item in the iterable passed.
            # NOTE: Substition is put last on purpose to discourage substition if other options are
            # available that cost the same.
            min_cost, min_paths = min(choices, key=lambda p: p[0])
            if all_alignments:
                min_paths = flatten([path for cost, path in choices if cost == min_cost])

            row_two[j - row_two_window[0]] = min_cost
            row_two_paths[j - row_two_window[0]] = min_paths

        row_one = row_two[:]
        row_one_paths = row_two_paths[:]
        row_one_window = row_two_window

    return row_one[-1], row_one_paths[-1] if all_alignments else row_one_paths[-1][0]
