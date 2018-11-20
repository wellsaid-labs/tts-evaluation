import matplotlib

matplotlib.use('Agg', warn=False)

from pathlib import Path
from math import isclose

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

from torch.utils.data.sampler import Sampler

import torch
import numpy as np

from src.hparams import configurable

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


def get_weighted_standard_deviation(tensor, dim=0, mask=None):
    """ Computed the average weighted standard deviation.

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
    biased_weighted_variance = (
        (indicies - weighted_mean.unsqueeze(dim=dim))**2 * tensor).sum(dim=dim)

    # Correct the bias
    bias = 1 - (tensor**2).sum(dim=dim)
    weighted_variance = biased_weighted_variance / bias
    weighted_standard_deviation = weighted_variance**0.5

    if mask is not None:
        weighted_standard_deviation = weighted_standard_deviation * mask
        divisor = mask.sum()
    else:
        divisor = weighted_standard_deviation.numel()

    return (weighted_standard_deviation.sum() / divisor).item()


def get_masked_average_norm(tensor, dim=0, mask=None, norm=2):
    """
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
        norm = norm * mask
        divisor = mask.sum()
    else:
        divisor = norm.numel()

    return (norm.sum() / divisor).item()


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
       beta (float, optional): Beta used to weight the exponential mean and standard deviation.
       sigma (float, optional): Number of standard deviations in order to classify as an anomaly.
       eps (float, optional): Minimum difference to be considered an anomaly used for numerical
          stability.
       min_steps (int, optional): Minimum number of steps to wait before detecting anomalies.
       type_ (str, optional): Detect anomalies that are too 'high', too 'low', or 'both'.
    """

    TYPE_HIGH = 'high'
    TYPE_LOW = 'low'
    TYPE_BOTH = 'both'

    # Below 10 samples there can be significant bias in the variance estimation causing it
    # to be underestimated.
    # LEARN MORE: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    @configurable
    def __init__(self, beta=0.99, sigma=6, eps=10**-6, min_steps=10, type_=TYPE_HIGH):
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


def torch_load(path, device=torch.device('cpu')):
    """ Using ``torch.load`` and ``dill`` load an object from ``path`` onto ``self.device``.

    Args:
        path (Path or str): Filename to load.

    Returns:
        (any): Object loaded.
    """
    logger.info('Loading: %s' % (path,))

    def remap(storage, loc):
        if 'cuda' in loc and device.type == 'cuda':
            return storage.cuda(device=device.index)
        return storage

    return torch.load(str(path), map_location=remap)


def torch_save(path, data):
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


@configurable
def split_signal(signal, bits=16):
    """ Compute the coarse and fine components of the signal.

    Args:
        signal (torch.FloatTensor [signal_length]): Signal with values ranging from [-1, 1]
        bits (int): Number of bits to encode signal in.

    Returns:
        coarse (torch.LongTensor [signal_length]): Top bits of the signal.
        fine (torch.LongTensor [signal_length]): Bottom bits of the signal.
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
    return coarse.long(), fine.long()


@configurable
def combine_signal(coarse, fine, bits=16):
    """ Compute the coarse and fine components of the signal.

    Args:
        coarse (torch.LongTensor [signal_length]): Top bits of the signal.
        fine (torch.LongTensor [signal_length]): Bottom bits of the signal.
        bits (int): Number of bits to encode signal in.

    Returns:
        signal (torch.FloatTensor [signal_length]): Signal with values ranging from [-1, 1]
    """
    assert len(coarse.shape) == 1 and len(fine.shape) == 1
    bins = int(2**(bits / 2))
    assert torch.min(coarse) >= 0 and torch.max(coarse) < bins
    assert torch.min(fine) >= 0 and torch.max(fine) < bins
    signal = coarse.float() * bins + fine.float() - 2**(bits - 1)
    return signal / 2**(bits - 1)


class Checkpoint():
    """ Model checkpoint object

    Args:
        directory (Path or str): Directory where to save the checkpoint.
        model (torch.nn.Module): Model to train and evaluate.
        step (int): Starting step, useful warm starts (i.e. checkpoints).
        **kwargs (dict, optional): Any other checkpoint attributes.
    """

    def __init__(self, directory, model, step, **kwargs):
        self.directory = Path(directory)
        self.model = model
        self.step = step

        for key, value in kwargs.items():
            setattr(self, key, value)

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

        instance = torch_load(str(path), device=device)
        instance.model.apply(
            lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None)
        setattr(instance, 'path', path)
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
                logger.warning('Failed to load checkpoint %s' % path)
                pass

        return None

    def save(self):
        """ Save a checkpoint. """
        name = 'step_{}.pt'.format(self.step)
        filename = Path(self.directory) / name
        self.path = filename
        torch_save(filename, self)
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
