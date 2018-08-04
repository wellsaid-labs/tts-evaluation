import matplotlib

matplotlib.use('Agg', warn=False)

import ast
import glob
import logging
import logging.config
import math
import os

from torchnlp.utils import shuffle as do_deterministic_shuffle

import torch
import numpy as np

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)

# Repository root path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))


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
    """

    # Below 10 samples there can be significant bias in the variance estimation causing it
    # to be underestimated.
    # LEARN MORE: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
    def __init__(self, beta=0.99, sigma=6, eps=10**-6, min_steps=10):
        super().__init__(beta=beta)
        self.sigma = sigma
        self.last_standard_deviation = 0.0
        self.last_average = 0.0
        self.min_steps = min_steps
        self.eps = eps

    def step(self, value):
        if not np.isfinite(value):
            return True

        if (self.step_counter + 1 >= self.min_steps and abs(value - self.last_average) >
                self.sigma * self.last_standard_deviation + self.eps):
            return True

        self.last_average, self.last_standard_deviation = super().step(value)
        return False


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

    return torch.load(path, map_location=remap)


def torch_save(path, data):
    """ Using ``torch.save`` and ``dill`` save an object to ``path``.

    Args:
        path (str): Filename to save to.
        data (any): Data to save into file.
    """
    torch.save(data, path)
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
