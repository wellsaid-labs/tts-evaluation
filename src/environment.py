""" Setup any global configurations like `logging`, `seed`, environment and disk configurations.
"""
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import functools
import logging
import os
import random
import subprocess
import sys

import numpy as np
import torch

from src.hparams import configurable
from src.hparams import ConfiguredArg

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[1].resolve()  # Repository root path

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

TTS_DISK_CACHE_NAME = '.tts_cache'  # Directory name for any disk cache's created by this repository

TEST_DATA_PATH = ROOT_PATH / 'tests' / '_test_data'

DATA_PATH = ROOT_PATH / 'data'

EXPERIMENTS_PATH = ROOT_PATH / 'experiments'

DEFAULT_TTS_DISK_CACHE = (TEST_DATA_PATH
                          if IS_TESTING_ENVIRONMENT else ROOT_PATH) / TTS_DISK_CACHE_NAME

TEMP_PATH = DEFAULT_TTS_DISK_CACHE / 'tmp'

TEMP_PATH.mkdir(exist_ok=True, parents=True)

RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state(cuda=torch.cuda.is_available()):
    """ Get the `torch`, `numpy` and `random` random generator state.

    Args:
        cuda (bool, optional): If `True` saves the `cuda` seed also. Note that getting and setting
            the random generator state for CUDA can be quite slow if you have a lot of GPUs.

    Returns:
        RandomGeneratorState
    """
    return RandomGeneratorState(random.getstate(), torch.random.get_rng_state(),
                                np.random.get_state(),
                                torch.cuda.get_rng_state_all() if cuda else None)


def set_random_generator_state(state):
    """ Set the `torch`, `numpy` and `random` random generator state.

    Args:
        state (RandomGeneratorState)
    """
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)
    if state.torch_cuda is not None and torch.cuda.is_available() and len(
            state.torch_cuda) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state.torch_cuda)


@contextmanager
def fork_rng(seed=None, cuda=torch.cuda.is_available()):
    """ Forks the `torch`, `numpy` and `random` random generators, so that when you return, the
    random generators are reset to the state that they were previously in.

    Args:
        seed (int or None, optional): If defined this sets the seed values for the random
            generator fork. This is a convenience parameter.
        cuda (bool, optional): If `True` saves the `cuda` seed also. Getting and setting the random
            generator state can be quite slow if you have a lot of GPUs.
    """
    state = get_random_generator_state(cuda)
    if seed is not None:
        set_seed(seed, cuda)
    try:
        yield
    finally:
        set_random_generator_state(state)


def fork_rng_wrap(function=None, **kwargs):
    """ Decorator alias for `fork_rng`.
    """
    if not function:
        return functools.partial(fork_rng_wrap, **kwargs)

    @functools.wraps(function)
    def wrapper():
        with fork_rng(**kwargs):
            return function()

    return wrapper


def set_basic_logging_config():
    """ Set up basic logging handlers. """
    logging.basicConfig(
        level=logging.INFO,
        format='\033[90m[%(asctime)s][%(process)d][%(name)s][%(levelname)s]\033[0m %(message)s')


@configurable
def set_seed(seed=ConfiguredArg(), cuda=torch.cuda.is_available()):
    """ Set seed values for random generators.

    Args:
        seed (int): Value used as a seed.
        cuda (bool, optional): If `True` sets the `cuda` seed also.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@configurable
def get_initial_seed(seed=ConfiguredArg()):
    return seed


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


def check_module_versions():
    """ Ensure installed modules respect ``requirements.txt`` """
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    freeze = freeze.decode('utf-8').split()
    requirements = Path(ROOT_PATH / 'requirements.txt').read_text()
    for line in requirements.split():
        line = line.strip()
        if '==' in line:
            specification = line.split()[0]
            package = specification.split('==')[0]
            installed = [p for p in freeze if p.split('==')[0] == package]
            if not len(installed) == 1:
                raise RuntimeError('%s not installed' % package)
            if not specification == installed[0]:
                # NOTE: RuntimeError could cause ``Illegal seek`` while running PyTest.
                raise RuntimeError('Versions are not compatible %s =/= %s' %
                                   (specification, installed[0]))
