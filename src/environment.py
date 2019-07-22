""" Setup any global configurations like `logging`, `seed`, environment and disk configurations.
"""
from collections import namedtuple
from pathlib import Path

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

TTS_DISK_CACHE_NAME = '.tts_cache'  # Directory name for any disk cache's created by this repository

TEMP_PATH = ROOT_PATH / TTS_DISK_CACHE_NAME / 'tmp'

TEMP_PATH.mkdir(exist_ok=True, parents=True)

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

# NOTE (Michael P., 07-22-2019): The torch cuda random generator state can be very large and can
# cause OOM errors; therefore, it's usage is optional and not recommended.
#
# RNG state size:
# >>> import torch
# >>> torch.cuda.get_rng_state().shape
# torch.Size([824016])
# >>> torch.random.get_rng_state().shape
# torch.Size([5048])
#
# That said, in PyTorch 1.2 (or the current master), this should be fixed.
RandomGeneratorState = namedtuple('RandomGeneratorState',
                                  ['random', 'torch', 'numpy', 'torch_cuda'])


def get_random_generator_state():
    """ Get the `torch`, `numpy` and `random` random generator state.

    Returns:
        RandomGeneratorState
    """
    return RandomGeneratorState(random.getstate(), torch.random.get_rng_state(),
                                np.random.get_state(), None)


def set_random_generator_state(state):
    """ Set the `torch`, `numpy` and `random` random generator state.

    Args:
        state (RandomGeneratorState)
    """
    logger.info('Setting the random state for `torch`, `numpy` and `random`.')
    random.setstate(state.random)
    torch.random.set_rng_state(state.torch)
    np.random.set_state(state.numpy)


def set_basic_logging_config():
    """ Set up basic logging handlers. """
    logging.basicConfig(
        level=logging.INFO,
        format='\033[90m[%(asctime)s][%(process)d][%(name)s][%(levelname)s]\033[0m %(message)s')


@configurable
def set_seed(seed=ConfiguredArg()):
    """ Set seed values for random generators.

    Args:
        seed (int): Value used as a seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
                raise RuntimeError(
                    'Versions are not compatible %s =/= %s' % (specification, installed[0]))
