""" Setup any global configurations like `logging`, `seed`, environment and disk configurations.
"""
from pathlib import Path

import logging
import os
import subprocess
import sys

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

NINJA_BUILD_PATH = TEMP_PATH / 'ninja_build'

NINJA_BUILD_PATH.mkdir(exist_ok=True, parents=True)


def set_basic_logging_config():
    """
    Inspired by: `logging.basicConfig`

    Do basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured. It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a `StreamHandler` which writes to
    `sys.stdout` and `sys.stderr`, set a formatter, and
    add the handler to the root logger.
    """
    root = logging.getLogger()
    if len(root.handlers) == 0:
        root.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '\033[90m[%(asctime)s][%(process)d][%(name)s][%(levelname)s]\033[0m %(message)s')

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        root.addHandler(handler)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        root.addHandler(handler)


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
