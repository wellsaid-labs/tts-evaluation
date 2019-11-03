from pathlib import Path

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[1].resolve()  # Repository root path

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

TTS_DISK_CACHE_NAME = '.tts_cache'  # Hidden directory stored in other directories for caching

TEST_DATA_PATH = ROOT_PATH / 'tests' / '_test_data'

DISK_PATH = TEST_DATA_PATH / '_disk' if IS_TESTING_ENVIRONMENT else ROOT_PATH / 'disk'

DATA_PATH = DISK_PATH / 'data'

EXPERIMENTS_PATH = DISK_PATH / 'experiments'

DISK_CACHE_PATH = DISK_PATH / 'other'

TEMP_PATH = DISK_PATH / 'temp'

NINJA_BUILD_PATH = DISK_CACHE_PATH / 'ninja_build'

NINJA_BUILD_PATH.mkdir(exist_ok=True, parents=True)


def set_basic_logging_config():
    """ Set up basic logging handlers. """
    logging.basicConfig(
        level=logging.INFO,
        format='\033[90m[%(asctime)s][%(process)d][%(name)s][%(levelname)s]\033[0m %(message)s')


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
