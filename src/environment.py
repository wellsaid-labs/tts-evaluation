from pathlib import Path

import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[1].resolve()  # Repository root path

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

TTS_DISK_CACHE_NAME = '.tts_cache'  # Hidden directory stored in other directories for caching

# NOTE: These paths help namespace the disk files.

TEST_DATA_PATH = ROOT_PATH / 'tests' / '_test_data'

DISK_PATH = TEST_DATA_PATH / '_disk' if IS_TESTING_ENVIRONMENT else ROOT_PATH / 'disk'

DATA_PATH = DISK_PATH / 'data'

EXPERIMENTS_PATH = DISK_PATH / 'experiments'

SIGNAL_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / 'signal_model'

SIGNAL_MODEL_EXPERIMENTS_PATH.mkdir(exist_ok=True)

SPECTROGRAM_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / 'spectrogram_model'

SPECTROGRAM_MODEL_EXPERIMENTS_PATH.mkdir(exist_ok=True)

OTHER_DISK_CACHE_PATH = DISK_PATH / 'other'

TEMP_PATH = DISK_PATH / 'temp'

NINJA_BUILD_PATH = OTHER_DISK_CACHE_PATH / 'ninja_build'

NINJA_BUILD_PATH.mkdir(exist_ok=True)

DISK_CACHE_PATH = OTHER_DISK_CACHE_PATH / 'disk_cache'

DISK_CACHE_PATH.mkdir(exist_ok=True)

SAMPLES_PATH = DISK_PATH / 'samples'

# NOTE: You can experiment with these codes in your console like so:
# `echo -e '\033[43m \033[30m hi \033[0m'`

COLORS = {  # Inspired by: https://godoc.org/github.com/whitedevops/colors
    'reset_all': '\033[0m',
    'bold': '\033[1m',
    'dim': '\033[2m',
    'underlined': '\033[4m',
    'blink': '\033[5m',
    'reverse': '\033[7m',
    'hidden': '\033[8m',
    'reset_bold': '\033[21m',
    'reset_dim': '\033[22m',
    'reset_underlined': '\033[24m',
    'reset_blink': '\033[25m',
    'reset_reverse': '\033[27m',
    'reset_hidden': '\033[28m',
    'default': '\033[39m',
    'black': '\033[30m',
    'red': '\033[31m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'blue': '\033[34m',
    'magenta': '\033[35m',
    'cyan': '\033[36m',
    'light_gray': '\033[37m',
    'dark_gray': '\033[90m',
    'light_red': '\033[91m',
    'light_green': '\033[92m',
    'light_yellow': '\033[93m',
    'light_blue': '\033[94m',
    'light_magenta': '\033[95m',
    'light_cyan': '\033[96m',
    'white': '\033[97m',
    'background_default': '\033[49m',
    'background_black': '\033[40m',
    'background_red': '\033[41m',
    'background_green': '\033[42m',
    'background_yellow': '\033[43m',
    'background_blue': '\033[44m',
    'background_magenta': '\033[45m',
    'background_cyan': '\033[46m',
    'background_lightgray': '\033[47m',
    'background_darkgray': '\033[100m',
    'background_lightred': '\033[101m',
    'background_lightgreen': '\033[102m',
    'background_lightyellow': '\033[103m',
    'background_lightblue': '\033[104m',
    'background_lightmagenta': '\033[105m',
    'background_lightcyan': '\033[106m',
    'background_white': '\033[107m',
}


class ColoredFormatter(logging.Formatter):
    """ Logging formatter with color.

    Args:
        id (int): An id to be printed along with all logs.
    """

    ID_COLOR_ROTATION = [  # Used to color the logging `id`.
        COLORS['red'], COLORS['green'], COLORS['yellow'], COLORS['blue'], COLORS['magenta'],
        COLORS['cyan'], COLORS['background_lightred'] + COLORS['black'],
        COLORS['background_lightgreen'] + COLORS['black'],
        COLORS['background_lightyellow'] + COLORS['black'],
        COLORS['background_lightblue'] + COLORS['black'],
        COLORS['background_lightmagenta'] + COLORS['black'],
        COLORS['background_lightcyan'] + COLORS['black']
    ]

    def __init__(self, id):
        super().__init__()

        process_id = COLORS['reset_all'] + ColoredFormatter.ID_COLOR_ROTATION[id % len(
            ColoredFormatter.ID_COLOR_ROTATION)] + '%s' % id + COLORS['reset_all']
        logging.Formatter.__init__(
            self, COLORS['dark_gray'] + '[%(asctime)s][' + process_id + COLORS['dark_gray'] +
            '][%(name)s][%(levelname)s' + COLORS['dark_gray'] + ']' + COLORS['reset_all'] +
            ' %(message)s')

    def format(self, record):
        if record.levelno > 30:  # Error
            record.levelname = COLORS['red'] + record.levelname + COLORS['reset_all']
        elif record.levelno > 20:  # Warning
            record.levelname = COLORS['yellow'] + record.levelname + COLORS['reset_all']

        return logging.Formatter.format(self, record)


class MaxLevelFilter(logging.Filter):
    """ Filter out logs above and equal to a certain level.

    This is the opposite of `setLevel` which sets a mininum threshold.
    """

    def __init__(self, maxLevel):
        super().__init__()
        self.maxLevel = maxLevel

    def filter(self, record):
        return record.levelno < self.maxLevel


def set_basic_logging_config(id=os.getpid()):
    """
    Inspired by: `logging.basicConfig`

    Do basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured. It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a `StreamHandler` which writes to
    `sys.stdout` and `sys.stderr`, set a formatter, and
    add the handler to the root logger.

    Args:
        id (int, optional): An id to be printed along with all logs.
        color_rotation (list, optional): A rotation of colors used to highlight the id.
    """
    root = logging.getLogger()
    if len(root.handlers) == 0:
        root.setLevel(0)

        formatter = ColoredFormatter(id)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(0)
        handler.setFormatter(formatter)
        handler.addFilter(MaxLevelFilter(logging.WARNING))
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
