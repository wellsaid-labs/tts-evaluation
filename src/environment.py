from pathlib import Path

import atexit
import logging
import os
import subprocess
import sys
import time

from hparams import configurable
from hparams import HParam

import torchnlp

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[1].resolve()  # Repository root path

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

TTS_DISK_CACHE_NAME = '.tts_cache'  # Hidden directory stored in other directories for caching

# NOTE: These paths help namespace the disk files.

TEST_DATA_PATH = ROOT_PATH / 'tests' / '_test_data'

if IS_TESTING_ENVIRONMENT:
    TEST_DATA_PATH.mkdir(exist_ok=True)

DISK_PATH = TEST_DATA_PATH / '_disk' if IS_TESTING_ENVIRONMENT else ROOT_PATH / 'disk'

DISK_PATH.mkdir(exist_ok=True)

DATA_PATH = DISK_PATH / 'data'

DATA_PATH.mkdir(exist_ok=True)

EXPERIMENTS_PATH = DISK_PATH / 'experiments'

EXPERIMENTS_PATH.mkdir(exist_ok=True)

SIGNAL_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / 'signal_model'

SIGNAL_MODEL_EXPERIMENTS_PATH.mkdir(exist_ok=True)

SPECTROGRAM_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / 'spectrogram_model'

SPECTROGRAM_MODEL_EXPERIMENTS_PATH.mkdir(exist_ok=True)

DATABASES_PATH = DISK_PATH / 'databases'

DATABASES_PATH.mkdir(exist_ok=True)

TEMP_PATH = DISK_PATH / 'temp'

TEMP_PATH.mkdir(exist_ok=True)

SAMPLES_PATH = DISK_PATH / 'samples'

SAMPLES_PATH.mkdir(exist_ok=True)

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
        id_ (int): An id to be printed along with all logs.
    """

    ID_COLOR_ROTATION = [
        COLORS['background_white'] + COLORS['black'],
        COLORS['background_lightgreen'] + COLORS['black'],
        COLORS['background_lightblue'] + COLORS['black'],
        COLORS['background_lightmagenta'] + COLORS['black'],
        COLORS['background_lightcyan'] + COLORS['black'], COLORS['green'], COLORS['blue'],
        COLORS['magenta'], COLORS['cyan']
    ]

    def __init__(self, id_):
        super().__init__()

        id_ = COLORS['reset_all'] + ColoredFormatter.ID_COLOR_ROTATION[id_ % len(
            ColoredFormatter.ID_COLOR_ROTATION)] + '%s' % id_ + COLORS['reset_all']
        logging.Formatter.__init__(
            self, COLORS['dark_gray'] + '[%(asctime)s][' + id_ + COLORS['dark_gray'] +
            '][%(name)s][%(levelname)s' + COLORS['dark_gray'] + ']' + COLORS['reset_all'] +
            ' %(message)s')

    def format(self, record):
        if record.levelno > 30:  # Error
            record.levelname = COLORS['red'] + record.levelname + COLORS['reset_all']
        elif record.levelno > 20:  # Warning
            record.levelname = COLORS['yellow'] + record.levelname + COLORS['reset_all']

        return logging.Formatter.format(self, record)


class MaxLevelFilter(logging.Filter):
    """ Filter out logs above a certain level.

    This is the opposite of `setLevel` which sets a mininum threshold.
    """

    def __init__(self, maxLevel):
        super().__init__()
        self.maxLevel = maxLevel

    def filter(self, record):
        return record.levelno <= self.maxLevel


def set_basic_logging_config(id_=os.getpid()):
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
        id_ (int, optional): An id to be printed along with all logs.
        color_rotation (list, optional): A rotation of colors used to highlight the id.
    """
    root = logging.getLogger()
    if len(root.handlers) == 0:
        root.setLevel(logging.INFO)

        formatter = ColoredFormatter(id_)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler.addFilter(MaxLevelFilter(logging.INFO))
        root.addHandler(handler)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        root.addHandler(handler)


@configurable
def assert_enough_disk_space(min_space=HParam()):
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


@configurable
def set_seed(seed=HParam()):
    """ Set a process seed to help ensure consistency. """
    logger.info('Setting process seed to be %d', seed)
    torchnlp.random.set_seed(seed)


def bash_time_label(add_pid=True):
    """ Get a bash friendly string representing the time and process.

    NOTE: This string is optimized for sorting by ordering units of time from largest to smallest.
    NOTE: This string avoids any special bash characters, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    NOTE: `os.getpid` is often used by routines that generate unique identifiers, learn more:
    http://manpages.ubuntu.com/manpages/cosmic/man2/getpid.2.html

    Args:
        add_pid (bool, optional): If `True` add the process PID to the label.

    Returns:
        (str)
    """
    label = str(time.strftime('DATE-%Yy%mm%dd-%Hh%Mm%Ss', time.localtime()))

    if add_pid:
        label += '_PID-%s' % str(os.getpid())

    return label


def text_to_label(text):
    """ Get a label like 'hilary_noriega' from text like 'Hilary Noriega'. """
    return text.lowercase().replace(' ', '_')


def _duplicate_stream(from_, to):
    """ Writes any messages to file object ``from_`` in file object ``to`` as well.

    NOTE:
        With the various references below, we were unable to add C support. Find more details
        here: https://travis-ci.com/wellsaid-labs/Text-to-Speech/jobs/152504931

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

        # Tee flush / close / terminate
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


class RecordStandardStreams():
    """ Record output of `sys.stdout` and `sys.stderr` to a log file over an entire Python process.

    Args:
        directory (Path or str): Directory to save log files in.
        log_filename (str, optional)
    """

    # TODO: Ensure `bash_time_label()` is executed when `start` is called so that the time
    # is accurate.
    def __init__(self, directory=TEMP_PATH, log_filename='%s.log' % bash_time_label()):
        directory = Path(directory)

        self._check_invariants(directory, log_filename)

        self.log_path = directory / log_filename
        self.stop_stream_stdout = None
        self.stop_stream_stderr = None
        self.first_start = True

    def _check_invariants(self, directory, log_filename):
        assert directory.exists()
        # NOTE: Stream must be duplicated to a new file.
        assert not (directory / log_filename).exists()

    # TODO: Add `directory` and `log_filename` parameters to `start` so that it's consistent with
    # `update`.
    def start(self):
        """ Start recording `sys.stdout` and `sys.stderr` at the start of the process. """
        self.stop_stream_stdout = _duplicate_stream(sys.stdout, self.log_path)
        self.stop_stream_stderr = _duplicate_stream(sys.stderr, self.log_path)

        # NOTE: This is unable to capture the command line arguments without explicitly logging
        # them.
        if self.first_start:
            logger.info('The command line arguments are: %s', str(sys.argv))
            self.first_start = False

        return self

    def _stop(self):
        """ Stop recording `sys.stdout` and `sys.stderr`.

        NOTE: This is a private method because `RecordStandardStreams` is not intended to be
        stopped outside of an exit event.
        """
        if self.stop_stream_stdout is None or self.stop_stream_stderr is None:
            raise ValueError('Recording has already been stopped or has not been started.')

        self.stop_stream_stderr()
        self.stop_stream_stdout()

        return self

    def update(self, directory, log_filename=None):
        """ Update the recorder to use new log paths without losing any logs.

        Args:
            directory (Path or str): Directory to save log files in.
            log_filename (str, optional)
        """
        if log_filename is None:
            log_filename = self.log_path.name

        self._check_invariants(directory, log_filename)

        self._stop()

        self.log_path.replace(directory / log_filename)
        self.log_path = directory / log_filename

        self.start()
        return self
