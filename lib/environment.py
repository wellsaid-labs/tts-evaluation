# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from pathlib import Path

import atexit
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import typing
import typing_extensions

from hparams import configurable
from hparams import HParam

import torch
import torchnlp.random

logger = logging.getLogger(__name__)

ROOT_PATH = Path(__file__).parents[1].resolve()  # Repository root path

TEST_DATA_PATH = ROOT_PATH / 'tests' / '_test_data'  # TODO: Move to tests/_utils.py?

IS_TESTING_ENVIRONMENT = 'pytest' in sys.modules

# NOTE: You can experiment with these codes in your console like so:
# `echo -e '\033[43m \033[30m hi \033[0m'`


# Inspired by:
# https://godoc.org/github.com/whitedevops/colors
# https://github.com/tartley/colorama
class AnsiCodes:
    RESET_ALL: typing_extensions.Final = '\033[0m'
    BOLD: typing_extensions.Final = '\033[1m'
    DIM: typing_extensions.Final = '\033[2m'
    UNDERLINED: typing_extensions.Final = '\033[4m'
    BLINK: typing_extensions.Final = '\033[5m'
    REVERSE: typing_extensions.Final = '\033[7m'
    HIDDEN: typing_extensions.Final = '\033[8m'
    RESET_BOLD: typing_extensions.Final = '\033[21m'
    RESET_DIM: typing_extensions.Final = '\033[22m'
    RESET_UNDERLINED: typing_extensions.Final = '\033[24m'
    RESET_BLINK: typing_extensions.Final = '\033[25m'
    RESET_REVERSE: typing_extensions.Final = '\033[27m'
    RESET_HIDDEN: typing_extensions.Final = '\033[28m'
    DEFAULT: typing_extensions.Final = '\033[39m'
    BLACK: typing_extensions.Final = '\033[30m'
    RED: typing_extensions.Final = '\033[31m'
    GREEN: typing_extensions.Final = '\033[32m'
    YELLOW: typing_extensions.Final = '\033[33m'
    BLUE: typing_extensions.Final = '\033[34m'
    MAGENTA: typing_extensions.Final = '\033[35m'
    CYAN: typing_extensions.Final = '\033[36m'
    LIGHT_GRAY: typing_extensions.Final = '\033[37m'
    DARK_GRAY: typing_extensions.Final = '\033[90m'
    LIGHT_RED: typing_extensions.Final = '\033[91m'
    LIGHT_GREEN: typing_extensions.Final = '\033[92m'
    LIGHT_YELLOW: typing_extensions.Final = '\033[93m'
    LIGHT_BLUE: typing_extensions.Final = '\033[94m'
    LIGHT_MAGENTA: typing_extensions.Final = '\033[95m'
    LIGHT_CYAN: typing_extensions.Final = '\033[96m'
    WHITE: typing_extensions.Final = '\033[97m'
    BACKGROUND_DEFAULT: typing_extensions.Final = '\033[49m'
    BACKGROUND_BLACK: typing_extensions.Final = '\033[40m'
    BACKGROUND_RED: typing_extensions.Final = '\033[41m'
    BACKGROUND_GREEN: typing_extensions.Final = '\033[42m'
    BACKGROUND_YELLOW: typing_extensions.Final = '\033[43m'
    BACKGROUND_BLUE: typing_extensions.Final = '\033[44m'
    BACKGROUND_MAGENTA: typing_extensions.Final = '\033[45m'
    BACKGROUND_CYAN: typing_extensions.Final = '\033[46m'
    BACKGROUND_LIGHT_GRAY: typing_extensions.Final = '\033[47m'
    BACKGROUND_DARK_GRAY: typing_extensions.Final = '\033[100m'
    BACKGROUND_LIGHT_RED: typing_extensions.Final = '\033[101m'
    BACKGROUND_LIGHT_GREEN: typing_extensions.Final = '\033[102m'
    BACKGROUND_LIGHT_YELLOW: typing_extensions.Final = '\033[103m'
    BACKGROUND_LIGHT_BLUE: typing_extensions.Final = '\033[104m'
    BACKGROUND_LIGHT_MAGENTA: typing_extensions.Final = '\033[105m'
    BACKGROUND_LIGHT_CYAN: typing_extensions.Final = '\033[106m'
    BACKGROUND_WHITE: typing_extensions.Final = '\033[107m'


class _ColoredFormatter(logging.Formatter):
    """ Logging formatter with color.

    Args:
        id_: An id to be printed along with all logs.
    """

    ID_COLOR_ROTATION = [
        AnsiCodes.BACKGROUND_LIGHT_BLUE + AnsiCodes.BLACK,
        AnsiCodes.BACKGROUND_LIGHT_CYAN + AnsiCodes.BLACK,
        AnsiCodes.BACKGROUND_LIGHT_GREEN + AnsiCodes.BLACK,
        AnsiCodes.BACKGROUND_LIGHT_MAGENTA + AnsiCodes.BLACK,
        AnsiCodes.BACKGROUND_WHITE + AnsiCodes.BLACK,
        AnsiCodes.BLUE,
        AnsiCodes.CYAN,
        AnsiCodes.GREEN,
        AnsiCodes.MAGENTA,
    ]

    def __init__(self, id_: int):
        super().__init__()
        logging.Formatter.__init__(
            self, f"{AnsiCodes.DARK_GRAY}[%(asctime)s][{AnsiCodes.RESET_ALL}"
            f"{self.ID_COLOR_ROTATION[id_ % len(self.ID_COLOR_ROTATION)]}{id_}{AnsiCodes.RESET_ALL}"
            f"{AnsiCodes.DARK_GRAY}][%(name)s][%(levelname)s]{AnsiCodes.RESET_ALL} %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno > logging.WARNING:
            record.levelname = AnsiCodes.RED + record.levelname + AnsiCodes.RESET_ALL
        elif record.levelno > logging.INFO:
            record.levelname = AnsiCodes.YELLOW + record.levelname + AnsiCodes.RESET_ALL
        return logging.Formatter.format(self, record)


class _MaxLevelFilter(logging.Filter):
    """ Filter out logs above a certain level.

    NOTE: This is the opposite of `logging.Handler.setLevel` which sets a mininum threshold.
    """

    def __init__(self, maxLevel: int):
        super().__init__()
        self.maxLevel = maxLevel

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.maxLevel


def set_basic_logging_config(id_: int = os.getpid()):
    """
    Inspired by: `logging.basicConfig`

    Set a basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured. It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a `StreamHandler` which writes to
    `sys.stdout` and `sys.stderr`, set a formatter, and
    add the handler to the root logger.

    Args:
        id_: An id to be printed along with all logs.
    """
    root = logging.getLogger()
    if len(root.handlers) == 0:
        root.setLevel(logging.INFO)

        formatter = _ColoredFormatter(id_)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        handler.addFilter(_MaxLevelFilter(logging.INFO))
        root.addHandler(handler)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(formatter)
        root.addHandler(handler)


@configurable
def assert_enough_disk_space(min_space: float = HParam()):
    """ Check if there is enough disk space.

    Args:
        min_space: Minimum percentage of free disk space.
    """
    st = os.statvfs(ROOT_PATH)
    free = st.f_bavail * st.f_frsize
    total = st.f_blocks * st.f_frsize
    available = free / total
    assert available > min_space, f"Not enough available ({available} < {min_space}) disk space."


def check_module_versions(requirements_path: Path = ROOT_PATH / 'requirements.txt'):
    """ Check if installed module versions respect `requirements.txt` specifications. """
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    split = freeze.decode('utf-8').split()
    for line in requirements_path.read_text().split():
        line = line.strip()
        if '==' in line:
            specification = line.split()[0]
            package = specification.split('==')[0]
            installed = [p for p in split if p.split('==')[0] == package]
            if not len(installed) == 1:
                raise RuntimeError(f"{package} not installed")
            if not specification == installed[0]:
                # NOTE: RuntimeError could cause "Illegal seek" while running `pytest`.
                raise RuntimeError(
                    f"Found incorrect installed version ({specification} =/= {installed[0]})")


@configurable
def set_seed(seed: int = HParam()):
    """ Set package random generator seed(s). """
    logger.info('Setting seed to be %d', seed)
    torchnlp.random.set_seed(seed)


def bash_time_label(add_pid: bool = True) -> str:
    """ Get a bash friendly string representing the time and process pid.

    NOTE: This string is optimized for sorting by ordering units of time from largest to smallest.
    NOTE: This string avoids any special bash characters, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    NOTE: `os.getpid` is often used by routines that generate unique identifiers, learn more:
    http://manpages.ubuntu.com/manpages/cosmic/man2/getpid.2.html

    Args:
        add_pid: If `True` add the process PID to the label.
    """
    label = str(time.strftime('DATE-%Yy%mm%dd-%Hh%Mm%Ss', time.localtime()))
    if add_pid:
        label += f"_PID-{os.getpid()}"
    return label


def text_to_label(text: str) -> str:
    """ Get a label like 'hilary_noriega' from text like 'Hilary Noriega'. """
    return text.lower().replace(' ', '_')


def _duplicate_stream(from_: typing.TextIO, to: Path) -> typing.Callable[[], None]:
    """ Duplicates writes to file object `from_` and writes them to `to`.

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
        from_
        to: File to write to.

    Returns:
        callable: If called the duplication is stopped.
    """
    from_.flush()

    to = Path(to)
    to.touch()

    # Keep a file descriptor open to the original file object
    original_fileno = os.dup(from_.fileno())
    tee = subprocess.Popen(['tee', '-a', str(to)], stdin=subprocess.PIPE)
    time.sleep(0.01)  # NOTE: `tee` needs time to open
    assert tee.stdin is not None
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
        directory: Directory to save log files in.
        log_filename
    """

    def __init__(self,
                 directory: Path = Path(tempfile.gettempdir()),
                 log_filename: str = f"{bash_time_label()}.log"):
        self._check_invariants(directory, log_filename)
        self.log_path = directory / log_filename
        self._start()

    def _check_invariants(self, directory: Path, log_filename: str):
        assert directory.exists()
        assert not (directory / log_filename).exists(), 'Cannot overwrite existing file.'

    def _start(self) -> RecordStandardStreams:
        """ Start recording `sys.stdout` and `sys.stderr`. """
        self.stop_stream_stdout = _duplicate_stream(sys.stdout, self.log_path)
        self.stop_stream_stderr = _duplicate_stream(sys.stderr, self.log_path)
        return self

    def _stop(self) -> RecordStandardStreams:
        """ Stop recording `sys.stdout` and `sys.stderr`.

        NOTE: `RecordStandardStreams` was designed to record a process from start to finish;
        therefore, there is no public `stop` function.
        """
        self.stop_stream_stderr()
        self.stop_stream_stdout()
        return self

    def update(self,
               directory: Path,
               log_filename: typing.Optional[str] = None) -> RecordStandardStreams:
        """ Update the recorder to use new log paths without losing any logs.

        Args:
            directory: Directory to save log files in.
            log_filename
        """
        log_filename = self.log_path.name if log_filename is None else log_filename
        self._check_invariants(directory, log_filename)
        self._stop()
        self.log_path.replace(directory / log_filename)
        self.log_path = directory / log_filename
        self._start()
        return self


def get_untracked_files() -> str:
    """ Get a formatted string describing untracked files by `git`.

    Learn more:
    https://stackoverflow.com/questions/3801321/git-list-only-untracked-files-also-custom-commands

    Example:
        >>> get_untracked_files()
        'lib/untracked.py'
    """
    command = 'git ls-files --others --exclude-standard'
    return subprocess.check_output(command, shell=True).decode().strip()


def has_untracked_files() -> bool:
    """ Return `True` is the `git` service has untracked files.

    Learn more:
    https://stackoverflow.com/questions/3801321/git-list-only-untracked-files-also-custom-commands
    """
    return len(get_untracked_files()) > 0


def get_last_git_commit_date() -> str:
    """ Get a formatted string describing the last commit date by `git`.

    Learn more:
    https://stackoverflow.com/questions/25563455/how-do-i-get-last-commit-date-from-git-repository

    Example:
        >>> get_last_git_commit_date()
        'Wed Sep 9 18:56:27 2020 -0700'
    """
    return subprocess.check_output('git log -1 --format=%cd', shell=True).decode().strip()


def get_git_branch_name() -> str:
    """ Get a the name of the current `git` branch.

    Learn more:
    https://stackoverflow.com/questions/6245570/how-to-get-the-current-branch-name-in-git

    Example:
        >>> get_git_branch_name()
        'master'
    """
    return subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()


def get_tracked_changes() -> str:
    """ Get a list of changed (created, deleted, modified) tracked files by `git`.

    Learn more:
    https://stackoverflow.com/questions/9915543/git-list-of-new-modified-deleted-files/38677776

    Example:
        >>> print(get_tracked_changes())
        M bin/README.md
         M lib/environment.py
    """
    return subprocess.check_output(
        'git status --porcelain --untracked-files=no', shell=True).decode().strip()


def has_tracked_changes() -> bool:
    """ Return `True` if there are active tracked changes by `git` in the working area. """
    return len(get_tracked_changes()) > 0


def get_cuda_gpus() -> str:
    """ Get a formatted string listing the system Nvidia CUDA enabled GPUs.

    Example:
        >>> get_gpus()
        GPU 0: Tesla T4 (UUID: GPU-4b9a3bdb-5109-27d9-635c-2825eb5f0bc6)
        GPU 1: Tesla T4 (UUID: GPU-d3c032d0-dd32-6b9b-8b74-70ecf9f9551b)
        GPU 2: Tesla T4 (UUID: GPU-debffe99-e0cf-af1f-67f8-6f00f588d7dd)
        GPU 3: Tesla T4 (UUID: GPU-2f28bad6-87a7-d0c2-60cf-91e7fa7f6ee8)
    """
    if torch.cuda.is_available():
        return subprocess.check_output('nvidia-smi --list-gpus', shell=True).decode().strip()
    return ''


def get_num_cuda_gpus() -> int:
    """ Get the number of Nvidia CUDA enabled GPUs. """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def get_disks() -> typing.Optional[str]:
    """ Get a formatted string listing the system disks.

    NOTE: This only supports `Linux` and it'll return `None` otherwise.

    Learn more:
    https://unix.stackexchange.com/questions/4561/how-do-i-find-out-what-hard-disks-are-in-the-system

    Example:
        >>> get_disks()
        *-storage:0
            description: Non-Volatile memory controller
            product: Amazon.com, Inc.
            vendor: Amazon.com, Inc.
            physical id: 4 bus
            info: pci@0000:00:04.0
            version: 00
            width: 32 bits
            clock: 33MHz
            capabilities: storage nvm_express bus_master cap_list
            configuration: driver=nvme latency=0
            resources: irq:11
            memory:fe010000-fe013fff
        *-storage:1
            description: Non-Volatile memory controller
            product: NVMe SSD Controller
            vendor: Amazon.com, Inc. physical
            id: 1f bus info: pci@0000:00:1f.0
            version: 00 width: 32 bits
            clock: 33MHz
            capabilities: storage nvm_express bus_master cap_list
            configuration: driver=nvme latency=0
            resources: irq:0
            memory:fe018000-fe01bfff
            memory:fe900000-fe901fff
    """
    if platform.system() == 'Linux':
        return subprocess.check_output(
            'lshw -class disk -class storage', shell=True).decode().strip()
    return None


def get_unique_cpus() -> typing.Optional[str]:
    """ Get a formatted string listing the system unique CPUs.

    NOTE: This only supports `Linux` and it'll return `None` otherwise.

    Example:
        >>> get_unique_cpus()
        'Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz'
    """
    if platform.system() == 'Linux':
        return subprocess.check_output(
            "awk '/model name/ {$1=$2=$3=\"\"; print $0}' /proc/cpuinfo | uniq",
            shell=True).decode().strip()
    return None


def get_total_physical_memory() -> typing.Optional[int]:
    """ Get a system's total physical memory in kilobytes.

    NOTE: This only supports `Linux` and it'll return `None` otherwise.

    Learn more:
    https://stackoverflow.com/questions/20348007/how-can-i-find-out-the-total-physical-memory-ram-of-my-linux-box-suitable-to-b

    Example:
        >>> get_total_physical_memory()
        195688856
    """
    if platform.system() == 'Linux':
        command = "awk '/MemTotal/ {print $2}' /proc/meminfo"
        return int(subprocess.check_output(command, shell=True).decode().strip())
    return None
