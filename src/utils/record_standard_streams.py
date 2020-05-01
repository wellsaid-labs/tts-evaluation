from pathlib import Path

import atexit
import os
import sys
import time
import subprocess
import logging

from src.environment import TEMP_PATH
from src.utils.utils import bash_time_label

logger = logging.getLogger(__name__)


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
