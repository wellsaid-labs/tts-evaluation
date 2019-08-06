import logging
import os
import sys

from src.environment import TEST_DATA_PATH
from src.utils import RecordStandardStreams
from src.utils.record_standard_streams import _duplicate_stream

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'utils'


def test__duplicate_stream(capsys):
    stdout_log = TEST_DATA_PATH_LOCAL / 'stdout.log'
    with capsys.disabled():  # Disable capsys because it messes with sys.stdout
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        stop = _duplicate_stream(sys.stdout, stdout_log)

        print('1')
        logger.info('2')
        os.system('echo 3')

        # Flush and close
        stop()
        logger.removeHandler(handler)

    assert stdout_log.is_file()
    output = stdout_log.read_text()
    assert set(output.split()) == set(['1', '2', '3'])


def test_save_standard_streams(capsys):
    with capsys.disabled():
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        stdout_name = 'stdout.log'
        stderr_name = 'stderr.log'

        recorder = RecordStandardStreams(TEST_DATA_PATH_LOCAL, stdout_name, stderr_name).start()

        # Check if stdout gets captured
        print('Test')
        sys.stderr.write('Error\n')
        logger.info('Test Logger')

        assert (TEST_DATA_PATH_LOCAL / stdout_name).is_file()
        assert (TEST_DATA_PATH_LOCAL / stderr_name).is_file()

        new_stdout_name = 'stdout_new.log'
        new_stderr_name = 'stderr_new.log'
        recorder.update(TEST_DATA_PATH_LOCAL, new_stdout_name, new_stderr_name)

        assert not (TEST_DATA_PATH_LOCAL / stdout_name).is_file()
        assert not (TEST_DATA_PATH_LOCAL / stderr_name).is_file()
        assert (TEST_DATA_PATH_LOCAL / new_stdout_name).is_file()
        assert (TEST_DATA_PATH_LOCAL / new_stderr_name).is_file()

        print('Test Update')

        # Just `Test` print in stdout
        lines = set((TEST_DATA_PATH_LOCAL / new_stdout_name).read_text().strip().split('\n'))
        assert 'Test' in lines
        assert 'Test Logger' in lines
        assert 'Test Update' in lines

        # TODO: Investigate how `stderr` also ends up in the `stdout` and `stderr` logs.
        # assert 'Error' not in lines

        # Nothing in stderr
        lines = set((TEST_DATA_PATH_LOCAL / new_stderr_name).read_text().strip().split('\n'))
        assert 'Error' in lines

        assert 'Test Logger' not in lines
        assert 'Test Update' not in lines
        assert 'Test' not in lines
