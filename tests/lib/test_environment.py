from pathlib import Path
from unittest import mock

import logging
import os
import sys
import tempfile

import pytest

import lib


def _make_log_record(name='logger_name',
                     level=logging.INFO,
                     pathname='path/file.py',
                     lineno=0,
                     message='the message',
                     args=tuple(),
                     exc_info=None,
                     func='function_name',
                     sinfo=''):
    return logging.LogRecord(name, level, pathname, lineno, message, args, exc_info, func, sinfo)


def test__colored_formatter__warning():
    """ Test `_ColoredFormatter` runs for a warning.
    """
    id_ = 0
    formatter = lib.environment._ColoredFormatter(id_)
    record = _make_log_record(level=logging.WARNING)
    formatted = formatter.format(record)
    assert record.message in formatted
    assert record.name in formatted
    assert 'WARNING' in formatted


def test__colored_formatter__error__large_id():
    """ Test `_ColoredFormatter` runs for a error with a large id.
    """
    id_ = 10000
    formatter = lib.environment._ColoredFormatter(id_)
    record = _make_log_record(level=logging.ERROR)
    formatted = formatter.format(record)
    assert record.message in formatted
    assert record.name in formatted
    assert 'ERROR' in formatted


def test__max_level_filter():
    """ Test `_MaxLevelFilter` filters correctly.
    """
    filter_ = lib.environment._MaxLevelFilter(logging.INFO)
    assert filter_.filter(_make_log_record(level=logging.INFO))
    assert not filter_.filter(_make_log_record(level=logging.WARNING))


def test_set_basic_logging_config():
    """ Test if `set_basic_logging_config` executes without error. """
    # `set_basic_logging_config` requires the root logger to have no handlers.
    logger = logging.getLogger()
    handlers = logger.handlers.copy()
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    lib.environment.set_basic_logging_config()

    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    for handler in handlers:
        logger.addHandler(handler)


def test_assert_enough_disk_space():
    """ Test that `assert_enough_disk_space` passes. """
    lib.environment.assert_enough_disk_space(min_space=0)


def test_assert_enough_disk_space__not_enough():
    """ Test that `assert_enough_disk_space` fails. """
    with pytest.raises(AssertionError):
        lib.environment.assert_enough_disk_space(min_space=1.0)


@mock.patch('lib.environment.subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('lib.environment.Path.read_text', return_value='torch==0.4.1\n')
def test_check_module_versions(_, __):
    """ Test that `check_module_versions` passes. """
    lib.environment.check_module_versions()


@mock.patch('lib.environment.subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('lib.environment.Path.read_text', return_value='torch==0.4.0\n')
def test_check_module_versions__wrong_version(_, __):
    """ Test that `check_module_versions` if the install version is incorrect. """
    with pytest.raises(RuntimeError):
        lib.environment.check_module_versions()


@mock.patch('lib.environment.subprocess.check_output', return_value='tensorflow==0.4.0'.encode())
@mock.patch('lib.environment.Path.read_text', return_value='torch==0.4.0\n')
def test_check_module_versions__missing_install(_, __):
    """ Test that `check_module_versions` if a installation is missing. """
    with pytest.raises(RuntimeError):
        lib.environment.check_module_versions()


def test_set_seed():
    """ Test that `set_seed` executes. """
    lib.environment.set_seed(123)


def test_bash_time_label():
    label = lib.environment.bash_time_label()
    assert 'PID-' in label
    assert 'DATE-' in label


def test_bash_time_label__no_pid():
    label = lib.environment.bash_time_label(False)
    assert 'PID-' not in label
    assert 'DATE-' in label


def test_bash_time_label__special_characters():
    """ Test to ensure that no bash special characters appear in the label, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    """
    label = lib.environment.bash_time_label()
    # NOTE (michael p): `:` and `=` wasn't mentioned explicitly; however, in my shell it required
    # an escape.
    for character in ([
            '`', '~', '!', '#', '$', '&', '*', '(', ')', ' ', '\t', '\n', '{', '}', '[', ']', '|',
            ';', '\'', '"', '<', '>', '?', '='
    ] + [':']):
        assert character not in label


def test_text_to_label():
    assert lib.environment.text_to_label('Michael P') == 'michael_p'


def test_get_root_path():
    """ Assuming there is `.git` directory on the root level, test if the `ROOT_PATH` is correct.
    """
    assert (lib.environment.ROOT_PATH / '.git').is_dir()


def test__duplicate_stream(capsys):
    """ Test if `_duplicate_stream` duplicates `sys.stdout`. """
    with tempfile.TemporaryDirectory() as directory:
        file_path = Path(directory) / 'stdout.log'
        with capsys.disabled():  # Disable capsys because it messes with sys.stdout
            logger = logging.getLogger(__name__)
            handler = logging.StreamHandler(sys.stdout)
            logger.addHandler(handler)
            stop = lib.environment._duplicate_stream(sys.stdout, file_path)

            print('1')
            logger.info('2')
            os.system('echo 3')

            # Flush and close
            stop()
            logger.removeHandler(handler)

        assert file_path.is_file()
        output = file_path.read_text()
        assert set(output.split()) == set(['1', '2', '3'])


def test_record_standard_streams(capsys):
    """ Test if `RecordStandardStreams` duplicates the standard streams. """
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        with capsys.disabled():
            logger = logging.getLogger(__name__)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            logger.addHandler(handler)

            log_filename = 'test.log'

            recorder = lib.environment.RecordStandardStreams(directory_path, log_filename)

            # Check if stdout gets captured
            print('Test')
            sys.stderr.write('Error\n')
            logger.info('Test Logger')

            assert (directory_path / log_filename).is_file()

            new_log_filename = 'new.log'
            recorder.update(directory_path, new_log_filename)

            assert not (directory_path / log_filename).is_file()
            assert (directory_path / new_log_filename).is_file()

            print('Test Update')

            # Just `Test` print in stdout
            lines = set((directory_path / new_log_filename).read_text().strip().split('\n'))
            assert 'Test' in lines
            assert 'Test Logger' in lines
            assert 'Test Update' in lines
            assert 'Error' in lines


def test_get_untracked_files():
    """ Test `get_untracked_files` finds the new file. """
    file_ = tempfile.NamedTemporaryFile(dir=lib.environment.ROOT_PATH)
    assert Path(file_.name).name in lib.environment.get_untracked_files().split()


def test_has_untracked_files():
    """ Test `has_untracked_files` finds the new file. """
    _ = tempfile.NamedTemporaryFile(dir=lib.environment.ROOT_PATH)
    assert lib.environment.has_untracked_files()


def test_get_last_git_commit_date():
    """ Test if `get_last_git_commit_date` executes without error. """
    assert isinstance(lib.environment.get_last_git_commit_date(), str)


def test_get_git_branch_name():
    """ Test if `get_git_branch_name` executes without error. """
    assert isinstance(lib.environment.get_git_branch_name(), str)


def test_get_tracked_changes():
    """ Test if `get_tracked_changes` executes without error. """
    assert isinstance(lib.environment.get_tracked_changes(), str)


def test_has_tracked_changes():
    """ Test if `has_tracked_changes` executes without error. """
    assert isinstance(lib.environment.has_tracked_changes(), bool)


def test_get_cuda_gpus():
    """ Test if `get_cuda_gpus` executes without error. """
    assert isinstance(lib.environment.get_cuda_gpus(), str)


def test_get_num_cuda_gpus():
    """ Test if `get_num_cuda_gpus` executes without error. """
    assert isinstance(lib.environment.get_num_cuda_gpus(), int)


def test_get_disks():
    """ Test if `get_disks` executes without error. """
    assert isinstance(lib.environment.get_disks(), (type(None), str))


def test_get_unique_cpus():
    """ Test if `get_unique_cpus` executes without error. """
    assert isinstance(lib.environment.get_unique_cpus(), (type(None), str))


def test_get_total_physical_memory():
    """ Test if `get_total_physical_memory` executes without error. """
    assert isinstance(lib.environment.get_total_physical_memory(), (type(None), int))
