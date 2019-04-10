from unittest import mock

import logging
import os
import shutil
import sys

import pytest
import torch

from src.training_context_manager import TrainingContextManager
from src.utils import ROOT_PATH

# TODO: Add a fixture to delete any files created similar too:
# https://stackoverflow.com/questions/51737334/pytest-deleting-files-created-by-the-tested-function
# Furthermore, then deleting empty directories from which the files were already deleted.


@pytest.fixture()
def root():
    root_ = ROOT_PATH / 'experiments'
    before = set(list(root_.iterdir())) if root_.exists() else set()
    yield root_
    after = set(list(root_.iterdir())) if root_.exists() else set()

    for path in after.difference(before):
        if not path.exists():
            continue

        if path.is_dir():
            shutil.rmtree(str(path))
        elif path.is_file():
            path.unlink()

    assert before == set(list(root_.iterdir()))


@mock.patch('src.training_context_manager.assert_enough_disk_space', return_value=True)
def test_save_standard_streams(_, root):
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    print('Before Context')
    with TrainingContextManager(min_time=-1) as context:
        # Check if stdout gets captured
        print('Test')
        sys.stderr.write('Error\n')

        context.set_context_root(root / 'test_save_standard_streams')

        logger.info('Test Logger')
    print('After Context')

    assert context.stdout_filename.is_file()
    assert context.stderr_filename.is_file()

    # Just `Test` print in stdout
    lines = set(context.stdout_filename.read_text().strip().split('\n'))
    assert 'Test' in lines
    assert 'Test Logger' in lines

    assert 'Error' not in lines
    assert 'Before Context' not in lines
    assert 'After Context' not in lines

    # Nothing in stderr
    lines = set(context.stderr_filename.read_text().strip().split('\n'))
    assert 'Error' in lines

    assert 'Test Logger' not in lines
    assert 'Test' not in lines
    assert 'Before Context' not in lines
    assert 'After Context' not in lines


@mock.patch('src.training_context_manager.assert_enough_disk_space', return_value=True)
def test_experiment(_, root):
    with TrainingContextManager(device=torch.device('cpu')) as context:
        context.set_context_root(root / 'test_experiment')

        # Check context directory was created
        assert context.root_directory.is_dir()
        assert context.checkpoints_directory.is_dir()

        context.clean_up()

    # Automatically cleaned up
    assert not context.root_directory.is_dir()


@mock.patch('src.training_context_manager.assert_enough_disk_space', return_value=True)
def test_duplicate(_, root):
    with TrainingContextManager(device=torch.device('cpu')) as context:
        context.set_context_root(root / 'test_experiment')

    with TrainingContextManager(device=torch.device('cpu')) as context:
        with pytest.raises(TypeError):
            context.set_context_root(root / 'test_experiment')


# Patch inspired by:
# https://stackoverflow.com/questions/24779893/customizing-unittest-mock-mock-open-for-iteration
def mock_open(*args, **kargs):
    file_ = mock.mock_open(*args, **kargs)
    file_.return_value.__iter__ = lambda self: iter(self.readline, '')
    return file_


@mock.patch('subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('builtins.open', new_callable=mock_open, read_data='torch==0.4.0\n')
def test_check_module_versions__wrong_version(_, __):
    with pytest.raises(ValueError):
        TrainingContextManager._check_module_versions(None)


@mock.patch('subprocess.check_output', return_value='tensorflow==0.4.0'.encode())
@mock.patch('builtins.open', new_callable=mock_open, read_data='torch==0.4.0\n')
def test_check_module_versions__missing_install(_, __):
    with pytest.raises(ValueError):
        TrainingContextManager._check_module_versions(None)


@mock.patch('src.training_context_manager.assert_enough_disk_space', return_value=True)
def test_clean_up(_, root):
    directory = root / 'test_clean_up'

    with TrainingContextManager() as context:
        context.set_context_root(directory)

        file_ = directory / 'abc.txt'
        file_.write_text('')

        # Confirm more than one file was added
        assert sum([len(files) for _, _, files in os.walk(str(directory))]) > 1

        context.clean_up()

    # Confirm the first file was not deleted during ``clean_up``
    assert sum([len(files) for _, _, files in os.walk(str(directory))]) == 1


@mock.patch('src.training_context_manager.assert_enough_disk_space', return_value=True)
def test_clean_up__exception_clean_up(_, root):
    directory = root / 'test_clean_up'

    try:
        with TrainingContextManager() as context:
            context.set_context_root(directory)

            file_ = directory / 'abc.txt'
            file_.write_text('')

            # Confirm more than one file was added
            assert sum([len(files) for _, _, files in os.walk(str(directory))]) > 1

            raise ValueError()
    except ValueError:
        # Confirm the ``abc.txt`` file was not deleted during ``clean_up``
        assert sum([len(files) for _, _, files in os.walk(str(directory))]) == 1
