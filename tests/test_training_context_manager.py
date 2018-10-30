from pathlib import Path
from unittest import mock

import logging
import os
import shutil
import sys

import pytest
import torch

from src.training_context_manager import TrainingContextManager
from src.utils import ROOT_PATH


def test_save_standard_streams():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    with TrainingContextManager(
            root_directory=ROOT_PATH / 'experiments' / 'test_save_standard_streams',
            min_time=-1) as context:
        # Check if 'Test' gets captured
        print('Test')
        logger.info('Test Logger')

    assert context.stdout_filename.is_file()
    assert context.stderr_filename.is_file()

    # Just `Test` print in stdout
    lines = set(context.stdout_filename.read_text().strip().split('\n'))
    assert 'Test' in lines
    assert 'Test Logger' in lines

    # Nothing in stderr
    lines = set(context.stderr_filename.read_text().strip().split('\n'))
    assert 'Test Logger' not in lines
    assert 'Test' not in lines

    # Clean up files
    shutil.rmtree(str(context.root_directory))


def test_experiment():
    with TrainingContextManager(
            root_directory=ROOT_PATH / 'experiments' / 'test_experiment',
            device=torch.device('cpu')) as context:
        # Check context directory was created
        assert context.root_directory.is_dir()
        assert context.checkpoints_directory.is_dir()

        context.clean_up()

    # Automatically cleaned up
    assert not context.root_directory.is_dir()


# Patch inspired by:
# https://stackoverflow.com/questions/24779893/customizing-unittest-mock-mock-open-for-iteration
def mock_open(*args, **kargs):
    file_ = mock.mock_open(*args, **kargs)
    file_.return_value.__iter__ = lambda self: iter(self.readline, '')
    return file_


@mock.patch('subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('builtins.open', new_callable=mock_open, read_data='torch==0.4.0\n')
def test_check_module_versions(_, __):
    with pytest.raises(ValueError):
        TrainingContextManager._check_module_versions(None)


def test_clean_up():
    directory = Path('experiments') / 'test_clean_up'

    with TrainingContextManager(root_directory=directory) as context:
        file_ = directory / 'abc.txt'
        file_.write_text('')

        # Confirm more than one file was added
        assert sum([len(files) for _, _, files in os.walk(str(directory))]) > 1

        context.clean_up()

    # Confirm the first file was not deleted during ``clean_up``
    assert sum([len(files) for _, _, files in os.walk(str(directory))]) == 1

    # Clean up
    file_.unlink()
    directory.rmdir()
