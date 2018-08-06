import logging
import os
import shutil

import torch

from src.utils.experiment_context_manager import ExperimentContextManager
from src.utils import ROOT_PATH

logger = logging.getLogger(__name__)


def test_save_standard_streams():
    with ExperimentContextManager(label='test_save_standard_streams', min_time=-1) as context:
        # Check if 'Test' gets captured
        print('Test')
        logger.info('Test Logger')

    assert os.path.isfile(context.stdout_filename)
    assert os.path.isfile(context.stderr_filename)

    # Just `Test` print in stdout
    lines = '\n'.join([l.strip() for l in open(context.stdout_filename, 'r')])
    assert 'Test' in lines
    assert 'Test Logger' in lines

    # Nothing in stderr
    lines = [l.strip() for l in open(context.stderr_filename, 'r')]
    assert len(lines) == 0

    # Clean up files
    shutil.rmtree(context.directory)


def test_experiment():
    with ExperimentContextManager(label='test_experiment', device=torch.device('cpu')) as context:
        # Check context directory was created
        assert os.path.isdir(context.directory)
        assert os.path.isdir(context.checkpoints_directory)

        context.clean_up()

    # Automatically cleaned up
    assert not os.path.isdir(context.directory)


def test_clean_up():
    directory = os.path.join(ROOT_PATH, 'experiments', 'test_clean_up')
    os.makedirs(directory)

    file_ = os.path.join(directory, 'abc.txt')
    open(file_, 'w').close()

    with ExperimentContextManager(directory=directory) as context:
        # Confirm more than files were added
        assert sum([len(files) for _, _, files in os.walk(directory)]) > 1

        context.clean_up()

    # Confirm the first file was not deleted during ``clean_up``
    assert sum([len(files) for _, _, files in os.walk(directory)]) == 1

    # Clean up
    os.remove(file_)
    os.rmdir(directory)
