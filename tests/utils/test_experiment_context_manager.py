import argparse
import logging
import mock
import os
import shutil

import torch

from src.utils.experiment_context_manager import ExperimentContextManager

logger = logging.getLogger(__name__)


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
def test_save_standard_streams(*_):
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


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
def test_experiment(*_):
    with ExperimentContextManager(label='test_experiment', device=torch.device('cpu')) as context:
        # Check context directory was created
        assert os.path.isdir(context.directory)
        assert os.path.isdir(context.checkpoints_directory)

    # Automatically cleaned up
    assert not os.path.isdir(context.directory)
