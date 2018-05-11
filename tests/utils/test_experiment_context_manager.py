import argparse
import logging
import mock
import os

import torch

from src.utils.experiment_context_manager import ExperimentContextManager

logger = logging.getLogger(__name__)


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
def test_save_standard_streams(*_):
    with ExperimentContextManager(label='test_save_standard_streams') as context:
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
    os.remove(context.stdout_filename)
    os.remove(context.stderr_filename)
    os.rmdir(context.directory)


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
def test_experiment(*_):
    with ExperimentContextManager(label='test_experiment', device=-1) as context:
        # Check context directory was created
        assert os.path.isdir(context.directory)

        # Check save and load into the directory work
        path = context.save('test.pt', {'test': True})
        data = context.load('test.pt')
        assert data['test']

        # Smoke test
        context.maybe_cuda(torch.LongTensor([1, 2]))

        context.epoch(1)
        assert os.path.isdir(context.epoch_directory)

        # Clean up
        os.remove(path)

    os.rmdir(context.epoch_directory)
    os.rmdir(os.path.dirname(context.epoch_directory))
    os.remove(context.stdout_filename)
    os.remove(context.stderr_filename)
    os.rmdir(context.directory)
