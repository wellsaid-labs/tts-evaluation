import argparse
import mock
import os
import sys

import torch

from src.experiment_context_manager import ExperimentContextManager


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
@mock.patch('src.experiment_context_manager.ExperimentContextManager.git_commit')
def test_save_standard_streams(*_):
    with ExperimentContextManager(label='test_save_standard_streams') as context:
        # Check if 'Test' gets captured
        print('Test')

        stdout_filename = context.stdout_filename
        stderr_filename = context.stderr_filename

    # Reset streams
    sys.stdout = sys.stdout.stream
    sys.stderr = sys.stderr.stream

    assert os.path.isfile(stdout_filename)
    assert os.path.isfile(stderr_filename)

    # Just `Test` print in stdout
    lines = [l.strip() for l in open(stdout_filename, 'r')]
    assert lines[0] == 'Test'

    # Nothing in stderr
    lines = [l.strip() for l in open(stderr_filename, 'r')]
    assert len(lines) == 0

    # Clean up files
    os.remove(stdout_filename)
    os.remove(stderr_filename)


@mock.patch('argparse.ArgumentParser.parse_args', return_value=argparse.Namespace(message='test'))
@mock.patch('src.experiment_context_manager.ExperimentContextManager.git_commit')
def test_experiment(*_):
    with ExperimentContextManager(label='test_experiment') as context:
        # Check context directory was created
        assert os.path.isdir(context.directory)

        # Check save and load into the directory work
        path = context.save('test.pt', {'test': True})
        data = context.load('test.pt')
        assert data['test']

        # Smore test
        context.maybe_cuda(torch.LongTensor([1, 2]))

        # Clear up
        os.remove(path)
        os.rmdir(context.directory)
        os.remove(context.stdout_filename)
        os.remove(context.stderr_filename)
