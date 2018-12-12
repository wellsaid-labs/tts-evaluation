from collections import namedtuple
from pathlib import Path
from unittest import mock

import logging
import os
import sys

from torch import nn
from torch.nn import functional

import pytest
import torch

from src.optimizer import Optimizer
from src.utils import AnomalyDetector
from src.utils import get_total_parameters
from src.utils import Checkpoint
from src.utils import parse_hparam_args
from src.utils import ROOT_PATH
from src.utils import get_weighted_standard_deviation
from src.utils import chunks
from src.utils import duplicate_stream
from src.utils import get_masked_average_norm
from src.utils import evaluate
from src.utils import collate_sequences
from src.utils import tensors_to
from src.utils import identity
from src.utils import DataLoader


def test_data_loader():
    dataset = [1]
    for batch in DataLoader(
            dataset, trial_run=True, post_processing_fn=lambda x: x + 1, load_fn=lambda x: x + 1):
        assert len(batch) == 1
        assert batch[0] == 3


def test_collate_sequences():
    TestTuple = namedtuple('TestTuple', ['t'])

    tensor = torch.Tensor(1)
    assert collate_sequences([tensor, tensor])[0].shape == (2, 1)
    assert collate_sequences([[tensor], [tensor]])[0][0].shape == (2, 1)
    assert collate_sequences([{'t': tensor}, {'t': tensor}])['t'][0].shape == (2, 1)
    assert collate_sequences([TestTuple(t=tensor), TestTuple(t=tensor)]).t[0].shape == (2, 1)
    assert collate_sequences(['test', 'test']) == ['test', 'test']


@mock.patch('torch.is_tensor')
def test_tensors_to(mock_is_tensor):
    TestTuple = namedtuple('TestTuple', ['t'])

    mock_tensor = mock.Mock()
    mock_is_tensor.side_effect = lambda m, **kwargs: m == mock_tensor
    tensors_to(mock_tensor, device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()

    returned = tensors_to({'t': [mock_tensor]}, device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, dict)

    returned = tensors_to([mock_tensor], device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, list)

    returned = tensors_to(tuple([mock_tensor]), device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, tuple)

    returned = tensors_to(TestTuple(t=mock_tensor), device=torch.device('cpu'))
    mock_tensor.to.assert_called_once()
    mock_tensor.to.reset_mock()
    assert isinstance(returned, TestTuple)


def test_evaluate():
    model = nn.LSTM(10, 10).train(mode=True)
    with evaluate(model, device=torch.device('cpu')):
        assert not model.training
    assert model.training


def test_identity():
    input = 2
    assert identity(input) == input


stdout_log = Path('tests/_test_data/stdout.log')


@pytest.fixture
def unlink_stdout_log():
    yield
    if stdout_log.is_file():
        stdout_log.unlink()


@pytest.mark.usefixtures('unlink_stdout_log')
def test_duplicate_stream(capsys):
    assert not stdout_log.is_file()  # Ensure everything is clean up

    with capsys.disabled():  # Disable capsys because it messes with sys.stdout
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        stop = duplicate_stream(sys.stdout, stdout_log)

        print('1')
        logger.info('2')
        os.system('echo 3')

        # Flush and close
        stop()
        logger.removeHandler(handler)

    assert stdout_log.is_file()
    output = stdout_log.read_text()
    assert set(output.split()) == set(['1', '2', '3'])


def test_chunks():
    assert list(chunks([1, 2, 3], 2)) == [[1, 2], [3]]


def test_get_masked_average_norm():
    tensor = torch.Tensor([0.5, 0.2, 0.3])
    expanded = tensor.unsqueeze(1).expand(3, 4)
    assert get_masked_average_norm(tensor) == get_masked_average_norm(expanded)


def test_get_masked_average_norm_masked():
    tensor = torch.randn(3, 4, 5)
    mask = torch.FloatTensor(3, 5).fill_(1)
    assert get_masked_average_norm(
        tensor, dim=1) == get_masked_average_norm(
            tensor, mask=mask, dim=1)


def test_get_weighted_standard_deviation_constant():
    tensor = torch.Tensor([0, 1, 0])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=0)
    # Standard deviation for a constant
    assert standard_deviation == 0.0


def test_get_weighted_standard_deviation_bias():
    tensor = torch.Tensor([.25, .25, .25, .25])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=0)
    # Population standard deviation for 1,2,3,4
    assert standard_deviation == pytest.approx(1.1180339887499)


def test_get_weighted_standard_deviation():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=2)
    assert standard_deviation == pytest.approx(0.5791246294975281)


def test_get_weighted_standard_deviation_masked():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    mask = torch.Tensor([[1, 0], [0, 0]])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=2, mask=mask)
    # Population standard deviation for 1,2,3
    assert standard_deviation == pytest.approx(0.81649658093, rel=1.0e-04)


class MockModel(nn.Module):
    # REFERENCE: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_anomaly_detector():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)


def test_anomaly_detector_type_low():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_LOW)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert not anomaly_detector.step(2)


def test_anomaly_detector_type_both():
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_BOTH)
    for _ in range(min_steps):
        assert not anomaly_detector.step(1)
    assert anomaly_detector.step(2)
    assert anomaly_detector.step(0)


def test_get_total_parameters():
    model = MockModel()
    assert 62006 == get_total_parameters(model)


def test_get_root_path():
    assert (ROOT_PATH / 'requirements.txt').is_file()


def test_parse_hparam_args():
    hparam_args = ['--foo 0.01', '--bar WaveNet', '--moo=1']
    assert parse_hparam_args(hparam_args) == {'foo': 0.01, 'bar': 'WaveNet', 'moo': 1}


def test_load_most_recent_checkpoint():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.pt')
    assert isinstance(checkpoint, Checkpoint)
    assert 'tests/_test_data/checkpoint.pt' in str(checkpoint.path)


def test_load_most_recent_checkpoint_none():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.abc')
    assert checkpoint is None


def test_load_save_checkpoint():
    model = nn.LSTM(10, 10)
    optimizer = Optimizer(
        torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
    checkpoint = Checkpoint(
        directory='tests/_test_data/', model=model, step=10, optimizer=optimizer)
    filename = checkpoint.save()
    assert filename.is_file()

    # Smoke test
    Checkpoint.from_path(filename)

    # Clean up
    filename.unlink()
