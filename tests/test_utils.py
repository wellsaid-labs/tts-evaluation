from pathlib import Path

import logging
import os
import sys

from torch import nn
from torch.nn import functional

import numpy as np
import pytest
import torch

from src.optimizer import Optimizer
from src.utils import AnomalyDetector
from src.utils import combine_signal
from src.utils import get_total_parameters
from src.utils import Checkpoint
from src.utils import parse_hparam_args
from src.utils import ROOT_PATH
from src.utils import split_signal
from src.utils import get_weighted_standard_deviation
from src.utils import chunks
from src.utils import duplicate_stream
from src.utils import get_masked_average_norm

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
    tensor = torch.Tensor([0.5, 0.2, 0.3])
    mask = torch.Tensor([1])
    assert get_masked_average_norm(tensor) == get_masked_average_norm(tensor, mask=mask)


def test_get_weighted_standard_deviation():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]], [[0, 0.5, 0.5],
                                                                          [0, 0.5, 0.5]]])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=2)
    assert standard_deviation == pytest.approx(0.7803307175636292)


def test_get_weighted_standard_deviation_masked():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]], [[0, 0.5, 0.5],
                                                                          [0, 0.5, 0.5]]])
    mask = torch.Tensor([[1, 0], [0, 0]])
    standard_deviation = get_weighted_standard_deviation(tensor, dim=2, mask=mask)
    assert standard_deviation == pytest.approx(1.0, rel=1.0e-04)


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


def test_split_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    assert torch.equal(coarse, torch.LongTensor([255, 0, 128, 129, 128]))
    assert torch.equal(fine, torch.LongTensor([255, 0, 0, 0, 2**7]))


def test_combine_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    new_signal = combine_signal(coarse, fine, 16)
    # NOTE: 1.0 gets clipped to ``(2**15 - 1) / 2**15``
    expected_signal = torch.FloatTensor([(2**15 - 1) / 2**15, -1.0, 0, 2**-7, 2**-8])
    np.testing.assert_allclose(expected_signal.numpy(), new_signal.numpy())


def test_split_combine_signal():
    signal = torch.FloatTensor(1000).uniform_(-1.0, 1.0)
    reconstructed_signal = combine_signal(*split_signal(signal))
    np.testing.assert_allclose(signal.numpy(), reconstructed_signal.numpy(), atol=1e-04)


def test_load_most_recent_checkpoint():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.pt')
    assert isinstance(checkpoint, Checkpoint)
    assert 'tests/_test_data/checkpoint.pt' in checkpoint.path


def test_load_most_recent_checkpoint_none():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.abc')
    assert checkpoint is None


def test_load_save_checkpoint():
    model = nn.LSTM(10, 10)
    optimizer = Optimizer(
        torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
    checkpoint = Checkpoint('tests/_test_data/', model=model, step=10, optimizer=optimizer)
    filename = checkpoint.save()
    assert filename.is_file()

    # Smoke test
    Checkpoint.from_path(filename)

    # Clean up
    filename.unlink()
