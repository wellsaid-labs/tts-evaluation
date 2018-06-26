import os

from torch import nn
from torch.nn import functional
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import figure_to_numpy_array
from src.utils import get_total_parameters
from src.utils import parse_hparam_args
from src.utils import plot_attention
from src.utils import plot_log_mel_spectrogram
from src.utils import plot_stop_token
from src.utils import plot_waveform
from src.utils import ROOT_PATH
from src.utils import spectrogram_to_image
from src.utils import split_dataset
from src.utils import split_signal
from src.utils import combine_signal


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


def test_get_total_parameters():
    model = MockModel()
    assert 62006 == get_total_parameters(model)


def test_get_root_path():
    root_path = ROOT_PATH
    assert os.path.isfile(os.path.join(root_path, 'requirements.txt'))


def test_split_dataset():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits, deterministic_shuffle=False) == [[1, 2, 3], [4], [5]]


def test_split_dataset_shuffle():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits) == [[4, 2, 5], [3], [1]]


def test_plot_log_mel_spectrogram():
    arr = torch.rand(5, 6)
    figure = plot_log_mel_spectrogram(arr)
    assert isinstance(figure, np.ndarray)


def test_spectrogram_to_image():
    arr = np.random.rand(5, 6)
    image = spectrogram_to_image(arr)
    assert image.shape == (6, 5, 3)


def test_plot_attention():
    arr = np.random.rand(5, 6)
    figure = plot_attention(arr)
    assert isinstance(figure, np.ndarray)


def test_plot_waveform():
    arr = np.random.rand(5)
    figure = plot_waveform(arr)
    assert isinstance(figure, np.ndarray)


def test_plot_stop_token():
    arr = np.random.rand(5)
    figure = plot_stop_token(arr)

    filename = 'tests/_test_data/sample_plot.png'
    image = Image.fromarray(figure, 'RGB')
    image.save(filename)

    assert os.path.isfile(filename)

    # Clean up
    os.remove(filename)


def test_figure_to_numpy_array():
    y = [1, 4, 5, 6]
    figure = plt.figure()
    plt.plot(list(range(len(y))), y)
    plt.close(figure)

    assert figure_to_numpy_array(figure).shape == (480, 640, 3)


def test_parse_hparam_args():
    hparam_args = ['--foo 0.01', '--bar WaveNet', '--moo=1']
    assert parse_hparam_args(hparam_args) == {'foo': 0.01, 'bar': 'WaveNet', 'moo': 1}


def test_split_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    assert torch.equal(coarse, torch.FloatTensor([255, 0, 128, 129, 128]))
    assert torch.equal(fine, torch.FloatTensor([255, 0, 0, 0, 2**7]))


def test_combine_signal():
    signal = torch.FloatTensor([1.0, -1.0, 0, 2**-7, 2**-8])
    coarse, fine = split_signal(signal, 16)
    new_signal = combine_signal(coarse, fine, 16)
    # NOTE: 1.0 gets clipped to ``(2**15 - 1) / 2**15``
    expected_signal = torch.FloatTensor([(2**15 - 1) / 2**15, -1.0, 0, 2**-7, 2**-8])
    np.testing.assert_allclose(expected_signal.numpy(), new_signal.numpy())
