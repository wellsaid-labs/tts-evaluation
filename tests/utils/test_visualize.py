from matplotlib import pyplot
from pathlib import Path

import matplotlib
import numpy as np
import shutil
import torch

from src.utils import plot_attention
from src.utils import plot_log_mel_spectrogram
from src.utils import plot_stop_token
from src.utils import plot_waveform
from src.utils import spectrogram_to_image
from src.utils import Tensorboard


def test_tensorboard():
    directory = Path('tests/_test_data/tensorboard')

    # Smoke tests
    tensorboard = Tensorboard(log_dir=directory)
    tensorboard.add_scalar('scalar', 0, 0)
    tensorboard.add_stop_token('stop_token', torch.rand(6), 0)
    tensorboard.add_waveform('waveform', torch.rand(6), 0)
    tensorboard.add_log_mel_spectrogram('log_mel_spectrogram', torch.rand(5, 6), 0)
    tensorboard.add_attention('attention', torch.rand(5, 6), 0)
    tensorboard.add_audio('audio', 'waveform', torch.rand(6), 0)
    with tensorboard.set_step(0):
        tensorboard.add_text('text', 'text', 0)

    tensorboard.close()

    assert directory.is_dir()
    shutil.rmtree(str(directory))


def test_plot_log_mel_spectrogram():
    arr = torch.rand(5, 6)
    figure = plot_log_mel_spectrogram(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_spectrogram_to_image():
    arr = np.random.rand(5, 6)
    image = spectrogram_to_image(arr)
    assert image.shape == (6, 5, 3)


def test_plot_attention():
    arr = np.random.rand(5, 6)
    figure = plot_attention(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_waveform():
    arr = np.random.rand(5)
    figure = plot_waveform(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_stop_token():
    arr = np.random.rand(5)
    figure = plot_stop_token(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)
