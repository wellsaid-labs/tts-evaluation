from matplotlib import pyplot

import matplotlib
import torch

from src.visualize import plot_attention
from src.visualize import plot_spectrogram
from src.visualize import plot_stop_token
from src.visualize import plot_waveform
from src.visualize import spectrogram_to_image
from src.visualize import CometML


def test_comet_ml():
    # Smoke tests
    visualizer = CometML('', disabled=True, api_key='')
    visualizer.set_step(0)
    visualizer.log_text_and_audio('audio', 'test input', torch.rand(100))
    visualizer.log_audio('audio', torch.rand(100))


def test_plot_spectrogram():
    arr = torch.rand(5, 6)
    figure = plot_spectrogram(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_spectrogram_to_image():
    arr = torch.rand(5, 6)
    image = spectrogram_to_image(arr)
    assert image.shape == (6, 5, 3)


def test_plot_attention():
    arr = torch.rand(5, 6)
    figure = plot_attention(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_waveform():
    arr = torch.rand(5)
    figure = plot_waveform(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_stop_token():
    arr = torch.rand(5)
    figure = plot_stop_token(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)
