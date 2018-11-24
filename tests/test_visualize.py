from matplotlib import pyplot
from unittest import mock

import matplotlib
import torch

from src.datasets import Speaker
from src.visualize import AccumulatedMetrics
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_spectrogram
from src.visualize import plot_stop_token
from src.visualize import plot_waveform
from src.visualize import spectrogram_to_image


@mock.patch('torch.distributed.is_initialized', return_value=True)
@mock.patch('torch.distributed.reduce', return_value=None)
def test_accumulated_metrics(_, __):
    metrics = AccumulatedMetrics(type_=torch)
    metrics.add_multiple_metrics({'test': torch.tensor([.25])}, 2)
    metrics.add_multiple_metrics({'test': torch.tensor([.5])}, 3)

    def callable_(key, value):
        assert key == 'test' and value == 0.4

    metrics.log_step_end(callable_)
    metrics.log_epoch_end(callable_)

    called = False

    def not_called():
        nonlocal called
        called = True

    metrics.log_step_end(not_called)
    metrics.log_epoch_end(not_called)


def test_comet_ml():
    # Smoke tests
    visualizer = CometML('', disabled=True, api_key='')
    visualizer.set_step(0)
    visualizer.log_text_and_audio('audio', 'test input', Speaker.LINDA_JOHNSON, torch.rand(100))
    figure = pyplot.figure()
    pyplot.close(figure)
    visualizer.log_multiple_figures({'figure': figure}, overwrite=True)
    visualizer.set_context('train')
    assert visualizer.context == 'train'


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
