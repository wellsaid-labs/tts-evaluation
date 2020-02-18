from matplotlib import pyplot

import matplotlib
import torch

from src.datasets import Gender
from src.datasets import Speaker
from src.visualize import CometML
from src.visualize import plot_attention
from src.visualize import plot_loss_per_frame
from src.visualize import plot_spectrogram
from src.visualize import plot_mel_spectrogram
from src.visualize import plot_stop_token
from src.visualize import plot_waveform
from src.visualize import spectrogram_to_image


def test_comet_ml():
    # Smoke tests
    visualizer = CometML('', disabled=True)
    visualizer.set_step(0)
    visualizer.log_audio(
        tag='audio',
        text='test input',
        speaker=Speaker('Test Speaker', Gender.MALE),
        predicted_audio=torch.rand(100),
        gold_audio=torch.rand(100))
    figure = pyplot.figure()
    pyplot.close(figure)
    visualizer.log_figures({'figure': figure}, overwrite=True)
    visualizer.set_context('train')
    assert visualizer.context == 'train'
    visualizer.log_current_epoch(0)
    visualizer.log_epoch_end(0)


def test_plot_spectrogram():
    arr = torch.rand(5, 6)
    figure = plot_spectrogram(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_mel_spectrogram():
    arr = torch.rand(5, 6)
    figure = plot_mel_spectrogram(arr)
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


def test_plot_loss_per_frame():
    arr = torch.rand(5)
    figure = plot_loss_per_frame(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)


def test_plot_stop_token():
    arr = torch.rand(5)
    figure = plot_stop_token(arr)
    assert isinstance(figure, matplotlib.figure.Figure)
    pyplot.close(figure)
