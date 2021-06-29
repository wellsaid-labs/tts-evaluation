import matplotlib
import matplotlib.figure
import torch

import lib


def test_plot_alignments():
    assert isinstance(lib.visualize.plot_alignments(torch.rand(5, 6)), matplotlib.figure.Figure)


def test_plot_logits():
    assert isinstance(lib.visualize.plot_logits(torch.rand(5)), matplotlib.figure.Figure)


def test_plot_waveform():
    assert isinstance(lib.visualize.plot_waveform(torch.rand(5), 24000), matplotlib.figure.Figure)


def test_plot_mel_spectrogram():
    figure = lib.visualize.plot_mel_spectrogram(
        torch.rand(5, 6),
        sample_rate=24000,
        frame_hop=256,
        lower_hertz=None,
        upper_hertz=None,
    )
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_spectrogram():
    figure = lib.visualize.plot_spectrogram(torch.rand(5, 6), sample_rate=24000, frame_hop=256)
    assert isinstance(figure, matplotlib.figure.Figure)
