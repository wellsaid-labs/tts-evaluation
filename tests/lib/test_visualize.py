from matplotlib import pyplot

import hparams
import matplotlib
import pytest
import torch

import lib


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration for `lib.visualize`. """
    sample_rate = 24000
    hparams.add_config({
        lib.audio.write_audio: hparams.HParams(sample_rate=sample_rate),
    })
    yield
    hparams.clear_config()


def test_plot_alignments():
    assert isinstance(lib.visualize.plot_alignments(torch.rand(5, 6)), matplotlib.figure.Figure)


def test_plot_line_graph_of_logits():
    assert isinstance(
        lib.visualize.plot_line_graph_of_logits(torch.rand(5)), matplotlib.figure.Figure)


def test_plot_waveform():
    assert isinstance(lib.visualize.plot_waveform(torch.rand(5), 24000), matplotlib.figure.Figure)


def test_plot_mel_spectrogram():
    figure = lib.visualize.plot_mel_spectrogram(
        torch.rand(5, 6), sample_rate=24000, frame_hop=256, lower_hertz=None, upper_hertz=None)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_plot_spectrogram():
    figure = lib.visualize.plot_spectrogram(torch.rand(5, 6), sample_rate=24000, frame_hop=256)
    assert isinstance(figure, matplotlib.figure.Figure)


def test_comet_ml_experiment():
    """ Test if `lib.visualize.CometMLExperiment` initializes, and the patched functions execute.
    """
    comet = lib.visualize.CometMLExperiment(disabled=True)
    with comet.set_context('train'):
        assert comet.context == 'train'
        comet.set_step(0)
        comet.set_step(0)
        comet.set_step(1)
        comet.log_audio(
            metadata='random metadata',
            audio={
                'predicted_audio': torch.rand(100),
                'gold_audio': torch.rand(100)
            })
        figure = pyplot.figure()
        pyplot.close(figure)
        comet.log_figures({'figure': figure}, overwrite=True)
        comet.log_current_epoch(0)
        comet.log_epoch_end(0)
        comet.set_name('name')
        comet.add_tags(['tag'])
