# NOTE: Must import `comet_ml` before `torch`
import comet_ml  # noqa

import logging
import matplotlib
import os
import torch
import io
import base64

matplotlib.use('Agg', warn=False)

from dotenv import load_dotenv
from comet_ml import Experiment
from comet_ml import ExistingExperiment
from matplotlib import cm as colormap
from matplotlib import pyplot

import librosa.display
import numpy as np

from src.hparams import configurable

logger = logging.getLogger(__name__)


def plot_attention(alignment):
    """ Plot alignment of attention.

    Args:
        alignment (numpy.array or torch.Tensor [decoder_timestep, encoder_timestep]): Attention
            alignment weights computed at every timestep of the decoder.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    if torch.is_tensor(alignment):
        alignment = alignment.detach().cpu().numpy()

    alignment = np.transpose(alignment)
    pyplot.style.use('ggplot')
    figure, axis = pyplot.subplots()
    im = axis.imshow(alignment, aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=1)
    figure.colorbar(im, ax=axis, orientation='horizontal')
    pyplot.xlabel('Decoder timestep')
    pyplot.ylabel('Encoder timestep')
    pyplot.close(figure)
    return figure


def plot_stop_token(stop_token):
    """ Plot probability of the stop token over time.

    Args:
        stop_token (numpy.array or torch.Tensor [decoder_timestep]): Stop token probablity per
            decoder timestep.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    if torch.is_tensor(stop_token):
        stop_token = stop_token.detach().cpu().numpy()

    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    pyplot.plot(list(range(len(stop_token))), stop_token, marker='.', linestyle='solid')
    pyplot.ylabel('Probability')
    pyplot.xlabel('Timestep')
    pyplot.close(figure)
    return figure


def spectrogram_to_image(spectrogram):
    """ Create an image array form a spectrogram.

    Args:
        spectrogram (torch.Tensor or numpy.array): A ``[frames, num_mel_bins]`` ``Tensor`` of
            ``complex64`` STFT values.

    Returns:
        image (numpy.array [width, height, 3]): Spectrogram image.
    """
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.detach().cpu().numpy()

    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    spectrogram = np.uint8(colormap.viridis(spectrogram.T) * 255)
    spectrogram = spectrogram[:, :, :-1]  # RGBA → RGB
    return spectrogram


@configurable
def plot_waveform(signal, sample_rate=24000):
    """ Save image of spectrogram to disk.

    Args:
        signal (torch.Tensor or numpy.array [signal_length]): Signal to plot.
        sample_rate (int): Sample rate of the associated wave.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    if torch.is_tensor(signal):
        signal = signal.detach().cpu().numpy()

    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    librosa.display.waveplot(signal, sr=sample_rate)
    pyplot.ylabel('Energy')
    pyplot.xlabel('Time')
    pyplot.ylim(-1, 1)
    pyplot.close(figure)
    return figure


@configurable
def plot_spectrogram(spectrogram,
                     sample_rate=24000,
                     frame_hop=300,
                     lower_hertz=125,
                     upper_hertz=7600,
                     y_axis='mel'):
    """ Get image of log mel spectrogram.

    Args:
        spectrogram (numpy.array or torch.FloatTensor [frames, num_mel_bins])
        sample_rate (int): Sample rate for the signal.
        frame_hop (int): The frame hop in samples.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    assert len(spectrogram.shape) == 2, 'Log mel spectrogram must be 2D.'
    figure = pyplot.figure()
    pyplot.style.use('ggplot')
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.detach().cpu().numpy()
    spectrogram = spectrogram.transpose()
    librosa.display.specshow(
        spectrogram,
        hop_length=frame_hop,
        sr=sample_rate,
        fmin=lower_hertz,
        fmax=upper_hertz,
        cmap='viridis',
        y_axis=y_axis,
        x_axis='time')
    pyplot.colorbar(format='%.2f')
    pyplot.close(figure)
    return figure


_BASE_TEMPLATE = """
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.css"
        type="text/css">
  <style>
    body {
      background-color: #f4f4f5;
    }

    p {
      font-family: 'Roboto', system-ui, sans-serif;
      margin-bottom: .5em;
    }

    b {
      font-weight: bold
    }

    section {
      padding: 1.5em;
      border-bottom: 2px solid #E8E8E8;
      background: white;
    }
  </style>
  %s
"""


def _encode_audio(audio):
    """ Encode audio into a base64 string.

    Args:
        audio (torch.Tensor): Audio in the form of a tensor
        **kwargs: Other arguments for `write_wav`

    Returns
        (str): The encoded audio.
    """
    assert torch.max(audio) <= 1.0 and torch.min(
        audio) >= -1.0, "Should be [-1, 1] it is [%f, %f]" % (torch.max(audio), torch.min(audio))
    in_memory_file = io.BytesIO()
    audio = audio.detach().cpu().numpy()
    librosa.output.write_wav(in_memory_file, audio, sr=24000)
    audio = base64.b64encode(in_memory_file.read()).decode('utf-8')
    return audio


def CometML(project_name, experiment_key=None, api_key=None, workspace=None, **kwargs):
    """
    Initiate a Comet.ml visualizer with several monkey patched methods.

    Args:
        project_name (str)
        experiment_key (str)
        api_key (str)
        workspace (str)
        **kwargs: Other kwargs to pass to comet `Experiment` or `ExistingExperiment`

    Returns:
        (Experiment or ExistingExperiment): Object for visualization with comet.
    """
    load_dotenv()
    api_key = os.getenv('COMET_ML_API_KEY') if api_key is None else api_key
    workspace = os.getenv('COMET_ML_WORKSPACE') if workspace is None else workspace

    kwargs.update({'project_name': project_name, 'api_key': api_key, 'workspace': workspace})
    if experiment_key is None:
        _remote_visualizer = Experiment(**kwargs)
    else:
        _remote_visualizer = ExistingExperiment(previous_experiment=experiment_key, **kwargs)

    def log_text_and_audio(self, tag, text, audio, step=None):
        """ Add text and audio to remote visualizer in one entry.

        TODO: Consider logging the Waveform as well

        Args:
            tag (str): Tag for this event.
            text (str)
            audio (torch.Tensor)
            step (int, optional)
        """
        step = self.curr_step if step is None else step
        assert step is not None
        _remote_visualizer.log_html(_BASE_TEMPLATE % """
        <section>
            <p><b>Step:</b> {step}</p>
            <p><b>Tag:</b> {tag}</p>
            <p><b>Text:</b> {text}</p>
            <p><b>Audio:</b></p>
            <audio controls="" src="data:audio/wav;base64,{base64_audio}"></audio>
        </section>
        """.format(step=step, tag=tag, text=text, base64_audio=_encode_audio(audio)))

    _remote_visualizer.log_text_and_audio = log_text_and_audio.__get__(_remote_visualizer)

    def log_audio(self, tag, audio, step=None):
        """
        TODO: Consider logging the Waveform as well

        Args:
            tag (str): Tag for this event.
            audio (torch.Tensor)
            step (int, optional)
        """
        step = self.curr_step if step is None else step
        assert step is not None
        _remote_visualizer.log_html(_BASE_TEMPLATE % """
        <section>
            <p><b>Step:</b> {step}</p>
            <p><b>Tag:</b> {tag}</p>
            <p><b>Audio:</b></p>
            <audio controls="" src="data:audio/wav;base64,{base64_audio}"></audio>
        </section>
        """.format(step=step, tag=tag, base64_audio=_encode_audio(audio)))

    _remote_visualizer.log_audio = log_audio.__get__(_remote_visualizer)
    return _remote_visualizer
