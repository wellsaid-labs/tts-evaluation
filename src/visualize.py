import io
import logging
import matplotlib
import os
import subprocess
import time

matplotlib.use('Agg', warn=False)

from comet_ml import ExistingExperiment
from comet_ml import Experiment
from dotenv import load_dotenv
from matplotlib import cm as colormap
from matplotlib import pyplot

import librosa.display
import numpy as np
import scipy
import torch

from src.hparams import configurable
from src.hparams import ConfiguredArg

import src.distributed

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
def plot_waveform(signal, sample_rate=ConfiguredArg()):
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
                     sample_rate=ConfiguredArg(),
                     frame_hop=ConfiguredArg(),
                     lower_hertz=ConfiguredArg(),
                     upper_hertz=ConfiguredArg(),
                     y_axis=ConfiguredArg()):
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


# Comet HTML for ``log_html`` base styles.
_BASE_HTML_STYLING = """
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
"""


def CometML(project_name,
            experiment_key=None,
            api_key=None,
            workspace=None,
            log_git_patch=None,
            **kwargs):
    """
    Initiate a Comet.ml visualizer with several monkey patched methods.

    Args:
        project_name (str)
        experiment_key (str, optional): Comet.ml existing experiment identifier.
        api_key (str, optional)
        workspace (str, optional)
        log_git_patch (bool or None, optional): If ``True``
        **kwargs: Other kwargs to pass to comet `Experiment` or `ExistingExperiment`

    Returns:
        (Experiment or ExistingExperiment): Object for visualization with comet.
    """
    load_dotenv()

    api_key = os.getenv('COMET_ML_API_KEY') if api_key is None else api_key
    workspace = os.getenv('COMET_ML_WORKSPACE') if workspace is None else workspace

    # NOTE: Comet ensures reproducibility if all files are tracked via git.
    untracked_files = subprocess.check_output(
        'git ls-files --others --exclude-standard', shell=True).decode().strip()
    if len(untracked_files) > 0:
        raise ValueError(('Experiment is not reproducible, Comet does not track untracked files. '
                          'Please track these files:\n%s') % untracked_files)

    kwargs.update({
        'project_name': project_name,
        'api_key': api_key,
        'workspace': workspace,
        'log_git_patch': log_git_patch,
    })
    if experiment_key is None:
        experiment = Experiment(**kwargs)
        experiment.log_html(_BASE_HTML_STYLING)
    else:
        experiment = ExistingExperiment(previous_experiment=experiment_key, **kwargs)

    # NOTE: Unlike the usage of ``_upload_git_patch`` in ``Experiment``, this does not catch
    # any exceptions thrown by ``_upload_git_patch``; therefore, exiting the program if git patch
    # fails to upload.
    experiment._upload_git_patch()

    experiment.log_other('is_distributed', src.distributed.is_initialized())
    # Log the last git commit date
    experiment.log_other(
        'last_git_commit',
        subprocess.check_output('git log -1 --format=%cd', shell=True).decode().strip())
    experiment.log_parameter('num_gpu', torch.cuda.device_count())

    start_epoch_time = None
    start_epoch_step = None

    other_log_current_epoch = experiment.log_current_epoch

    def log_current_epoch(self, *args, **kwargs):
        nonlocal start_epoch_time
        nonlocal start_epoch_step
        start_epoch_step = self.curr_step
        start_epoch_time = time.time()

        return other_log_current_epoch(*args, **kwargs)

    experiment.log_current_epoch = log_current_epoch.__get__(experiment)

    other_log_epoch_end = experiment.log_epoch_end

    def log_epoch_end(self, *args, **kwargs):
        if start_epoch_step is not None and start_epoch_time is not None:
            self.log_metric('epoch/steps_per_second',
                            (self.curr_step - start_epoch_step) / (time.time() - start_epoch_time))

        return other_log_epoch_end(*args, **kwargs)

    experiment.log_epoch_end = log_epoch_end.__get__(experiment)

    def _write_wav(file_name, data, sample_rate):
        """ Write wav from a tensor to ``io.BytesIO``.

        Args:
            file_name (str): File name to use with comet.ml
            data (np.array or torch.tensor)
            sample_rate (int)

        Returns:
            (str): String url to the asset.
        """
        if torch.is_tensor(data):
            data = data.numpy()

        file_ = io.BytesIO()
        scipy.io.wavfile.write(filename=file_, data=data, rate=sample_rate)
        asset = experiment.log_asset(file_, file_name=file_name)
        return asset['web'] if asset is not None else asset

    @configurable
    def log_audio(self,
                  gold_audio=None,
                  predicted_audio=None,
                  step=None,
                  sample_rate=ConfiguredArg(),
                  **kwargs):
        """ Add text and audio to Comet via their HTML tab.

        TODO: Consider logging the waveform visualized also.

        Args:
            gold_audio (torch.Tensor, optional)
            predicted_audio (torch.Tensor, optional)
            step (int, optional)
            sample_rate (int, optional)
            **kwargs: Additional arguments to be printed.
        """
        step = self.curr_step if step is None else step
        assert step is not None
        items = ['<p><b>Step:</b> {}</p>'.format(step)]
        for key, value in kwargs.items():
            items.append('<p><b>{}:</b> {}</p>'.format(key.title(), value))
        if gold_audio is not None:
            url = _write_wav('gold.wav', gold_audio, sample_rate)
            items.append('<p><b>Gold Audio:</b></p>')
            items.append('<audio controls preload="metadata" src="{}"></audio>'.format(url))
        if predicted_audio is not None:
            url = _write_wav('predicted.wav', predicted_audio, sample_rate)
            items.append('<p><b>Predicted Audio:</b></p>')
            items.append('<audio controls preload="metadata" src="{}"></audio>'.format(url))
        experiment.log_html('<section>{}</section>'.format('\n'.join(items)))

    experiment.log_audio = log_audio.__get__(experiment)

    def log_figures(self, dict_, **kwargs):
        """ Convenience function to log multiple figures.

        Args:
            dict_ (dict): Dictionary with figure (e.g. value) and figure names (e.g. key).

        Returns:
            (list): List of the ``log_figure`` returned values.
        """
        return [self.log_figure(k, v, **kwargs) for k, v in dict_.items()]

    experiment.log_figures = log_figures.__get__(experiment)

    def set_context(self, context):
        """ Set the context (i.e. train, dev, test) for further logs.

        Args:
            context (str): Some context for all further logs.
        """
        self.context = context

    experiment.set_context = set_context.__get__(experiment)

    return experiment