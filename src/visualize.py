import io
import logging
import matplotlib
import os
import platform
import subprocess
import time

matplotlib.use('Agg', warn=False)

from comet_ml import ExistingExperiment
from comet_ml import Experiment
from comet_ml.config import get_config
from hparams import configurable
from hparams import HParam
from matplotlib import cm as colormap
from matplotlib import pyplot
from third_party import LazyLoader

import numpy as np
import torch
librosa_display = LazyLoader('librosa_display', globals(), 'librosa.display')

from src.audio import write_audio
from src.utils import log_runtime

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
def plot_waveform(signal, sample_rate=HParam()):
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
    librosa_display.waveplot(signal, sr=sample_rate)
    pyplot.ylabel('Energy')
    pyplot.xlabel('Time')
    pyplot.ylim(-1, 1)
    pyplot.close(figure)
    return figure


@configurable
def plot_spectrogram(spectrogram,
                     sample_rate=HParam(),
                     frame_hop=HParam(),
                     lower_hertz=HParam(),
                     upper_hertz=HParam(),
                     y_axis=HParam()):
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
    librosa_display.specshow(
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


@log_runtime
@configurable
def CometML(project_name=HParam(), experiment_key=None, log_git_patch=None, **kwargs):
    """
    Initiate a Comet.ml visualizer with several monkey patched methods.

    Args:
        project_name (str)
        experiment_key (str, optional): Comet.ml existing experiment identifier.
        log_git_patch (bool or None, optional): If ``True``
        **kwargs: Other kwargs to pass to comet `Experiment` or `ExistingExperiment`

    Returns:
        (Experiment or ExistingExperiment): Object for visualization with comet.
    """
    # NOTE: Comet ensures reproducibility if all files are tracked via git.
    untracked_files = subprocess.check_output(
        'git ls-files --others --exclude-standard', shell=True).decode().strip()
    if len(untracked_files) > 0:
        raise ValueError(('Experiment is not reproducible, Comet does not track untracked files. '
                          'Please track these files via `git`:\n%s') % untracked_files)

    kwargs.update({'log_git_patch': log_git_patch, 'workspace': get_config()['comet.workspace']})
    if project_name is not None:
        kwargs.update({'project_name': project_name})
    if experiment_key is None:
        experiment = Experiment(**kwargs)
        experiment.log_html(_BASE_HTML_STYLING)
    else:
        experiment = ExistingExperiment(previous_experiment=experiment_key, **kwargs)

    # NOTE: Unlike the usage of ``_upload_git_patch`` in ``Experiment``, this does not catch
    # any exceptions thrown by ``_upload_git_patch``; therefore, exiting the program if git patch
    # fails to upload.
    experiment._upload_git_patch()

    # Log the last git commit date
    experiment.log_other(
        'last_git_commit',
        subprocess.check_output('git log -1 --format=%cd', shell=True).decode().strip())
    # TODO: Instead of using different approaches for extracting CPU, GPU, and disk information
    # consider standardizing to use `lshw` instead.
    if torch.cuda.is_available():
        experiment.log_other(
            'list_gpus',
            subprocess.check_output('nvidia-smi --list-gpus', shell=True).decode().strip())
    if platform.system() == 'Linux':
        experiment.log_other(
            'list_disks',
            subprocess.check_output('lshw -class disk -class storage', shell=True).decode().strip())
    if platform.system() == 'Linux':
        experiment.log_other(
            'list_unique_cpus',
            subprocess.check_output(
                "awk '/model name/ {$1=$2=$3=\"\"; print $0}' /proc/cpuinfo | uniq",
                shell=True).decode().strip())
    experiment.log_parameter('num_gpu', torch.cuda.device_count())
    experiment.log_parameter('num_cpu', os.cpu_count())
    if platform.system() == 'Linux':
        experiment.log_parameter(
            'total_physical_memory_in_kb',
            subprocess.check_output("awk '/MemTotal/ {print $2}' /proc/meminfo",
                                    shell=True).decode().strip())

    last_step_time = None
    last_step = None

    other_set_step = experiment.set_step

    def set_step(self, *args, **kwargs):
        return_ = other_set_step(*args, **kwargs)

        nonlocal last_step_time
        nonlocal last_step

        if last_step_time is not None and last_step is not None and self.curr_step > last_step:
            seconds_per_step = (time.time() - last_step_time) / (self.curr_step - last_step)
            last_step_time = time.time()
            last_step = self.curr_step
            # NOTE: Ensure that `last_step` is updated before `log_metric` to ensure that
            # recursion is prevented via `self.curr_step > last_step`.
            self.log_metric('step/seconds_per_step', seconds_per_step)
        elif last_step_time is None and last_step is None:
            last_step_time = time.time()
            last_step = self.curr_step

        return return_

    experiment.set_step = set_step.__get__(experiment)

    start_epoch_time = None
    start_epoch_step = None
    first_epoch_time = None
    first_epoch_step = None

    other_log_current_epoch = experiment.log_current_epoch

    def log_current_epoch(self, *args, **kwargs):
        nonlocal start_epoch_time
        nonlocal start_epoch_step
        nonlocal first_epoch_time
        nonlocal first_epoch_step

        start_epoch_step = self.curr_step
        start_epoch_time = time.time()

        if first_epoch_time is None and first_epoch_step is None:
            first_epoch_step = self.curr_step
            first_epoch_time = time.time()

        return other_log_current_epoch(*args, **kwargs)

    experiment.log_current_epoch = log_current_epoch.__get__(experiment)

    other_log_epoch_end = experiment.log_epoch_end

    def log_epoch_end(self, *args, **kwargs):
        # NOTE: Logs an average `steps_per_second` for each epoch.
        if start_epoch_step is not None and start_epoch_time is not None:
            self.log_metric('epoch/steps_per_second',
                            (self.curr_step - start_epoch_step) / (time.time() - start_epoch_time))

        # NOTE: Logs an average `steps_per_second` since the training started.
        if first_epoch_time is not None and first_epoch_step is not None:
            old_context = self.context
            self.context = None
            self.log_metric('steps_per_second',
                            (self.curr_step - first_epoch_step) / (time.time() - first_epoch_time))
            self.context = old_context

        return other_log_epoch_end(*args, **kwargs)

    experiment.log_epoch_end = log_epoch_end.__get__(experiment)

    def _write_wav(file_name, data):
        """ Write wav from a tensor to ``io.BytesIO``.

        Args:
            file_name (str): File name to use with comet.ml
            data (np.array or torch.tensor)

        Returns:
            (str): String url to the asset.
        """
        if torch.is_tensor(data):
            data = data.numpy()

        file_ = io.BytesIO()
        write_audio(file_, data)
        asset = experiment.log_asset(file_, file_name=file_name)
        return asset['web'] if asset is not None else asset

    def log_audio(self, gold_audio=None, predicted_audio=None, step=None, **kwargs):
        """ Add text and audio to Comet via their HTML tab.

        TODO: Consider logging a visualized waveform also.

        Args:
            gold_audio (torch.Tensor, optional)
            predicted_audio (torch.Tensor, optional)
            step (int, optional)
            **kwargs: Additional arguments to be printed.
        """
        step = self.curr_step if step is None else step
        assert step is not None
        items = ['<p><b>Step:</b> {}</p>'.format(step)]
        for key, value in kwargs.items():
            items.append('<p><b>{}:</b> {}</p>'.format(key.title(), value))
        if gold_audio is not None:
            url = _write_wav('gold.wav', gold_audio)
            items.append('<p><b>Gold Audio:</b></p>')
            items.append('<audio controls preload="metadata" src="{}"></audio>'.format(url))
        if predicted_audio is not None:
            url = _write_wav('predicted.wav', predicted_audio)
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
