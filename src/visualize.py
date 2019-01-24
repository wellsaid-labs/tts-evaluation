from collections import defaultdict

import io
import logging
import matplotlib
import os
import subprocess
import time

matplotlib.use('Agg', warn=False)

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


class AccumulatedMetrics():
    """
    Args:
        type_: Default torch tensor type.
    """

    def __init__(self, type_=torch.cuda):
        self.type_ = type_
        self.reset()

    def reset(self):
        self.metrics = {
            'epoch_total': defaultdict(float),
            'epoch_count': defaultdict(float),
            'step_total': defaultdict(float),
            'step_count': defaultdict(float)
        }

    def add_metric(self, name, value, count=1):
        """ Add metric as part of the current step.

        Args:
            name (str)
            value (number)
            count (int): Number of times to add value / frequency of value.
        """
        if torch.is_tensor(value):
            value = value.item()

        if torch.is_tensor(count):
            count = count.item()

        assert count > 0, '%s count, %f, must be a positive number' % (name, count)

        self.metrics['step_total'][name] += value * count
        self.metrics['step_count'][name] += count

    def add_metrics(self, dict_, count=1):
        """ Add multiple metrics as part of the current step.

        Args:
            dict_ (dict): Metrics in the form of key value pairs.
            count (int): Number of times to add value / frequency of value.
        """
        for metric, value in dict_.items():
            self.add_metric(metric, value, count)

    def log_step_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_step_end``.

        Note that in the distributed case, only the master node gets the accurate metric.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        # Accumulate metrics between multiple processes.
        if src.distributed.in_use():
            metrics_total_items = sorted(self.metrics['step_total'].items(), key=lambda t: t[0])
            metrics_total_values = [value for _, value in metrics_total_items]

            metrics_count_items = sorted(self.metrics['step_count'].items(), key=lambda t: t[0])
            metrics_count_values = [value for _, value in metrics_count_items]

            packed = self.type_.FloatTensor(metrics_total_values + metrics_count_values)
            torch.distributed.reduce(packed, dst=src.distributed.get_master_rank())
            packed = packed.tolist()

            for (key, _), value in zip(metrics_total_items, packed[:len(metrics_total_items)]):
                self.metrics['step_total'][key] = value

            for (key, _), value in zip(metrics_count_items, packed[len(metrics_total_items):]):
                self.metrics['step_count'][key] = value

        # Log step metrics and update epoch metrics.
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['step_total'].items(), self.metrics['step_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)

            self.metrics['epoch_total'][total_key] += total_value
            self.metrics['epoch_count'][total_key] += count_value

        # Reset step metrics
        self.metrics['step_total'] = defaultdict(float)
        self.metrics['step_count'] = defaultdict(float)

    def get_epoch_metric(self, metric):
        """ Get the current epoch value for a metric.
        """
        return self.metrics['epoch_total'][metric] / self.metrics['epoch_count'][metric]

    def log_epoch_end(self, log_metric):
        """ Log all metrics that have been accumulated since the last ``log_epoch_end``.

        Args:
            log_metric (callable(key, value)): Callable to log a metric.
        """
        self.log_step_end(lambda *args, **kwargs: None)

        # Log epoch metrics
        for (total_key, total_value), (count_key, count_value) in zip(
                self.metrics['epoch_total'].items(), self.metrics['epoch_count'].items()):

            assert total_key == count_key, 'AccumulatedMetrics invariant failed.'
            assert count_value > 0, 'AccumulatedMetrics invariant failed (%s, %f, %f)'
            log_metric(total_key, total_value / count_value)


_BASE_HTML = """
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
    # NOTE: To prevent inadvertently triggering the ``comet_ml``
    # ``SyntaxError: Please import comet before importing any torch modules``, we only import
    # ``comet_ml`` if this module is executed.
    from comet_ml import Experiment
    from comet_ml import ExistingExperiment
    load_dotenv()
    api_key = os.getenv('COMET_ML_API_KEY') if api_key is None else api_key
    workspace = os.getenv('COMET_ML_WORKSPACE') if workspace is None else workspace

    untracked_files = subprocess.check_output(
        'git ls-files --others --exclude-standard', shell=True).decode().strip()
    if len(untracked_files) > 0:
        raise ValueError(('Experiment is not reproducible, Comet does not track untracked files. '
                          'Please track these files:\n%s') % untracked_files)

    kwargs.update({
        'project_name': project_name,
        'api_key': api_key,
        'workspace': workspace,
        'log_git_patch': False
    })
    if experiment_key is None:
        experiment = Experiment(**kwargs)
        experiment.log_html(_BASE_HTML)
    else:
        experiment = ExistingExperiment(previous_experiment=experiment_key, **kwargs)

    # NOTE: Unlike the usage of this function in ``Experiment``, we do not catch an exception
    # thrown by this function; therefore, exiting the program if git patch fails to upload.
    experiment._upload_git_patch()
    experiment.log_other('is_distributed', src.distributed.in_use())
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
        self.log_metric('epoch/steps_per_second',
                        (self.curr_step - start_epoch_step) / (time.time() - start_epoch_time))
        return other_log_epoch_end(*args, **kwargs)

    experiment.log_epoch_end = log_epoch_end.__get__(experiment)

    def _write_wav(file_name, data, sample_rate):
        """ Write wav from a tensor to `io.BytesIO`.

        Args:
            file_name (str): File name to use with comet.ml
            data (np.array or torch.tensor)

        Returns:
            (str): String url to the asset.
        """
        if torch.is_tensor(data):
            data = data.numpy()

        file_ = io.BytesIO()
        scipy.io.wavfile.write(filename=file_, data=data, rate=sample_rate)
        asset = experiment.log_asset(file_like_object=file_, file_name=file_name)
        return asset['web'] if asset is not None else asset

    @configurable
    def log_audio(self,
                  gold_audio=None,
                  predicted_audio=None,
                  step=None,
                  sample_rate=ConfiguredArg(),
                  **kwargs):
        """ Add text and audio to remote visualizer in one entry.

        TODO: Consider logging the waveform as well.

        Args:
            gold_audio (torch.Tensor)
            predicted_audio (torch.Tensor)
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
        """ Convience function to log multiple figures """
        for key, value in dict_.items():
            self.log_figure(key, value, **kwargs)

    experiment.log_figures = log_figures.__get__(experiment)

    def set_context(self, context):
        """ Set the context (i.e. train, dev, test) for further logs. """
        self.context = context

    experiment.set_context = set_context.__get__(experiment)

    return experiment
