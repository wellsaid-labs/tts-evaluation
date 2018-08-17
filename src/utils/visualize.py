from contextlib import contextmanager

import logging
import matplotlib
import os
import torch

matplotlib.use('Agg', warn=False)

from matplotlib import cm as colormap
from matplotlib import pyplot
from tensorboardX import SummaryWriter
from tensorboardX.src.event_pb2 import Event
from tensorboardX.src.event_pb2 import SessionLog
from tensorboardX.utils import figure_to_image

import librosa.display
import numpy as np

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


def plot_attention(alignment):
    """ Plot alignment of attention.

    Args:
        alignment (numpy.array([decoder_timestep, encoder_timestep])): Attention alignment weights
            computed at every timestep of the decoder.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    alignment = np.transpose(alignment)
    pyplot.style.use('ggplot')
    figure, axis = pyplot.subplots()
    im = axis.imshow(alignment, aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=1)
    figure.colorbar(im, ax=axis, orientation='horizontal')
    pyplot.xlabel('Decoder timestep')
    pyplot.ylabel('Encoder timestep')
    return figure


def plot_stop_token(stop_token):
    """ Plot probability of the stop token over time.

    Args:
        stop_token (numpy.array([decoder_timestep])): Stop token probablity per decoder timestep.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    pyplot.plot(list(range(len(stop_token))), stop_token, marker='.', linestyle='solid')
    pyplot.ylabel('Probability')
    pyplot.xlabel('Timestep')
    return figure


def spectrogram_to_image(spectrogram):
    """ Create an image array form a spectrogram.

    Args:
        spectrogram (Tensor): A ``[frames, num_mel_bins]`` ``Tensor`` of ``complex64`` STFT
            values.

    Returns:
        image (np.array [width, height, 3]): Spectrogram image.
    """
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    spectrogram = np.uint8(colormap.viridis(spectrogram.T) * 255)
    spectrogram = spectrogram[:, :, :-1]  # RGBA → RGB
    return spectrogram


@configurable
def plot_waveform(signal, sample_rate=24000):
    """ Save image of spectrogram to disk.

    Args:
        signal (Tensor [signal_length]): Signal to plot.
        sample_rate (int): Sample rate of the associated wave.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    librosa.display.waveplot(signal, sr=sample_rate)
    pyplot.ylabel('Energy')
    pyplot.xlabel('Time')
    pyplot.ylim(-1, 1)
    return figure


@configurable
def plot_log_mel_spectrogram(log_mel_spectrogram,
                             sample_rate=24000,
                             frame_hop=300,
                             lower_hertz=125,
                             upper_hertz=7600):
    """ Get image of log mel spectrogram.

    Args:
        log_mel_spectrogram (torch.FloatTensor [frames, num_mel_bins])
        sample_rate (int): Sample rate for the signal.
        frame_hop (int): The frame hop in samples.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        (matplotlib.figure.Figure): Matplotlib figure representing the plot.
    """
    figure = pyplot.figure()
    pyplot.style.use('ggplot')
    if torch.is_tensor(log_mel_spectrogram):
        log_mel_spectrogram = log_mel_spectrogram.numpy()
    log_mel_spectrogram = log_mel_spectrogram.transpose()
    librosa.display.specshow(
        log_mel_spectrogram,
        hop_length=frame_hop,
        sr=sample_rate,
        fmin=lower_hertz,
        fmax=upper_hertz,
        cmap='viridis',
        y_axis='mel',
        x_axis='time')
    pyplot.colorbar(format='%.2f')
    return figure


class Tensorboard(SummaryWriter):

    def __init__(self, *args, log_dir=None, step=0, **kwargs):
        """
        Args:
            log_dir (string, optional): The save location for tensorboard events.
            step (int, optional): Global step tensorboard uses to assign continuity to events.
            *args: Other arguments used to initialize ``SummaryWriter``.
            **kwargs: Other key word arguments used to initialize ``SummaryWriter``.
        """
        self.writer = SummaryWriter(*args, log_dir=log_dir, **kwargs)
        self._step = None

        # Setup tensorboard
        os.makedirs(log_dir, exist_ok=True)

        # LEARN MORE:
        # * ``SessionLog.START`` as related to purge:
        #   https://github.com/tensorflow/tensorboard/blob/master/README.md#how-should-i-handle-tensorflow-restarts
        # * File version as related to purge:
        #   https://github.com/tensorflow/tensorboard/blob/8eaf24e79a2f2de7c324034e73c990af20ec5979/tensorboard/backend/event_processing/event_accumulator.py#L571
        #   https://github.com/tensorflow/tensorboard/blob/a856e61d39d231e38d45e32e92b28be596afbb58/tensorboard/plugins/debugger/constants.py#L33
        # * File version naming ``brain.Event:``
        #   https://github.com/tensorflow/tensorboard/blob/8eaf24e79a2f2de7c324034e73c990af20ec5979/tensorboard/backend/event_processing/event_accumulator.py#L742
        # * Tensorflow usage of Tensorboard SessionLog
        #   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/supervisor.py
        # * For debugging, read Tensorboard event files via:
        #   https://github.com/tensorflow/tensorboard/blob/634f93ab0396bf10098fbc1bfb5bccacbedbb7e4/tensorboard/backend/event_processing/event_file_loader.py
        self.writer.file_writer.event_writer.add_event(
            Event(file_version='brain.Event:2', step=step))
        self.writer.file_writer.event_writer.add_event(
            Event(session_log=SessionLog(status=SessionLog.START), step=step))

        logger.info('Started Tensorboard at step %d with log directory ``%s/*tfevents*``', step,
                    log_dir)

    def add_text(self, path, text, step=None):
        """ Add text to tensorboard.

        Args:
            path (str): List of tags to use as label.
            text (str): Text to add to tensorboard.
            step (int, optional): Step value to record.
        """
        step = self._step if step is None else step
        assert step is not None

        self.writer.add_text(path, text, step)

    def add_scalar(self, path, scalar, step=None):
        """ Add scalar to tensorboard.

        Args:
            path (str): List of tags to use as label.
            scalar (number): Scalar to add to tensorboard.
            step (int, optional): Step value to record.
        """
        step = self._step if step is None else step
        assert step is not None

        self.writer.add_scalar(path, scalar, step)

    def _add_image(self, path, step, plot, *data):
        """ Plot data and add image to tensorboard.

        Args:
            path (str): List of tags to use as label.
            step (int): Step value to record.
            plot (callable): Callable that returns an ``matplotlib.figure.Figure`` given numpy data.
            *tensors (torch.Tensor): Tensor to visualize.
        """
        step = self._step if step is None else step
        assert step is not None

        data = [row.detach().cpu().numpy() if torch.is_tensor(row) else row for row in data]
        image = figure_to_image(plot(*data))
        self.writer.add_image(path, image, step)

    def add_stop_token(self, path, stop_token, step=None):
        """ Plot probability of the stop token over time in Tensorboard.

        Args:
            path (str): List of tags to use as label.
            stop_token (torch.FloatTensor([decoder_timestep])): Stop token probablity per decoder
                timestep.
            step (int, optional): Step value to record.
        """
        self._add_image(path, step, plot_stop_token, stop_token)

    def add_waveform(self, path, signal, step=None):
        """ Add image of a waveform to Tensorboard.

        Args:
            path (str): List of tags to use as label.
            signal (torch.FloatTensor [signal_length]): Signal to plot.
            step (int, optional): Step value to record.
        """
        self._add_image(path, step, plot_waveform, signal)

    def add_log_mel_spectrogram(self, path, log_mel_spectrogram, step=None):
        """ Add image of a log mel spectrogram to Tensorboard.

        Args:
            path (str): List of tags to use as label.
            log_mel_spectrogram (torch.FloatTensor [frames, num_mel_bins])
            step (int, optional): Step value to record.
        """
        self._add_image(path, step, plot_log_mel_spectrogram, log_mel_spectrogram)

    def add_attention(self, path, alignment, step=None):
        """ Add image of an attention alignment to Tensorboard.

        Args:
            path (str): List of tags to use as label.
            alignment (torch.FloatTensor([decoder_timestep, encoder_timestep])): Attention alignment
                weights computed at every timestep of the decoder.
            step (int, optional): Step value to record.
        """
        self._add_image(path, step, plot_attention, alignment)

    @configurable
    def add_audio(self, path_audio, path_image, signal, step=None, sample_rate=24000):
        """ Add audio to tensorboard.

        Args:
            path_audio (list): List of tags to use as label for the audio file.
            path_image (list): List of tags to use as label for the waveform image.
            signal (torch.Tensor): Signal to add to tensorboard as audio.
            step (int, optional): Step value to record.
            sample_rate (int): Sample rate of the associated wave.
        """
        step = self._step if step is None else step
        assert step is not None

        signal = signal.detach().cpu()
        assert torch.max(signal) <= 1.0 and torch.min(
            signal) >= -1.0, "Should be [-1, 1] it is [%f, %f]" % (torch.max(signal),
                                                                   torch.min(signal))
        self.writer.add_audio(path_audio, signal, step, sample_rate)
        self.add_waveform(path_image, signal, step)

    def close(self):
        """ Flushes the event file to disk and close the file. Call this method when you do not
        need the Tensorboard anymore.
        """
        self.writer.file_writer.event_writer.add_event(
            Event(session_log=SessionLog(status=SessionLog.STOP)))

        # Prevent references to file_writer from being lost
        file_writer = self.writer.file_writer
        all_writers = self.writer.all_writers

        self.writer.close()

        self.writer.file_writer = file_writer
        self.writer.all_writers = all_writers

    @contextmanager
    def set_step(self, step):
        """ Set a global step to be used for tensorboard events unless step is explicitly declared.

        Args:
            step (int): Global step to set.
        """
        self._step = step
        yield self
        self._step = None
