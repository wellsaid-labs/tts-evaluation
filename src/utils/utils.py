import matplotlib

matplotlib.use('Agg', warn=False)

import logging
import logging.config
import os

from matplotlib import pyplot
from matplotlib import cm as colormap
from torchnlp.utils import shuffle as do_deterministic_shuffle

import dill
import librosa.display
import numpy as np
import torch

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)

# Repository root path
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')


def get_total_parameters(model):
    """ Return the total number of trainable parameters in model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def split_dataset(dataset, splits, deterministic_shuffle=True, random_seed=123):
    """
    Args:
        dataset (list): Dataset to split.
        splits (tuple): Tuple of percentages determining dataset splits.
        shuffle (bool, optional): If ``True`` determinisitically shuffle the dataset before
            splitting.

    Returns:
        (list): splits of the dataset

    Example:
        >>> dataset = [1, 2, 3, 4, 5]
        >>> splits = (.6, .2, .2)
        >>> split_dataset(dataset, splits)
        [[1, 2, 3], [4], [5]]
    """
    if deterministic_shuffle:
        do_deterministic_shuffle(dataset, random_seed=random_seed)
    assert sum(splits) == 1, 'Splits must sum to 100%'
    splits = [round(s * len(dataset)) for s in splits]
    datasets = []
    for split in splits:
        datasets.append(dataset[:split])
        dataset = dataset[split:]
    return datasets


def figure_to_numpy_array(figure):
    """ Turn a figure into an image array.

    References:
        * https://github.com/lanpa/tensorboard-pytorch/blob/master/examples/matplotlib_demo.py
        * https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        * https://stackoverflow.com/questions/20051160/renderer-problems-using-matplotlib-from-within-a-script # noqa: E501

    Args:
        figure (matplotlib.figure): Figure with the plot.

    Returns:
        array (np.array [H, W, 3]): Numpy array representing the image of the figure.
    """
    figure.canvas.draw()
    data = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(figure.canvas.get_width_height()[::-1] + (3,))


def plot_attention(alignment, plot_to_numpy=True):
    """ Plot alignment of attention.

    Args:
        alignment (numpy.array([decoder_timestep, encoder_timestep])): Attention alignment weights
            computed at every timestep of the decoder.
        title (str, optional): Title of the plot.
        plot_to_numpy (bool, optional): If ``True``, return as a numpy array; otherwise, ``None``
            is returned.

    Returns:
        If ``plot_to_numpy=True`` returns image (np.array [width, height, 3]) otherwise returns
            ``None``.
    """
    alignment = np.transpose(alignment)
    pyplot.style.use('ggplot')
    figure, axis = pyplot.subplots()
    im = axis.imshow(alignment, aspect='auto', origin='lower', interpolation='none', vmin=0, vmax=1)
    figure.colorbar(im, ax=axis, orientation='horizontal')
    pyplot.xlabel('Decoder timestep')
    pyplot.ylabel('Encoder timestep')
    pyplot.close(figure)
    return figure_to_numpy_array(figure)


def plot_stop_token(stop_token, plot_to_numpy=True):
    """ Plot probability of the stop token over time.

    Args:
        stop_token (numpy.array([decoder_timestep])): Stop token probablity per decoder timestep.
        title (str, optional): Title of the plot.
        plot_to_numpy (bool, optional): If ``True``, return as a numpy array; otherwise, ``None``
            is returned.

    Returns:
        If ``plot_to_numpy=True`` returns image (np.array [width, height, 3]) otherwise returns
            ``None``.
    """
    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    pyplot.plot(list(range(len(stop_token))), stop_token, marker='.', linestyle='solid')
    pyplot.ylabel('Probability')
    pyplot.xlabel('Timestep')
    if plot_to_numpy:
        # LEARN MORE: https://github.com/matplotlib/matplotlib/issues/8560/
        # ``pyplot.close`` removes the figure from it's internal state and (if there is a gui)
        # closes the window. However it does not clear the figure and if you are still holding a
        # reference to the ``Figure`` object (and it holds references to all of it's children) so it
        # is not garbage collected.
        pyplot.close(figure)
        return figure_to_numpy_array(figure)


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
def plot_waveform(signal, sample_rate=24000, plot_to_numpy=True):
    """ Save image of spectrogram to disk.

    Args:
        signal (Tensor [signal_length]): Signal to plot.
        sample_rate (int): Sample rate of the associated wave.
        plot_to_numpy (bool, optional): If ``True``, return as a numpy array; otherwise, ``None``
            is returned.

    Returns:
        If ``plot_to_numpy=True`` returns image (np.array [width, height, 3]) otherwise returns
            ``None``.
    """
    pyplot.style.use('ggplot')
    figure = pyplot.figure()
    librosa.display.waveplot(signal, sr=sample_rate)
    pyplot.ylabel('Energy')
    pyplot.xlabel('Time')
    pyplot.ylim(-1, 1)
    if plot_to_numpy:
        pyplot.close()
        return figure_to_numpy_array(figure)


@configurable
def plot_log_mel_spectrogram(log_mel_spectrogram,
                             sample_rate=24000,
                             frame_hop=300,
                             lower_hertz=125,
                             upper_hertz=7600,
                             plot_to_numpy=True):
    """ Get image of log mel spectrogram.

    Args:
        log_mel_spectrogram (torch.FloatTensor [frames, num_mel_bins])
        plot_to_numpy (bool, optional): If ``True``, return as a numpy array; otherwise, ``None``
            is returned.

    Returns:
        If ``plot_to_numpy=True`` returns image (np.array [width, height, 3]) otherwise returns
            ``None``.
    """
    aspect = log_mel_spectrogram.shape[0] / log_mel_spectrogram.shape[1]
    figure = pyplot.figure(figsize=(2 * aspect, 2))
    pyplot.style.use('ggplot')
    if torch.is_tensor(log_mel_spectrogram):
        log_mel_spectrogram = log_mel_spectrogram.numpy()
    log_mel_spectrogram = log_mel_spectrogram.transpose()
    mel_spectrogram = np.exp(log_mel_spectrogram)
    librosa.display.specshow(
        librosa.power_to_db(mel_spectrogram, ref=np.max),
        hop_length=frame_hop,
        sr=sample_rate,
        fmin=lower_hertz,
        fmax=upper_hertz,
        cmap='viridis',
        y_axis='mel',
        x_axis='time',
    )
    pyplot.title('Mel spectrogram')
    pyplot.colorbar(format='%+2.0f dB')
    if plot_to_numpy:
        pyplot.close(figure)
        return figure_to_numpy_array(figure)


def torch_load(path, device=torch.device('cpu')):
    """ Using ``torch.load`` and ``dill`` load an object from ``path`` onto ``self.device``.

    Args:
        path (str): Filename to load.

    Returns:
        (any): Object loaded.
    """
    logger.info('Loading: %s' % (path,))

    def remap(storage, loc):
        if 'cuda' in loc and device.type == 'cuda':
            return storage.cuda(device=device.index)
        return storage

    return torch.load(path, map_location=remap, pickle_module=dill)


def torch_save(path, data):
    """ Using ``torch.save`` and ``dill`` save an object to ``path``.

    Args:
        path (str): Filename to save to.
        data (any): Data to save into file.
    """
    torch.save(data, path, pickle_module=dill)
    logger.info('Saved: %s' % (path,))
