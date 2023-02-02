import logging
import typing

import numpy as np
import torch
from third_party import LazyLoader

if typing.TYPE_CHECKING:  # pragma: no cover
    import matplotlib
    import matplotlib.figure
    from librosa import display as librosa_display
    from matplotlib import pyplot
else:
    librosa_display = LazyLoader("librosa_display", globals(), "librosa.display")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")
    pyplot = LazyLoader("pyplot", globals(), "matplotlib.pyplot")

logger = logging.getLogger(__name__)

try:
    pyplot.style.use("ggplot")  # type: ignore
except (ModuleNotFoundError, NameError):
    logger.info("Ignoring optional `matplotlib` dependency.")


def _to_numpy(array: typing.Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    return array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array


def plot_alignments(
    alignment: typing.Union[torch.Tensor, np.ndarray]
) -> "matplotlib.figure.Figure":
    """Plot an alignment of two sequences.

    Args:
        alignment (numpy.ndarray or torch.Tensor [decoded_sequence, encoded_sequence]): Alignment
            weights between two sequences.
    """
    alignment = np.transpose(_to_numpy(alignment))
    figure, axis = pyplot.subplots()
    im = axis.imshow(  # type: ignore
        alignment,
        cmap="turbo",
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=0,
        vmax=1,
    )
    figure.colorbar(im, ax=axis, orientation="horizontal")  # type: ignore
    pyplot.xlabel("Decoded Sequence")
    pyplot.ylabel("Encoded Sequence")
    pyplot.close(figure)
    return figure


def plot_logits(logits: typing.Union[torch.Tensor, np.ndarray]) -> "matplotlib.figure.Figure":
    """Given a time-series of logits, plot a line graph.

    Args:
        logits (numpy.array or torch.Tensor [sequence_length])
    """
    logits = logits if isinstance(logits, torch.Tensor) else torch.tensor(logits)
    logits = torch.sigmoid(logits)
    logits = logits.detach().cpu().numpy()
    figure = pyplot.figure()
    pyplot.plot(list(range(len(logits))), logits, marker=".", linestyle="solid")
    pyplot.ylabel("Probability")
    pyplot.xlabel("Timestep")
    pyplot.close(figure)
    return figure


def plot_loudness(loudness: typing.Union[torch.Tensor, np.ndarray]) -> "matplotlib.figure.Figure":
    """Given a time-series of decibel values, plot a line graph.
    Args:
        loudness (numpy.array or torch.Tensor [sequence_length])
    """
    logits = loudness if isinstance(loudness, torch.Tensor) else torch.tensor(loudness)
    logits = logits.detach().cpu().numpy()
    figure = pyplot.figure()
    pyplot.plot(list(range(len(logits))), logits, marker=".", linestyle="solid")
    pyplot.ylabel("Loudness")
    pyplot.xlabel("Timestep")
    pyplot.close(figure)
    return figure


def plot_waveform(
    signal: typing.Union[torch.Tensor, np.ndarray], sample_rate: int
) -> "matplotlib.figure.Figure":
    """Plot the amplitude envelope of a waveform.

    Args:
        signal (torch.Tensor or numpy.array [signal_length]): Signal to plot as a waveform.
        sample_rate: The number of samples or points per second.
    """
    figure = pyplot.figure()
    librosa_display.waveplot(_to_numpy(signal), sr=sample_rate)
    pyplot.ylabel("Amplitude")
    pyplot.xlabel("Time")
    pyplot.ylim(-1.0, 1.0)
    # Learn more: https://stackoverflow.com/questions/21884271/warning-about-too-many-open-figures
    pyplot.close(figure)
    return figure


def plot_mel_spectrogram(
    spectrogram: typing.Union[torch.Tensor, np.ndarray],
    lower_hertz: typing.Optional[int],
    upper_hertz: typing.Optional[int],
    y_axis: str = "mel",
    **kwargs,
) -> "matplotlib.figure.Figure":
    """Plot a mel spectrogram.

    Args:
        spectrogram (numpy.array or torch.FloatTensor [num_frames, num_mel_bins])
        lower_hertz: Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz: The desired top edge of the highest frequency band.
        **kwargs: Any additional keyword arguments are passed onto `visualize.plot_spectrogram`.
    """
    return plot_spectrogram(
        spectrogram, fmin=lower_hertz, fmax=upper_hertz, y_axis=y_axis, **kwargs
    )


def plot_spectrogram(
    spectrogram: typing.Union[torch.Tensor, np.ndarray],
    sample_rate: int,
    frame_hop: int,
    cmap: str = "turbo",
    y_axis: str = "linear",
    x_axis: str = "time",
    fmax: typing.Optional[float] = None,
    **kwargs,
) -> "matplotlib.figure.Figure":
    """Plot a spectrogram.

    Args:
        spectrogram (numpy.array or torch.FloatTensor [num_frames, num_frequencies])
        sample_rate: Sample rate for the signal.
        frame_hop: The frame hop in samples.
        **kwargs: Any additional keyword arguments are passed onto `librosa_display.specshow`.
    """
    assert len(spectrogram.shape) == 2, "Spectrogram must be 2-dimensional."
    figure = pyplot.figure()
    spectrogram = np.transpose(_to_numpy(spectrogram))
    fmax = fmax if fmax is None else min(fmax, float(sample_rate) / 2)
    librosa_display.specshow(
        spectrogram,
        hop_length=frame_hop,
        sr=sample_rate,
        cmap=cmap,
        y_axis=y_axis,
        x_axis=x_axis,
        fmax=fmax,
        **kwargs,
    )
    pyplot.colorbar(format="%.2f")
    pyplot.close(figure)
    return figure
