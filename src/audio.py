import logging
import math

from tqdm import tqdm

import librosa
import numpy as np

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


@configurable
def read_audio(filename, sample_rate=None):
    """ Read an audio file.

    Tacotron 1 Reference:
        We use 24 kHz sampling rate for all experiments.

    Notes:
        * To keep consistent with Tacotron 2 ensure audio files are mono WAVs with subformat PCM
          16 bit and a 24 kHz sampling rate.

    References:
        * WAV specs:
          http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
        * Resampy the Librosa resampler.
          https://github.com/bmcfee/resampy
        * All Python audio resamplers:
          https://livingthing.danmackinlay.name/python_audio.html
        * Issue on scaling amplitude:
          https://github.com/bmcfee/resampy/issues/61

    Args:
        filename (Path or str): Name of the file to load.
        sample_rate (int or None): Assert this target sample rate.

    Returns:
        numpy.ndarray [n,]: Audio time series.
    """
    signal, observed_sample_rate = librosa.core.load(str(filename), sr=None)
    if sample_rate is not None:
        assert sample_rate == observed_sample_rate, (
            "Sample rate must be set to %d (!= %s) before hand for file %s" %
            (sample_rate, observed_sample_rate, filename))
    assert len(signal.shape) == 1, "Signal must be mono."
    assert np.max(signal) <= 1 and np.min(signal) >= -1, "Signal must be in range [-1, 1]."
    return signal


@configurable
def mel_filters(sample_rate, fft_length=2048, num_mel_bins=80, lower_hertz=125, upper_hertz=7600):
    """ Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Reference:
        * The API written by RJ Skerry-Ryan a Tacotron author.
          https://www.tensorflow.org/api_docs/python/tf/contrib/signal/linear_to_mel_weight_matrix

    Args:
        sample_rate (int): The sample rate of the signal.
        fft_length (int): The size of the FFT to apply. If not provided, uses the smallest power of
          2 enclosing `frame_length`.
        num_mel_bins (int): Number of Mel bands to generate.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        (np.ndarray [num_mel_bins, 1 + fft_length / 2): Mel transform matrix.
    """
    # NOTE: The Tacotron 2 model likely did not normalize the filterbank; otherwise, the 0.01
    # minimum mentioned in their paper for the dynamic range is too high. NVIDIA/tacotron2 includes
    # norm and had to set their minimum to 10**-5 to compensate.
    # NOTE: ``htk=True`` because normalization of the mel filterbank is from Slaney's algorithm.
    return librosa.filters.mel(
        sample_rate,
        fft_length,
        n_mels=num_mel_bins,
        fmin=lower_hertz,
        fmax=upper_hertz,
        norm=None,
        htk=True)


@configurable
def get_log_mel_spectrogram(signal,
                            sample_rate,
                            frame_size=1200,
                            frame_hop=300,
                            fft_length=2048,
                            window='hann',
                            min_magnitude=0.01):
    """ Compute a log-mel-scaled spectrogram from signal.

    Tacotron 2 Reference:
        As in Tacotron, mel spectrograms are computed through a shorttime Fourier transform (STFT)
        using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.

        We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
        spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression. Prior to log
        compression, the filterbank output magnitudes are clipped to a minimum value of 0.01 in
        order to limit dynamic range in the logarithmic domain.

    Tacotron 1 Reference:
        We use log magnitude spectrogram  with Hann windowing, 50 ms frame length, 12.5 ms frame
        shift, and 2048-point Fourier transform. We also found pre-emphasis (0.97) to be helpful.
        We use 24 kHz sampling rate for all experiments.

    Librosa vs Tensorflow:
        * ``n_fft`` for Tensorflow STFT is rounded to the closests power of two while for Librosa
          it defaults to ``frame_size``.
        * Bugs filed against Tensorflow STFT comparing it librosa:
          https://github.com/tensorflow/tensorflow/issues/16465
          https://github.com/tensorflow/tensorflow/issues/15134
        * Librosa normalizes the mel back filter; therefore, then range of the mel basis is between
          ~(0, 0.03) while for Tensorflow it is between ~(0, 0.99) unnormalized.
        * Frames are segmented in Tensorflow with a frame size of ``frame_size`` while librosa uses
        ``fft_length``.

    Reference:
        * DSP MFCC Tutorial:
          http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
        * Tacotron Paper:
          https://arxiv.org/pdf/1703.10135.pdf
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
        * Tacotron 2 Author Spectrogram Code:
          https://github.com/tensorflow/tensorflow/blob/r1.7/tensorflow/contrib/signal/python/ops/spectral_ops.py
        * Tacotron 2 Authors:
          https://github.com/rryan
        * Tensorflow Commits by Tacotron 2 Authors:
          https://github.com/tensorflow/tensorflow/commits?author=rryan
        * Tacotron 2 Author Spectrogram Guide:
          https://www.tensorflow.org/api_guides/python/contrib.signal

    Args:
        signal (np.array [signal_length]): A batch of float32 time-domain signals in the range
            [-1, 1].
        sample_rate (int): Sample rate for the signal.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        fft_length (int): The window size used by the fourier transform.
        window (str, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        min_magnitude (float): Stabilizing minimum to avoid high dynamic ranges caused by the
            singularity at zero in the mel spectrograms.

    Returns:
        log_mel_spectrograms (np.ndarray [frames, num_mel_bins]): Log mel spectrogram.
        pad (tuple): Number of zeros to pad the left and right of the signal, such that:
            ``(signal.shape[0] + pad) / frame_hop == log_mel_spectrograms.shape[0]``
    """
    # NOTE: Check ``notebooks/Comparing Mel Spectrogram to Signal.ipynb`` for the correctness
    # of this padding algorithm.
    # NOTE: Pad signal so that is divisable by ``frame_hop``
    remainder = frame_hop - signal.shape[0] % frame_hop
    padding = (math.ceil(remainder / 2), math.floor(remainder / 2))
    padded_signal = np.pad(signal, padding, mode='constant', constant_values=0)
    assert padded_signal.shape[0] % frame_hop == 0

    # NOTE: The number of spectrogram frames generated is, with ``center=True``:
    # ``(padded_signal.shape[0] + frame_hop) // frame_hop``.
    # Otherewise, it's: ``(padded_signal.shape[0] - frame_size + frame_hop) // frame_hop``
    spectrogram = librosa.stft(
        padded_signal, n_fft=fft_length, hop_length=frame_hop, win_length=frame_size, window=window)

    # NOTE: Return number of padding needed to pad signal such that
    # ``spectrogram.shape[0] * num_frames == signal.shape[0]``
    # This is padding is partly determined by ``center=True`` librosa padding.
    ret_pad = frame_hop + remainder
    assert ret_pad <= frame_size
    assert (signal.shape[0] + ret_pad) % frame_hop == 0
    ret_pad = (math.ceil(ret_pad / 2), math.floor(ret_pad / 2))

    # SOURCE (Tacotron 2):
    # "STFT magnitude"
    magnitude_spectrogram = np.abs(spectrogram)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
    # spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    mel_basis = mel_filters(sample_rate)
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrogram).transpose()

    # SOURCE (Tacotron 2):
    # Prior to log compression, the filterbank output magnitudes are clipped to a minimum value of
    # 0.01 in order to limit dynamic range in the logarithmic domain.
    mel_spectrogram = np.maximum(0.01, mel_spectrogram)

    # SOURCE (Tacotron 2):
    # followed by log dynamic range compression.
    log_mel_spectrogram = np.log(mel_spectrogram)

    log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)  # ``np.float64`` → ``np.float32``

    return log_mel_spectrogram, ret_pad


def _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, sample_rate):
    """ Transform log mel spectrogram to spectrogram (lossy).

    Args:
        log_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate (int): Sample rate of the ``log_mel_spectrogram``.

    Returns:
        (np.ndarray [frames, num_spectrogram_bins]): Spectrogram.
    """
    mel_spectrogram = np.exp(log_mel_spectrogram)
    num_mel_bins = mel_spectrogram.shape[1]
    mel_basis = mel_filters(sample_rate, num_mel_bins=num_mel_bins)

    # ``np.linalg.pinv`` creates approximate inverse matrix of ``mel_basis``
    inverse_mel_basis = np.linalg.pinv(mel_basis)
    return np.dot(inverse_mel_basis, mel_spectrogram.transpose())


@configurable
def griffin_lim(log_mel_spectrogram,
                sample_rate,
                frame_size=1200,
                frame_hop=300,
                fft_length=2048,
                window='hann',
                power=1.2,
                iterations=30):
    """ Transform log mel spectrogram to wav file with the Griffin-Lim algorithm.

    Given a magnitude spectrogram as input, reconstruct the audio signal and return it using the
    Griffin-Lim algorithm from the paper:
    "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
    in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April
    1984.

    Tacotron 1 Reference:
        We use the Griffin-Lim algorithm (Griffin & Lim, 1984) to synthesize waveform from the
        predicted spectrogram. We found that raising the predicted magnitudes by a power of 1.2
        before feeding to Griffin-Lim reduces artifacts, likely due to its harmonic enhancement
        effect. We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
        iterations seems to be enough), which is reasonably fast.

    Reference:
        * Tacotron Paper:
          https://arxiv.org/pdf/1703.10135.pdf
        * Griffin and Lim Paper:
          https://ieeexplore.ieee.org/document/1164317/?reload=true

    Args:
        log_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate (int): Sample rate of the spectrogram and the resulting wav file.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        fft_length (int): The size of the FFT to apply. If not provided, uses the smallest power of
          2 enclosing `frame_length`.
        window (str, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        power (float): Amplification float used to reduce artifacts.
        iterations (int): Number of iterations of griffin lim to run.
    """
    spectrogram = _log_mel_spectrogram_to_spectrogram(
        log_mel_spectrogram=log_mel_spectrogram, sample_rate=sample_rate)

    # SOURCE (Tacotron 1):
    # We found that raising the predicted magnitudes by a power of 1.2 before feeding to
    # Griffin-Lim reduces artifacts, likely due to its harmonic enhancement effect.
    magnitude_spectrogram = np.abs(spectrogram)
    magnitude_spectrogram = np.power(magnitude_spectrogram, power)

    len_samples = int((magnitude_spectrogram.shape[1] - 1) * frame_hop)
    waveform = np.random.uniform(size=(len_samples,))
    for i in tqdm(range(iterations)):
        reconstruction_spectrogram = librosa.stft(
            waveform, n_fft=fft_length, hop_length=frame_hop, win_length=frame_size, window=window)
        reconstruction_angle = np.angle(reconstruction_spectrogram).astype(np.complex64)
        # Discard magnitude part of the reconstruction and use the supplied magnitude
        # spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram * np.exp(1j * reconstruction_angle)
        waveform = librosa.istft(
            proposal_spectrogram, hop_length=frame_hop, win_length=frame_size, window=window)

    waveform = np.real(waveform)
    large_values = (waveform < -1).sum() + (waveform > 1).sum()
    logger.warning('Griffin-lim waveform clipped %d samples.', large_values)
    return np.clip(waveform, -1, 1)
