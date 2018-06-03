import copy
import functools
import logging
import math

import librosa

import numpy as np

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


def _asint(x):
    """
    Args:
        x (torch.tensor, np.ndarray, python number): Some value or array x

    Returns:
        x converted to an integer for the appropriate library
    """
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.int) if isnumpy else int(x) if isscalar else x.long()


def _asfloat(x):
    """
    Args:
        x (torch.tensor, np.ndarray, python number): Some value or array x

    Returns:
        x converted to an float for the appropriate library
    """
    isnumpy = isinstance(x, np.ndarray)
    isscalar = np.isscalar(x)
    return x.astype(np.float32) if isnumpy else float(x) if isscalar else x.float()


def inverse_mu_law(y, mu=255):
    """ Inverse of mu-law companding (mu-law expansion).

    .. math::
        f^{-1}(x) = sign(y) (1 / \mu) (1 + \mu)^{|y|} - 1)

    Args:
        y (array-like): Compressed signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Uncomprresed signal (-1 <= x <= 1)
    """
    return np.sign(y) * (1.0 / mu) * ((1.0 + mu)**np.abs(y) - 1.0)


@configurable
def inverse_mu_law_quantize(y, mu=255):
    """ Inverse of mu-law companding and quantize.

    Args:
        y (array-like): Quantized signal (∈ [0, mu]).
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Uncompressed signal ([-1, 1])

    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> x_hat = inverse_mu_law_quantize(P.mulaw_quantize(x))
        >>> x_hat = (x_hat * 32768).astype(np.int16)
    """
    # [0, m) to [-1, 1]
    y = 2 * _asfloat(y) / mu - 1
    return inverse_mu_law(y, mu)


@configurable
def mu_law_quantize(x, mu=255):
    """ Mu-Law companding and quantize.

    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Quantized signal (dtype=int)
          - y ∈ [0, mu] if x ∈ [-1, 1]
          - y ∈ [0, mu) if x ∈ [-1, 1)

    .. note::
        If you want to get quantized values of range [0, mu) (not [0, mu]),
        then you need to provide input signal of range [-1, 1).

    Examples:
        >>> from scipy.io import wavfile
        >>> import pysptk
        >>> import numpy as np
        >>> fs, x = wavfile.read(pysptk.util.example_audio_file())
        >>> x = (x / 32768.0).astype(np.float32)
        >>> y = mulaw_quantize(x)
        >>> print(y.min(), y.max(), y.dtype)
        15 246 int64
    """
    y = mu_law(x, mu=mu)
    # scale [-1, 1] to [0, mu]
    return _asint((y + 1) / 2 * mu)


def mu_law(x, mu=255):
    """ Mu-Law companding.

    Method described in paper [1]_.
    .. math::
        f(x) = sign(x) \ln (1 + \mu |x|) / \ln (1 + \mu)

    Args:
        x (array-like): Input signal. Each value of input signal must be in
          range of [-1, 1].
        mu (number): Compression parameter ``μ``.

    Returns:
        array-like: Compressed signal ([-1, 1])

    .. [1] Brokish, Charles W., and Michele Lewis. "A-law and mu-law companding
        implementations using the tms320c54x." SPRA163 (1997).
    """
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)


@configurable
def read_audio(filename, sample_rate=None):
    """ Read an audio file into a mono signal.

    Tacotron 1 Reference:
        We use 24 kHz sampling rate for all experiments.

    Notes:
        * To keep consistent with Tensorflow audio API (possibily Tacotron Tensorflow
          implementation) ensure audio files are mono WAVs with subformat PCM 16 bit and a 24 kHz
          sampling rate.
        * ``tests/test_spectrogram.py#test_librosa_tf_decode_wav`` tests that ``librosa`` and ``tf``
          decode outputs are similar.

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
        filename (str): Name of the file to load.
        sample_rate (int or None): Assert this target sample rate.
    Returns:
        numpy.ndarray [shape=(n,)]: Audio time series.
    """
    signal, observed_sample_rate = librosa.core.load(filename)
    if sample_rate is not None:
        assert sample_rate == observed_sample_rate, (
            "Sample rate must be set to %d before hand." % sample_rate)
    assert len(signal.shape) == 1, "Signal must be mono."
    assert max(signal) <= 1 and min(signal) >= -1, "Signal must be in range [-1, 1]."
    return signal


def _milliseconds_to_samples(milliseconds, sample_rate):
    """ Convert between milliseconds to the number of samples.

    Args:
        milliseconds (float)
        sample_rate (int): The number of samples per second.

    Returns
        int: Number of samples.
    """
    return int(round(milliseconds * (sample_rate / 1000)))


@configurable
def wav_to_log_mel_spectrogram(signal,
                               sample_rate,
                               frame_size=1200,
                               frame_hop=300,
                               window='hann',
                               num_mel_bins=80,
                               lower_hertz=125,
                               upper_hertz=7600,
                               min_magnitude=0.01):
    """ Transform wav file to a log mel spectrogram.

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

    Notes:
        * One difference between the Tensorflow STFT implementation and librosa is ``n_fft`` for
          Tensorflow STFT is rounded to the closests power of two. Before
          applying ``int(1 + n_fft // 2)`` to determine the number of spectrogram bins.
        * Some other differences mentioned between librosa and Tensorflow STFT:
          https://github.com/tensorflow/tensorflow/issues/16465
          https://github.com/tensorflow/tensorflow/issues/15134

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
        winow (string, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        num_mel_bins (int): How many bands in the resulting mel spectrum.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.
        min_magnitude (float): Stabilizing minimum to avoid high dynamic ranges caused by the
            singularity at zero in the mel spectrograms.

    Returns:
        log_mel_spectrograms (np.ndarray [frames, num_mel_bins]): Log mel spectrogram.
        right_pad (int): Number of zeros padding the end of the signal.
    """
    # Pad the signal by up to frame_size samples based on how many samples are remaining starting
    # from the last frame.
    # FYI: Originally, the number of spectrogram frames was
    # ``(signal.shape[0] - frame_size + frame_hop) // frame_hop``.
    # Following padding the number of spectrogram frames is
    # ``math.ceil(signal.shape[0] / frame_hop)`` a tad bit larger than the original signal.
    num_frames = math.ceil(signal.shape[0] / frame_hop)
    remainder = frame_hop * (num_frames - 1) - signal.shape[0]
    # TODO: Consider center padding instead of left padding similar to librosa
    signal = np.pad(signal, (0, frame_size + remainder), mode='constant')
    assert signal.shape[0] % frame_hop == 0

    spectrogram = librosa.stft(
        signal, n_fft=frame_size, hop_length=frame_hop, window=window, center=False)

    assert spectrogram.shape[1] == num_frames
    # Return number of padding needed to pad signal such that
    # ``spectrogram.shape[0] * num_frames == signal.shape[0]``
    ret_pad = frame_hop + remainder
    assert ret_pad <= frame_size

    # SOURCE (Tacotron 2):
    # "STFT magnitude"
    magnitude_spectrogram = np.abs(spectrogram)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
    # spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    mel_basis = librosa.filters.mel(
        sample_rate, frame_size, n_mels=num_mel_bins, fmin=lower_hertz, fmax=upper_hertz)
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrogram).transpose()

    # SOURCE (Tacotron 2):
    # Prior to log compression, the filterbank output magnitudes are clipped to a minimum value of
    # 0.01 in order to limit dynamic range in the logarithmic domain.
    mel_spectrogram = np.maximum(0.01, mel_spectrogram)

    # SOURCE (Tacotron 2):
    # followed by log dynamic range compression.
    log_mel_spectrogram = np.log(mel_spectrogram)

    return log_mel_spectrogram, ret_pad


@configurable
def _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, frame_size, sample_rate, lower_hertz,
                                        upper_hertz):
    """ Transform log mel spectrogram to spectrogram (lossy).

    Args:
        log_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        sample_rate (int): Sample rate of the ``log_mel_spectrogram``.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        (np.ndarray [frames, num_spectrogram_bins]) Spectrogram.
    """
    mel_spectrogram = np.exp(log_mel_spectrogram)
    num_mel_bins = mel_spectrogram.shape[1]
    mel_basis = librosa.filters.mel(
        sample_rate, frame_size, n_mels=num_mel_bins, fmin=lower_hertz, fmax=upper_hertz)
    # ``np.linalg.pinv`` creates approximate inverse matrix of ``mel_basis``
    mel_spectrogram = mel_spectrogram.transpose()
    inverse_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(10**-10, np.dot(inverse_mel_basis, mel_spectrogram))


@configurable
def griffin_lim(log_mel_spectrogram,
                filename,
                sample_rate,
                frame_size=1200,
                frame_hop=300,
                window='hann',
                lower_hertz=125,
                upper_hertz=7600,
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
        filename (:class:`list` of :class:`str`): Filename of the resulting wav file.
        sample_rate (int): Sample rate of the spectrogram and the resulting wav file.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        winow (string, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.
        power (float): Amplification float used to reduce artifacts.
        iterations (int): Number of iterations of griffin lim to run.
    """
    assert '.wav' in filename, "Filename must be a .wav file"

    spectrogram = _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, frame_size, sample_rate,
                                                      lower_hertz, upper_hertz)

    # SOURCE (Tacotron 1):
    # We found that raising the predicted magnitudes by a power of 1.2 before feeding to
    # Griffin-Lim reduces artifacts, likely due to its harmonic enhancement effect.
    spectrogram = spectrogram**power

    istft = functools.partial(
        librosa.istft, hop_length=frame_hop, win_length=frame_size, window='hann', center=False)
    stft = functools.partial(
        librosa.stft,
        n_fft=frame_size,
        hop_length=frame_hop,
        win_length=frame_size,
        window='hann',
        center=False)

    reconstruction_spectrogram = copy.deepcopy(spectrogram)
    for i in range(iterations):
        waveform = istft(reconstruction_spectrogram)
        estimate = stft(waveform)

        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram
        # instead.
        phase = estimate / np.maximum(1e-8, np.abs(estimate))
        reconstruction_spectrogram = spectrogram * phase

    waveform = istft(reconstruction_spectrogram)
    waveform = np.real(waveform)
    librosa.output.write_wav(filename, waveform, sr=sample_rate)
