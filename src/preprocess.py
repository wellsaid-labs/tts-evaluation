from functools import partial

import functools
import logging
import math

from tensorflow.contrib.signal.python.ops import window_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops

import librosa

import numpy as np
import tensorflow as tf

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


@configurable
def mu_law_quantize(x, mu=255):
    """Mu-Law companding + quantize
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
    return ((y + 1) / 2 * mu).astype(np.int)


def mu_law(x, mu=255):
    """Mu-Law companding
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
def find_silence(quantized, silence_threshold=15):
    """ Given a Mu-Law companding quantized signal, trim the silence off the audio.

    Args:
        quantized (np.array dtype=int): Quantized signal.
        silence_threshold (int): Threshold for silence.

    Returns:
        start (int): End of silence in the start of the signal
        end (int): Start of silence at the end of the signal
    """
    for start in range(quantized.size):
        if abs(quantized[start] - mu_law_quantize(0)) > silence_threshold:
            break

    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - mu_law_quantize(0)) > silence_threshold:
            break

    return start, end


@configurable
def read_audio(filename, sample_rate=None, normalize=True):
    """ Read an audio file into a mono signal.

    Tacotron 1 Reference:
        We use 24 kHz sampling rate for all experiments.

    Notes:
        * To keep consistent with Tensorflow audio API (possibily Tacotron Tensorflow
          implementation) ensure audio files are mono WAVs with subformat PCM 16 bit and a 24 kHz
          sampling rate.
        * ``tests/test_spectrogram.py#test_librosa_tf_decode_wav`` tests that ``librosa`` and ``tf``
          decode outputs are similar.
        * Scaling is done because resampling can push the Waveform past [-1, 1] limits.

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
        sample_rate (int or None): Target sample rate or None to keep native sample rate.
        normalize (bool): If ``True``, rescale audio from [1, -1].
    Returns:
        numpy.ndarray [shape=(n,)]: Audio time series.
        int: Sample rate of the file.
    """
    signal, sample_rate = librosa.core.load(filename, sr=sample_rate, mono=True)
    if normalize:
        signal = signal / np.abs(signal).max()  # Normalize to [1, -1]
    return signal, sample_rate


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
                               window_function=functools.partial(
                                   window_ops.hann_window, periodic=True),
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
        frame_size (float): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (float): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        window_function (callable): A callable that takes a window length and a dtype keyword
            argument and returns a [window_length] Tensor of samples in the provided datatype. If
            set to None, no windowing is used.
        num_mel_bins (int): How many bands in the resulting mel spectrum.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.
        min_magnitude (float): Stabilizing minimum to avoid high dynamic ranges caused by the
            singularity at zero in the mel spectrograms.

    Returns:
        log_mel_spectrograms (np.array [frames, num_mel_bins]): Log mel spectrogram.
        right_pad (int): Number of zeros padding the end of the signal.
    """
    # TODO: Update tests after args update
    signals = np.expand_dims(signal, axis=0)

    # A batch of float32 time-domain signal in the range [-1, 1] with shape
    # [1, signal_length].
    signals = tf.convert_to_tensor(signals)

    # Simplifies padding mathematics.
    assert frame_size % frame_hop == 0

    # NOTE: Tacotron 2 authors confirmed they padded the signal over GChat to fullfil requirements
    # for Wavenet.

    # Specotrogram shape ``[1, (signals.shape[1] - frame_size + frame_hop) // frame_hop]``.
    # We need the shape to be divisable by ``signals.shape[1] % frame_hop == 0`` for Wavenet
    # upsampling. First, we deal with ``- frame_size + frame_hop``:
    front_pad = tf.zeros([1, frame_size - frame_hop])

    # Next, we deal with ``// frame_hop`` (fyi ``//`` is floor division):
    remainder = (signals.shape[1] + frame_size - frame_hop) % frame_hop
    remainder = (frame_hop - remainder)
    right_pad = remainder
    end_pad = tf.zeros([1, remainder])

    signals = tf.concat([front_pad, signals, end_pad], 1)
    assert signals.shape[1] % frame_hop == 0

    # NOTE: Zero padding affects the ``stft`` for the particular frames it's applied. We apply it
    # consistently at the beginning ``frame_size - frame_hop`` but inconsistently at the end
    # ``remainder``; therefore, it may be useful to mask the loss for Wavenet.
    spectrograms = tf.contrib.signal.stft(
        signals,
        frame_length=frame_size,
        frame_step=frame_hop,
        window_fn=window_function,
    )

    # Finally, we need ``spectrograms.shape[1] * frame_hop == signals.shape[1]``. At this point:
    # ``spectrograms.shape[1] = (signals.shape[1] - frame_size + frame_hop) / frame_hop``
    # ``(signals.shape[1] - frame_size + frame_hop) == signals.shape[1]``
    # The ``signals.shape[1]`` is ``frame_size - frame_hop`` too big; therefore, we cut out the
    # padding from before:
    signals = signals[:, (frame_size - frame_hop):]
    assert spectrograms.shape[1] * frame_hop == signals.shape[1]

    # SOURCE (Tacotron 2):
    # "STFT magnitude"
    magnitude_spectrograms = tf.abs(spectrograms)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
    # spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_hertz, upper_hertz)
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # SOURCE (Tacotron 2):
    # Prior to log compression, the filterbank output magnitudes are clipped to a minimum value of
    # 0.01 in order to limit dynamic range in the logarithmic domain.
    mel_spectrograms = tf.maximum(0.01, mel_spectrograms)

    # SOURCE (Tacotron 2):
    # followed by log dynamic range compression.
    log_mel_spectrograms = tf.log(mel_spectrograms)

    return log_mel_spectrograms[0].numpy(), right_pad


# INSPIRED BY:
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/signal/python/ops/spectral_ops.py
def _enclosing_power_of_two(value):
    """ Return 2**N for integer N such that 2**N >= value. """
    value = ops.convert_to_tensor(value, name='frame_length')
    value_static = tensor_util.constant_value(value)
    if value_static is not None:
        return constant_op.constant(
            int(2**np.ceil(np.log(value_static) / np.log(2.0))), value.dtype)
    return math_ops.cast(
        math_ops.pow(2.0, math_ops.ceil(
            math_ops.log(math_ops.to_float(value)) / math_ops.log(2.0))), value.dtype)


def _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, frame_size, sample_rate, lower_hertz,
                                        upper_hertz):
    """ Transform log mel spectrogram to spectrogram (lossy).

    Args:
        log_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        frame_size (float): The frame size in samples.
        sample_rate (int): Sample rate of the ``log_mel_spectrogram``.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        A ``[frames, num_spectrogram_bins]`` ``Tensor`` of ``complex64`` STFT values.
    """
    log_mel_spectrogram = tf.convert_to_tensor(log_mel_spectrogram, dtype=tf.complex64)
    log_mel_spectrograms = tf.expand_dims(log_mel_spectrogram, 0)

    # Reverse the operations from ``log_mel_spectrograms`` to ``spectrograms``
    mel_spectrograms = tf.exp(log_mel_spectrograms)  # `tf.log`` is the natural log
    num_mel_bins = mel_spectrograms.shape[-1].value
    # Documentation of Tensorflow mentions:
    # "num_spectrogram_bins ... understood to be fft_size // 2 + 1"
    # https://www.tensorflow.org/api_docs/python/tf/contrib/signal/linear_to_mel_weight_matrix
    # "fft_unique_bins is fft_length // 2 + 1"
    # https://www.tensorflow.org/api_docs/python/tf/contrib/signal/stft
    # "fft_length ...  uses the smallest power of 2 enclosing frame_length."
    num_spectrogram_bins = _enclosing_power_of_two(frame_size) // 2 + 1
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate, lower_hertz, upper_hertz)
    # ``np.linalg.pinv`` creates approximate inverse matrix of ``linear_to_mel_weight_matrix``
    mel_to_linear_weight_matrix = tf.py_func(np.linalg.pinv, [linear_to_mel_weight_matrix],
                                             tf.float32)
    mel_to_linear_weight_matrix = tf.cast(mel_to_linear_weight_matrix, tf.complex64)
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    spectrograms = tf.tensordot(mel_spectrograms, mel_to_linear_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        mel_to_linear_weight_matrix.shape[-1:]))
    spectrograms = tf.squeeze(spectrograms, 0)
    return spectrograms


@configurable
def log_mel_spectrogram_to_wav(log_mel_spectrogram,
                               filename,
                               sample_rate,
                               frame_size=1200,
                               frame_hop=300,
                               window_function=functools.partial(
                                   window_ops.hann_window, periodic=True),
                               lower_hertz=125,
                               upper_hertz=7600,
                               power=1.2,
                               iterations=30,
                               log=False):
    """ Transform log mel spectrogram to wav file with the Griffin-Lim algorithm.

    # TODO: Try using Mozillas/TTS Griffin lim

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
        frame_size (float): The frame size in milliseconds.
        frame_hop (float): The frame hop in milliseconds.
        window_function (callable): A callable that takes a window length and a dtype keyword
            argument and returns a [window_length] Tensor of samples in the provided datatype. If
            set to None, no windowing is used.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.
        power (float): Amplification float used to reduce artifacts
        loss_diff (float): Difference in loss used to determine convergance.
        log (bool): If bool is True, prints the RMSE as the algorithm runs.
    """
    assert '.wav' in filename, "Filename must be a .wav file"

    # Complex operations are not defined for GPU
    with tf.device('/cpu'):
        # Convert hertz to more relevant units like samples
        spectrogram = _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, frame_size,
                                                          sample_rate, lower_hertz, upper_hertz)
        spectrograms = tf.expand_dims(spectrogram, 0)

        inverse_stft = partial(
            tf.contrib.signal.inverse_stft,
            frame_length=frame_size,
            frame_step=frame_hop,
            window_fn=tf.contrib.signal.inverse_stft_window_fn(
                frame_step=frame_hop, forward_window_fn=window_function))
        stft = partial(
            tf.contrib.signal.stft,
            frame_length=frame_size,
            frame_step=frame_hop,
            window_fn=window_function)

        # SOURCE (Tacotron 1):
        # We found that raising the predicted magnitudes by a power of 1.2 before feeding to
        # Griffin-Lim reduces artifacts, likely due to its harmonic enhancement effect.
        spectrograms = tf.pow(spectrograms, power)

        # Run the Griffin-Lim algorithm
        spectrograms = tf.cast(tf.abs(spectrograms), dtype=tf.complex64)
        time_slices = spectrograms.shape[1] - 1
        len_samples = int(time_slices * frame_hop + frame_size)
        waveform = tf.random_uniform((1, len_samples))
        for i in range(iterations):
            reconstruction_spectrogram = stft(waveform)
            reconstruction_angle = tf.cast(tf.angle(reconstruction_spectrogram), dtype=tf.complex64)
            # Discard magnitude part of the reconstruction and use the supplied magnitude
            # spectrogram instead.
            proposal_spectrogram = spectrograms * tf.exp(
                tf.complex(0.0, 1.0) * reconstruction_angle)
            previous_waveform = waveform
            waveform = inverse_stft(proposal_spectrogram)
            if log:
                loss = tf.reduce_sum((waveform - previous_waveform)**2)
                loss = math.sqrt(loss / tf.size(waveform, out_type=tf.float32))
                logger.info('Reconstruction iteration: {} RMSE: {} '.format(i, loss))

        waveform = tf.real(waveform)
        librosa.output.write_wav(filename, waveform[0].numpy(), sr=sample_rate)


def plot_spectrogram(spectrogram, filename, title='Mel-Spectrogram'):
    """ Save image of spectrogram to disk.

    Args:
        spectrogram (Tensor): A ``[frames, num_mel_bins]`` ``Tensor`` of ``complex64`` STFT
            values.
        filename (str): Name of the file to save to.
        title (str): Title of the plot.

    Returns:
        None
    """
    import matplotlib.pyplot as plt

    assert '.png' in filename.lower(), "Filename saves in PNG format"

    plt.figure()
    plt.style.use('ggplot')
    plt.imshow(np.rot90(spectrogram))
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Mel-Channels')
    xlabel = 'Frames'
    plt.xlabel(xlabel)
    plt.title(title)
    plt.savefig(filename, format='png', bbox_inches='tight')
    plt.close()
