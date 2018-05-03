from functools import partial

import argparse
import glob
import os

from matplotlib import cm
from PIL import Image
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops

import librosa
import numpy as np
import tensorflow as tf

from src.configurable import configurable


@configurable
def _read_audio(filename, sample_rate=None):
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

    Args:
        filename (str): Name of the file to load.
        sample_rate (int or None): Target sample rate or None to keep native sample rate.
    Returns:
        numpy.ndarray [shape=(n, 1)]: Audio time series.
        int: Sample rate of the file.
    """
    audio, sample_rate = librosa.core.load(filename, sr=sample_rate, mono=True)
    audio = np.expand_dims(audio, axis=1)
    return audio, sample_rate


def _milliseconds_to_samples(milliseconds, sample_rate):
    """ Convert between milliseconds to the number of samples.

    Args:
        milliseconds (float)
        sample_rate (int): The number of samples per second.

    Returns
        int: Number of samples.
    """
    return int(round(milliseconds * (sample_rate / 1000)))


def _get_wav_filenames_from_path(path):
    """ Get a list of WAV files from a path.

    Args:
        path (str): Path to a directory of WAV files or WAV file.

    Returns:
        :class:`list` of :class`str`: List of WAV files.
    """
    if os.path.isfile(path):
        assert '.wav' in path, "Path must be a directory of WAV files or a WAV file."
        filenames = [path]
    elif os.path.isdir(path):
        filenames = [f for f in glob.iglob(os.path.join(path, '**/*.wav'), recursive=True)]
        assert len(filenames) != 0, "Path must be a directory of WAV files or a WAV file."
    else:
        raise ValueError('Pass either a directory or file')
    return filenames


@configurable
def wav_to_log_mel_spectrogram(filename, frame_size, frame_hop, window_function, num_mel_bins,
                               lower_hertz, upper_hertz, min_magnitude):
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

    TODO: This function runs slowly (23 minutes on a GPU for the LJSpeech dataset) due to running
    one signal at a time. Consider, batching Waveforms. That requires Waveforms to be padded with
    zero on the end to remove length variability. Using the original length, we compute the expected
    frames of the Spectrogram. Finally, we generate the Spectrogram and cut it at the appropriate
    length. To ensure correctness of this approach, we can easily test batched vs non-batched
    spectrograms.

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
        filenames (:class:`list` of :class:`str`): Names of the files to load.
        frame_size (float): The frame size in milliseconds.
        frame_hop (float): The frame hop in milliseconds.
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
        A ``[frames, num_mel_bins]`` ``Tensor`` of ``complex64`` STFT values.
    """
    # A batch of float32 time-domain signal in the range [-1, 1] with shape
    # [signal_length].
    signals, sample_rate = _read_audio(filename)
    signals = tf.convert_to_tensor(signals)

    # [signal_length, batch_size] -> [batch_size, signal_length]
    signals = tf.transpose(signals)

    # SOURCE (Tacotron 2):
    # As in Tacotron, mel spectrograms are computed through a shorttime Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
    frame_size = _milliseconds_to_samples(frame_size, sample_rate)
    frame_hop = _milliseconds_to_samples(frame_hop, sample_rate)
    print('frame_size', frame_size)
    print('frame_hop', frame_hop)

    spectrograms = tf.contrib.signal.stft(
        signals,
        frame_length=frame_size,
        frame_step=frame_hop,
        window_fn=window_function,
    )

    # SOURCE (Tacotron 2):
    # "STFT magnitude"
    magnitude_spectrograms = tf.abs(spectrograms)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
    # spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    print('num_spectrogram_bins', num_spectrogram_bins)
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

    return log_mel_spectrograms[0].numpy()


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


@configurable
def log_mel_spectrogram_to_wav(log_mel_spectrogram, filename, frame_size, frame_hop,
                               window_function, sample_rate, lower_hertz, upper_hertz, iterations):
    """ Transform log mel spectrogram to wav file with the Griffin-Lim algorithm.

    Tacotron 1 Reference:
        We use the Griffin-Lim algorithm (Griffin & Lim, 1984) to synthesize waveform from the
        predicted spectrogram. We found that raising the predicted magnitudes by a power of 1.2
        before feeding to Griffin-Lim reduces artifacts, likely due to its harmonic enhancement
        effect. We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
        iterations seems to be enough), which is reasonably fast.


    Reference:
        * Tacotron Paper:
          https://arxiv.org/pdf/1703.10135.pdf

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
    """
    log_mel_spectrogram = tf.convert_to_tensor(log_mel_spectrogram, dtype=tf.complex64)
    frame_size = _milliseconds_to_samples(frame_size, sample_rate)
    frame_hop = _milliseconds_to_samples(frame_hop, sample_rate)
    log_mel_spectrograms = tf.expand_dims(log_mel_spectrogram, 0)

    # Reverse the log operation
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
    mel_to_linear_weight_matrix = tf.py_func(np.linalg.pinv, [linear_to_mel_weight_matrix],
                                             tf.float32)
    mel_to_linear_weight_matrix = tf.cast(mel_to_linear_weight_matrix, tf.complex64)
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    spectrograms = tf.tensordot(mel_spectrograms, mel_to_linear_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        mel_to_linear_weight_matrix.shape[-1:]))
    # spectrograms = tf.maximum(1e-10, spectrograms)
    print('spectrograms', tf.reduce_sum(spectrograms))

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

    angles = tf.cast(np.exp(2j * np.pi * np.random.rand(*spectrograms.shape)), dtype=tf.complex64)
    spectrogram_complex = tf.cast(tf.abs(spectrograms), dtype=tf.complex64)
    y = inverse_stft(spectrogram_complex * angles)
    for i in range(iterations):
        angles = tf.cast(np.exp(1j * np.angle(stft(y).numpy())), dtype=tf.complex64)
        y = inverse_stft(spectrogram_complex * angles)

    waveform = tf.real(y)
    librosa.output.write_wav(filename, waveform[0].numpy(), sr=sample_rate)


def save_image_of_spectrogram(spectrogram, filename):
    """ Save image of spectrogram to disk.

    Args:
        spectrogram (Tensor): A ``[frames, num_mel_bins]`` ``Tensor`` of ``complex64`` STFT
            values.
        filename (str): Name of the file to save to.

    Returns:
        None
    """
    assert '.png' in filename, "Filename must be a PNG"
    spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=1)  # flip against freq axis
    spectrogram = np.uint8(cm.viridis(spectrogram.T) * 255)
    image = Image.fromarray(spectrogram)
    image.save(filename, 'png')


def command_line_interface():
    """ Command line interface to convert a directory of WAV files or WAV file to spectrograms.
    """
    from src.hparams import set_hparams

    set_hparams()

    parser = argparse.ArgumentParser(description='Convert WAV to a log mel spectrogram CLI.')
    parser.add_argument('path', type=str, help='filename or directory of WAVs to convert')
    args = parser.parse_args()

    filenames = _get_wav_filenames_from_path(args.path)

    for filename in filenames:
        spectrogram = wav_to_log_mel_spectrogram(filename)
        filename = filename.replace('.wav', '_spectrogram.png')
        save_image_of_spectrogram(spectrogram, filename)


if __name__ == "__main__":  # pragma: no cover
    tf.enable_eager_execution()
    command_line_interface()
