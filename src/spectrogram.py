import argparse
import glob
import os

from matplotlib import cm
from PIL import Image

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
def wav_to_log_mel_spectrograms(filename, frame_size, frame_hop, window_function, num_mel_bins,
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
    # TODO: Concat multiple audio files and run spectrogram computation on a larger batch size

    # A batch of float32 time-domain signal in the range [-1, 1] with shape
    # [signal_length].
    signals, sample_rate = _read_audio(filename)

    # [signal_length, batch_size] -> [batch_size, signal_length]
    signals = tf.transpose(signals)

    # SOURCE (Tacotron 2):
    # As in Tacotron, mel spectrograms are computed through a shorttime Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
    frame_size = _milliseconds_to_samples(frame_size, sample_rate)
    frame_hop = _milliseconds_to_samples(frame_hop, sample_rate)

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
        spectrogram = wav_to_log_mel_spectrograms(filename)
        filename = filename.replace('.wav', '_spectrogram.png')
        save_image_of_spectrogram(spectrogram, filename)


if __name__ == "__main__":  # pragma: no cover
    tf.enable_eager_execution()
    command_line_interface()
