from collections import namedtuple
from pathlib import Path

import logging
import math
import os
import struct
import subprocess

from hparams import configurable
from hparams import HParam
from third_party import LazyLoader
from tqdm import tqdm

import numpy as np
import torch

# NOTE: `LazyLoader` allows for this repository to run without these dependancies. Also,
# it side-steps this issue: https://github.com/librosa/librosa/issues/924.
librosa = LazyLoader('librosa', globals(), 'librosa')
scipy_wavfile = LazyLoader('scipy_wavfile', globals(), 'scipy.io.wavfile')

from src.environment import TTS_DISK_CACHE_NAME
from src.utils import disk_cache
from src.utils import get_chunks
from src.utils import make_arg_key
from src.utils import Pool
from src.utils import assert_no_overwritten_files

logger = logging.getLogger(__name__)


def get_num_seconds(audio_path):
    """ Get the number of seconds for a WAV audio file.

    Args:
        audio_path (str)

    Returns:
        (float): The number of seconds in audio.
    """
    metadata = get_audio_metadata(Path(audio_path))
    BITS_PER_BYTE = 8
    # Learn more: http://www.topherlee.com/software/pcm-tut-wavformat.html
    WAV_HEADER_LENGTH_IN_BYTES = 44
    bytes_per_second = (metadata.sample_rate * (metadata.bits / BITS_PER_BYTE)) * metadata.channels
    return (os.path.getsize(audio_path) - WAV_HEADER_LENGTH_IN_BYTES) / bytes_per_second


@configurable
def read_audio(filename, assert_metadata=HParam()):
    """ Read an audio file.

    TODO: Rename considering the tighter specification.

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
        assert_metadata (WavFileMetadata): Assert this metadata for any audio file read.

    Returns:
        See the second return value of `scipy.io.wavfile.read`.
    """
    metadata = get_audio_metadata(Path(filename))
    assert metadata == assert_metadata, (
        "The filename (%s) metadata `%s` does not match `assert_metadata` `%s`." %
        (filename, metadata, assert_metadata))
    _, signal = scipy_wavfile.read(str(filename))
    assert len(signal) > 0, "Signal (%s) length is zero." % filename
    return signal


def integer_to_floating_point_pcm(signal):
    """ Convert a `int32` or `int16` PCM signal representation to a `float32`
    PCM representation.

    Learn more about the common data types:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#r7015bff88555-1

    Implementation inspired by:
    https://librosa.github.io/librosa/_modules/librosa/util/utils.html#buf_to_float

    Args:
        signal (np.ndarry or torch.Tensor): Signal with a dtype of `float32`, `int32` or
          `int16`.

    Returns:
        (np.ndarray or torch.Tensor): Signal with a dtype of `float32` with a range of [-1.0, 1.0].
    """
    is_tensor = torch.is_tensor(signal)

    if is_tensor and signal.dtype == torch.float32:
        return signal

    signal = signal.detach().cpu().numpy() if is_tensor else signal

    if signal.dtype != np.float32:
        assert signal.dtype in (np.int32, np.int16)
        scale = 1. / float(1 << ((8 * signal.dtype.itemsize) - 1))  # Invert the scale of the data
        signal = scale * np.frombuffer(signal, '<i{:d}'.format(signal.dtype.itemsize)).astype(
            np.float32)

    return torch.tensor(signal) if is_tensor else signal


@configurable
def write_audio(filename, audio, sample_rate=HParam(int), overwrite=False):
    """ Write a numpy array as a WAV file.

    Args:
        filename (Path or str or open file handle): Name of the file to load.
        audio (np.ndarray or torch.Tensor): A 1-D or 2-D matrix of either integer or float
            data-type.
        sample_rate (int, optional): The sample rate (in samples/sec).
        overwrite (bool, optional): If `True` this allows for `path` to be overwritten.

    Returns:
        See `scipy.io.wavfile.write`.
    """
    if not overwrite and Path(filename).exists():
        raise ValueError('A file already exists at %s' % filename)

    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()

    assert audio.dtype in (np.int32, np.int16, np.uint8, np.float32)
    if audio.dtype == np.float32 and audio.shape[0] > 0:
        assert np.max(audio) <= 1.0 and np.min(audio) >= -1.0, (
            "Signal (%s) must be in range [-1, 1]." % filename)

    return scipy_wavfile.write(filename, sample_rate, audio)


@configurable
def _mel_filters(sample_rate, num_mel_bins, fft_length, lower_hertz=HParam(), upper_hertz=HParam()):
    """ Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    Reference:
        * The API written by RJ Skerry-Ryan a Tacotron author.
          https://www.tensorflow.org/api_docs/python/tf/contrib/signal/linear_to_mel_weight_matrix

    Args:
        sample_rate (int): The sample rate of the signal.
        num_mel_bins (int): Number of Mel bands to generate.
        fft_length (int): The size of the FFT to apply. If not provided, uses the smallest
            power of 2 enclosing `frame_length`.
        lower_hertz (int): Lower bound on the frequencies to be included in the mel
            spectrum. This corresponds to the lower edge of the lowest triangular band.
        upper_hertz (int): The desired top edge of the highest frequency band.

    Returns:
        (np.ndarray [num_mel_bins, 1 + fft_length / 2]): Mel transform matrix.
    """
    # NOTE: The Tacotron 2 model likely did not normalize the filterbank; otherwise, the 0.01
    # minimum mentioned in their paper for the dynamic range is too high. NVIDIA/tacotron2 includes
    # norm and had to set their minimum to 10**-5 to compensate.
    # NOTE: ``htk=True`` because normalization of the mel filterbank is from Slaney's algorithm.
    lower_hertz = 0.0 if lower_hertz is None else lower_hertz
    return librosa.filters.mel(
        sample_rate,
        fft_length,
        n_mels=num_mel_bins,
        fmin=lower_hertz,
        fmax=upper_hertz,
        norm=None,
        htk=True)


class SignalToMelSpectrogram(torch.nn.Module):
    """ Compute a log-mel-scaled spectrogram from signal.

    This function guarantees similar results to `get_log_mel_spectrogram`; however, it's implemented
    in PyTorch, is differentiable, and can batch process.
    """

    @configurable
    def __init__(self,
                 fft_length=HParam(),
                 frame_hop=HParam(),
                 sample_rate=HParam(),
                 num_mel_bins=HParam(),
                 window=HParam(),
                 min_magnitude=HParam()):
        super().__init__()

        mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=fft_length)
        mel_basis = torch.from_numpy(mel_basis).float()

        self.register_buffer('mel_basis', mel_basis)
        self.register_buffer('window', window)
        self.register_buffer('min_magnitude', torch.tensor(min_magnitude).float())

        self.fft_length = fft_length
        self.frame_hop = frame_hop
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins

    def forward(self, signal):
        """
        Args:
            signal (torch.FloatTensor [batch_size, signal_length])

        Returns:
            log_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
        """
        signal = signal.view(-1, signal.shape[-1])
        spectrogram = torch.stft(
            signal,
            n_fft=self.fft_length,
            hop_length=self.frame_hop,
            win_length=self.window.shape[0],
            window=self.window,
            center=False)
        # NOTE: The below `norm` line is equal to a numerically stable version of the below...
        # >>> real_part, imag_part = spectrogram.unbind(-1)
        # >>> magnitude_spectrogram = torch.sqrt(real_part**2 + imag_part**2)
        magnitude_spectrogram = torch.norm(spectrogram, dim=-1)
        mel_spectrogram = torch.matmul(self.mel_basis, magnitude_spectrogram).transpose(0, 1)
        return torch.max(self.min_magnitude, mel_spectrogram).permute(1, 2, 0).squeeze()


class SignalToLogMelSpectrogram(SignalToMelSpectrogram):

    def forward(self, signal):
        return torch.log(super().forward(signal))


def _get_spectrogram(signal, sample_rate, frame_size, frame_hop, fft_length, window, center=True):
    """ Helper function for `get_log_mel_spectrogram`. """
    if center:
        # NOTE: Check ``notebooks/Signal_to_Spectrogram_Consistency.ipynb`` for the correctness
        # of this padding algorithm.
        # NOTE: Pad signal so that is divisable by ``frame_hop``
        remainder = frame_hop - signal.shape[0] % frame_hop
        padding = (math.ceil(remainder / 2), math.floor(remainder / 2))
        padded_signal = np.pad(signal, padding, mode='constant', constant_values=0)
        assert padded_signal.shape[0] % frame_hop == 0

        # NOTE: Center the signal such that the resulting spectrogram and audio are aligned. Learn
        # more here: https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
        padded_signal = np.pad(
            padded_signal, int(fft_length // 2), mode='constant', constant_values=0)
    else:
        padded_signal = signal

    # NOTE: The number of spectrogram frames generated is, with ``center=True``:
    # ``(maybe_padded_signal.shape[0] + frame_hop) // frame_hop``.
    # Otherewise, it's: ``(maybe_padded_signal.shape[0] - frame_size + frame_hop) // frame_hop``
    spectrogram = librosa.stft(
        integer_to_floating_point_pcm(padded_signal),
        n_fft=fft_length,
        hop_length=frame_hop,
        win_length=frame_size,
        window=window,
        center=False)

    if center:
        # NOTE: Return number of padding needed to pad signal such that
        # ``spectrogram.shape[0] * num_frames == signal.shape[0]``
        # This is padding is partly determined by ``center=True`` librosa padding.
        ret_pad = frame_hop + remainder
        assert ret_pad <= frame_size
        assert (signal.shape[0] + ret_pad) % frame_hop == 0
        ret_pad = (math.ceil(ret_pad / 2), math.floor(ret_pad / 2))
        ret_signal = np.pad(signal, ret_pad, mode='constant', constant_values=0)

    if center:
        return spectrogram, ret_signal
    else:
        return spectrogram


@configurable
def get_log_mel_spectrogram(signal,
                            sample_rate=HParam(),
                            frame_size=HParam(),
                            frame_hop=HParam(),
                            fft_length=HParam(),
                            window=HParam(),
                            min_magnitude=HParam(),
                            num_mel_bins=HParam(),
                            center=True):
    """ Compute a log-mel-scaled spectrogram from signal.

    TODO: Remove this in favor of `SignalToLogMelSpectrogram`.

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
        signal (np.array [signal_length]): `float32`, `int32` or `int16 time-domain
            signal in the range [-1, 1].
        sample_rate (int): Sample rate for the signal.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        fft_length (int): The window size used by the fourier transform.
        window (str, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        min_magnitude (float): Stabilizing minimum to avoid high dynamic ranges caused by
            the singularity at zero in the mel spectrograms.
        num_mel_bins (int): Number of Mel bands to generate.
        center (bool, optional): Return a spectrogram and audio that are aligned.

    Returns:
        log_mel_spectrograms (np.ndarray [frames, num_mel_bins]): Log mel spectrogram.
        signal (np.ndarray): Returns if `center` is `True` a `signal` with padding such that:
            `(signal.shape[0] + pad) / frame_hop == log_mel_spectrograms.shape[0]`
    """
    if center:
        spectrogram, ret_signal = _get_spectrogram(
            signal, sample_rate, frame_size, frame_hop, fft_length, window, center=center)
    else:
        spectrogram = _get_spectrogram(
            signal, sample_rate, frame_size, frame_hop, fft_length, window, center=center)

    # SOURCE (Tacotron 2):
    # "STFT magnitude"
    magnitude_spectrogram = np.abs(spectrogram)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank
    # spanning 125 Hz to 7.6 kHz, followed by log dynamic range compression.
    mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=fft_length)
    mel_spectrogram = np.dot(mel_basis, magnitude_spectrogram).transpose()

    # SOURCE (Tacotron 2):
    # Prior to log compression, the filterbank output magnitudes are clipped to a minimum value of
    # 0.01 in order to limit dynamic range in the logarithmic domain.
    mel_spectrogram = np.maximum(0.01, mel_spectrogram)

    # SOURCE (Tacotron 2):
    # followed by log dynamic range compression.
    log_mel_spectrogram = np.log(mel_spectrogram)

    log_mel_spectrogram = log_mel_spectrogram.astype(np.float32)  # ``np.float64`` → ``np.float32``

    if center:
        return log_mel_spectrogram, ret_signal
    else:
        return log_mel_spectrogram


def _log_mel_spectrogram_to_spectrogram(log_mel_spectrogram, sample_rate, fft_length):
    """ Transform log mel spectrogram to spectrogram (lossy).

    Args:
        log_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate (int): Sample rate of the ``log_mel_spectrogram``.
        fft_length (int): The size of the FFT to apply. If not provided, uses the smallest
            power of 2 enclosing `frame_length`.

    Returns:
        (np.ndarray [frames, num_spectrogram_bins]): Spectrogram.
    """
    mel_spectrogram = np.exp(log_mel_spectrogram)
    num_mel_bins = mel_spectrogram.shape[1]
    mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=fft_length)

    # ``np.linalg.pinv`` creates approximate inverse matrix of ``mel_basis``
    inverse_mel_basis = np.linalg.pinv(mel_basis)
    return np.dot(inverse_mel_basis, mel_spectrogram.transpose()).transpose()


@configurable
def griffin_lim(log_mel_spectrogram,
                sample_rate=HParam(),
                frame_size=HParam(),
                frame_hop=HParam(),
                fft_length=HParam(),
                window=HParam(),
                power=HParam(),
                iterations=HParam(),
                use_tqdm=False):
    """ Transform log mel spectrogram to waveform with the Griffin-Lim algorithm.

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
        fft_length (int): The size of the FFT to apply. If not provided, uses the smallest
            power of 2 enclosing `frame_length`.
        window (str, tuple, number, callable): Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        power (float): Amplification float used to reduce artifacts.
        iterations (int): Number of iterations of griffin lim to run.
        use_tqdm (bool, optional): If `True` attach a progress bar during iteration.

    Returns:
        (np.ndarray [num_samples]): Predicted waveform.
    """
    try:
        logger.info('Running Griffin-Lim....')
        spectrogram = _log_mel_spectrogram_to_spectrogram(
            log_mel_spectrogram=log_mel_spectrogram, sample_rate=sample_rate, fft_length=fft_length)
        spectrogram = spectrogram.transpose()
        waveform = librosa.core.griffinlim(
            spectrogram,
            n_iter=iterations,
            hop_length=frame_hop,
            win_length=frame_size,
            window=window)
        # NOTE: Pad to ensure spectrogram and waveform align.
        waveform = np.pad(waveform, int(frame_hop // 2), mode='constant', constant_values=0)
        large_values = (waveform < -1).sum() + (waveform > 1).sum()
        if large_values > 0:
            logger.warning('Griffin-lim waveform clipped %d samples.', large_values)
        return np.clip(waveform, -1, 1).astype(np.float32)
    # TODO: Be more specific with the cases that this is capturing so that we don't have silent
    # failures for invalid inputs.
    except Exception:
        logger.warning('Griffin-lim encountered an issue and was unable to render audio.')
        # NOTE: Return no audio for valid inputs that fail due to an overflow error or a small
        # spectrogram.
        return np.array([], dtype=np.float32)


@configurable
def build_wav_header(num_frames,
                     frame_rate=HParam(),
                     wav_format=0x0001,
                     num_channels=1,
                     sample_width=2):
    """ Create a WAV file header.

    Args:
        num_frames (int): Number of frames. A frame includes one sample per channel.
        frame_rate (int): Number of frames per second.
        wav_format (int): Format of the audio file, 1 indiciates PCM format.
        num_channels (int): Number of audio channels.
        sample_width (int): Number of bytes per sample, typically 1 (8-bit), 2 (16-bit)
            or 4 (32-bit).

    Returns:
        (bytes): Bytes representing the WAV header.
        (int): File size in bytes.
    """
    # Inspired by: https://github.com/python/cpython/blob/master/Lib/wave.py
    # Inspired by: https://github.com/scipy/scipy/blob/v1.2.0/scipy/io/wavfile.py#L284-L396
    header_length = 36
    data_length = num_frames * num_channels * sample_width
    file_size = header_length + data_length + 8
    bytes_per_second = num_channels * frame_rate * sample_width
    block_align = num_channels * sample_width
    bit_depth = sample_width * 8

    header = b'RIFF'  # RIFF identifier
    header += struct.pack('<I', header_length + data_length)  # RIFF chunk length
    header += b'WAVE'  # RIFF type
    header += b'fmt '  # Format chunk identifier
    header += struct.pack('<I', 16)  # Format chunk length
    header += struct.pack('<HHIIHH', wav_format, num_channels, frame_rate, bytes_per_second,
                          block_align, bit_depth)

    if ((len(header) - 4 - 4) + (4 + 4 + data_length)) > 0xFFFFFFFF:
        raise ValueError('Data exceeds wave file size limit.')

    header += b'data'  # Data chunk identifier
    header += struct.pack('<I', data_length)  # Data chunk length
    return header, file_size


@configurable
def split_signal(signal, bits=HParam()):
    """ Compute the coarse and fine components of the signal.

    Args:
        signal (torch.FloatTensor): Signal with values ranging from [-1, 1]
        bits (int): Total number of bits to encode signal in.

    Returns:
        coarse (torch.LongTensor): Top bits of the signal.
        fine (torch.LongTensor): Bottom bits of the signal.
    """
    assert torch.min(signal) >= -1.0 and torch.max(signal) <= 1.0
    assert (bits %
            2 == 0), 'To support an even split between coarse and fine, use an even number of bits'
    range_ = int((2**(bits - 1)))
    signal = torch.round(signal * range_)
    signal = torch.clamp(signal, -1 * range_, range_ - 1)
    unsigned = signal + range_  # Move range minimum to 0
    bins = int(2**(bits / 2))
    coarse = torch.floor(unsigned / bins)
    fine = unsigned % bins
    return coarse.long(), fine.long()


@configurable
def combine_signal(coarse, fine, bits=HParam(), return_int=False):
    """ Compute the coarse and fine components of the signal.

    Args:
        coarse (torch.LongTensor): Top bits of the signal.
        fine (torch.LongTensor): Bottom bits of the signal.
        bits (int): Total number of bits to encode signal in.
        return_int (bool, optional): Return in the range of integer min to max instead of [-1, 1].

    Returns:
        signal (torch.FloatTensor): Signal with values ranging from [-1, 1] if ``return_int`` is
            ``False``; Otherwise, return an integer value from integer min to max.
    """
    bins = int(2**(bits / 2))
    assert torch.min(coarse) >= 0 and torch.max(coarse) < bins
    assert torch.min(fine) >= 0 and torch.max(fine) < bins
    signal = coarse * bins + fine - 2**(bits - 1)

    if return_int:
        if bits == 16:
            return signal.type(torch.int16)
        else:
            raise ValueError('Only 16-bit fidelity is supported.')

    return signal.float() / 2**(bits - 1)  # Scale to [-1, 1] range.


# Args:
#   sample_rate (int): The sample rate of the audio.
#   bits (int): The bit depth of the audio.
#   channels (int): The number of audio channels in the audio file.
#   encoding (str): The encoding of the audio file: ['signed-integer', 'unsigned-integer',
#     'floating-point']. The encoding options are based off SoX's encoding options. Learn more:
#     http://sox.sourceforge.net/sox.html
WavFileMetadata = namedtuple('WavFileMetadata', ['sample_rate', 'bits', 'channels', 'encoding'])


def _parse_audio_metadata(metadata):
    """ Parse audio metadata returned by `sox --i`.

    Example:

        >>> metadata = '''
        ... Input File     : 'data/Heather Doe/03 Recordings/Heather_4-21.wav'
        ... Channels       : 1
        ... Sample Rate    : 44100
        ... Precision      : 24-bit
        ... Duration       : 03:46:28.09 = 599234761 samples = 1.01911e+06 CDDA sectors
        ... File Size      : 1.80G
        ... Bit Rate       : 1.06M
        ... Sample Encoding: 24-bit Signed Integer PCM
        ... '''
        >>> str(_parse_audio_metadata(metadata)[0])
        'data/Heather Doe/03 Recordings/Heather_4-21.wav'
        >>> _parse_audio_metadata(metadata)[1]
        WavFileMetadata(sample_rate=44100, bits=24, channels=1, encoding='signed-integer')

    Args:
        metadata (str)

    Returns:
        (Path): The audio path of the parsed `metadata`.
        (WavFileMetadata): The parsed `metadata`.
    """
    # NOTE: Parse the output of `sox --i` instead individual requests like `sox --i -r` and
    # `sox --i -b` to conserve on the number  of requests made via `subprocess`.
    metadata = [s.split(':')[1].strip() for s in metadata.strip().split('\n')]
    audio_path = str(metadata[0][1:-1])
    channels = int(metadata[1])
    sample_rate = int(metadata[2])
    encoding = metadata[-1].split()
    bits = encoding[0]
    assert bits[-4:] == '-bit', 'Unexpected format.'
    bits = int(bits[:-4])
    assert encoding[-1] == 'PCM', 'Unexpected format.'
    encoding = '-'.join(encoding[1:-1]).lower().strip()
    assert encoding in ['signed-integer', 'unsigned-integer',
                        'floating-point'], 'Unexpected format.'
    return Path(audio_path), WavFileMetadata(sample_rate, bits, channels, encoding)


@assert_no_overwritten_files
@disk_cache
def get_audio_metadata(audio_path):
    """ Get metadata on `audio_path`.

    NOTE: Launching `subprocess` via `get_audio_metadata` is slow, try to avoid it.

    Args:
        audio_path (Path): WAV audio file to get metadata on.

    Returns:
        WavFileMetadata
    """
    assert isinstance(audio_path, Path), '`audio_path` must be a Path for caching.'
    # NOTE: `sox --i` behaves like `soxi`
    # Learn more about `soxi`: http://sox.sourceforge.net/soxi.html
    metadata = subprocess.run(['sox', '--i', audio_path], check=False, stdout=subprocess.PIPE)
    metadata = metadata.stdout.decode('utf-8').strip()
    return _parse_audio_metadata(metadata)[1]


def _cache_get_audio_metadata_helper(chunk):
    # NOTE: Glob may find unused files like:
    # `M-AILABS/ ... /judy_bieber/ozma_of_oz/wavs/._ozma_of_oz_10_f000095.wav`
    # Those errors are ignored along with any other output from `stderr`.
    command = ['sox', '--i'] + chunk
    metadatas = subprocess.run(command, stdout=subprocess.PIPE).stdout.decode('utf-8')
    metadatas = metadatas.strip().split('\n\n')
    metadatas = metadatas[:-1] if 'Total Duration' in metadatas[-1] else metadatas
    return [_parse_audio_metadata(metadata) for metadata in metadatas]


def cache_get_audio_metadata(paths):
    """ Cache all `paths` in `_get_audio_metadata.disk_cache`.

    NOTE: This method is more performant due to batching than calling `get_audio_metadata` the same
    number of times for the same paths.

    Args:
        paths (iterable): List of `Path`s to cache.
    """
    function = get_audio_metadata.__wrapped__.__wrapped__
    paths = sorted(list(paths))
    paths = [
        p for p in paths if make_arg_key(function, Path(p)) not in get_audio_metadata.disk_cache
    ]
    if len(paths) == 0:
        return

    logger.info('Caching audio metadata for %d audio files.', len(paths))

    # NOTE: It's difficult to determine the bash maximum argument length, learn more:
    # https://unix.stackexchange.com/questions/45143/what-is-a-canonical-way-to-find-the-actual-maximum-argument-list-length
    # https://stackoverflow.com/questions/19354870/bash-command-line-and-input-limit
    # NOTE: 1024 was choosen empirically to be less then the bash maximum argument length on most
    # systems.
    chunks = list(get_chunks(paths, 1024))
    progress_bar = tqdm(total=len(paths))
    with Pool() as pool:
        for result in pool.imap_unordered(_cache_get_audio_metadata_helper, chunks):
            for audio_path, metadata in result:
                get_audio_metadata.disk_cache.set(make_arg_key(function, audio_path), metadata)
                progress_bar.update()

    get_audio_metadata.disk_cache.save()


@configurable
def normalize_audio(audio_path,
                    sample_rate=HParam(),
                    bits=HParam(),
                    channels=HParam(),
                    encoding=HParam()):
    """ Normalize audio on disk with the SoX library.

    Args:
        audio_path (Path or str): Path to a audio file.
        sample_rate (int or None, optional): Change the audio sampling rate
            (i.e. resample the audio).
        bits (int, optional): Change the bit-depth of the audio file.
        channels (bool, optional): The channels effect should be invoked in order to change
            the number of channels in an audio signal to `channels`.
        encoding (str, optional): Changing the audio encoding type: ['signed-integer',
          'unsigned-integer', 'floating-point'].

    Returns:
        (str): Filename of the processed file.
    """
    audio_path = Path(audio_path)

    # TODO: Consider adding support for `--show-progress`.
    metadata = get_audio_metadata(audio_path)

    _channels = None if metadata.channels == channels else channels
    _sample_rate = None if metadata.sample_rate == sample_rate else sample_rate
    _bits = None if metadata.bits == bits else bits
    _encoding = None if metadata.encoding == encoding else encoding

    stem = audio_path.stem
    stem = stem if _sample_rate is None else ('rate(%s,%d)' % (stem, _sample_rate))
    stem = stem if _bits is None else ('bits(%s,%d)' % (stem, _bits))
    stem = stem if _channels is None else ('channels(%s,%d)' % (stem, _channels))
    stem = stem if _encoding is None else ('encoding(%s,%s)' % (stem, _encoding))

    if stem == audio_path.stem:
        return audio_path

    parent = audio_path.parent
    parent = parent if parent.name == TTS_DISK_CACHE_NAME else parent / TTS_DISK_CACHE_NAME
    destination = parent / '{}{}'.format(stem, audio_path.suffix)

    if destination.is_file():
        return destination

    destination.parent.mkdir(exist_ok=True)

    # `-G`: Automatically invoke the gain effect to guard against clipping.
    commands = ['sox', '-G', audio_path, destination]
    if _encoding is not None:
        commands[-1:1] = ['-e', _encoding]
    if _bits is not None:
        commands[-1:1] = ['-b', str(_bits)]
    commands = commands if _sample_rate is None else commands + ['rate', str(_sample_rate)]
    commands = commands if _channels is None else commands + ['channels', str(_channels)]
    subprocess.run(commands, check=True)

    return destination


class MelSpectrogramLoss(torch.nn.Module):
    """ Compute loss comparing two Mel Spectrograms.

    TODO: There are a number of interesting loss functions for spectrograms defined, here:
    - Power loss from the Parallel WaveNet paper https://arxiv.org/pdf/1711.10433.pdf
    - "Fast Spectrogram Inversion using Multi-head Convolutional Neural Networks"
      https://arxiv.org/pdf/1808.06719.pdf

    Args:
        fft_length (int): The window size used by the fourier transform.
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        window (torch.FloatTensor [window_length]): Window function to be applied to each frame.
    """

    def __init__(self, fft_length, frame_hop, window):
        super(MelSpectrogramLoss, self).__init__()

        self.fft_length = fft_length
        self.frame_hop = frame_hop
        self.window = window

        self.signal_to_mel_spectrogram = SignalToMelSpectrogram(
            fft_length=fft_length, frame_hop=frame_hop, window=window)

    def forward(self, predicted, target):
        """

        Args:
            predicted (torch.FloatTensor [batch_size, signal_length])
            target (torch.FloatTensor [batch_size, signal_length])

        Returns:
            torch.FloatTensor: Spectral convergence loss value.
            torch.FloatTensor: Log STFT magnitude loss value.
        """
        predicted_mel_spectrogram = self.signal_to_mel_spectrogram(predicted)
        target_mel_spectrogram = self.signal_to_mel_spectrogram(target)

        # [batch_size, num_frames, frame_channels]
        predicted_mel_spectrogram = predicted_mel_spectrogram.view(
            -1, *predicted_mel_spectrogram.shape[-2:])

        target_mel_spectrogram = target_mel_spectrogram.view(-1, *target_mel_spectrogram.shape[-2:])

        # NOTE: Frobenius matrix norm is not supported in CUDA. For a matrix, the Frobenius matrix
        # norm is equal to the L2 norm.
        spectral_convergence_loss = torch.norm(
            target_mel_spectrogram - predicted_mel_spectrogram, dim=(1, 2), p=2)
        spectral_convergence_loss = spectral_convergence_loss / (
            torch.norm(target_mel_spectrogram, dim=(1, 2), p=2) + 1e-8)
        spectral_convergence_loss = spectral_convergence_loss.mean()

        log_mel_spectrogram_magnitude_loss = torch.nn.functional.l1_loss(
            torch.log(predicted_mel_spectrogram), torch.log(target_mel_spectrogram))

        return spectral_convergence_loss, log_mel_spectrogram_magnitude_loss


class MultiResolutionMelSpectrogramLoss(torch.nn.Module):
    """ Multi resolution STFT loss criterion.

    Similar to the one described here: https://arxiv.org/abs/1910.11480

    Args:
        get_stft_losses (lambda: list of STFTLoss)
    """

    def __init__(self,
                 get_stft_losses=lambda: [
                     MelSpectrogramLoss(512, 50, torch.hann_window(300)),
                     MelSpectrogramLoss(1024, 150, torch.hann_window(600)),
                     MelSpectrogramLoss(2048, 300, torch.hann_window(1200))
                 ]):
        super(MultiResolutionMelSpectrogramLoss, self).__init__()

        self.stft_losses = torch.nn.ModuleList(get_stft_losses())

    def forward(self, predicted, target):
        """ Calculate forward propagation.

        TODO: Support masking the loss by aligning a signal mask with a spectrogram.

        Args:
            predicted (torch.FloatTensor [batch_size, signal_length])
            target (torch.FloatTensor [batch_size, signal_length])

        Returns:
            torch.FloatTensor: Multi resolution spectral convergence loss value.
            torch.FloatTensor: Multi resolution log STFT magnitude loss value.
        """
        spectral_convergence_loss_total = torch.tensor(0.0, device=predicted.device)
        log_mel_spectrogram_magnitude_loss_total = torch.tensor(0.0, device=predicted.device)

        for module in self.stft_losses:
            spectral_convergence_loss, log_mel_spectrogram_magnitude_loss = module(
                predicted, target)
            spectral_convergence_loss_total += spectral_convergence_loss
            log_mel_spectrogram_magnitude_loss_total += log_mel_spectrogram_magnitude_loss

        spectral_convergence_loss_total /= len(self.stft_losses)
        log_mel_spectrogram_magnitude_loss_total /= len(self.stft_losses)

        return spectral_convergence_loss_total, log_mel_spectrogram_magnitude_loss_total
