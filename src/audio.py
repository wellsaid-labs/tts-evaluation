from collections import namedtuple
from functools import lru_cache
from pathlib import Path

import logging
import math
import os
import struct
import subprocess

from hparams import configurable
from hparams import HParam
from third_party import LazyLoader
from third_party.iso226 import iso226_spl_itpl
from tqdm import tqdm

import numpy as np
import torch

# NOTE: `LazyLoader` allows for this repository to run without these dependancies. Also,
# it side-steps this issue: https://github.com/librosa/librosa/issues/924.
librosa = LazyLoader('librosa', globals(), 'librosa')
scipy_wavfile = LazyLoader('scipy_wavfile', globals(), 'scipy.io.wavfile')
scipy_signal = LazyLoader('scipy_signal', globals(), 'scipy.signal')

from src.environment import IS_TESTING_ENVIRONMENT
from src.environment import TTS_DISK_CACHE_NAME
from src.utils import assert_no_overwritten_files
from src.utils import disk_cache
from src.utils import get_chunks
from src.utils import get_file_metadata
from src.utils import make_arg_key
from src.utils import Pool

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
    """ Convert a `int32` or `int16` PCM signal representation to a `float32` PCM representation.

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

    assert signal.ndim == 1

    if signal.dtype != np.float32:
        assert signal.dtype in (np.int32, np.int16), '`%s` `dtype` is not supported' % signal.dtype
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
    if not overwrite and (isinstance(filename, str) or
                          isinstance(filename, Path)) and Path(filename).exists():
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
        fft_length (int): The size of the FFT to apply.
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
    upper_hertz = min(upper_hertz, float(sample_rate) / 2)
    return librosa.filters.mel(
        sample_rate,
        fft_length,
        n_mels=num_mel_bins,
        fmin=lower_hertz,
        fmax=upper_hertz,
        norm=None,
        htk=True)


# Learn more with regards to `full_scale_sine_wave`, `full_scale_square_wave` and
# `REFERENCE_FREQUENCY`:
# - Wikipedia describing decibels relative to full scale (dBFS) -
#   https://en.wikipedia.org/wiki/DBFS
# - Definition of decibels relative to full scale (dBFS) -
#   http://www.digitizationguidelines.gov/term.php?term=decibelsrelativetofullscale
#   http://www.prismsound.com/define.php?term=Full-scale_amplitude
#   https://github.com/dechamps/audiotools (Opinionated)
# - Question on generating a 0 dBFS full scale sine-wave -
#   https://sound.stackexchange.com/questions/44910/how-do-i-generate-a-0-dbfs-sine-wave-in-sox-output-clipped-warning

REFERENCE_FREQUENCY = 997
REFERENCE_SAMPLE_RATE = 48000


def full_scale_sine_wave(sample_rate=REFERENCE_SAMPLE_RATE, frequency=REFERENCE_FREQUENCY):
    """ Full-scale sine wave is used to define the maximum peak level for a dBFS unit.

    Learn more:
    https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave/34442729
    https://github.com/makermovement/3.5-Sensor2Phone/blob/master/generate_any_audio.py

    Args:
        sample_rate (int)
        frequency (int)

    Returns:
        np.ndarray: Array of length `sample_rate`
    """
    x = np.arange(sample_rate)
    return np.sin(2 * np.pi * frequency * (x / sample_rate)).astype(np.float32)


def full_scale_square_wave(sample_rate=REFERENCE_SAMPLE_RATE, frequency=REFERENCE_FREQUENCY):
    """ Full-scale square wave is also used to define the maximum peak level for a dBFS unit.

    Args:
        sample_rate (int)
        frequency (int)

    Returns:
        np.ndarray: Array of length `sample_rate`
    """
    x = np.arange(sample_rate)
    return scipy_signal.square(2 * np.pi * frequency * (x / sample_rate)).astype(np.float32)


def _k_weighting(frequencies, fs):
    # pre-filter 1
    f0 = 1681.9744509555319
    G = 3.99984385397
    Q = 0.7071752369554193
    K = np.tan(np.pi * f0 / fs)
    Vh = np.power(10.0, G / 20.0)
    Vb = np.power(Vh, 0.499666774155)
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K - Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_

    h1 = scipy_signal.freqz([b0, b1, b2], [a0, a1, a2], worN=frequencies, fs=fs)[1]
    h1 = 20 * np.log10(abs(h1))

    # pre-filter 2
    f0 = 38.13547087613982
    Q = 0.5003270373253953
    K = np.tan(np.pi * f0 / fs)
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0

    h2 = scipy_signal.freqz([b0, b1, b2], [a0, a1, a2], worN=frequencies, fs=fs)[1]
    h2 = 20 * np.log10(abs(h2))

    return h1 + h2


def k_weighting(frequencies, sample_rate, offset=None):
    """ K-Weighting as specified in EBU R-128 / ITU BS.1770-4.

    Learn more:
    - Original implementation of sample rate independent EBU R128 / ITU-R BS.1770 -
      https://github.com/BrechtDeMan/loudness.py
    - Python loudness library with implementation of EBU R128 / ITU-R BS.1770 -
      https://www.christiansteinmetz.com/projects-blog/pyloudnorm
      https://github.com/csteinmetz1/pyloudnorm
    - Original document describing ITU-R BS.1770 -
      https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-4-201510-I!!PDF-E.pdf
    - MATLAB implementation of K-Weighting -
      https://www.mathworks.com/help/audio/ref/weightingfilter-system-object.html
    - Wikipedia describing different weightings -
      https://en.wikipedia.org/wiki/Weighting_filter
    - Google TTS on LUFS / EBU R128 / ITU-R BS.1770 -
      https://developers.google.com/assistant/tools/audio-loudness

    This implementation is largely borrowed from https://github.com/BrechtDeMan/loudness.py.

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.
        sample_rate (int)
        offset (optional, float)

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    # SOURCE (ITU-R BS.1770):
    # The constant −0.691 in equation (2) cancels out the K-weighting gain for 997 Hz.
    offset = -_k_weighting(np.array([REFERENCE_FREQUENCY]),
                           sample_rate) if offset is None else offset
    return _k_weighting(frequencies, sample_rate) + offset


def a_weighting(frequencies):
    """ Wrapper around `librosa.core.A_weighting`.

    Learn more:
    - Wikipedia describing A-weighting -
      https://en.wikipedia.org/wiki/A-weighting

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    return librosa.core.A_weighting(
        frequencies, min_db=None) - librosa.core.A_weighting(
            np.array([REFERENCE_FREQUENCY]), min_db=None)


def iso226_weighting(frequencies):
    """ Get the ISO226 weights for `frequencies`.

    Learn more:
    - Wikipedia describing the purposes of weighting -
      https://en.wikipedia.org/wiki/Equal-loudness_contour
    - Wikipedia graphic compared ISO 226 with A-Weighting -
      https://commons.wikimedia.org/wiki/File:A-weighting,_ISO_226_and_ITU-R_468.svg

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    interpolator = iso226_spl_itpl(hfe=True)
    # SOURCE: https://en.wikipedia.org/wiki/A-weighting
    # The offsets ensure the normalisation to 0 dB at 1000 Hz.
    return -interpolator(frequencies) + interpolator(np.array([REFERENCE_FREQUENCY]))


def identity_weighting(frequencies):
    """ Get identity weighting, it doesn't change the frequency weighting.

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    return np.zeros_like(frequencies)


@lru_cache()
def get_signal_to_db_mel_spectrogram(*args, **kwargs):
    """ Get cached `SignalTodBMelSpectrogram` module. """
    return SignalTodBMelSpectrogram(*args, **kwargs)


class SignalTodBMelSpectrogram(torch.nn.Module):
    """ Compute a dB-mel-scaled spectrogram from signal.

    The spectrogram is an important representation of audio data because human hearing is based on a
    kind of real-time spectrogram encoded by the cochlea of the inner ear.

    TODO: Instead of using the decibel scale, it likely makes more sense to use the sone scale. The
    reason is that the sone scale is intended to be linear to perceived loudness; therefore, it'd
    make more sense to compute a linear loss like L1 or L2 on a linear scale. The decibel scale
    has still a logarithmic relationship with perceived loudness because every additional 10 dB
    the perceived loudness doubles.
    Learn more:
    https://community.sw.siemens.com/s/article/sound-quality-metrics-loudness-and-sones
    https://en.wikipedia.org/wiki/Sone
    http://www.physics.mcgill.ca/~guymoore/ph224/notes/lecture12.pdf

    Learn more:
    - Loudness Spectrogram:
      https://www.dsprelated.com/freebooks/sasp/Loudness_Spectrogram_Examples.html
    - Loudness Spectrogram: https://ccrma.stanford.edu/~jos/sasp/Loudness_Spectrogram.html
    - MFCC Preprocessing Steps: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
    - MFCC Preprocessing Steps:
      https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
    - Perceptual Loss: https://github.com/magenta/ddsp/issues/12
    - Compute Loudness: https://github.com/librosa/librosa/issues/463
    - Compute Loudness: https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py#L171
    - Ampltidue To dB:
      https://librosa.github.io/librosa/generated/librosa.core.amplitude_to_db.html
    - Into To Speech Science: http://www.cas.usf.edu/~frisch/SPA3011_L07.html
    - Frequency Scales:
    https://www.researchgate.net/figure/Comparison-between-Mel-Bark-ERB-and-linear-frequency-scales-Since-the-units-of-the_fig4_283044643
    - Frequency Scales: https://www.vocal.com/noise-reduction/perceptual-noise-reduction/
    - A just-noticeable difference (JND) in amplitude level is on the order of a quarter dB.
      (https://www.dsprelated.com/freebooks/mdft/Decibels.html)
    - https://librosa.github.io/librosa/generated/librosa.core.amplitude_to_db.html
    - The human perception of the intensity of sound and light approximates the logarithm of
      intensity rather than a linear relationship (Weber–Fechner law), making the dB scale a useful
      measure. (https://en.wikipedia.org/wiki/Decibel)
    - http://msp.ucsd.edu/techniques/v0.08/book-html/node6.html
    - Spectrogram Basics: https://www.dsprelated.com/freebooks/sasp/Classic_Spectrograms.html

    Args:
        fft_length (int): See `n_fft` here: https://pytorch.org/docs/stable/torch.html#torch.stft
        frame_hop (int): See `hop_length` here:
            https://pytorch.org/docs/stable/torch.html#torch.stft
        sample_rate (int): The sample rate of the audio.
        num_mel_bins (int): See `src.audio._mel_filters`. The mel scale is applied to mimic the the
            non-linear human ear perception of sound, by being more discriminative at lower
            frequencies and less discriminative at higher frequencies.
        window (torch.FloatTensor): See `window` here:
            https://pytorch.org/docs/stable/torch.html#torch.stft
        min_decibel (float): The minimum decible to limit the lower range. Since decibel's is on
            the log scale, the lower range can extend to −∞ as the amplitude gets closer to 0.
        get_weighting (callable): Given a `np.ndarray` of frequencies this returns a weighting in
            decibels. Weighting in an effort to account for the relative loudness perceived by the
            human ear, as the ear is less sensitive to low audio frequencies.
        eps (float): The minimum amplitude to `log` avoiding the discontinuity at `log(0)`. This
            is similar to `min_decibel` but it operates on the amplitude scale.
        **kwargs: Additional arguments passed to `_mel_filters`.
    """

    @configurable
    def __init__(self,
                 fft_length=HParam(),
                 frame_hop=HParam(),
                 sample_rate=HParam(),
                 num_mel_bins=HParam(),
                 window=HParam(),
                 min_decibel=HParam(),
                 get_weighting=HParam(),
                 eps=1e-10,
                 **kwargs):
        super().__init__()

        self.register_buffer('window', window)
        self.register_buffer('min_decibel', torch.tensor(min_decibel).float())
        self.register_buffer('eps', torch.tensor(eps).float())
        self.fft_length = fft_length
        self.frame_hop = frame_hop
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins

        mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=self.fft_length, **kwargs)
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.fft_length)
        weighting = torch.tensor(get_weighting(frequencies)).float().view(-1, 1)
        weighting = db_to_power(weighting)
        self.register_buffer('mel_basis', torch.tensor(mel_basis).float())
        self.register_buffer('weighting', weighting)

    def forward(self, signal, intermediate=False, aligned=False):
        """ Compute a dB-mel-scaled spectrogram from signal.

        Args:
            signal (torch.FloatTensor [batch_size, signal_length])
            intermediate (bool, optional): If `True`, along with a `db_mel_spectrogram`, this
                returns a `db_spectrogram` and `spectrogram`.
            aligned (bool, optional): If `True` the returned spectrogram is aligned to the signal
                such that `signal.shape[1] / self.frame_hop == db_mel_spectrogram.shape[1]`

        Returns:
            db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins]): A
                spectrogram with the mel scale for frequency, decibel scale for power, and a regular
                time scale.
            db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1]): This
                is only  returned iff `intermediate=True`.
            spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1]): This is
                only returned iff `intermediate=True`.
        """
        assert signal.dtype == torch.float32, 'Invalid argument.'

        has_batch_dim = signal.dim() == 2
        signal = signal.view(-1, signal.shape[-1])

        if aligned:
            assert signal.shape[1] % self.frame_hop == 0, (
                'The signal must be a multiple of `frame_hop` to be aligned to the spectrogram.')
            assert (self.fft_length - self.frame_hop) % 2 == 0, (
                '`self.fft_length - self.frame_hop` must be even for the signal '
                'to be aligned to the spectrogram.')
            # NOTE: Check ``notebooks/Signal_to_Spectrogram_Consistency.ipynb`` for the correctness
            # of this padding algorithm.
            # NOTE: Center the signal such that the resulting spectrogram and audio are aligned.
            # Learn more here:
            # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
            # NOTE: The number of spectrogram frames generated is:
            # `(signal.shape[1] - frame_size + frame_hop) // frame_hop`
            padding = (self.fft_length - self.frame_hop) // 2
            padded_signal = torch.nn.functional.pad(
                signal, (padding, padding), mode='constant', value=0)
        else:
            padded_signal = signal

        spectrogram = torch.stft(
            padded_signal,
            n_fft=self.fft_length,
            hop_length=self.frame_hop,
            win_length=self.window.shape[0],
            window=self.window,
            center=False)

        if aligned:
            assert spectrogram.shape[-2] * self.frame_hop == signal.shape[1], 'Invariant failure.'

        # NOTE: `torch.norm` is too slow to use in this case
        # https://github.com/pytorch/pytorch/issues/34279
        # spectrogram [batch_size, fft_length // 2 + 1, num_frames]
        power_spectrogram = spectrogram.pow(2).sum(-1)

        # NOTE: Perceived loudness (for example, the sone scale) corresponds fairly well to the dB
        # scale, suggesting that human perception of loudness is roughly logarithmic with
        # intensity; therefore, we convert our "ampltitude spectrogram" to the dB scale.
        # NOTE: A "weighting" can we added to the dB scale such as A-weighting to adjust it so
        # that it more representative of the human perception of loudness.
        # NOTE: A multiplication is equal to an addition in the log space / dB space.
        # power_spectrogram [batch_size, fft_length // 2 + 1, num_frames]
        weighted_power_spectrogram = power_spectrogram * self.weighting
        # power_mel_spectrogram [batch_size, num_mel_bins, num_frames]
        power_mel_spectrogram = torch.matmul(self.mel_basis, weighted_power_spectrogram)
        db_mel_spectrogram = power_to_db(power_mel_spectrogram, eps=self.eps)
        db_mel_spectrogram = torch.max(self.min_decibel, db_mel_spectrogram).transpose(-2, -1)
        db_mel_spectrogram = db_mel_spectrogram if has_batch_dim else db_mel_spectrogram.squeeze(0)

        if intermediate:
            # TODO: Simplify the `tranpose` and `squeeze`s.
            db_spectrogram = power_to_db(weighted_power_spectrogram).transpose(-2, -1)
            db_spectrogram = torch.max(self.min_decibel, db_spectrogram)
            spectrogram = torch.sqrt(torch.max(power_spectrogram, self.eps)).transpose(-2, -1)
            db_spectrogram = db_spectrogram if has_batch_dim else db_spectrogram.squeeze(0)
            spectrogram = spectrogram if has_batch_dim else spectrogram.squeeze(0)
            return db_mel_spectrogram, db_spectrogram, spectrogram
        else:
            return db_mel_spectrogram


def rms_from_signal(signal):
    """ Compute the root mean square from a signal.

    Learn more:
    - Implementations of RMS:
      https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#rms
      https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/_common.py#L116
    - Wikipedia on RMS:
      https://en.wikipedia.org/wiki/Root_mean_square

    Args:
        signal (np.ndarray [signal_length])

    Returns:
        np.ndarray [1]
    """
    return np.sqrt(np.mean(np.abs(signal)**2))


@configurable
def framed_rms_from_signal(signal, frame_length=HParam(), hop_length=HParam()):
    """ Compute the framed root mean square from a signal.

    Args:
        signal (np.ndarray [signal_length])
        frame_length (int)
        hop_length (int)

    Returns:
        np.ndarray [1]
    """
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(np.abs(frames**2), axis=0))


@configurable
def framed_rms_from_power_spectrogram(power_spectrogram, window=HParam()):
    """ Compute the root mean square from a spectrogram.

    Learn more:
    - Implementations of RMS:
      https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#rms
    - Opinionated discussion between LUFS and RMS:
      https://www.gearslutz.com/board/mastering-forum/1142602-lufs-really-better-than-rms-measure-loudness.html
      Also, see `test_loudness` in `test_audio.py` that replicates LUFS via RMS.
    - Compare LUFS to Decibels:
      https://backtracks.fm/blog/whats-the-difference-between-decibels-and-lufs/

    Args:
        power_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
        window (torch.FloatTensor [window_size])

    Returns:
        (torch.FloatTensor [batch_size, num_frames])
    """
    has_batch_dim = power_spectrogram.dim() == 3
    power_spectrogram = power_spectrogram.view(-1, *power_spectrogram.shape[-2:])

    # Learn more:
    # https://community.sw.siemens.com/s/article/window-correction-factors
    # https://www.mathworks.com/matlabcentral/answers/372516-calculate-windowing-correction-factor
    window_correction_factor = (
        torch.ones(*window.shape).pow(2).mean().sqrt() / window.pow(2).mean().sqrt())

    # TODO: This adjustment might have an unintended affect on the a mel spectrogram.
    # TODO: This adjustment might be related to repairing constant-overlap-add, see here:
    # https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_Decomposition.html
    # Adjust the DC and half sample rate component
    power_spectrogram[:, :, 0] *= 0.5
    if window.shape[0] % 2 == 0:
        power_spectrogram[:, :, -1] *= 0.5

    # Calculate power
    power = 2 * power_spectrogram.sum(dim=-1) / window.shape[0]**2
    frame_rms = power.sqrt() * window_correction_factor
    return frame_rms if has_batch_dim else frame_rms.squeeze(0)


def power_to_db(tensor, eps=1e-10):
    """ Convert power (https://www.dsprelated.com/freebooks/mdft/Decibels.html) units to decibel
    units.

    Args:
        tensor (torch.FloatTensor)
        eps (float or torch.FloatTensor): The minimum amplitude to `log` avoiding the discontinuity
            at `log(0)`.

    Returns:
        (torch.FloatTensor)
    """
    eps = eps if torch.is_tensor(eps) else torch.tensor(eps, device=tensor.device)
    return 10.0 * torch.log10(torch.max(eps, tensor))


def amplitude_to_db(tensor, **kwargs):
    """ Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to decibel units.

    Args:
        tensor (torch.FloatTensor)
        **kwargs: Other keyword arguments passed to `power_to_db`.

    Returns:
        (torch.FloatTensor)
    """
    return power_to_db(tensor, **kwargs) * 2


def amplitude_to_power(tensor):
    """ Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to power units.

    Args:
        tensor (torch.FloatTensor)

    Returns:
        (torch.FloatTensor)
    """
    return tensor**2


def power_to_amplitude(tensor):
    """ Convert power units to amplitude units.

    Args:
        tensor (torch.FloatTensor)

    Returns:
        (torch.FloatTensor)
    """
    return tensor.sqrt()


def db_to_power(tensor):
    """ Convert decibel units to power units.

    Args:
        tensor (torch.FloatTensor)

    Returns:
        (torch.FloatTensor)
    """
    return 10**(tensor / 10.0)


def db_to_amplitude(tensor):
    """ Convert decibel units to amplitude units.

    Args:
        tensor (torch.FloatTensor)

    Returns:
        (torch.FloatTensor)
    """
    return db_to_power(tensor / 2)


@configurable
def pad_remainder(signal, multiple=HParam(), mode=HParam(), constant_values=HParam(), **kwargs):
    """ Pad signal such that `signal.shape[0] % multiple == 0`.

    Args:
        signal (np.array [signal_length]): One-dimensional signal to pad.
        multiple (int): The returned signal shape is divisible by `multiple`.
        **kwargs: Key word arguments passed to `np.pad`.

    Returns:
        np.array [padded_signal_length]
    """
    assert isinstance(signal, np.ndarray)
    remainder = signal.shape[0] % multiple
    remainder = multiple - remainder if remainder != 0 else remainder
    padding = (math.ceil(remainder / 2), math.floor(remainder / 2))
    padded_signal = np.pad(signal, padding, mode=mode, constant_values=constant_values, **kwargs)
    assert padded_signal.shape[0] % multiple == 0
    return padded_signal


@configurable
def _db_mel_spectrogram_to_spectrogram(db_mel_spectrogram,
                                       sample_rate,
                                       fft_length,
                                       get_weighting=HParam(),
                                       **kwargs):
    """ Transform dB mel spectrogram to spectrogram (lossy).

    Args:
        db_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate (int): Sample rate of the `db_mel_spectrogram`.
        fft_length (int): The size of the FFT to apply.
        **kwargs: Additional arguments passed to `_mel_filters`.

    Returns:
        (np.ndarray [frames, fft_length // 2 + 1]): Spectrogram.
    """
    num_mel_bins = db_mel_spectrogram.shape[1]
    mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=fft_length, **kwargs)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_length)
    weighting = get_weighting(frequencies)
    weighting = db_to_power(weighting)
    inverse_mel_basis = np.linalg.pinv(mel_basis)  # NOTE: Approximate inverse matrix of `mel_basis`
    power_mel_spectrogram = db_to_power(db_mel_spectrogram)
    power_spectrogram = np.dot(inverse_mel_basis, power_mel_spectrogram.transpose()).transpose()
    power_spectrogram = np.maximum(0.0, power_spectrogram)
    return np.sqrt(power_spectrogram / weighting)


@configurable
def griffin_lim(db_mel_spectrogram,
                sample_rate=HParam(),
                frame_size=HParam(),
                frame_hop=HParam(),
                fft_length=HParam(),
                window=HParam(),
                power=HParam(),
                iterations=HParam(),
                use_tqdm=False):
    """ Transform dB mel spectrogram to waveform with the Griffin-Lim algorithm.

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
        db_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate (int): Sample rate of the spectrogram and the resulting wav file.
        frame_size (int): The frame size in samples. (e.g. 50ms * 24,000 / 1000 == 1200)
        frame_hop (int): The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        fft_length (int): The size of the FFT to apply.
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
        assert isinstance(db_mel_spectrogram, np.ndarray)
        spectrogram = _db_mel_spectrogram_to_spectrogram(
            db_mel_spectrogram=db_mel_spectrogram, sample_rate=sample_rate, fft_length=fft_length)
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
    except Exception:
        logger.exception('Griffin-lim encountered an issue and was unable to render audio.')
        # NOTE: Return no audio for valid inputs that fail due to an overflow error or a small
        # spectrogram.
        return np.array([], dtype=np.float32)


WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003


@configurable
def build_wav_header(num_frames,
                     frame_rate=HParam(),
                     wav_format=HParam(),
                     num_channels=HParam(),
                     sample_width=HParam()):
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
    header_length = 36 if wav_format == WAVE_FORMAT_PCM else 50
    data_length = num_frames * num_channels * sample_width
    file_size = header_length + data_length + 8
    bytes_per_second = num_channels * frame_rate * sample_width
    block_align = num_channels * sample_width
    bit_depth = sample_width * 8

    header = b'RIFF'  # RIFF identifier
    header += struct.pack('<I', header_length + data_length)  # RIFF chunk length
    header += b'WAVE'  # RIFF type
    header += b'fmt '  # Format chunk identifier

    fmt_chunk_data = struct.pack('<HHIIHH', wav_format, num_channels, frame_rate, bytes_per_second,
                                 block_align, bit_depth)
    if wav_format != WAVE_FORMAT_PCM:
        fmt_chunk_data += b'\x00\x00'  # Add `cbSize` field for non-PCM files

    header += struct.pack('<I', len(fmt_chunk_data))  # Format chunk length
    header += fmt_chunk_data

    if wav_format != WAVE_FORMAT_PCM:  # Add fact chunk (non-PCM files)
        header += b'fact'
        header += struct.pack('<II', 4, num_frames)

    if ((len(header) - 4 - 4) + (4 + 4 + data_length)) > 0xFFFFFFFF:
        raise ValueError('Data exceeds wave file size limit.')

    header += b'data'  # Data chunk identifier
    header += struct.pack('<I', data_length)  # Data chunk length
    return header, file_size


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
    with Pool(1 if IS_TESTING_ENVIRONMENT else os.cpu_count()) as pool:
        for result in pool.imap_unordered(_cache_get_audio_metadata_helper, chunks):
            for audio_path, metadata in result:
                get_audio_metadata.disk_cache.set(make_arg_key(function, audio_path), metadata)
                progress_bar.update()

        for i, result in tqdm(enumerate(pool.imap(get_file_metadata, paths)), total=len(paths)):
            get_audio_metadata.__wrapped__.assert_no_overwritten_files_cache.set(paths[i], result)

    get_audio_metadata.__wrapped__.assert_no_overwritten_files_cache.save()
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
