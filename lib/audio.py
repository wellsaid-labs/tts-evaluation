import dataclasses
import enum
import itertools
import logging
import math
import multiprocessing.pool
import os
import subprocess
import typing
from functools import lru_cache, partial
from pathlib import Path

import numpy as np
import torch
import torch.nn
import torch.nn.functional
from hparams import HParam, configurable
from third_party import LazyLoader
from third_party.iso226 import iso226_spl_itpl
from tqdm import tqdm

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    import librosa.core
    import librosa.util
    import pyloudnorm
    from scipy import signal as scipy_signal
    from scipy.io import wavfile
else:
    pyloudnorm = LazyLoader("pyloudnorm", globals(), "pyloudnorm")
    librosa = LazyLoader("librosa", globals(), "librosa")
    scipy_signal = LazyLoader("scipy_signal", globals(), "scipy.signal")
    wavfile = LazyLoader("wavfile", globals(), "scipy.io.wavfile")


logger = logging.getLogger(__name__)


def milli_to_sec(milli: float) -> float:
    """Covert milliseconds to seconds."""
    return milli / 1000


def sec_to_milli(sec: float) -> float:
    """Covert seconds to milliseconds."""
    return sec * 1000


def milli_to_sample(milli: float, sample_rate: int) -> int:
    """Covert milliseconds to samples."""
    return int(round(milli_to_sec(milli) * sample_rate))


def sample_to_milli(sample: int, sample_rate: int) -> float:
    """Covert samples to milliseconds."""
    return sec_to_milli(float(sample) / sample_rate)


def sample_to_sec(sample: int, sample_rate: int) -> float:
    """Covert samples to seconds."""
    return float(sample) / sample_rate


def sec_to_sample(sec: float, sample_rate: int) -> int:
    """Covert seconds to samples."""
    return int(round(sec * sample_rate))


class AudioEncoding(enum.Enum):
    MPEG: typing.Final = "MPEG audio (layer I, II or III)"
    PCM_INT_8_BIT: typing.Final = "8-bit Signed Integer PCM"
    PCM_INT_16_BIT: typing.Final = "16-bit Signed Integer PCM"
    PCM_INT_24_BIT: typing.Final = "24-bit Signed Integer PCM"
    PCM_INT_32_BIT: typing.Final = "32-bit Signed Integer PCM"
    PCM_FLOAT_32_BIT: typing.Final = "32-bit Floating Point PCM"


class AudioDataType(enum.Enum):
    SIGNED_INTEGER: typing.Final = "signed-integer"
    UNSIGNED_INTEGER: typing.Final = "unsigned-integer"
    FLOATING_POINT: typing.Final = "floating-point"


@dataclasses.dataclass(frozen=True)
class AudioFormat:
    """
    TODO: The `sample_rate` does not change in porportion to the number of channels; therefore, the
    `sample_rate` should technically be called the `frame_rate` because it measures the number of
    frames per second.

    Learn more: http://sox.sourceforge.net/soxi.html

    Args:
        sample_rate: The sample rate of the audio.
        num_channels: The number of audio channels in the audio file.
        encoding: The encoding of the audio file (e.g. "32-bit Floating Point PCM").
        bit_rate: The number of bits per second.
        precision: The estimated sample precision in bits.
    """

    sample_rate: int
    num_channels: int
    encoding: AudioEncoding
    bit_rate: str
    precision: str


@dataclasses.dataclass(frozen=True)
class AudioMetadata(AudioFormat):
    """
    Args:
        ...
        path: The audio file path.
        num_samples: The duration of the audio file in samples.
    """

    path: Path
    num_samples: int

    @property
    def length(self):
        return sample_to_sec(self.num_samples, self.sample_rate)


def _parse_audio_metadata(metadata: str) -> AudioMetadata:
    """Parse audio metadata returned by `sox --i`.

    NOTE: This parses the output of `sox --i` instead individual requests like `sox --i -r` and
    `sox --i -b` to conserve on the number  of requests made via `subprocess`.

    TODO: Adapt `ffmpeg --i` for consistency.
    """
    lines = metadata.strip().split("\n")[:8]
    assert [s.split(":", maxsplit=1)[0].strip() for s in lines] == [
        "Input File",
        "Channels",
        "Sample Rate",
        "Precision",
        "Duration",
        "File Size",
        "Bit Rate",
        "Sample Encoding",
    ], f"This `metadata` is incompatible with `AudioMetadata`: {metadata}"
    splits = [s.split(":", maxsplit=1)[1].strip() for s in lines]
    assert splits[4].split()[3] == "samples"
    sample_rate = int(splits[2])
    num_samples = int(splits[4].split()[2])
    encoding = next((e for e in AudioEncoding if e.value == splits[7]), None)
    assert encoding is not None, f"Format '{splits[7]}' isn't supported."
    return AudioMetadata(
        path=Path(str(splits[0][1:-1])),
        sample_rate=sample_rate,
        num_channels=int(splits[1]),
        encoding=encoding,
        bit_rate=splits[6],
        precision=splits[3],
        num_samples=num_samples,
    )


def _get_audio_metadata_helper(chunk: typing.List[Path]) -> typing.List[AudioMetadata]:
    # NOTE: `-V1` ignores non-actionable warnings, SoX tends to spam the command line with strict
    # formating warnings like: "sox WARN wav: wave header missing extended part of fmt chunk".
    command = ["sox", "--i", "-V1"] + [str(p) for p in chunk]
    metadatas = subprocess.check_output(command).decode()
    splits = metadatas.strip().split("\n\n")
    splits = splits[:-1] if "Total Duration" in splits[-1] else splits
    return [_parse_audio_metadata(metadata) for metadata in splits]


def _get_audio_metadata(
    *paths: Path,
    max_arg_length: int = 2 ** 16,
    max_parallel: int = typing.cast(int, os.cpu_count()),
    add_tqdm: bool = False,
) -> typing.Iterator[AudioMetadata]:
    """
    NOTE: It's difficult to determine the bash maximum argument length, learn more:
    https://unix.stackexchange.com/questions/45143/what-is-a-canonical-way-to-find-the-actual-maximum-argument-list-length
    https://stackoverflow.com/questions/19354870/bash-command-line-and-input-limit
    """
    if len(set(paths)) != len(paths):
        logger.warning("`_get_audio_metadata` was called with duplicate paths.")

    if len(paths) == 0:
        return

    len_ = lambda p: len(str(p))
    splits = [float(max_arg_length)] * (sum([len_(p) for p in paths]) // max_arg_length)
    chunks = list(lib.utils.split(list(paths), splits, len_))
    if len(chunks) == 1:
        yield from _get_audio_metadata_helper(chunks[0])
    else:
        message = "Getting audio metadata for %d audio files in %d chunks..."
        logger.info(message, len(paths), len(chunks))
        with tqdm(total=len(paths), disable=not add_tqdm) as progress_bar:
            with multiprocessing.pool.ThreadPool(min(max_parallel, len(chunks))) as pool:
                for result in pool.imap(_get_audio_metadata_helper, chunks):
                    yield from result
                    progress_bar.update(len(result))


@typing.overload
def get_audio_metadata(paths: typing.List[Path], **kwargs) -> typing.List[AudioMetadata]:
    ...


@typing.overload
def get_audio_metadata(paths: Path, **kwargs) -> AudioMetadata:
    ...


def get_audio_metadata(paths, **kwargs):
    """Get the audio metadatas for a list of files."""
    is_list = isinstance(paths, list)
    metadatas = list(_get_audio_metadata(*tuple(paths if is_list else [paths]), **kwargs))
    return metadatas if is_list else metadatas[0]


def clip_waveform(waveform: np.ndarray) -> np.ndarray:
    """Clip audio at the maximum and minimum amplitude.

    NOTE: Clipping will cause distortion to the waveform, learn more:
    https://en.wikipedia.org/wiki/Clipping_(audio)
    """
    dtype = waveform.dtype
    is_floating = np.issubdtype(dtype, np.floating)
    min_: typing.Union[int, float] = -1.0 if is_floating else np.iinfo(dtype).min
    max_: typing.Union[int, float] = 1.0 if is_floating else np.iinfo(dtype).max
    num_clipped_samples = (waveform < min_).sum() + (waveform > max_).sum()
    if num_clipped_samples > 0:
        max_sample = np.max(np.absolute(waveform))
        logger.debug("%d samples clipped (%f max sample)", num_clipped_samples, max_sample)
    return np.clip(waveform, min_, max_)


def read_audio(
    path: Path, start: float = 0, length: float = math.inf, dtype=("f32le", "pcm_f32le", np.float32)
) -> np.ndarray:
    """Read an audio file slice into a `numpy` array.

    NOTE: Audio files with multiple channels will be mixed into a mono channel.
    NOTE: `ffmpeg` may load audio that's not clipped.
    NOTE: Learn more about efficiently selecting a slice of audio with `ffmpeg`:
    https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg

    TODO: Should we implement automatic gain control?
    https://en.wikipedia.org/wiki/Automatic_gain_control
    TODO: Should there be a look up table from numpy dtype to ffmpeg data types?

    Args:
        path: Path to load.
        start: The start of the audio segment, in seconds.
        length: The length of the audio segment, in seconds.
        dtype: The output `dtype` with the corresponding ffmpeg audio codec.
    """
    if length == 0:
        return np.array([], dtype=dtype[2])
    command = ["ffmpeg"]
    command += [] if start == 0 else ["-ss", start]
    command += [] if math.isinf(length) else ["-t", length]
    command += ["-i", path, "-f", dtype[0], "-acodec", dtype[1], "-ac", "1", "pipe:"]
    command = [str(c) for c in command]
    ndarray = np.frombuffer(subprocess.check_output(command, stderr=subprocess.DEVNULL), dtype[2])
    return clip_waveform(ndarray)


def read_wave_audio(
    metadata: AudioMetadata, start: float = 0, length: float = -1, memmap=False
) -> np.ndarray:
    """Fast read and seek WAVE file (for supported formats).

    Args:
        metadata: The audio file to load.
        start: The start of the audio segment, in seconds.
        length: The length of the audio segment, in seconds.
        memmap: Load audio into memory mapped storage.
    """
    assert metadata.path.suffix == ".wav"
    assert metadata.num_channels == 1
    # NOTE: Use `.value` because of this bug:
    # https://github.com/streamlit/streamlit/issues/2379
    lookup = {
        AudioEncoding.PCM_FLOAT_32_BIT.value: np.float32,
        AudioEncoding.PCM_INT_32_BIT.value: np.int32,
        AudioEncoding.PCM_INT_16_BIT.value: np.int16,
        AudioEncoding.PCM_INT_8_BIT.value: np.int8,
    }
    assert metadata.encoding.value in lookup, f"Metadata encoding '{metadata}' is not supported."
    dtype = lookup[metadata.encoding.value]
    bytes_per_sample = np.dtype(dtype).itemsize
    sec_to_sample_ = partial(sec_to_sample, sample_rate=metadata.sample_rate)
    header_size = os.path.getsize(metadata.path) - bytes_per_sample * metadata.num_samples
    start = lib.utils.round_(sec_to_sample_(start) * bytes_per_sample, bytes_per_sample)
    length = sec_to_sample_(length) if length > 0 else metadata.num_samples - sec_to_sample_(start)
    if memmap:
        ndarray = np.memmap(metadata.path, dtype=dtype, shape=(length,), offset=start + header_size)
    else:
        ndarray = np.fromfile(metadata.path, dtype=dtype, count=length, offset=start + header_size)
    return clip_waveform(ndarray)


@configurable
def write_audio(
    path: typing.Union[Path, typing.BinaryIO],
    audio: typing.Union[np.ndarray, torch.Tensor],
    sample_rate: int = HParam(),
    overwrite: bool = False,
):
    """Write a `np.float32` array in a Waveform Audio File Format (WAV) format at `path`.

    Args:
        path: Path to save audio at.
        audio: A 1-D or 2-D `np.float32` array.
        sample_rate: The audio sample rate (samples/sec).
        overwrite: If `True` and there is an existing file at `path`, it'll be overwritten.
    """
    if not overwrite and isinstance(path, Path) and path.exists():
        raise ValueError(f"File exists at {path}.")
    audio = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    assert audio.dtype == np.float32  # type: ignore
    if typing.cast(int, audio.size) > 0:
        error = f"Signal must be in range [-1, 1] rather than [{np.min(audio)}, {np.max(audio)}]."
        assert np.max(audio) <= 1.0 and np.min(audio) >= -1.0, error
    wavfile.write(path, sample_rate, audio)


def normalize_audio(
    source: Path,
    destination: Path,
    suffix: str,
    data_type: AudioDataType,
    bits: int,
    sample_rate: int,
    num_channels: int,
):
    """Create a new file at `destination` with an updated audio file format.

    NOTE: SoX tends to be better than ffmpeg, especially the resampler. SoX also includes
    features like guard that ffmpeg does not include. Learn more:
    https://trac.ffmpeg.org/wiki/FFmpeg%20and%20the%20SoX%20Resampler
    https://www.reddit.com/r/audiophile/comments/308023/highest_quality_resampling_and_bit_depth/
    https://transcoding.wordpress.com/2011/11/16/careful-with-audio-resampling-using-ffmpeg/
    https://forum.kodi.tv/showthread.php?tid=318686

    Args:
        ...
        suffix: The expected `suffix` of `destination`.
        ...
    """
    assert destination.suffix == suffix, f'The normalized file must be of type "{suffix}".'
    command = ["sox", "-G", source.absolute(), "-e", data_type.value, "-b", bits]
    command += [destination.absolute(), "rate", sample_rate, "channels", num_channels]
    subprocess.run([str(c) for c in command], check=True)


AudioFilter = typing.NewType("AudioFilter", str)


def format_ffmpeg_audio_filter(name: str, **kwargs: typing.Union[str, int, float]) -> AudioFilter:
    """Format ffmpeg audio filter flag "-af"."""
    return typing.cast(AudioFilter, f"{name}=" + ":".join([f"{k}={v}" for k, v in kwargs.items()]))


AudioFilters = typing.NewType("AudioFilters", str)


def format_ffmpeg_audio_filters(filters: typing.List[AudioFilter]) -> AudioFilters:
    """Format ffmpeg audio filter flag "-af"."""
    return typing.cast(AudioFilters, ",".join(filters))


def apply_audio_filters(source: AudioMetadata, destination: Path, audio_filters: AudioFilters):
    """Apply `audio_filters` to `source` and save at `destination`.

    NOTE: Learn more about `-hide_banner`, `-loglevel` and `-nostats`:
    https://superuser.com/questions/326629/how-can-i-make-ffmpeg-be-quieter-less-verbose
    NOTE: Audio filters may change audio metadata, learn more:
    https://trac.ffmpeg.org/ticket/6570
    https://superuser.com/questions/1218471/converting-and-normalizing-audio-file-creates-unusable-file
    NOTE: Learn more about `ffprobe`:
    https://stackoverflow.com/questions/5618363/is-there-a-way-to-use-ffmpeg-to-determine-the-encoding-of-a-file-before-transcod/5619907
    """
    command = "ffprobe -hide_banner -stats -i".split() + [str(source.path.absolute())]
    command += "-show_entries stream=codec_name -of default=nokey=1:noprint_wrappers=1".split()
    command += ["-v", "error"]
    acodec = subprocess.check_output([str(c) for c in command]).decode().strip()

    command = "ffmpeg -hide_banner -loglevel error -nostats -i".split()
    command += [source.path.absolute(), "-acodec", acodec, "-ar", source.sample_rate, "-af"]
    command += [audio_filters, destination.absolute()]
    subprocess.run([str(c) for c in command], check=True)


@configurable
def pad_remainder(
    signal: np.ndarray, multiple: int = HParam(), center: bool = True, **kwargs
) -> np.ndarray:
    """Pad signal such that `signal.shape[0] % multiple == 0`.

    Args:
        signal (np.array [signal_length]): One-dimensional signal to pad.
        multiple: The returned signal shape is divisible by `multiple`.
        center: If `True` both sides are padded, else the right side is padded.
        **kwargs: Key word arguments passed to `np.pad`.

    Returns:
        np.array [padded_signal_length]
    """
    assert isinstance(signal, np.ndarray)
    remainder = signal.shape[0] % multiple
    remainder = multiple - remainder if remainder != 0 else remainder
    padding = (math.ceil(remainder / 2), math.floor(remainder / 2)) if center else (0, remainder)
    padded_signal = np.pad(signal, padding, **kwargs)
    assert padded_signal.shape[0] % multiple == 0
    return padded_signal


def _mel_filters(
    sample_rate: int,
    num_mel_bins: int,
    fft_length: int,
    lower_hertz: float,
    upper_hertz: float,
) -> np.ndarray:
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins.

    NOTE: The Tacotron 2 model likely did not normalize the filterbank; otherwise, the 0.01
    minimum mentioned in their paper for the dynamic range is too high. NVIDIA/tacotron2 includes
    norm and had to set their minimum to 10**-5 to compensate.
    NOTE: ``htk=True`` because normalization of the mel filterbank is from Slaney's algorithm.

    Reference:
        * The API written by RJ Skerry-Ryan a Tacotron author.
          https://www.tensorflow.org/api_docs/python/tf/contrib/signal/linear_to_mel_weight_matrix

    Args:
        sample_rate: The sample rate of the signal.
        num_mel_bins: Number of Mel bands to generate.
        fft_length: The size of the FFT to apply.
        lower_hertz: Lower bound on the frequencies to be included in the mel spectrum. This
            corresponds to the lower edge of the lowest triangular band.
        upper_hertz: The desired top edge of the highest frequency band.

    Returns:
        (np.ndarray [num_mel_bins, 1 + fft_length / 2]): Mel transform matrix.
    """
    lower_hertz = 0.0 if lower_hertz is None else lower_hertz
    upper_hertz = min(upper_hertz, float(sample_rate) / 2)
    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=fft_length,
        n_mels=num_mel_bins,
        fmin=lower_hertz,
        fmax=upper_hertz,
        norm=None,
        htk=True,
    )


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


def full_scale_sine_wave(
    sample_rate: int = REFERENCE_SAMPLE_RATE, frequency: float = REFERENCE_FREQUENCY
) -> np.ndarray:
    """Full-scale sine wave is used to define the maximum peak level for a dBFS unit.

    Learn more:
    https://stackoverflow.com/questions/22566692/python-how-to-plot-graph-sine-wave/34442729
    https://github.com/makermovement/3.5-Sensor2Phone/blob/master/generate_any_audio.py

    Returns:
        `np.ndarray` of length `sample_rate`
    """
    x = np.arange(sample_rate, dtype=np.float32)  # type: ignore
    return np.sin(2 * np.pi * frequency * (x / sample_rate)).astype(np.float32)  # type: ignore


def full_scale_square_wave(
    sample_rate: int = REFERENCE_SAMPLE_RATE, frequency: float = REFERENCE_FREQUENCY
) -> np.ndarray:
    """Full-scale square wave is also used to define the maximum peak level for a dBFS unit.

    Returns:
        `np.ndarray` of length `sample_rate`
    """
    x = np.arange(sample_rate, dtype=np.float32)  # type: ignore
    x = scipy_signal.square(2 * np.pi * frequency * (x / sample_rate))  # type: ignore
    return x.astype(np.float32)  # type: ignore


def _k_weighting(frequencies: np.ndarray, fs: int) -> np.ndarray:
    # pre-filter 1
    f0 = 1681.9744509555319
    G = 3.99984385397
    Q = 0.7071752369554193
    K = np.tan(np.pi * f0 / fs)  # type: ignore
    Vh = np.power(10.0, G / 20.0)  # type: ignore
    Vb = np.power(Vh, 0.499666774155)  # type: ignore
    a0_ = 1.0 + K / Q + K * K
    b0 = (Vh + Vb * K / Q + K * K) / a0_
    b1 = 2.0 * (K * K - Vh) / a0_
    b2 = (Vh - Vb * K / Q + K * K) / a0_
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / a0_
    a2 = (1.0 - K / Q + K * K) / a0_

    h1 = scipy_signal.freqz([b0, b1, b2], [a0, a1, a2], worN=frequencies, fs=fs)[1]
    h1 = 20 * np.log10(np.absolute(h1))  # type: ignore

    # pre-filter 2
    f0 = 38.13547087613982
    Q = 0.5003270373253953
    K = np.tan(np.pi * f0 / fs)  # type: ignore
    a0 = 1.0
    a1 = 2.0 * (K * K - 1.0) / (1.0 + K / Q + K * K)
    a2 = (1.0 - K / Q + K * K) / (1.0 + K / Q + K * K)
    b0 = 1.0
    b1 = -2.0
    b2 = 1.0

    h2 = scipy_signal.freqz([b0, b1, b2], [a0, a1, a2], worN=frequencies, fs=fs)[1]
    h2 = 20 * np.log10(np.absolute(h2))  # type: ignore

    return h1 + h2


def k_weighting(
    frequencies: np.ndarray, sample_rate: int, offset: typing.Optional[float] = None
) -> np.ndarray:
    """K-Weighting as specified in EBU R-128 / ITU BS.1770-4.

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
        sample_rate
        offset

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    # SOURCE (ITU-R BS.1770):
    # The constant −0.691 in equation (2) cancels out the K-weighting gain for 997 Hz.
    offset = (
        -_k_weighting(np.array([REFERENCE_FREQUENCY]), sample_rate) if offset is None else offset
    )
    return _k_weighting(frequencies, sample_rate) + offset


def a_weighting(frequencies: np.ndarray, *_) -> np.ndarray:
    """Wrapper around `librosa.core.A_weighting`.

    Learn more:
    - Wikipedia describing A-weighting -
      https://en.wikipedia.org/wiki/A-weighting

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    return librosa.core.A_weighting(frequencies, min_db=None) - librosa.core.A_weighting(
        np.array([REFERENCE_FREQUENCY]), min_db=None
    )


def iso226_weighting(frequencies: np.ndarray, *_) -> np.ndarray:
    """Get the ISO226 weights for `frequencies`.

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
    return -typing.cast(np.ndarray, interpolator(frequencies)) + interpolator(
        np.array([REFERENCE_FREQUENCY])
    )


def identity_weighting(frequencies: np.ndarray, *_) -> np.ndarray:
    """Get identity weighting, it doesn't change the frequency weighting.

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    return np.zeros_like(frequencies)


_TensorOrArrayOrFloat = typing.TypeVar("_TensorOrArrayOrFloat", torch.Tensor, np.ndarray, float)


def power_to_db(tensor: _TensorOrArrayOrFloat, eps: float = 1e-10) -> _TensorOrArrayOrFloat:
    """Convert power (https://www.dsprelated.com/freebooks/mdft/Decibels.html) units to decibel
    units.

    TODO: Fix numerical instability: https://github.com/pytorch/audio/issues/611

    Args:
        tensor (torch.FloatTensor)
        eps (float or torch.FloatTensor): The minimum amplitude to `log` avoiding the discontinuity
            at `log(0)`.
    """
    if isinstance(tensor, torch.Tensor):
        result = torch.log10(torch.clamp(tensor, min=eps))
    elif isinstance(tensor, float):
        result = math.log10(max(tensor, eps))
    else:
        result = np.log10(np.clip(tensor, eps, None))
    return typing.cast(_TensorOrArrayOrFloat, 10 * result)


def amp_to_db(tensor: _TensorOrArrayOrFloat, **kwargs) -> _TensorOrArrayOrFloat:
    """Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to decibel units.

    Args:
        tensor
        **kwargs: Other keyword arguments passed to `power_to_db`.
    """
    return power_to_db(tensor, **kwargs) * 2


def amp_to_power(tensor: _TensorOrArrayOrFloat) -> _TensorOrArrayOrFloat:
    """Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to power units."""
    return tensor ** 2


def power_to_amp(tensor: _TensorOrArrayOrFloat) -> _TensorOrArrayOrFloat:
    """Convert power units to amplitude units."""
    return tensor ** 0.5


def db_to_power(tensor: _TensorOrArrayOrFloat) -> _TensorOrArrayOrFloat:
    """Convert decibel units to power units."""
    return typing.cast(_TensorOrArrayOrFloat, 10 ** (tensor / 10.0))


def db_to_amp(tensor: _TensorOrArrayOrFloat) -> _TensorOrArrayOrFloat:
    """Convert decibel units to amplitude units."""
    return db_to_power(tensor / 2)


def signal_to_rms_power(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the RMS power from a signal."""
    return np.mean(amp_to_power(np.abs(signal)), axis=axis)  # type: ignore


def signal_to_rms(signal: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute the root mean square from a signal.

    Learn more:
    - Implementations of RMS:
      https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#rms
      https://github.com/endolith/waveform_analysis/blob/master/waveform_analysis/_common.py#L116
    - Wikipedia on RMS:
      https://en.wikipedia.org/wiki/Root_mean_square
    """
    return power_to_amp(signal_to_rms_power(signal, axis=axis))


@configurable
def signal_to_framed_rms(
    signal: np.ndarray, frame_length: int = HParam(), hop_length: int = HParam()
) -> np.ndarray:
    """Compute the framed root mean square from a signal.

    Args:
        signal (np.ndarray [signal_length])
        frame_length
        hop_length

    Returns:
        np.ndarray [num_frames]
    """
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    return signal_to_rms(frames, axis=0)  # type: ignore


def framed_rms_to_rms(
    frame_rms: _TensorOrArrayOrFloat, frame_hop: int, signal_length: int
) -> _TensorOrArrayOrFloat:
    """Convert framed RMS to RMS."""
    rms = amp_to_power(frame_rms) * frame_hop / signal_length
    rms = rms if isinstance(rms, float) else rms.sum()
    return power_to_amp(typing.cast(_TensorOrArrayOrFloat, rms))


def get_window_correction_factor(window: torch.Tensor) -> float:
    """Energy correction factor for `window` distortion.

    Learn more:
    https://community.sw.siemens.com/s/article/window-correction-factors
    https://www.mathworks.com/matlabcentral/answers/372516-calculate-windowing-correction-factor
    """
    window_correction_factor = torch.ones(*window.shape, device=window.device).pow(2).mean().sqrt()
    window_correction_factor = window_correction_factor / window.pow(2).mean().sqrt()
    return float(window_correction_factor)


@configurable
def power_spectrogram_to_framed_rms(
    power_spectrogram: torch.Tensor,
    window: torch.Tensor = HParam(),
    window_correction_factor: typing.Optional[float] = None,
) -> torch.Tensor:
    """Compute the root mean square from a spectrogram.

    Learn more:
    - Implementations of RMS:
      https://librosa.github.io/librosa/_modules/librosa/feature/spectral.html#rms
    - Opinionated discussion between LUFS and RMS:
      https://www.gearslutz.com/board/mastering-forum/1142602-lufs-really-better-than-rms-measure-loudness.html
      Also, see `test_loudness` in `test_audio.py` that replicates LUFS via RMS.
    - Compare LUFS to Decibels:
      https://backtracks.fm/blog/whats-the-difference-between-decibels-and-lufs/

    Args:
        power_spectrogram (torch.FloatTensor
            [batch_size (optional), num_frames, fft_length // 2 + 1])
        ...

    Returns:
        (torch.FloatTensor [batch_size (optional), num_frames])
    """
    has_batch_dim = power_spectrogram.dim() == 3
    batch_size = power_spectrogram.shape[0] if has_batch_dim else 1
    power_spectrogram = power_spectrogram.view(batch_size, *power_spectrogram.shape[-2:])

    window_correction_factor = (
        get_window_correction_factor(window)
        if window_correction_factor is None
        else window_correction_factor
    )

    # TODO: This adjustment might be related to repairing constant-overlap-add, see here:
    # https://ccrma.stanford.edu/~jos/sasp/Overlap_Add_Decomposition.html. It should be better
    # documented and tested. We've included it mostly because `librosa` also included it.
    # Adjust the DC and half sample rate component
    power_spectrogram[:, :, 0] *= 0.5
    if window.shape[0] % 2 == 0:
        power_spectrogram[:, :, -1] *= 0.5

    # Calculate power
    power = 2 * power_spectrogram.sum(dim=-1) / window.shape[0] ** 2
    frame_rms = power.sqrt() * window_correction_factor
    return frame_rms if has_batch_dim else frame_rms.squeeze(0)


class Spectrograms(typing.NamedTuple):
    """
    Args:
        db_mel (torch.FloatTensor [batch_size (optional), num_frames, num_mel_bins]):
            A spectrogram with the mel scale for frequency and decibel scale for loudness.
        db (torch.FloatTensor [batch_size (optional), num_frames, fft_length // 2 + 1]):
            A spectrogram with a decibel scale for loudness.
        amp (torch.FloatTensor [batch_size (optional), num_frames, fft_length // 2 + 1]):
            A spectrogram with a amplitude scale for loudness.
    """

    db_mel: torch.Tensor
    db: torch.Tensor
    amp: torch.Tensor


class SignalTodBMelSpectrogram(torch.nn.Module):
    """Compute a dB-mel-scaled spectrogram from signal.

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
    https://github.com/tuwien-musicir/DeepLearning_Tutorial/blob/master/rp_extract.py#L320
    https://github.com/tuwien-musicir/DeepLearning_Tutorial/blob/master/rp_extract.py#L363
    http://hyperphysics.phy-astr.gsu.edu/hbase/Sound/phon.html

    TODO: Use LEAF instead, to learn, a spectrogram representation that works well for TTS:
    https://github.com/denfed/leaf-audio-pytorch/tree/main/leaf_audio_pytorch
    The LEAF module could be pretrained by the signal model. The module would be trained to create
    the input and to discriminate. Since both the output and the input are differentiable, in order
    to prevent the network from collapsing, we could also have a secondary loss based on a basic
    spectrogram.

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
        fft_length: See `n_fft` here: https://pytorch.org/docs/stable/torch.html#torch.stft
        frame_hop: See `hop_length` here:
            https://pytorch.org/docs/stable/torch.html#torch.stft
        sample_rate: The sample rate of the audio.
        num_mel_bins: See `_mel_filters`. The mel scale is applied to mimic the the
            non-linear human ear perception of sound, by being more discriminative at lower
            frequencies and less discriminative at higher frequencies.
        window: See `window` here:
            https://pytorch.org/docs/stable/torch.html#torch.stft
        min_decibel: The minimum decible to limit the lower range. Since decibel's is on
            the log scale, the lower range can extend to −∞ as the amplitude gets closer to 0.
        get_weighting: Given a `np.ndarray` of frequencies this returns a weighting in
            decibels. Weighting in an effort to account for the relative loudness perceived by the
            human ear, as the ear is less sensitive to low audio frequencies.
        eps: The minimum amplitude to `log` avoiding the discontinuity at `log(0)`. This
            is similar to `min_decibel` but it operates on the amplitude scale.
        **kwargs: Additional arguments passed to `_mel_filters`.
    """

    @configurable
    def __init__(
        self,
        fft_length: int = HParam(),
        frame_hop: int = HParam(),
        sample_rate: int = HParam(),
        num_mel_bins: int = HParam(),
        window: torch.Tensor = HParam(),
        min_decibel: float = HParam(),
        get_weighting: typing.Callable[[np.ndarray, int], np.ndarray] = HParam(),
        eps: float = 1e-10,
        **kwargs,
    ):
        super().__init__()

        self.register_buffer("window", window)
        self.register_buffer("min_decibel", torch.tensor(min_decibel).float())
        self.fft_length = fft_length
        self.frame_hop = frame_hop
        self.sample_rate = sample_rate
        self.num_mel_bins = num_mel_bins
        self.eps = eps
        self.get_weighting = get_weighting

        mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=self.fft_length, **kwargs)
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.fft_length)  # type: ignore
        weighting = torch.tensor(get_weighting(frequencies, sample_rate)).float().view(-1, 1)
        weighting = db_to_power(weighting)
        self.register_buffer("mel_basis", torch.tensor(mel_basis).float())
        self.register_buffer("weighting", weighting)

    @typing.overload
    def __call__(
        self,
        signal: torch.Tensor,
        intermediate: typing.Literal[False] = False,
        aligned: bool = False,
    ) -> torch.Tensor:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        signal: torch.Tensor,
        intermediate: typing.Literal[True],
        aligned: bool = False,
    ) -> Spectrograms:
        ...  # pragma: no cover

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)

    def forward(
        self, signal: torch.Tensor, intermediate: bool = False, aligned: bool = False
    ) -> typing.Union[Spectrograms, torch.Tensor]:
        """Compute a dB-mel-scaled spectrogram from signal.

        NOTE: Iff a batch of signals is padded sufficiently with zeros and the signal length is a
        multiple of `self.frame_hop`, then this function is invariant to batch size.

        Args:
            signal (torch.FloatTensor [batch_size (optional), signal_length])
            intermediate: If `True`, along with a `db_mel_spectrogram`, this
                returns a `db_spectrogram` and `spectrogram`.
            aligned: If `True` the returned spectrogram is aligned to the signal
                such that `signal.shape[1] // self.frame_hop == db_mel_spectrogram.shape[1]`
        """
        assert signal.dtype == torch.float32, "Invalid argument."
        assert isinstance(self.window, torch.Tensor)
        assert isinstance(self.min_decibel, torch.Tensor)
        assert isinstance(self.mel_basis, torch.Tensor)
        assert isinstance(self.weighting, torch.Tensor)

        has_batch_dim = signal.dim() == 2
        signal = signal.view(-1, signal.shape[-1])

        if aligned:
            assert (
                signal.shape[1] % self.frame_hop == 0
            ), "The signal must be a multiple of `frame_hop` to be aligned to the spectrogram."
            assert (self.fft_length - self.frame_hop) % 2 == 0, (
                "`self.fft_length - self.frame_hop` must be even for the signal "
                "to be aligned to the spectrogram."
            )
            # NOTE: Check ``notebooks/Signal_to_Spectrogram_Consistency.ipynb`` for the correctness
            # of this padding algorithm.
            # NOTE: Center the signal such that the resulting spectrogram and audio are aligned.
            # Learn more here:
            # https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#stft
            padding_ = (self.fft_length - self.frame_hop) // 2
            padding = [padding_, padding_]
            padded_signal = torch.nn.functional.pad(signal, padding, mode="constant", value=0)
        else:
            padded_signal = signal

        # NOTE: The number of spectrogram frames generated is:
        # `(signal.shape[1] - frame_size + frame_hop) // frame_hop`
        spectrogram = torch.stft(
            padded_signal,
            n_fft=self.fft_length,
            hop_length=self.frame_hop,
            win_length=self.window.shape[0],
            window=self.window,
            center=False,
        )

        if aligned:
            assert spectrogram.shape[-2] * self.frame_hop == signal.shape[1], "Invariant failure."

        # NOTE: `torch.norm` is too slow to use in this case
        # https://github.com/pytorch/pytorch/issues/34279
        # power_spectrogram [batch_size, fft_length // 2 + 1, num_frames]
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
            spectrogram = torch.sqrt(torch.clamp(power_spectrogram, min=self.eps)).transpose(-2, -1)
            db_spectrogram = db_spectrogram if has_batch_dim else db_spectrogram.squeeze(0)
            spectrogram = spectrogram if has_batch_dim else spectrogram.squeeze(0)
            return Spectrograms(db_mel_spectrogram, db_spectrogram, spectrogram)
        else:
            return db_mel_spectrogram


@lru_cache(maxsize=None)
def get_signal_to_db_mel_spectrogram(*args, **kwargs) -> SignalTodBMelSpectrogram:
    """Get cached `SignalTodBMelSpectrogram` module."""
    return SignalTodBMelSpectrogram(*args, **kwargs)


@configurable
def get_pyloudnorm_meter(
    sample_rate: int, filter_class: str = HParam(), **kwargs
) -> "pyloudnorm.Meter":
    return _get_pyloudnorm_meter(sample_rate=sample_rate, filter_class=filter_class, **kwargs)


@lru_cache(maxsize=None)
def _get_pyloudnorm_meter(sample_rate: int, filter_class: str, **kwargs) -> "pyloudnorm.Meter":
    """Get cached `pyloudnorm.Meter` module.

    NOTE: `pyloudnorm.Meter` is expensive to import, so we try to avoid it.
    """
    return pyloudnorm.Meter(rate=sample_rate, filter_class=filter_class, **kwargs)


def _db_mel_spectrogram_to_spectrogram(
    db_mel_spectrogram: np.ndarray,
    sample_rate: int,
    fft_length: int,
    get_weighting: typing.Callable[[np.ndarray], np.ndarray],
    **kwargs,
) -> np.ndarray:
    """Transform dB mel spectrogram to spectrogram (lossy).

    Args:
        db_mel_spectrogram (np.array [frames, num_mel_bins]): Numpy array with the spectrogram.
        sample_rate: Sample rate of the `db_mel_spectrogram`.
        fft_length: The size of the FFT to apply.
        get_weighting: Get weighting to weight frequencies.
        **kwargs: Additional arguments passed to `_mel_filters`.

    Returns:
        (np.ndarray [frames, fft_length // 2 + 1]): Spectrogram.
    """
    num_mel_bins = db_mel_spectrogram.shape[1]
    mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=fft_length, **kwargs)
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_length)  # type: ignore
    weighting = get_weighting(frequencies)
    weighting = db_to_power(weighting)
    inverse_mel_basis = np.linalg.pinv(mel_basis)  # NOTE: Approximate inverse matrix of `mel_basis`
    power_mel_spectrogram = db_to_power(db_mel_spectrogram)
    assert isinstance(power_mel_spectrogram, np.ndarray)
    power_spectrogram = np.transpose(np.dot(inverse_mel_basis, np.transpose(power_mel_spectrogram)))
    power_spectrogram = np.maximum(0.0, power_spectrogram)  # type: ignore
    return np.sqrt(power_spectrogram / weighting)  # type: ignore


@lib.utils.log_runtime
@configurable
def griffin_lim(
    db_mel_spectrogram: np.ndarray,
    sample_rate: int = HParam(),
    fft_length: int = HParam(),
    frame_hop: int = HParam(),
    window: np.ndarray = HParam(),
    power: float = HParam(),
    iterations: int = HParam(),
    **kwargs,
) -> np.ndarray:
    """Transform dB mel spectrogram to waveform with the Griffin-Lim algorithm.

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
        sample_rate: Sample rate of the spectrogram and the resulting wav file.
        fft_length: The size of the FFT to apply.
        frame_hop: The frame hop in samples. (e.g. 12.5ms * 24,000 / 1000 == 300)
        window: Window function to be applied to each
            frame. See the full specification for window at ``librosa.filters.get_window``.
        power: Amplification float used to reduce artifacts.
        iterations: Number of iterations of griffin lim to run.

    Returns:
        (np.ndarray [num_samples]): Predicted waveform.
    """
    try:
        logger.info("Running Griffin-Lim....")
        spectrogram = _db_mel_spectrogram_to_spectrogram(
            db_mel_spectrogram=db_mel_spectrogram,
            sample_rate=sample_rate,
            fft_length=fft_length,
            **kwargs,
        )
        spectrogram = spectrogram.transpose()
        spectrogram = np.power(spectrogram, power)
        waveform = librosa.core.griffinlim(
            spectrogram,
            n_iter=iterations,
            hop_length=frame_hop,
            win_length=window.shape[0],
            window=window,
        )
        # NOTE: Pad to ensure spectrogram and waveform align.
        waveform = np.pad(waveform, int(frame_hop // 2), mode="constant", constant_values=0)
        return clip_waveform(waveform).astype(np.float32)  # type: ignore
    except Exception:
        logger.exception("Griffin-lim encountered an issue and was unable to render audio.")
        # NOTE: Return no audio for valid inputs that fail due to an overflow error or a small
        # spectrogram.
        return np.array([], dtype=np.float32)  # type: ignore


def highpass_filter(signal: np.ndarray, freq: int, sample_rate: int, order: int = 5) -> np.ndarray:
    """A high-pass filter passes signals with a frequency higher than `freq` and attenuates signals
    with frequencies lower than `freq`.

    Based on:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    NOTE: This isn't memory efficient because it loads the entire audio file into memory,
    learn more:
    https://github.com/scipy/scipy/issues/6669
    https://dsp.stackexchange.com/questions/73937/memory-efficient-filtering-with-scipy-signal-in-python
    """
    nyquist = 0.5 * sample_rate
    sos = scipy_signal.butter(order, freq / nyquist, analog=False, btype="highpass", output="sos")
    return scipy_signal.sosfiltfilt(sos, signal).astype(signal.dtype)


_GroupAudioFramesVar = typing.TypeVar("_GroupAudioFramesVar")
_GroupAudioFramesIterator = typing.Iterator[typing.Tuple[_GroupAudioFramesVar, typing.List[int]]]


def group_audio_frames(
    sample_rate: int,
    classifications: typing.Iterable[_GroupAudioFramesVar],
    indicies: typing.Iterable[typing.List[int]],
    is_include: typing.Callable[[_GroupAudioFramesVar], bool],
) -> typing.List[typing.Tuple[float, float]]:
    """Group audio frames by `classification`.

    Args:
        ...
        classification: A classification for each frame.
        indicies: The sample indicies represented in each frame.
        is_include: Callable to determine if to include a class.
    """
    iterator: _GroupAudioFramesIterator[_GroupAudioFramesVar] = zip(classifications, indicies)
    groups = typing.cast(
        typing.Iterable[typing.Tuple[_GroupAudioFramesVar, _GroupAudioFramesIterator]],
        itertools.groupby(iterator, key=lambda i: i[0]),
    )
    results = []
    for classification, group in groups:
        group = list(group)
        if is_include(classification):
            start = sample_to_sec(group[0][1][0], sample_rate)
            stop = sample_to_sec(group[-1][1][-1], sample_rate)
            results.append((start, stop))
    return results


def _get_non_speech_segments_helper(
    audio: np.ndarray,
    audio_file: AudioMetadata,
    low_cut: int,
    frame_length: float,
    hop_length: float,
):
    """Get non-speech segments in `audio` helper for framing, and measuring RMS."""
    assert audio.dtype == np.float32
    sample_rate = audio_file.sample_rate
    milli_to_sample_ = partial(milli_to_sample, sample_rate=sample_rate)
    hop_length = milli_to_sample_(hop_length)
    frame_length = milli_to_sample_(frame_length)
    frame = partial(librosa.util.frame, frame_length=frame_length, hop_length=hop_length, axis=0)
    audio = highpass_filter(audio, low_cut, sample_rate)  # NOTE: Noise reduction
    rms_level_power = signal_to_rms_power(frame(audio), axis=1)
    min_indicies = np.arange(0, len(audio), hop_length)[: len(rms_level_power)]
    max_indicies = np.arange(frame_length - 1, len(audio), hop_length)[: len(rms_level_power)]
    indicies = np.stack([min_indicies, max_indicies], axis=1)
    return indicies, rms_level_power


def get_non_speech_segments(
    audio: np.ndarray,
    audio_file: AudioMetadata,
    low_cut: int,
    frame_length: float,
    hop_length: float,
    threshold: float,
) -> typing.List[typing.Tuple[float, float]]:
    """Get non-speech segments in `audio`.

    Args:
        ...
        low_cut: This attenuates frequencies lower than `low_cut`.
        frame_length: The `audio` is sliced into overlapping `frame_length` milliseconds frames.
        hop_length: The number of milliseconds to advance between each frame.
        threshold: This is a decision threshold, in decibels, for deciding if a frame is to be
            classified as speech, or non-speech.

    Returns: This returns an iterable of `audio` slices in seconds representing non-speech segments.
    """
    audio = np.pad(audio, milli_to_sample(frame_length, sample_rate=audio_file.sample_rate))
    indicies, rms_level_power = _get_non_speech_segments_helper(
        audio, audio_file, low_cut, frame_length, hop_length
    )
    power_threshold = db_to_power(threshold)
    is_not_speech: typing.Callable[[bool], bool] = lambda is_speech: not is_speech
    is_speech: typing.List[bool] = list(rms_level_power > power_threshold)
    segments = group_audio_frames(audio_file.sample_rate, is_speech, indicies, is_not_speech)
    offset = lambda a: lib.utils.clamp(a - milli_to_sec(frame_length), 0, audio_file.length)
    return [(offset(a), offset(b)) for a, b in segments]
