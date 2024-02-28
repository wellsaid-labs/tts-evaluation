import dataclasses
import enum
import logging
import math
import os
import subprocess
import typing
from functools import partial
from pathlib import Path

import numpy as np
import package_utils.numeric
from third_party import LazyLoader

SAMPLE_RATE = 24000


def sample_to_sec(sample: int, sample_rate: int) -> float:
    """Covert samples to seconds."""
    return float(sample) / sample_rate


def sec_to_sample(sec: float, sample_rate: int) -> int:
    """Covert seconds to samples."""
    return int(round(sec * sample_rate))


def read_audio(
    path: Path,
    start: float = 0,
    length: float = math.inf,
    dtype=("f32le", "pcm_f32le", np.float32),
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
    command += [
        "-i",
        path,
        "-f",
        dtype[0],
        "-acodec",
        dtype[1],
        "-ac",
        "1",
        "pipe:",
    ]
    command = [str(c) for c in command]
    ndarray = np.frombuffer(
        subprocess.check_output(command, stderr=subprocess.DEVNULL), dtype[2]
    )
    return clip_waveform(ndarray)


def write_audio(
    path: typing.Union[Path, typing.BinaryIO],
    audio: np.ndarray,
    sample_rate: int = 24000,
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
    audio = (
        audio.detach().cpu().numpy()
        if not isinstance(audio, np.ndarray)
        else audio
    )
    if typing.cast(int, audio.size) > 0:
        error = f"Signal must be in range [-1, 1] rather than [{np.min(audio)}, {np.max(audio)}]."
        assert np.max(audio) <= 1.0 and np.min(audio) >= -1.0, error
    wavfile.write(path, sample_rate, audio)


def clip_waveform(waveform: np.ndarray) -> np.ndarray:
    """Clip audio at the maximum and minimum amplitude.

    NOTE: Clipping will cause distortion to the waveform, learn more:
    https://en.wikipedia.org/wiki/Clipping_(audio)
    """
    dtype = waveform.dtype
    is_floating = np.issubdtype(dtype, np.floating)
    min_: typing.Union[int, float] = (
        -1.0 if is_floating else np.iinfo(dtype).min
    )
    max_: typing.Union[int, float] = 1.0 if is_floating else np.iinfo(dtype).max
    num_clipped_samples = (waveform < min_).sum() + (waveform > max_).sum()
    if num_clipped_samples > 0:
        max_sample = np.max(np.absolute(waveform))
        logger.debug(
            "%d samples clipped (%f max sample)",
            num_clipped_samples,
            max_sample,
        )
    return np.clip(waveform, min_, max_)


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

    @property
    def len(self):
        # TODO: Deprecate `length` eventually, it's too verbose and not pythonic.
        return self.length

    @property
    def format(self):
        return AudioFormat(
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            encoding=self.encoding,
            bit_rate=self.bit_rate,
            precision=self.precision,
        )


wavfile = LazyLoader("wavfile", globals(), "scipy.io.wavfile")
logger = logging.getLogger(__name__)


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
    assert (
        metadata.encoding.value in lookup
    ), f"Metadata encoding '{metadata}' is not supported."
    dtype = lookup[metadata.encoding.value]
    bytes_per_sample = np.dtype(dtype).itemsize
    sec_to_sample_ = partial(sec_to_sample, sample_rate=metadata.sample_rate)
    header_size = (
        os.path.getsize(metadata.path) - bytes_per_sample * metadata.num_samples
    )
    start = int(
        package_utils.numeric.round_(
            sec_to_sample_(start) * bytes_per_sample, bytes_per_sample
        )
    )
    length = (
        sec_to_sample_(length)
        if length > 0
        else metadata.num_samples - sec_to_sample_(start)
    )
    if memmap:
        ndarray = np.memmap(
            metadata.path,
            dtype=dtype,
            shape=(length,),
            offset=start + header_size,
        )
    else:
        ndarray = np.fromfile(
            metadata.path, dtype=dtype, count=length, offset=start + header_size
        )
    return clip_waveform(ndarray)
