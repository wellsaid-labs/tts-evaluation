from __future__ import annotations

import logging
import math
import os
import subprocess
import typing
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from hparams import HParam, configurable
from third_party import LazyLoader
from third_party.iso226 import iso226_spl_itpl
from tqdm import tqdm

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    import pyloudnorm
    from scipy import signal as scipy_signal
    from scipy.io.wavfile import scipy_wavfile
else:
    pyloudnorm = LazyLoader("pyloudnorm", globals(), "pyloudnorm")
    librosa = LazyLoader("librosa", globals(), "librosa")
    scipy_signal = LazyLoader("scipy_signal", globals(), "scipy.signal")
    scipy_wavfile = LazyLoader("scipy_wavfile", globals(), "scipy.io.wavfile")


logger = logging.getLogger(__name__)


class AudioFileMetadata(typing.NamedTuple):
    """
    TODO: The `sample_rate` does not change in porportion to the number of channels; therefore, the
    `sample_rate` should technically be called the `frame_rate` because it measures the number of
    frames per second.

    Args:
      path: The audio file path.
      sample_rate: The sample rate of the audio.
      channels: The number of audio channels in the audio file.
      encoding: The encoding of the audio file (e.g. "32-bit Floating Point PCM").
      length: The duration of the audio file in seconds.
    """

    path: Path
    sample_rate: int
    channels: int
    encoding: str
    length: float


def _parse_audio_metadata(metadata: str) -> AudioFileMetadata:
    """Parse audio metadata returned by `sox --i`.

    NOTE: This parses the output of `sox --i` instead individual requests like `sox --i -r` and
    `sox --i -b` to conserve on the number  of requests made via `subprocess`.

    TODO: Adapt `ffmpeg --i` for consistency.
    """
    splits = [s.split(":", maxsplit=1)[1].strip() for s in metadata.strip().split("\n")]
    audio_path = str(splits[0][1:-1])
    channels = int(splits[1])
    sample_rate = int(splits[2])
    assert splits[4].split()[3] == "samples"
    length = float(splits[4].split()[2]) / sample_rate
    encoding = splits[-1]
    return AudioFileMetadata(Path(audio_path), sample_rate, channels, encoding, length)


def _get_audio_metadata_helper(
    chunk: typing.List[Path],
) -> typing.List[AudioFileMetadata]:
    command = ["sox", "--i"] + [str(p) for p in chunk]
    metadatas = subprocess.check_output(command).decode()
    splits = metadatas.strip().split("\n\n")
    splits = splits[:-1] if "Total Duration" in splits[-1] else splits
    return [_parse_audio_metadata(metadata) for metadata in splits]


def get_audio_metadata(
    paths: typing.List[Path],
    max_arg_length: int = 2 ** 16,
    num_processes: int = (
        1 if lib.environment.IS_TESTING_ENVIRONMENT else typing.cast(int, os.cpu_count())
    ),
) -> typing.List[AudioFileMetadata]:
    """Get the audio metadatas for a list of files.

    NOTE: It's difficult to determine the bash maximum argument length, learn more:
    https://unix.stackexchange.com/questions/45143/what-is-a-canonical-way-to-find-the-actual-maximum-argument-list-length
    https://stackoverflow.com/questions/19354870/bash-command-line-and-input-limit
    """
    len_ = lambda p: len(str(p))
    total = sum([len_(p) for p in paths])
    chunks = list(
        lib.utils.accumulate_and_split(paths, [max_arg_length] * (total // max_arg_length), len_)
    )
    if len(chunks) == 1:
        return _get_audio_metadata_helper(chunks[0])

    logger.info("Getting audio metadata for %d audio files.", len(paths))
    return_ = []
    with tqdm(total=len(paths)) as progress_bar:
        with lib.utils.Pool(num_processes) as pool:
            for result in pool.imap_unordered(_get_audio_metadata_helper, chunks):
                return_.extend(result)
                progress_bar.update(len(result))
    return return_


def read_audio(path: Path) -> np.ndarray:
    """Read an audio file slice into a `np.float32` array.

    NOTE: Audio files with multiple channels will be mixed into a mono channel.
    """
    command = f"ffmpeg -i {path} -f f32le -acodec pcm_f32le -ac 1 pipe:"
    return np.frombuffer(
        subprocess.check_output(command.split(), stderr=subprocess.DEVNULL), np.float32
    )


def read_audio_slice(path: Path, start: float, length: float) -> np.ndarray:
    """Read an audio file slice into a `np.float32` array.

    NOTE: Audio files with multiple channels will be mixed into a mono channel.
    NOTE: Learn more about efficiently selecting a slice of audio with `ffmpeg`:
    https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg

    Args:
        path: Path to load.
        start: The start of the audio segment.
        length: The length of the audio segment.
    """
    command = f"ffmpeg -ss {start} -i {path} -to {length} -f f32le -acodec pcm_f32le -ac 1 pipe:"
    return np.frombuffer(
        subprocess.check_output(command.split(), stderr=subprocess.DEVNULL), np.float32
    )


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
    assert audio.dtype == np.float32
    if audio.size > 0:
        assert np.max(audio) <= 1.0 and np.min(audio) >= -1.0, "Signal must be in range [-1, 1]."
    scipy_wavfile.write(path, sample_rate, audio)


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
        sample_rate,
        fft_length,
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
    x = np.arange(sample_rate, dtype=np.float32)
    return np.sin(2 * np.pi * frequency * (x / sample_rate)).astype(np.float32)


def full_scale_square_wave(
    sample_rate: int = REFERENCE_SAMPLE_RATE, frequency: float = REFERENCE_FREQUENCY
) -> np.ndarray:
    """Full-scale square wave is also used to define the maximum peak level for a dBFS unit.

    Returns:
        `np.ndarray` of length `sample_rate`
    """
    x = np.arange(sample_rate, dtype=np.float32)
    return scipy_signal.square(2 * np.pi * frequency * (x / sample_rate)).astype(np.float32)


def _k_weighting(frequencies: np.ndarray, fs: int) -> np.ndarray:
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


def a_weighting(frequencies: np.ndarray) -> np.ndarray:
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


def iso226_weighting(frequencies: np.ndarray) -> np.ndarray:
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
    return -interpolator(frequencies) + interpolator(np.array([REFERENCE_FREQUENCY]))


def identity_weighting(frequencies: np.ndarray) -> np.ndarray:
    """Get identity weighting, it doesn't change the frequency weighting.

    Args:
        frequencies (np.ndarray [*]): Frequencies for which to get weights.

    Returns:
        np.ndarray [*frequencies.shape]: Weighting for each frequency.
    """
    return np.zeros_like(frequencies)


def power_to_db(tensor: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Convert power (https://www.dsprelated.com/freebooks/mdft/Decibels.html) units to decibel
    units.

    Args:
        tensor (torch.FloatTensor)
        eps (float or torch.FloatTensor): The minimum amplitude to `log` avoiding the discontinuity
            at `log(0)`.
    """
    return 10.0 * torch.log10(torch.clamp(tensor, min=eps))


def amplitude_to_db(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to decibel units.

    Args:
        tensor
        **kwargs: Other keyword arguments passed to `power_to_db`.
    """
    return power_to_db(tensor, **kwargs) * 2


def amplitude_to_power(
    tensor: typing.Union[torch.Tensor, np.ndarray]
) -> typing.Union[torch.Tensor, np.ndarray]:
    """ Convert amplitude (https://en.wikipedia.org/wiki/Amplitude) units to power units. """
    return tensor ** 2


def power_to_amplitude(
    tensor: typing.Union[torch.Tensor, np.ndarray]
) -> typing.Union[torch.Tensor, np.ndarray]:
    """ Convert power units to amplitude units. """
    return tensor ** 0.5


def db_to_power(
    tensor: typing.Union[torch.Tensor, np.ndarray]
) -> typing.Union[torch.Tensor, np.ndarray]:
    """ Convert decibel units to power units. """
    return 10 ** (tensor / 10.0)


def db_to_amplitude(
    tensor: typing.Union[torch.Tensor, np.ndarray]
) -> typing.Union[torch.Tensor, np.ndarray]:
    """ Convert decibel units to amplitude units. """
    return db_to_power(tensor / 2)


def signal_to_rms(signal: np.ndarray) -> np.ndarray:
    """Compute the root mean square from a signal.

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
    return np.sqrt(np.mean(np.abs(signal) ** 2))


@configurable
def signal_to_framed_rms(
    signal: np.ndarray,
    frame_length: int = HParam(),
    hop_length: int = HParam(),
) -> np.ndarray:
    """Compute the framed root mean square from a signal.

    Args:
        signal (np.ndarray [signal_length])
        frame_length
        hop_length

    Returns:
        np.ndarray [1]
    """
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)
    return np.sqrt(np.mean(np.abs(frames ** 2), axis=0))


@configurable
def power_spectrogram_to_framed_rms(
    power_spectrogram: torch.Tensor,
    window: torch.Tensor = HParam(),
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
        torch.ones(*window.shape).pow(2).mean().sqrt() / window.pow(2).mean().sqrt()
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
        num_mel_bins: See `src.audio._mel_filters`. The mel scale is applied to mimic the the
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
        get_weighting: typing.Callable[[np.ndarray], np.ndarray] = HParam(),
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

        mel_basis = _mel_filters(sample_rate, num_mel_bins, fft_length=self.fft_length, **kwargs)
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=self.fft_length)
        weighting = torch.tensor(get_weighting(frequencies)).float().view(-1, 1)
        weighting = db_to_power(weighting)
        self.register_buffer("mel_basis", torch.tensor(mel_basis).float())
        self.register_buffer("weighting", weighting)

    @typing.overload
    def forward(
        self,
        signal: torch.Tensor,
        intermediate: typing.Literal[False],
        aligned: bool,
    ) -> torch.Tensor:
        ...  # pragma: no cover

    @typing.overload
    def forward(
        self,
        signal: torch.Tensor,
        intermediate: typing.Literal[True],
        aligned: bool,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...  # pragma: no cover

    def forward(
        self, signal: torch.Tensor, intermediate: bool = False, aligned: bool = False
    ) -> typing.Union[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        """Compute a dB-mel-scaled spectrogram from signal.

        Args:
            signal (torch.FloatTensor [batch_size (optional), signal_length])
            intermediate: If `True`, along with a `db_mel_spectrogram`, this
                returns a `db_spectrogram` and `spectrogram`.
            aligned: If `True` the returned spectrogram is aligned to the signal
                such that `signal.shape[1] / self.frame_hop == db_mel_spectrogram.shape[1]`

        Returns:
            db_mel_spectrogram (torch.FloatTensor
                [batch_size  (optional), num_frames, num_mel_bins]): A spectrogram with the mel
                scale for frequency, decibel scale for power, and a regular time scale.
            db_spectrogram (torch.FloatTensor
                [batch_size (optional), num_frames, fft_length // 2 + 1]): This is only  returned
                iff `intermediate=True`.
            spectrogram (torch.FloatTensor [batch_size (optional),
                num_frames, fft_length // 2 + 1]): This is only returned iff `intermediate=True`.
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
            # NOTE: The number of spectrogram frames generated is:
            # `(signal.shape[1] - frame_size + frame_hop) // frame_hop`
            padding = (self.fft_length - self.frame_hop) // 2
            padded_signal = torch.nn.functional.pad(
                signal, [padding, padding], mode="constant", value=0
            )
        else:
            padded_signal = signal

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
            spectrogram = torch.sqrt(torch.clamp(power_spectrogram, min=self.eps)).transpose(-2, -1)
            db_spectrogram = db_spectrogram if has_batch_dim else db_spectrogram.squeeze(0)
            spectrogram = spectrogram if has_batch_dim else spectrogram.squeeze(0)
            return db_mel_spectrogram, db_spectrogram, spectrogram
        else:
            return db_mel_spectrogram


@lru_cache(maxsize=None)
def get_signal_to_db_mel_spectrogram(*args, **kwargs) -> SignalTodBMelSpectrogram:
    """ Get cached `SignalTodBMelSpectrogram` module. """
    return SignalTodBMelSpectrogram(*args, **kwargs)


@lru_cache(maxsize=None)
def get_pyloudnorm_meter(*args, **kwargs) -> pyloudnorm.Meter:
    """Get cached `pyloudnorm.Meter` module.

    NOTE: `pyloudnorm.Meter` is expensive to import, so we try to avoid it.
    """
    return pyloudnorm.Meter(*args, **kwargs)


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
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=fft_length)
    weighting = get_weighting(frequencies)
    weighting = db_to_power(weighting)
    inverse_mel_basis = np.linalg.pinv(mel_basis)  # NOTE: Approximate inverse matrix of `mel_basis`
    power_mel_spectrogram = db_to_power(db_mel_spectrogram)
    assert isinstance(power_mel_spectrogram, np.ndarray)
    power_spectrogram = np.dot(inverse_mel_basis, power_mel_spectrogram.transpose()).transpose()
    power_spectrogram = np.maximum(0.0, power_spectrogram)
    return np.sqrt(power_spectrogram / weighting)


@configurable
def griffin_lim(
    db_mel_spectrogram: np.ndarray,
    sample_rate: int = HParam(),
    fft_length: int = HParam(),
    frame_hop: int = HParam(),
    window: np.ndarray = HParam(),
    power: float = HParam(),
    iterations: int = HParam(),
    use_tqdm: bool = False,
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
        use_tqdm: If `True` attach a progress bar during iteration.

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
        waveform = librosa.core.griffinlim(
            spectrogram,
            n_iter=iterations,
            hop_length=frame_hop,
            win_length=window.shape[0],
            window=window,
        )
        # NOTE: Pad to ensure spectrogram and waveform align.
        waveform = np.pad(waveform, int(frame_hop // 2), mode="constant", constant_values=0)
        large_values = (waveform < -1).sum() + (waveform > 1).sum()
        if large_values > 0:
            logger.warning("Griffin-lim waveform clipped %d samples.", large_values)
        return np.clip(waveform, -1, 1).astype(np.float32)
    except Exception:
        logger.exception("Griffin-lim encountered an issue and was unable to render audio.")
        # NOTE: Return no audio for valid inputs that fail due to an overflow error or a small
        # spectrogram.
        return np.array([], dtype=np.float32)
