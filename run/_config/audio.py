import logging
import math
import typing
from functools import partial

import config as cf
from config import Args
from third_party import LazyLoader

import lib
import run

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
else:
    librosa = LazyLoader("librosa", globals(), "librosa")

logger = logging.getLogger(__name__)


# TODO: Instead of using global variables, can we use `config`, easily?


# SOURCE (Tacotron 2):
# We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank spanning
# 125 Hz to 7.6 kHz, followed by log dynamic range compression.
# SOURCE (Tacotron 2 Author):
# Google mentioned they settled on [20, 12000] with 128 filters in Google Chat.
NUM_FRAME_CHANNELS = 128


# SOURCE (Tacotron 2):
# mel spectrograms are computed through a shorttime Fourier transform (STFT)
# using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
# TODO: Parameterizing frame sizes in milliseconds can help ensure that your code is invariant
# to the sample rate; however, we would need to assure that we're still using powers of two
# for performance.
# TODO: 50ms / 12.5ms spectrogram is not typical spectrogram parameterization, a more typical
# parameterization is 25ms / 10ms. Learn more:
# https://www.dsprelated.com/freebooks/sasp/Classic_Spectrograms.html
# https://github.com/pytorch/audio/issues/384#issuecomment-597020705
# https://pytorch.org/audio/compliance.kaldi.html
FRAME_SIZE = 4096  # NOTE: Frame size in samples.
FFT_LENGTH = 4096
assert FRAME_SIZE % 4 == 0
FRAME_HOP = FRAME_SIZE // 4


def _get_max_audio_len_in_frames(text: str, frames_per_second: float) -> int:
    """Get the max audio length in frames."""
    return math.ceil(run._config.lang.get_max_audio_length(text) * frames_per_second)


# NOTE: The annotation values and lengths can be reviewed in the
# `run/review/dataset_processing/annotations.py` workbook which provides basic statistics and
# and clips to review.
# NOTE: Usually, for training, it's helpful if the data is within a range of -1 to 1. These
# normalization functions help adjust for that.


def _norm_anno_len(val: float, avg_anno_length: float = 40.5) -> float:
    """Normalize a annotation length with an average around `avg_anno_length` characters.
    This is bounded to 1 so this value doesn't grow unbounded for longer annotations.
    """
    return min(val / avg_anno_length - 1, 1.0)


def _norm_anno_loudness(val: float, compression: float = 50) -> float:
    """Normalize a annotation loudness value which is from 35 to -60 db centered at 0 db."""
    return val / compression


def _norm_tempo(val: float, avg_val: float = 1.0) -> float:
    """Normalize a annotation tempo value which is from 0.5x to 5x with an average around 1x."""
    return val - avg_val


def _norm_sesh_loudness(val: float, avg_val: float = -21, compression: float = 50) -> float:
    """Normalize a session loudness value which is from -5 to -70 db with an average around -21 db.

    NOTE: LUFS by definition cannot be less than -70 db, or more than 0db.
    """
    return (val - avg_val) / compression


def configure(sample_rate: int = 24000, overwrite: bool = False):
    """Configure modules that process audio.

    SOURCE (Tacotron 1): We use 24 kHz sampling rate for all experiments.
    """
    # NOTE: The SoX and FFmpeg encodings are the same.
    # NOTE: The signal model output is 32-bit.
    suffix = ".wav"
    data_type = lib.audio.AudioDataType.FLOATING_POINT
    bits = 32
    format_ = lib.audio.AudioFormat(
        sample_rate=sample_rate,
        num_channels=1,  # NOTE: The signal model output is 1-channel, similar to Tacotron-2.
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
    )
    non_speech_segment_frame_length = 100
    frames_per_second = format_.sample_rate / FRAME_HOP
    # NOTE: Today pauses longer than one second are not used for emphasis or meaning; however,
    # Otis does tend to use long pauses for emphasis; however, he rarely pauses for longer than
    # one second.
    too_long_pause_length = 1.0

    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    try:
        window = run._utils.get_window("hann", FRAME_SIZE, FRAME_HOP)
        window_correction_factor = lib.audio.get_window_correction_factor(window)
        config = {
            lib.audio.power_spectrogram_to_framed_rms: Args(
                window=window,
                window_correction_factor=window_correction_factor,
            ),
            lib.audio.SignalTodBMelSpectrogram: Args(window=window),
            lib.audio.griffin_lim: Args(window=window.numpy()),
        }
        cf.add(config, overwrite)
    except ImportError:
        logger.info("Ignoring optional `scipy` and `librosa` configurations.")

    # NOTE: The human range is commonly given as 20 to 20,000 Hz
    # (https://en.wikipedia.org/wiki/Hearing_range).
    hertz_bounds = {"lower_hertz": 20, "upper_hertz": 20000}

    try:
        config = {
            librosa.effects.trim: Args(frame_length=FRAME_SIZE, hop_length=FRAME_HOP),
        }
        cf.add(config, overwrite)
    except ImportError:
        logger.info("Ignoring optional `librosa` configurations.")

    args = Args(sample_rate=sample_rate)
    config = {
        lib.visualize.plot_waveform: args,
        lib.visualize.plot_spectrogram: args,
        lib.visualize.plot_mel_spectrogram: args,
        lib.audio.write_audio: args,
        lib.audio.SignalTodBMelSpectrogram: args,
        lib.audio.griffin_lim: args,
        run.train.spectrogram_model._metrics.get_num_pause_frames: args,
        run.data._loader.utils.normalize_audio: args,
        run._tts.tts_ffmpeg_generator: args,
        lib.audio.get_pyloudnorm_meter: args,
        lib.audio.milli_to_sample: args,
        lib.audio.sample_to_milli: args,
        lib.audio.sample_to_sec: args,
        lib.audio.sec_to_sample: args,
        run._config.data._get_loudness_annotation: args,
    }
    cf.add(config, overwrite)

    # NOTE: The `DeMan` loudness implementation of ITU-R BS.1770 is sample rate independent.
    filter_class = "DeMan"
    sil_thresh = -50
    config = {
        lib.visualize.plot_spectrogram: Args(frame_hop=FRAME_HOP),
        lib.visualize.plot_mel_spectrogram: Args(frame_hop=FRAME_HOP, **hertz_bounds),
        lib.audio.pad_remainder: Args(multiple=FRAME_HOP, mode="constant", constant_values=0.0),
        lib.audio.signal_to_framed_rms: Args(frame_length=FRAME_SIZE, hop_length=FRAME_HOP),
        lib.audio.SignalTodBMelSpectrogram: Args(
            fft_length=FFT_LENGTH,
            frame_hop=FRAME_HOP,
            num_mel_bins=NUM_FRAME_CHANNELS,
            # SOURCE (Tacotron 2):
            # Prior to log compression, the filterbank output magnitudes are clipped to a
            # minimum value of 0.01 in order to limit dynamic range in the logarithmic domain.
            # NOTE: The `min_decibel` is set to ensure there is around 100 dB of dynamic range,
            # allowing us to make the most use of the maximum 96 dB dynamic range a 16-bit audio
            # file can provide. This is assuming that a full-scale 997 Hz sine wave is the maximum
            # dB which would be around ~47 dB. Tacotron 2's equivalent  of 0.01 (~ -40 dB).
            min_decibel=-50.0,
            # NOTE: ISO226 is one of the latest standards for determining loudness:
            # https://www.iso.org/standard/34222.html. It does have some issues though:
            # http://www.lindos.co.uk/cgi-bin/FlexiData.cgi?SOURCE=Articles&VIEW=full&id=2
            get_weighting=lib.audio.iso226_weighting,
            # NOTE: Ensure that the weighting isn't below -30 decibels; otherwise, a value may go
            # to zero which and it'll go to infinity when applying the operations in inverse.
            min_weight=-30.0,
            **hertz_bounds,
        ),
        lib.audio.griffin_lim: Args(
            fft_length=FFT_LENGTH,
            frame_hop=FRAME_HOP,
            # SOURCE (Tacotron 1):
            # We found that raising the predicted magnitudes by a power of 1.2 before feeding to
            # Griffin-Lim reduces artifacts
            power=1.20,
            # SOURCE (Tacotron 1):
            # We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
            # iterations seems to be enough), which is reasonably fast.
            iterations=60,
            get_weighting=lib.audio.iso226_weighting,
            min_weight=-30,
            # NOTE: Based on a brief analysis in April 2022, we found that "Fast Griffin-lim" clips
            # with momentum, where more difficult to understand. By setting `momentum` to zero,
            # we are recovering the original Griffin-lim algorithm which we believe is more
            # interpretable in a text-to-speech context.
            momentum=0.0,
            **hertz_bounds,
        ),
        lib.audio.get_pyloudnorm_meter: Args(filter_class=filter_class),
        # TODO: Some speakers, like Heather, speak at around -40 db to -50db,
        # and pauses are at best -60 db. While speakers like Donnla, may have silences around
        # -45 db. We should use session avg loudness, to create a threshold that is more general.
        # There is some preliminary work on that in `batch_griffin_lim.py`
        run.train.spectrogram_model._metrics.get_num_pause_frames: Args(
            frame_hop=FRAME_HOP,
            min_length=too_long_pause_length,
            max_loudness=sil_thresh,
        ),
        run.train.spectrogram_model._metrics.get_num_sil_frames: Args(sil_threshold=sil_thresh),
        run._models.signal_model.wrapper.SignalModelWrapper: Args(
            ratios=[2] * math.ceil(math.log2(FRAME_HOP)),
            pred_sample_rate=2 ** math.ceil(math.log2(FRAME_HOP)),
            out_sample_rate=FRAME_HOP,
        ),
        run.data._loader.utils.normalize_audio_suffix: Args(suffix=suffix),
        run.data._loader.utils.normalize_audio: Args(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            num_channels=format_.num_channels,
        ),
        run.data._loader.utils.is_normalized_audio_file: Args(audio_format=format_, suffix=suffix),
        # NOTE: `get_non_speech_segments` parameters are set based on `vad_workbook.py`. They
        # are applicable to most datasets with little to no noise.
        run.data._loader.utils.get_non_speech_segments_and_cache: Args(
            low_cut=300, frame_length=non_speech_segment_frame_length, hop_length=5, threshold=-60
        ),
        run.data._loader.structures._make_speech_segments_helper: Args(
            pad=lib.audio.milli_to_sec(non_speech_segment_frame_length / 2)
        ),
        run.data._loader.utils.maybe_normalize_audio_and_cache: Args(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            format_=format_,
        ),
        run.data._loader.utils.SpanGenerator: Args(max_pause=too_long_pause_length),
        # NOTE: A 0.400 `block_size` is standard for ITU-R BS.1770.
        # NOTE: We use `precision=0` because the just noticeable difference in sound
        # intensity for the human ear is about 1 decibel. For more intense sounds, this can be
        # as low as 1/2 or 1/3.
        # http://physics.gmu.edu/~dmaria/phys260summer03/sound/DB.HTML#c4
        run._config.data._get_loudness_annotation: Args(
            block_size=0.400, precision=0, filter_class=filter_class
        ),
        run._models.spectrogram_model.inputs.preprocess: Args(
            get_max_audio_len=partial(
                _get_max_audio_len_in_frames, frames_per_second=frames_per_second
            ),
            norm_anno_len=_norm_anno_len,
            norm_anno_loudness=_norm_anno_loudness,
            norm_sesh_loudness=_norm_sesh_loudness,
            norm_tempo=_norm_tempo,
        ),
        run._models.spectrogram_model.inputs.InputsWrapper.check_invariants: Args(
            # NOTE: The annotation min and max values can be reviewed in the
            # `run/review/dataset_processing/annotations.py` workbook.
            # NOTE: The LUFS range is approximately centered on our loudest a recording session
            # (-10db) and quietest recording session (-40db). With a range of -5db to -70db,
            # centered on those recording sessions, the quietest recording session could
            # be 35db louder and the loudest recording session could be -60db quieter.
            # TODO: We should consider adding filtering for outlier annotations and reducing the
            # allowable range.
            min_loudness=-60,
            max_loudness=35,
            # NOTE: The tempo range is approximately from 10% to 1000% slower than faster than
            # average.
            # TODO: We should refine this a bit more, and find stricter bounds.
            min_tempo=0.1,
            max_tempo=10,
        ),
        run.train.spectrogram_model._data._make_stop_token: Args(
            # NOTE: The stop token uncertainty was approximated by a fully trained model that
            # learned the stop token distribution. The distribution looked like a gradual increase
            # over 4 - 8 frames in January 2020, on Comet.
            # NOTE: This was rounded up to 10 after the spectrograms length was increased by 17%
            # on average.
            length=10,
            standard_deviation=0.75,
        ),
        run._models.spectrogram_model.wrapper.SpectrogramModelWrapper: cf.Args(
            # NOTE: The spectrogram values range from -50 to 50. Thie scalar rescales the
            # spectrogram to a more reasonable range for deep learning. This also ensures that
            # the output respects the minimum bound.
            output_scalar=50.0
        ),
    }
    cf.add(config, overwrite)
