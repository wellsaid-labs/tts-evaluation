import dataclasses
import logging
import math
import typing

from hparams import HParams, add_config
from third_party import LazyLoader

import lib
import run

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
else:
    librosa = LazyLoader("librosa", globals(), "librosa")

logger = logging.getLogger(__name__)


# TODO: Instead of using global variables, can we use `hparams`, easily?

# SOURCE (Tacotron 1):
# We use 24 kHz sampling rate for all experiments.
SAMPLE_RATE = 24000

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


def configure():
    """Configure modules that process audio."""
    # NOTE: The SoX and FFmpeg encodings are the same.
    # NOTE: The signal model output is 32-bit.
    suffix = ".wav"
    data_type = lib.audio.AudioDataType.FLOATING_POINT
    bits = 32
    format_ = lib.audio.AudioFormat(
        sample_rate=SAMPLE_RATE,
        num_channels=1,  # NOTE: The signal model output is 1-channel, similar to Tacotron-2.
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
    )
    non_speech_segment_frame_length = 50
    max_frames_per_token = 0.18 / (FRAME_HOP / format_.sample_rate)
    # NOTE: Today pauses longer than one second are not used for emphasis or meaning; however,
    # Otis does tend to use long pauses for emphasis; however, he rarely pauses for longer than
    # one second.
    too_long_pause_length = 1.0

    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    try:
        window = run._utils.get_window("hann", FRAME_SIZE, FRAME_HOP)
        config = {
            lib.audio.power_spectrogram_to_framed_rms: HParams(
                window=window,
                window_correction_factor=lib.audio.get_window_correction_factor(window),
            ),
            lib.audio.SignalTodBMelSpectrogram.__init__: HParams(window=window),
            lib.audio.griffin_lim: HParams(window=window.numpy()),
        }
        add_config(config)
    except ImportError:
        logger.info("Ignoring optional `scipy` and `librosa` configurations.")

    # NOTE: The human range is commonly given as 20 to 20,000 Hz
    # (https://en.wikipedia.org/wiki/Hearing_range).
    hertz_bounds = {"lower_hertz": 20, "upper_hertz": 20000}

    try:
        config = {
            librosa.effects.trim: HParams(frame_length=FRAME_SIZE, hop_length=FRAME_HOP),
        }
        add_config(config)
    except ImportError:
        logger.info("Ignoring optional `librosa` configurations.")

    config = {
        lib.visualize.plot_waveform: HParams(sample_rate=format_.sample_rate),
        lib.visualize.plot_spectrogram: HParams(
            sample_rate=format_.sample_rate, frame_hop=FRAME_HOP
        ),
        lib.visualize.plot_mel_spectrogram: HParams(**hertz_bounds),
        lib.audio.write_audio: HParams(sample_rate=format_.sample_rate),
        lib.audio.pad_remainder: HParams(multiple=FRAME_HOP, mode="constant", constant_values=0.0),
        lib.audio.signal_to_framed_rms: HParams(frame_length=FRAME_SIZE, hop_length=FRAME_HOP),
        lib.audio.SignalTodBMelSpectrogram.__init__: HParams(
            fft_length=FFT_LENGTH,
            frame_hop=FRAME_HOP,
            sample_rate=format_.sample_rate,
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
            **hertz_bounds,
        ),
        lib.audio.griffin_lim: HParams(
            sample_rate=format_.sample_rate,
            fft_length=FFT_LENGTH,
            frame_hop=FRAME_HOP,
            # SOURCE (Tacotron 1):
            # We found that raising the predicted magnitudes by a power of 1.2 before feeding to
            # Griffin-Lim reduces artifacts
            power=1.20,
            # SOURCE (Tacotron 1):
            # We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
            # iterations seems to be enough), which is reasonably fast.
            iterations=30,
            get_weighting=lib.audio.iso226_weighting,
            **hertz_bounds,
        ),
        # NOTE: The `DeMan` loudness implementation of ITU-R BS.1770 is sample rate independent.
        lib.audio.get_pyloudnorm_meter: HParams(filter_class="DeMan"),
        lib.spectrogram_model.SpectrogramModel.__init__: HParams(
            # NOTE: This is based on one of the slowest legitimate alignments in
            # `dataset_dashboard`. With a sample size of 8192, we found that 0.18 seconds per token
            # included everything but 3 alignments. The last three alignments were 0.19 "or",
            # 0.21 "or", and 0.24 "EEOC". The slowest alignment was the acronym "EEOC" with the
            # last letter taking 0.5 seconds.
            max_frames_per_token=max_frames_per_token,
        ),
        run.train.spectrogram_model._metrics.get_num_repeated: HParams(
            threshold=max_frames_per_token
        ),
        run.train.spectrogram_model._metrics.get_num_pause_frames: HParams(
            frame_hop=FRAME_HOP,
            sample_rate=format_.sample_rate,
            min_length=too_long_pause_length,
            max_loudness=-50,
        ),
        lib.signal_model.SignalModel.__init__: HParams(
            ratios=[2] * int(math.log2(FRAME_HOP)),
        ),
        run.data._loader.utils.normalize_audio_suffix: HParams(suffix=suffix),
        run.data._loader.utils.normalize_audio: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            sample_rate=format_.sample_rate,
            num_channels=format_.num_channels,
        ),
        run.data._loader.utils.is_normalized_audio_file: HParams(
            audio_format=format_, suffix=suffix
        ),
        # NOTE: `get_non_speech_segments` parameters are set based on `vad_workbook.py`. They
        # are applicable to most datasets with little to no noise.
        run.data._loader.utils.get_non_speech_segments_and_cache: HParams(
            low_cut=300, frame_length=non_speech_segment_frame_length, hop_length=5, threshold=-60
        ),
        run.data._loader.data_structures._make_speech_segments_helper: HParams(
            pad=lib.audio.milli_to_sec(non_speech_segment_frame_length / 2)
        ),
        run.data._loader.utils.maybe_normalize_audio_and_cache: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            **{f.name: getattr(format_, f.name) for f in dataclasses.fields(format_)},
        ),
        run.data._loader.utils.SpanGenerator.__init__: HParams(max_pause=too_long_pause_length),
        # NOTE: A 0.400 `block_size` is standard for ITU-R BS.1770.
        run.train.spectrogram_model._data._get_loudness: HParams(block_size=0.400, precision=0),
        run.train.spectrogram_model._data._random_loudness_annotations: HParams(max_annotations=10),
        run.train.spectrogram_model._data._random_speed_annotations: HParams(
            max_annotations=10, precision=2
        ),
        run.train.spectrogram_model._data._make_stop_token: HParams(
            # NOTE: The stop token uncertainty was approximated by a fully trained model that
            # learned the stop token distribution. The distribution looked like a gradual increase
            # over 4 - 8 frames in January 2020, on Comet.
            # NOTE: This was rounded up to 10 after the spectrograms length was increased by 17%
            # on average.
            # TODO: In July 2020, the spectrogram size was decreased by 2x, we should test
            # decreasing `length` by 2x, also.
            length=10,
            standard_deviation=2,
        ),
        run._tts.text_to_speech_ffmpeg_generator: HParams(sample_rate=format_.sample_rate),
    }
    add_config(config)
