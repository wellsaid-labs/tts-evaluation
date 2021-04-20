import copy
import dataclasses
import enum
import logging
import math
import pprint
import typing

import torch
import torch.nn
from hparams import HParams, add_config, configurable
from third_party import LazyLoader

import lib
import run
import run.data._loader.utils
from run.data import _loader
from run.data._loader import DATASETS, Passage, Span, Speaker

if typing.TYPE_CHECKING:  # pragma: no cover
    import IPython
    import IPython.display
    import librosa
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    IPython = LazyLoader("IPython", globals(), "IPython")

logger = logging.getLogger(__name__)
pprinter = pprint.PrettyPrinter(indent=4)

RANDOM_SEED = 1212212
PHONEME_SEPARATOR = "|"
DATASETS = copy.copy(DATASETS)
del DATASETS[_loader.ELLIOT_MILLER]  # NOTE: Elliot has unannotated character portrayals.
del DATASETS[_loader.ELIZABETH_KLETT]  # NOTE: Elizabeth has unannotated character portrayals.

# NOTE: It's theoretically impossible to know all the phonemes eSpeak might predict because
# the predictions vary with context. We cannot practically generate every possible permutation
# to generate the vocab.
# TODO: Remove this once `grapheme_to_phoneme` is deprecated.
# fmt: off
DATASET_PHONETIC_CHARACTERS = [
    '\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '[', ']', '=', 'aɪ',
    'aɪə', 'aɪɚ', 'aɪʊ', 'aɪʊɹ', 'aʊ', 'b', 'd', 'dʒ', 'eɪ', 'f', 'h', 'i', 'iə', 'iː', 'j',
    'k', 'l', 'm', 'n', 'nʲ', 'n̩', 'oʊ', 'oː', 'oːɹ', 'p', 'r', 's', 't', 'tʃ', 'uː', 'v', 'w',
    'x', 'z', 'æ', 'æː', 'ð', 'ø', 'ŋ', 'ɐ', 'ɐː', 'ɑː', 'ɑːɹ', 'ɑ̃', 'ɔ', 'ɔɪ', 'ɔː', 'ɔːɹ',
    'ə', 'əl', 'ɚ', 'ɛ', 'ɛɹ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɪɹ', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ',
    'ʒ', 'ʔ', 'ˈ', 'ˌ', 'θ', 'ᵻ', 'ɬ'
]
# fmt: on

TTS_DISK_CACHE_NAME = ".tts_cache"  # NOTE: Hidden directory stored in other directories for caching
DISK_PATH = lib.environment.ROOT_PATH / "disk"
DATA_PATH = DISK_PATH / "data"
EXPERIMENTS_PATH = DISK_PATH / "experiments"
TEMP_PATH = DISK_PATH / "temp"
SAMPLES_PATH = DISK_PATH / "samples"
SIGNAL_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "signal_model"
SPECTROGRAM_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "spectrogram_model"

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
FRAME_SIZE = 2048  # NOTE: Frame size in samples.
FFT_LENGTH = 2048
assert FRAME_SIZE % 4 == 0
FRAME_HOP = FRAME_SIZE // 4


class Cadence(enum.Enum):
    STEP: typing.Final = "step"
    MULTI_STEP: typing.Final = "multi_step"
    RUN: typing.Final = "run"  # NOTE: Measures statistic over the course of a "run"
    STATIC: typing.Final = "static"


class DatasetType(enum.Enum):
    TRAIN: typing.Final = "train"
    DEV: typing.Final = "dev"


class Device(enum.Enum):
    CUDA: typing.Final = "cuda"
    CPU: typing.Final = "cpu"


Label = typing.NewType("Label", str)
Dataset = typing.Dict[Speaker, typing.List[Passage]]


class GetLabel(typing.Protocol):
    def __call__(self, **kwargs) -> Label:
        ...


def _label(template: str, *args, **kwargs) -> Label:
    """Format `template` recursively, and return a `Label`.

    TODO: For recursive formatting, don't reuse arguments.
    """
    while True:
        formatted = template.format(*args, **kwargs)
        if formatted == template:
            return Label(formatted)
        template = formatted


def get_dataset_label(
    name: str,
    cadence: Cadence,
    type_: DatasetType,
    speaker: typing.Optional[Speaker] = None,
    **kwargs,
) -> Label:
    """ Label something related to a dataset. """
    kwargs = dict(cadence=cadence.value, type=type_.value, name=name, **kwargs)
    if speaker is None:
        return _label("{cadence}/dataset/{type}/{name}", **kwargs)
    return _label("{cadence}/dataset/{type}/{speaker}/{name}", speaker=speaker.label, **kwargs)


def get_model_label(
    name: str, cadence: Cadence, speaker: typing.Optional[Speaker] = None, **kwargs
) -> Label:
    """ Label something related to the model. """
    kwargs = dict(cadence=cadence.value, name=name, **kwargs)
    if speaker is None:
        return _label("{cadence}/model/{name}", **kwargs)
    return _label("{cadence}/model/{speaker}/{name}", speaker=speaker.label, **kwargs)


def get_config_label(name: str, cadence: Cadence = Cadence.STATIC, **kwargs) -> Label:
    """ Label something related to a configuration. """
    return _label("{cadence}/config/{name}", cadence=cadence.value, name=name, **kwargs)


def get_environment_label(name: str, cadence: Cadence = Cadence.STATIC, **kwargs) -> Label:
    """ Label something related to a environment. """
    return _label("{cadence}/environment/{name}", cadence=cadence.value, name=name, **kwargs)


def get_timer_label(
    name: str, device: Device = Device.CPU, cadence: Cadence = Cadence.STATIC, **kwargs
) -> Label:
    """ Label something related to a performance. """
    template = "{cadence}/timer/{device}/{name}"
    return _label(template, cadence=cadence.value, device=device.value, name=name, **kwargs)


def configurable_(func: typing.Callable):
    """`configurable` has issues if it's run twice, on the same function.

    TODO: Remove this, once, this issue is resolved: https://github.com/PetrochukM/HParams/issues/8
    """
    if not hasattr(func, "_configurable"):
        return configurable(func)
    return func


try:
    librosa.effects.trim = configurable_(librosa.effects.trim)
except ImportError:
    logger.info("Ignoring optional `librosa` configurations.")

try:
    IPython.display.Audio.__init__ = configurable_(IPython.display.Audio.__init__)
except ImportError:
    logger.info("Ignoring optional `IPython` configurations.")

torch.nn.modules.batchnorm._BatchNorm.__init__ = configurable_(
    torch.nn.modules.batchnorm._BatchNorm.__init__
)
torch.nn.LayerNorm.__init__ = configurable_(torch.nn.LayerNorm.__init__)


def _configure_audio_processing():
    """ Configure modules that process audio. """
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

    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    try:
        window = run._utils.get_window("hann", FRAME_SIZE, FRAME_HOP)
        config = {
            lib.audio.power_spectrogram_to_framed_rms: HParams(window=window),
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

    try:
        add_config({IPython.display.Audio.__init__: HParams(rate=format_.sample_rate)})
    except ImportError:
        logger.info("Ignoring optional `IPython` configurations.")

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
            # `dataset_dashboard`. With a sample size of 8192, we found that 0.18 frames per token
            # included everything but 3 alignments. The last three alignments were 0.19 "or",
            # 0.21 "or", and 0.24 "EEOC". The slowest alignment was the acronym "EEOC" with the
            # last letter taking 0.5 seconds.
            max_frames_per_token=(0.18 / (FRAME_HOP / format_.sample_rate)),
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
        run.data._loader.utils._cache_path: HParams(cache_dir=TTS_DISK_CACHE_NAME),
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
        # NOTE: Today pauses longer than one second are not used for emphasis or meaning; however,
        # Otis does tend to use long pauses for emphasis; however, he rarely pauses for longer than
        # one second.
        run.data._loader.utils.SpanGenerator.__init__: HParams(max_pause=1.0),
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
    }
    add_config(config)


def _configure_models():
    """ Configure spectrogram and signal model. """
    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    encoder_output_size = 128

    # SOURCE (Tacotron 2):
    # Specifically, generation completes at the first frame for which this
    # probability exceeds a threshold of 0.5.
    stop_threshold = 0.5

    # NOTE: Configure the model sizes.
    config = {
        lib.spectrogram_model.encoder.Encoder.__init__: HParams(
            # SOURCE (Tacotron 2):
            # Input characters are represented using a learned 512-dimensional character embedding
            # ...
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            # ...
            # The output of the final convolutional layer is passed into a single bi-directional
            # [19] LSTM [20] layer containing 512 units (256) in each direction) to generate the
            # encoded features.
            hidden_size=512,
            # SOURCE (Tacotron 2):
            # which are passed through a stack of 3 convolutional layers each containing
            # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
            num_convolution_layers=3,
            convolution_filter_size=5,
            # SOURCE (Tacotron 2)
            # The output of the final convolutional layer is passed into a single
            # bi-directional [19] LSTM [20] layer containing 512 units (256) in each
            # direction) to generate the encoded features.
            lstm_layers=1,
            out_size=encoder_output_size,
        ),
        lib.spectrogram_model.attention.Attention.__init__: HParams(
            # SOURCE (Tacotron 2):
            # Location features are computed using 32 1-D convolution filters of length 31.
            # SOURCE (Tacotron 2):
            # Attention probabilities are computed after projecting inputs and location
            # features to 128-dimensional hidden representations.
            hidden_size=128,
            convolution_filter_size=31,
            # NOTE: The alignment between text and speech is monotonic; therefore, the attention
            # progression should reflect that. The `window_length` ensures the progression is
            # limited.
            # NOTE: Comet visualizes the metric "attention_std", and this metric represents the
            # number of characters the model is attending too at a time. That metric can be used
            # to set the `window_length`.
            window_length=9,
        ),
        lib.spectrogram_model.decoder.Decoder.__init__: HParams(
            encoder_output_size=encoder_output_size,
            # SOURCE (Tacotron 2):
            # The prediction from the previous time step is first passed through a small
            # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
            pre_net_size=256,
            # SOURCE (Tacotron 2):
            # The prenet output and attention context vector are concatenated and
            # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
            lstm_hidden_size=1024,
        ),
        lib.spectrogram_model.pre_net.PreNet.__init__: HParams(
            # SOURCE (Tacotron 2):
            # The prediction from the previous time step is first passed through a small
            # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
            num_layers=2
        ),
        lib.spectrogram_model.SpectrogramModel.__init__: HParams(
            num_frame_channels=NUM_FRAME_CHANNELS,
            # SOURCE (Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech
            #         Synthesis):
            # The paper mentions their proposed model uses a 256 dimension embedding.
            # NOTE: See https://github.com/wellsaid-labs/Text-to-Speech/pull/258 to learn more about
            # this parameter.
            speaker_embedding_size=128,
        ),
        lib.signal_model.SignalModel.__init__: HParams(
            input_size=NUM_FRAME_CHANNELS, hidden_size=32, max_channel_size=512
        ),
        # NOTE: We found this hidden size to be effective on Comet in April 2020.
        lib.signal_model.SpectrogramDiscriminator.__init__: HParams(hidden_size=512),
    }
    add_config(config)

    # NOTE: Configure the model regularization.
    config = {
        # SOURCE (Tacotron 2):
        # In order to introduce output variation at inference time, dropout with probability 0.5 is
        # applied only to layers in the pre-net of the autoregressive decoder.
        lib.spectrogram_model.pre_net.PreNet.__init__: HParams(dropout=0.5),
        # NOTE: This dropout approach proved effective in Comet in March 2020.
        lib.spectrogram_model.SpectrogramModel.__init__: HParams(speaker_embed_dropout=0.1),
        lib.spectrogram_model.attention.Attention.__init__: HParams(dropout=0.1),
        lib.spectrogram_model.decoder.Decoder.__init__: HParams(stop_net_dropout=0.5),
        lib.spectrogram_model.encoder.Encoder.__init__: HParams(dropout=0.1),
    }
    add_config(config)

    config = {
        # NOTE: Window size smoothing parameter is not sensitive.
        lib.optimizers.AdaptiveGradientNormClipper.__init__: HParams(window_size=128, norm_type=2),
        # NOTE: The `beta` parameter is not sensitive.
        lib.optimizers.ExponentialMovingParameterAverage.__init__: HParams(beta=0.9999),
        lib.signal_model.SignalModel.__init__: HParams(
            # SOURCE https://en.wikipedia.org/wiki/%CE%9C-law_algorithm:
            # For a given input x, the equation for μ-law encoding is where μ = 255 in the North
            # American and Japanese standards.
            mu=255,
        ),
        lib.spectrogram_model.SpectrogramModel.__init__: HParams(
            # NOTE: The spectrogram values range from -50 to 50. Thie scalar rescales the
            # spectrogram to a more reasonable range for deep learning.
            output_scalar=10.0,
            stop_threshold=stop_threshold,
        ),
    }
    add_config(config)

    # NOTE: PyTorch and Tensorflow parameterize `BatchNorm` differently, learn more:
    # https://stackoverflow.com/questions/48345857/batchnorm-momentum-convention-pytorch?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    config = {
        # NOTE: `momentum=0.01` to match Tensorflow defaults.
        torch.nn.modules.batchnorm._BatchNorm.__init__: HParams(momentum=0.01),
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        torch.nn.LayerNorm.__init__: HParams(eps=1e-12),
    }
    add_config(config)


def _include_passage(passage: Passage) -> bool:
    """Return `True` iff `passage` should be included in the dataset."""
    repr_ = f"{passage.__class__.__name__}("
    repr_ += f"{passage.audio_file.path.relative_to(DATA_PATH)},"
    repr_ += f" {(passage.script[:50] + '...') if len(passage.script) > 50 else passage.script})"

    if len(passage.alignments) == 0:
        logger.warning("%s has zero alignments.", repr_)
        return False

    if len(passage.speech_segments) == 0:
        logger.warning("%s has zero speech segments.", repr_)
        return False

    span = passage[:]
    if span.audio_length == 0.0:
        logger.warning("%s has no aligned audio.", repr_)
        return False

    if len(span.script) == 0:
        logger.warning("%s has no aligned text.", repr_)
        return False

    # NOTE: Filter out passages(s) that don't have a lower case character because it'll make
    # it difficult to classify initialisms.
    if not any(c.islower() for c in passage.script):
        return False

    # NOTE: Filter out Midnight Passenger because it has an inconsistent acoustic setup compared to
    # other samples from the same speaker.
    # NOTE: Filter out the North & South book because it uses English in a way that's not consistent
    # with editor usage, for example: "To-morrow, you will-- Come back to-night, John!"
    books = (_loader.m_ailabs.MIDNIGHT_PASSENGER, _loader.m_ailabs.NORTH_AND_SOUTH)
    metadata = passage.other_metadata
    if metadata is not None and "books" in metadata and (metadata["books"] in books):
        return False

    return True


def _include_span(span: Span):
    """Return `True` iff `span` should be included in the dataset."""
    if "<" in span.script or ">" in span.script:
        return False

    # NOTE: Filter out any passage(s) with a slash because it's ambigious. It's not obvious if
    # it should be silent or verbalized.
    if "/" in span.script or "\\" in span.script:
        return False

    # NOTE: Filter out any passage(s) with digits because the pronunciation is fundamentally
    # ambigious, and it's much easier to handle this case with text normalization.
    # NOTE: See performance statistics here: https://stackoverflow.com/a/31861306/4804936
    if lib.text.has_digit(span.script):
        return False

    # NOTE: `Span`s which end with a short, or fast `Span`, tend to be error prone.
    is_not_aligned = lambda s: s.audio_length < 0.2 or (s.audio_length / len(s.script)) < 0.04
    if is_not_aligned(span[0]) or is_not_aligned(span[-1]):
        return False

    if _loader.has_a_mistranscription(span):
        return False

    return True


def _configure_data_processing():
    """Configure modules that process data, other than audio.

    TODO: Remove `BETH_CAMERON__CUSTOM` from the `WSL_DATASETS` groups because it has it's own
    custom script.
    """
    dev_speakers = run.data._loader.WSL_DATASETS.copy()
    # NOTE: The `MARI_MONGE__PROMO` dataset is too short for evaluation, at 15 minutes long.
    del dev_speakers[run.data._loader.MARI_MONGE__PROMO]
    # NOTE: The `ALICIA_HARRIS`, `JACK_RUTKOWSKI`, and `SAM_SCHOLL` datasets are duplicate datasets.
    # There is an improved version of their datasets already in `dev_speakers`.
    del dev_speakers[run.data._loader.ALICIA_HARRIS]
    del dev_speakers[run.data._loader.JACK_RUTKOWSKI]
    del dev_speakers[run.data._loader.SAM_SCHOLL]
    # NOTE: The `BETH_CAMERON__CUSTOM` dataset isn't included in the studio.
    del dev_speakers[run.data._loader.BETH_CAMERON__CUSTOM]
    dev_speakers = set(dev_speakers.keys())
    groups = [set(_loader.WSL_DATASETS.keys())]
    # NOTE: For other datasets like M-AILABS and LJ, this assumes that there is no duplication
    # between different speakers.
    groups += [{s} for s in _loader.DATASETS.keys() if s not in _loader.WSL_DATASETS]
    config = {
        lib.text.grapheme_to_phoneme: HParams(separator=PHONEME_SEPARATOR),
        run._utils.get_dataset: HParams(
            datasets=DATASETS,
            path=DATA_PATH,
            include_passage=_include_passage,
            handle_passage=lib.utils.identity,
        ),
        run._utils.split_dataset: HParams(
            groups=groups, dev_speakers=dev_speakers, approx_dev_len=30 * 60, min_sim=0.9
        ),
        run._utils.SpanGenerator.__init__: HParams(max_seconds=15, include_span=_include_span),
    }
    add_config(config)


def configure():
    """ Configure modules required for `run`. """
    for directory in [
        DISK_PATH,
        DATA_PATH,
        EXPERIMENTS_PATH,
        TEMP_PATH,
        SAMPLES_PATH,
        SIGNAL_MODEL_EXPERIMENTS_PATH,
        SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    ]:
        directory.mkdir(exist_ok=True)

    _configure_audio_processing()
    _configure_models()
    _configure_data_processing()
