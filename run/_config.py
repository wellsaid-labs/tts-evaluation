import collections
import copy
import enum
import functools
import logging
import math
import multiprocessing
import pathlib
import pprint
import random
import re
import typing

import torch
import torch.nn
import tqdm
from hparams import HParams, add_config, configurable
from Levenshtein.StringMatcher import StringMatcher
from third_party import LazyLoader
from torchnlp.random import fork_rng

import lib
import lib.datasets.m_ailabs
import run
from lib import datasets
from lib.datasets import DATASETS, WSL_DATASETS, Speaker

if typing.TYPE_CHECKING:  # pragma: no cover
    import IPython
    import IPython.display
    import librosa
    import scipy
    import scipy.signal
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    scipy = LazyLoader("scipy", globals(), "scipy")
    IPython = LazyLoader("IPython", globals(), "IPython")

logger = logging.getLogger(__name__)
pprinter = pprint.PrettyPrinter(indent=4)

RANDOM_SEED = 1212212
# SOURCE (Tacotron 1):
# We use 24 kHz sampling rate for all experiments.
SAMPLE_RATE = 24000
# SOURCE (Tacotron 2):
# We transform the STFT magnitude to the mel scale using an 80 channel mel filterbank spanning
# 125 Hz to 7.6 kHz, followed by log dynamic range compression.
# SOURCE (Tacotron 2 Author):
# Google mentioned they settled on [20, 12000] with 128 filters in Google Chat.
NUM_FRAME_CHANNELS = 128
PHONEME_SEPARATOR = "|"
DATASETS = copy.copy(DATASETS)
# NOTE: Elliot Miller is not included due to his unannotated character portrayals.
del DATASETS[datasets.ELLIOT_MILLER]

TTS_DISK_CACHE_NAME = ".tts_cache"  # NOTE: Hidden directory stored in other directories for caching
DISK_PATH = lib.environment.ROOT_PATH / "disk"
DATA_PATH = DISK_PATH / "data"
EXPERIMENTS_PATH = DISK_PATH / "experiments"
TEMP_PATH = DISK_PATH / "temp"
SAMPLES_PATH = DISK_PATH / "samples"
SIGNAL_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "signal_model"
SPECTROGRAM_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "spectrogram_model"
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


class Cadence(enum.Enum):
    STEP: typing.Final = "step"
    MULTI_STEP: typing.Final = "multi_step"
    RUN: typing.Final = "run"  # NOTE: Measures statistic over the course of a "run"
    STATIC: typing.Final = "static"


class DatasetType(enum.Enum):
    TRAIN: typing.Final = "train"
    DEV: typing.Final = "dev"


Label = typing.NewType("Label", str)
Dataset = typing.Dict[Speaker, typing.List[datasets.Passage]]


def get_dataset_label(
    name: str, cadence: Cadence, type_: DatasetType, speaker: typing.Optional[Speaker] = None
) -> Label:
    """ Label something related to a dataset. """
    kwargs = dict(cadence=cadence.value, type=type_.value, name=name)
    if speaker is None:
        return Label("{cadence}/dataset/{type}/{name}".format(**kwargs))
    label = "{cadence}/dataset/{type}/{speaker}/{name}"
    return Label(label.format(speaker=speaker.label, **kwargs))


def get_model_label(name: str, cadence: Cadence, speaker: typing.Optional[Speaker] = None) -> Label:
    """ Label something related to the model. """
    kwargs = dict(cadence=cadence.value, name=name)
    if speaker is None:
        return Label("{cadence}/model/{name}".format(**kwargs))
    return Label("{cadence}/model/{speaker}/{name}".format(speaker=speaker.label, **kwargs))


def get_config_label(name: str, cadence: Cadence = Cadence.STATIC) -> Label:
    """ Label something related to a configuration. """
    return Label("{cadence}/config/{name}".format(cadence=cadence.value, name=name))


def get_environment_label(name: str, cadence: Cadence = Cadence.STATIC) -> Label:
    """ Label something related to a environment. """
    return Label("{cadence}/environment/{name}".format(cadence=cadence.value, name=name))


def _get_window(window: str, window_length: int, window_hop: int) -> torch.Tensor:
    """Get a `torch.Tensor` window that passes `scipy.signal.check_COLA`.

    NOTE: `torch.hann_window` does not pass `scipy.signal.check_COLA`, for example. Learn more:
    https://github.com/pytorch/audio/issues/452
    """
    window = librosa.filters.get_window(window, window_length)
    assert scipy.signal.check_COLA(window, window_length, window_length - window_hop)
    return torch.tensor(window).float()


def configure_audio_processing():
    """ Configure modules that process audio. """
    num_channels = 1  # NOTE: The signal model output is 1-channel, similar to Tacotron-2.
    # NOTE: The SoX and FFmpeg encodings are the same.
    # NOTE: The signal model output is 32-bit.
    sox_encoding = "32-bit Floating Point PCM"
    ffmpeg_encoding = "pcm_f32le"
    suffix = ".wav"

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
    frame_size = 2048  # NOTE: Frame size in samples.
    fft_length = 2048
    assert frame_size % 4 == 0
    frame_hop = frame_size // 4

    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    try:
        window = _get_window("hann", frame_size, frame_hop)
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
        librosa.effects.trim = configurable(librosa.effects.trim)
        config = {
            librosa.effects.trim: HParams(frame_length=frame_size, hop_length=frame_hop),
        }
        add_config(config)
    except ImportError:
        logger.info("Ignoring optional `librosa` configurations.")

    try:
        IPython.display.Audio.__init__ = configurable(IPython.display.Audio.__init__)
        add_config({IPython.display.Audio.__init__: HParams(rate=SAMPLE_RATE)})
    except ImportError:
        logger.info("Ignoring optional `IPython` configurations.")

    config = {
        lib.audio.seconds_to_samples: HParams(sample_rate=SAMPLE_RATE),
        lib.audio.samples_to_seconds: HParams(sample_rate=SAMPLE_RATE),
        lib.visualize.plot_waveform: HParams(sample_rate=SAMPLE_RATE),
        lib.visualize.plot_spectrogram: HParams(sample_rate=SAMPLE_RATE, frame_hop=frame_hop),
        lib.visualize.plot_mel_spectrogram: HParams(**hertz_bounds),
        lib.audio.write_audio: HParams(sample_rate=SAMPLE_RATE),
        lib.audio.normalize_audio: HParams(
            suffix=suffix,
            encoding=ffmpeg_encoding,
            sample_rate=SAMPLE_RATE,
            num_channels=num_channels,
            audio_filters=lib.audio.AudioFilters(""),
        ),
        lib.audio.normalize_suffix: HParams(suffix=suffix),
        lib.audio.assert_audio_normalized: HParams(
            suffix=suffix, encoding=sox_encoding, sample_rate=SAMPLE_RATE, num_channels=num_channels
        ),
        lib.audio.pad_remainder: HParams(multiple=frame_hop, mode="constant", constant_values=0.0),
        lib.audio.signal_to_framed_rms: HParams(frame_length=frame_size, hop_length=frame_hop),
        lib.audio.SignalTodBMelSpectrogram.__init__: HParams(
            fft_length=fft_length,
            frame_hop=frame_hop,
            sample_rate=SAMPLE_RATE,
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
            sample_rate=SAMPLE_RATE,
            fft_length=fft_length,
            frame_hop=frame_hop,
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
        lib.audio.get_pyloudnorm_meter: HParams(sample_rate=SAMPLE_RATE, filter_class="DeMan"),
        lib.spectrogram_model.SpectrogramModel.__init__: HParams(
            # NOTE: This is based on one of the slowest legitimate alignments in
            # `dataset_dashboard`. With a sample size of 8192, we found that 0.18 frames per token
            # included everything but 3 alignments. The last three alignments were 0.19 "or",
            # 0.21 "or", and 0.24 "EEOC". The slowest alignment was the acronym "EEOC" with the
            # last letter taking 0.5 seconds.
            max_frames_per_token=(0.18 / (frame_hop / SAMPLE_RATE)),
        ),
        lib.signal_model.SignalModel.__init__: HParams(
            ratios=[2] * int(math.log2(frame_hop)),
        ),
        # NOTE: A 0.400 `block_size` is standard for ITU-R BS.1770.
        run._spectrogram_model._get_loudness: HParams(block_size=0.400, precision=0),
        run._spectrogram_model._random_loudness_annotations: HParams(max_annotations=10),
        run._spectrogram_model._random_speed_annotations: HParams(max_annotations=10, precision=2),
        run._spectrogram_model._make_stop_token: HParams(
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


def configure_models():
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
        lib.spectrogram_model.attention.LocationRelativeAttention.__init__: HParams(
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
        lib.spectrogram_model.decoder.AutoregressiveDecoder.__init__: HParams(
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
        lib.spectrogram_model.attention.LocationRelativeAttention.__init__: HParams(dropout=0.1),
        lib.spectrogram_model.decoder.AutoregressiveDecoder.__init__: HParams(stop_net_dropout=0.5),
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
    torch.nn.modules.batchnorm._BatchNorm.__init__ = configurable(  # type: ignore
        torch.nn.modules.batchnorm._BatchNorm.__init__
    )
    torch.nn.LayerNorm.__init__ = configurable(torch.nn.LayerNorm.__init__)  # type: ignore
    config = {
        # NOTE: `momentum=0.01` to match Tensorflow defaults.
        torch.nn.modules.batchnorm._BatchNorm.__init__: HParams(momentum=0.01),
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        torch.nn.LayerNorm.__init__: HParams(eps=1e-12),
    }
    add_config(config)


def configure():
    """ Configure modules. """
    configure_audio_processing()
    configure_models()

    config = {lib.text.grapheme_to_phoneme: HParams(separator=PHONEME_SEPARATOR)}
    add_config(config)


def _include_passage(passage: datasets.Passage) -> bool:
    """Return `True` iff `passage` should be included in the dataset."""
    details = passage.to_string("audio_file", "script", "other_metadata")

    if len(passage.alignments) == 0:
        logger.warning("Passage (%s) has little to no alignments.", details)
        return False

    span = passage[:]
    if span.audio_length == 0.0:
        logger.warning("Passage (%s) has no aligned audio.", details)
        return False

    if len(span.script) == 0:
        logger.warning("Passage (%s) has no aligned text.", details)
        return False

    # NOTE: Filter out passages(s) that don't have a lower case character because it'll make
    # it difficult to classify initialisms.
    if not any(c.islower() for c in passage.script):
        return False

    # NOTE: Filter out Midnight Passenger because it has an inconsistent acoustic setup compared to
    # other samples from the same speaker.
    # NOTE: Filter out the North & South book because it uses English in a way that's not consistent
    # with editor usage, for example: "To-morrow, you will-- Come back to-night, John!"
    books = (datasets.m_ailabs.MIDNIGHT_PASSENGER, datasets.m_ailabs.NORTH_AND_SOUTH)
    metadata = passage.other_metadata
    if metadata is not None and "books" in metadata and (metadata["books"] in books):
        return False

    return True


def _handle_passage(passage: datasets.Passage) -> datasets.Passage:
    """Update and/or check `passage`."""
    if passage.speaker in set([datasets.JUDY_BIEBER, datasets.MARY_ANN, datasets.ELIZABETH_KLETT]):
        script = lib.text.normalize_vo_script(passage.script)
        return lib.datasets.update_conventional_passage_script(passage, script)
    assert lib.text.is_normalized_vo_script(passage.script)
    return passage


def get_dataset(
    datasets: typing.Dict[lib.datasets.Speaker, lib.datasets.DataLoader] = DATASETS,
    path: pathlib.Path = DATA_PATH,
) -> Dataset:
    """Define a TTS dataset.

    TODO: `normalize_audio` could be used replicate datasets with different audio processing.

    Args:
        datasets: Dictionary of datasets to load.
        path: Directory to cache the dataset.
    """
    logger.info("Loading dataset...")
    with multiprocessing.pool.ThreadPool() as pool:
        handle_passages = lambda l: [_handle_passage(p) for p in l if _include_passage(p)]
        load_data = lambda i: (i[0], handle_passages(i[1](path)))
        items = list(pool.map(load_data, datasets.items()))
    dataset = {k: v for k, v in items}
    dataset = run._utils.normalize_audio(dataset)
    return dataset


@functools.lru_cache(maxsize=None)
def _is_duplicate(a: str, b: str, min_similarity: float) -> bool:
    """Helper function for `split_dataset` used to judge string similarity."""
    matcher = StringMatcher(seq1=a, seq2=b)
    return (
        matcher.real_quick_ratio() > min_similarity
        and matcher.quick_ratio() > min_similarity
        and matcher.ratio() > min_similarity
    )


def _find_duplicate_passages(
    dev_scripts: typing.Set[str],
    passages: typing.List[lib.datasets.Passage],
    min_similarity: float,
) -> typing.Tuple[typing.List[lib.datasets.Passage], typing.List[lib.datasets.Passage]]:
    """Find passages in `passages` that are a duplicate of a passage in `dev_scripts`.

    Args:
        dev_scripts: Set of unique scripts.
        passages: Passages that may have a `script` thats already included in `dev_scripts`.
        minimum_similarity: From 0 - 1, this is the minimum similarity two scripts must have to be
          considered duplicates.
    """
    duplicates, rest = [], []
    for passage in passages:
        if passage.script in dev_scripts:
            duplicates.append(passage)
            continue

        length = len(duplicates)
        for dev_script in dev_scripts:
            if _is_duplicate(dev_script, passage.script, min_similarity):
                duplicates.append(passage)
                break

        if length == len(duplicates):
            rest.append(passage)

    return duplicates, rest


DEV_SPEAKERS = [
    lib.datasets.ADRIENNE_WALKER_HELLER,
    lib.datasets.ALICIA_HARRIS__MANUAL_POST,
    lib.datasets.BETH_CAMERON,
    lib.datasets.ELISE_RANDALL,
    lib.datasets.FRANK_BONACQUISTI,
    lib.datasets.GEORGE_DRAKE_JR,
    lib.datasets.HANUMAN_WELCH,
    lib.datasets.HEATHER_DOE,
    lib.datasets.HILARY_NORIEGA,
    lib.datasets.JACK_RUTKOWSKI__MANUAL_POST,
    lib.datasets.JOHN_HUNERLACH__NARRATION,
    lib.datasets.JOHN_HUNERLACH__RADIO,
    lib.datasets.MARK_ATHERLAY,
    lib.datasets.MEGAN_SINCLAIR,
    lib.datasets.SAM_SCHOLL__MANUAL_POST,
    lib.datasets.STEVEN_WAHLBERG,
    lib.datasets.SUSAN_MURPHY,
]


@lib.utils.log_runtime
def split_dataset(
    dataset: Dataset,
    dev_speakers: typing.Set[lib.datasets.Speaker] = set(DEV_SPEAKERS),
    approximate_dev_length: int = 30 * 60,
    min_similarity: float = 0.9,
    seed: int = 123,
) -> typing.Tuple[Dataset, Dataset]:
    """Split the dataset into a train set and development set.

    NOTE: The RNG state should never change; otherwise, the training and dev datasets may be
    different from experiment to experiment.

    NOTE: `len_` assumes that the amount of data in each passage can be estimated with
    `aligned_audio_length`. For example, if there was a long pause within a passage, this estimate
    wouldn't make sense.

    NOTE: Passages are split between the train and development set in groups. The groups are
    dictated by textual similarity. The result of this is that the text in the train and
    development sets is distinct.

    NOTE: Any duplicate data for a speaker, not in `dev_speakers` will be discarded.

    Args:
        ...
        dev_speakers: Speakers to include in the development set.
        approximate_dev_length: Number of seconds per speaker in the development dataset. The
            deduping algorithm may add extra items above the `approximate_dev_length`.
        ...
    """
    logger.info("Splitting `dataset`...")
    dev: typing.Dict[lib.datasets.Speaker, list] = collections.defaultdict(list)
    train: typing.Dict[lib.datasets.Speaker, list] = collections.defaultdict(list)
    dev_scripts: typing.Set[str] = set()
    len_ = lambda _passage: _passage.aligned_audio_length()
    sum_ = lambda _passages: sum([len_(p) for p in _passages])
    with fork_rng(seed=seed):
        iterator = list(sorted(dataset.items(), key=lambda i: (len(i[1]), i[0])))
        for speaker, passages in tqdm.tqdm(iterator):
            if speaker not in dev_speakers:
                train[speaker] = passages
                continue

            duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_similarity)
            dev[speaker].extend(duplicates)

            random.shuffle(rest)
            seconds = max(approximate_dev_length - sum_(dev[speaker]), 0)
            splits = tuple(lib.utils.split(rest, [seconds, math.inf], len_))
            dev[speaker].extend(splits[0])
            train[speaker].extend(splits[1])
            dev_scripts.update(d.script for d in dev[speaker])

            message = "The `dev` dataset is larger than the `train` dataset."
            assert sum_(train[speaker]) >= sum_(dev[speaker]), message
            assert sum_(train[speaker]) > 0, "The train dataset has no aligned audio data."
            assert sum_(dev[speaker]) > 0, "The train dataset has no aligned audio data."

        # NOTE: Run the deduping algorithm until there are no more duplicates.
        length = None
        while length is None or length != len(dev_scripts):
            logger.info("Rerunning until there are no more duplicates...")
            length = len(dev_scripts)
            for speaker, _ in iterator:
                duplicates, rest = _find_duplicate_passages(
                    dev_scripts, train[speaker], min_similarity
                )
                if speaker not in dev_speakers and len(duplicates) > 0:
                    message = "Discarded %d duplicates for non-dev speaker %s. "
                    logger.warning(message, len(duplicates), speaker)
                elif speaker in dev_speakers:
                    dev[speaker].extend(duplicates)
                train[speaker] = rest
                dev_scripts.update(d.script for d in duplicates)

    _is_duplicate.cache_clear()

    return dict(train), dict(dev)


DIGIT_REGEX = re.compile(r"\d")
ALPHANUMERIC_REGEX = re.compile(r"[a-zA-Z0-9]")


def _include_span(span: datasets.Span):
    """Return `True` iff `span` should be included in the dataset.

    TODO: The `span` is still cut-off sometimes, and it's difficult to detect if it is. Instead
    of cutting `span`s via `Alignment`s, we should cut `span`s based on pausing.
    """
    if "<" in span.script or ">" in span.script:
        return False

    # NOTE: Filter out any passage(s) with digits because the pronunciation is fundamentally
    # ambigious, and it's much easier to handle this case with text normalization.
    # NOTE: See performance statistics here: https://stackoverflow.com/a/31861306/4804936
    if DIGIT_REGEX.search(span.script):
        return False

    is_not_aligned = lambda s: s.audio_length < 0.2 or (s.audio_length / len(s.script)) < 0.04
    if is_not_aligned(span[0]) or is_not_aligned(span[-1]):
        return False

    if any(
        ALPHANUMERIC_REGEX.search(a) or ALPHANUMERIC_REGEX.search(b)
        for a, b, _ in span.script_nonalignments()
    ):
        return False

    return True


class SpanGenerator(typing.Iterator[datasets.Span]):
    """Define the dataset generator to train and evaluate the TTS models on.

    Args:
        dataset
        max_seconds: The maximum seconds delimited by an `Span`.
    """

    @lib.utils.log_runtime
    def __init__(self, dataset: Dataset, max_seconds: int = 15):
        self.max_seconds = max_seconds
        self.dataset = dataset
        self.generators: typing.Dict[lib.datasets.Speaker, typing.Iterator[lib.datasets.Span]] = {}
        for speaker, passages in dataset.items():
            # NOTE: Some datasets are pre-cut, and this conditional preserves their distribution.
            is_singles = all([len(p.alignments) == 1 for p in passages])
            max_seconds_ = math.inf if is_singles else max_seconds
            self.generators[speaker] = datasets.SpanGenerator(passages, max_seconds_)
        self.speakers = list(dataset.keys())
        self.counter = {s: 0.0 for s in self.speakers}

    def __iter__(self) -> typing.Iterator[datasets.Span]:
        return self

    def __next__(self) -> datasets.Span:
        while True:  # NOTE: This samples speakers uniformly.
            speaker = lib.utils.corrected_random_choice(self.counter)
            span = next(self.generators[speaker])
            if span.audio_length < self.max_seconds and _include_span(span):
                self.counter[span.speaker] += span.audio_length
                return span


# NOTE: It's theoretically impossible to know all the phonemes eSpeak might predict because
# the predictions vary with context. We cannot practically generate every possible permutation
# to generate the vocab.
# TODO: Remove this once `grapheme_to_phoneme` is deprecated.
# fmt: off
DATASET_PHONETIC_CHARACTERS = [
    '\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '[', ']', 'aɪ',
    'aɪə', 'aɪɚ', 'aɪʊ', 'aɪʊɹ', 'aʊ', 'b', 'd', 'dʒ', 'eɪ', 'f', 'h', 'i', 'iə', 'iː', 'j',
    'k', 'l', 'm', 'n', 'nʲ', 'n̩', 'oʊ', 'oː', 'oːɹ', 'p', 'r', 's', 't', 'tʃ', 'uː', 'v', 'w',
    'x', 'z', 'æ', 'æː', 'ð', 'ø', 'ŋ', 'ɐ', 'ɐː', 'ɑː', 'ɑːɹ', 'ɑ̃', 'ɔ', 'ɔɪ', 'ɔː', 'ɔːɹ',
    'ə', 'əl', 'ɚ', 'ɛ', 'ɛɹ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɪɹ', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ',
    'ʒ', 'ʔ', 'ˈ', 'ˌ', 'θ', 'ᵻ', 'ɬ'
]
# fmt: on
