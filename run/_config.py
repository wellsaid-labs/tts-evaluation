import enum
import logging
import math
import pprint
import random
import typing

import torch
from hparams import HParams, add_config, configurable
from third_party import LazyLoader
from torchnlp.random import fork_rng

import lib
import run
from lib.datasets import Speaker

if typing.TYPE_CHECKING:  # pragma: no cover
    import IPython
    import librosa
    import scipy
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
    assert directory.exists(), "Directory has not been made."

DATABASE_PATH = TEMP_PATH / "database.db"


class Context(enum.Enum):
    """ Constants and labels for contextualizing the use-case. """

    TRAIN: typing.Final = "train"
    EVALUATE: typing.Final = "evaluate"
    EVALUATE_INFERENCE: typing.Final = "evaluate_inference"


class Cadence(enum.Enum):
    STEP: typing.Final = "step"
    MULTI_STEP: typing.Final = "multi_step"
    STATIC: typing.Final = "static"


class DatasetType(enum.Enum):
    TRAIN: typing.Final = "train"
    DEV: typing.Final = "dev"


Label = typing.NewType("Label", str)


def get_dataset_label(
    name: str, cadence: Cadence, type_: DatasetType, speaker: typing.Optional[Speaker] = None
) -> Label:
    """ Label something related to a dataset. """
    kwargs = dict(cadence=cadence, type=type_, name=name)
    if speaker is None:
        return Label("{cadence}/dataset/{type}/{name}".format(**kwargs))
    speaker_ = lib.environment.text_to_label(speaker.name)
    return Label("{cadence}/dataset/{type}/{speaker}/{name}".format(speaker=speaker_, **kwargs))


def get_model_label(name: str, cadence: Cadence, speaker: typing.Optional[Speaker] = None) -> Label:
    """ Label something related to the model. """
    kwargs = dict(cadence=cadence, name=name)
    if speaker is None:
        return Label("{cadence}/model/{name}".format(**kwargs))
    speaker_ = lib.environment.text_to_label(speaker.name)
    return Label("{cadence}/model/{speaker}/{name}".format(speaker=speaker_, **kwargs))


def get_config_label(name: str, cadence: Cadence) -> Label:
    """ Label something related to a configuration. """
    return Label("{cadence}/config/{name}".format(cadence=cadence, name=name))


def _get_window(window: str, window_length: int, window_hop: int) -> torch.Tensor:
    """Get a `torch.Tensor` window that passes `scipy.signal.check_COLA`.

    NOTE: `torch.hann_window` does not pass `scipy.signal.check_COLA`, for example. Learn more:
    https://github.com/pytorch/audio/issues/452
    """
    window = librosa.filters.get_window(window, window_length)
    window_tensor = torch.tensor(window).float()
    assert scipy.signal.check_COLA(window, window_length, window_length - window_hop)
    return window_tensor


def configure_audio_processing():
    """ Configure modules that process audio. """
    channels = 1  # NOTE: The signal model output is 1-channel, similar to Tacotron-2.
    # NOTE: The SoX and FFmpeg encodings are the same.
    # NOTE: The signal model output is 32-bit.
    sox_encoding = "32-bit Floating Point PCM"
    ffmpeg_encoding = "pcm_f32le"
    ffmpeg_format = "f32le"
    loud_norm_audio_filter = run._utils.format_ffmpeg_audio_filter(
        "loudnorm",
        integrated_loudness=-21,
        loudness_range=4,
        true_peak=-6.1,
        print_format="summary",
    )

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
        lib.visualize.plot_waveform: HParams(sample_rate=SAMPLE_RATE),
        lib.visualize.plot_spectrogram: HParams(sample_rate=SAMPLE_RATE, frame_hop=frame_hop),
        lib.visualize.plot_mel_spectrogram: HParams(**hertz_bounds),
        lib.audio.write_audio: HParams(sample_rate=SAMPLE_RATE),
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
        lib.spectrogram_model.model.SpectrogramModel.__init__: HParams(
            # NOTE: This is based on one of the slowest legitimate example in the dataset:
            # "rate(WSL_SMurphyScript34-39,24000)/script_52_chunk_9.wav"
            # NOTE: This configuration is related to the dataset preprocessing step:
            # `_filter_too_much_audio_per_character` # TODO: Remove
            # NOTE: This number was configured with the help of this notebook:
            # `QA_Datasets/Sample_Dataset.ipynb`
            max_frames_per_token=(0.16 / (frame_hop / SAMPLE_RATE)),
        ),
        lib.signal_model.SignalModel.__init__: HParams(
            ratios=[2] * int(math.log2(frame_hop)),
        ),
        run._utils.normalize_audio: HParams(
            format=ffmpeg_format,
            encoding=ffmpeg_encoding,
            sample_rate=SAMPLE_RATE,
            channels=channels,
            get_audio_filters=lambda _: ",".join([loud_norm_audio_filter]),
        ),
        run._utils.assert_audio_normalized: HParams(
            encoding=sox_encoding, sample_rate=SAMPLE_RATE, channels=channels
        ),
        run._utils.get_spectrogram_example: HParams(
            loudness_implementation="DeMan",
            max_loudness_annotations=5,
            loudness_precision=0,
            max_speed_annotations=5,
            speed_precision=0,
            # NOTE: The stop token uncertainty was approximated by a fully trained model that
            # learned the stop token distribution. The distribution looked like a gradual increase
            # over 4 - 8 frames in January 2020, on Comet.
            # NOTE: This was rounded up to 10 after the spectrograms length was increased by 17%
            # on average.
            # TODO: In July 2020, the spectrogram size was decreased by 2x, we should test
            # decreasing `length` by 2x, also.
            stop_token_range=10,
            stop_token_standard_deviation=2,
            sample_rate=SAMPLE_RATE,
        ),
        run._utils.get_rms_level: HParams(
            frame_hop=frame_hop,
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
        lib.spectrogram_model.model.SpectrogramModel.__init__: HParams(
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
        lib.spectrogram_model.model.SpectrogramModel.__init__: HParams(speaker_embed_dropout=0.1),
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
        lib.spectrogram_model.model.SpectrogramModel.__init__: HParams(
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
    config = {
        lib.text.grapheme_to_phoneme: HParams(seperator=PHONEME_SEPARATOR),
        lib.environment.set_seed: HParams(seed=RANDOM_SEED),
    }
    add_config(config)


Dataset = typing.Dict[Speaker, typing.List[lib.datasets.Example]]


def get_dataset(dev_size: int = 60 * 60) -> typing.Tuple[Dataset, Dataset]:
    """Define the dataset to train and evaluate the TTS models on.

    NOTE: Elliot Miller is not included due to his unannotated character portrayals.

    Args:
        dev_size: Number of seconds per speaker in the development dataset.

    Returns:
        train
        dev
    """
    # NOTE: This `seed` should never change; otherwise, the training and dev datasets may be
    # different from experiment to experiment.
    with fork_rng(seed=123):
        logger.info("Loading dataset...")
        dev = {}
        train = {}

        iterator = [
            (lib.datasets.HILARY_NORIEGA, lib.datasets.hilary_noriega_speech_dataset()),
        ]
        for speaker, examples in iterator:
            train[speaker], dev[speaker] = run._utils.split_examples(examples, dev_size)

        #  NOTE: Elliot Miller is not included due to his unannotated character portrayals.
        train[lib.datasets.LINDA_JOHNSON] = lib.datasets.lj_speech_dataset()
        train[lib.datasets.JUDY_BIEBER] = lib.datasets.m_ailabs_en_us_judy_bieber_speech_dataset()
        train[lib.datasets.MARY_ANN] = lib.datasets.m_ailabs_en_us_mary_ann_speech_dataset()
        train[
            lib.datasets.ELIZABETH_KLETT
        ] = lib.datasets.m_ailabs_en_uk_elizabeth_klett_speech_dataset()

        return train, dev


def _include_example(example: lib.datasets.Example) -> bool:
    """Return `True` iff `example` should be included in the dataset.

    TODO: Potential update(s) to our example filters:
    - Use the speed of the example with phonemes or graphemes.
    - Look for gaps in alignments:
      - Is the gap a long pause?
      - Is the gap loud?
      - Does the gap have text?
    - Does the starting and ending alignment make sense? Or does it cut off?
    - Get an accurate prediction of audio length by removing silence.
    - Don't filter out an entire example if it has numbers, and slice the example first.
    - Should we filter out an entire example if there is a very long pause or a lot of words
      missing? We might have a hard time with singular slices but we might know if a script
      has a big screw up.
    """
    assert example.alignments is not None
    assert lib.text.is_normalized_vo_script(example.text)

    text = example.text[example.alignments[0].text[0] : example.alignments[-1].text[-1]]

    # NOTE: Filter any example(s) that are invalid or zero length.
    if (
        len(text) == 0
        or not example.audio_path.is_file()
        or example.alignments[0].audio[0] == example.alignments[-1].audio[-1]
    ):
        return False

    # NOTE: Filter out any example(s) with digits because the pronunciation is fundamentally
    # ambigious, and it's much easier to handle this case with text normalization.
    if any(c.isdigit() for c in text):
        return False

    # NOTE: Filter out example(s) that don't have a lower case character because it'll make
    # it difficult to classify initialisms.
    if not any(c.islower() for c in example.text):
        return False

    # NOTE: Filter out Midnight Passenger because it has an inconsistent acoustic setup compared to
    # other samples from the same speaker.
    # NOTE: Filter out the North & South book because it uses English in a way that's not consistent
    # with editor usage, for example: "To-morrow, you will-- Come back to-night, John!"
    if example.metadata is not None and (
        example.metadata["books"] == lib.datasets.m_ailabs.MIDNIGHT_PASSENGER
        or example.metadata["books"] == lib.datasets.m_ailabs.NORTH_AND_SOUTH
    ):
        return False

    return True


def get_dataset_generator(
    dataset: Dataset,
    include_example: typing.Callable[[lib.datasets.Example], bool] = _include_example,
    max_seconds=15,
) -> typing.Generator[lib.datasets.Example, None, None]:
    """Define the dataset generator to train and evaluate the TTS models on.

    Args:
        dataset
        include_example
        max_seconds: The maximum seconds delimited by an `Example`.
    """
    generators = {}
    for speaker, examples in dataset.items():
        # NOTE: Some datasets were pre-cut, and `is_singles` preserves their distribution.
        is_singles = all([e.alignments is None or len(e.alignments) == 1 for e in examples])
        generators[speaker] = lib.datasets.dataset_generator(
            examples, max_seconds=max_seconds if is_singles else math.inf
        )

    speakers = list(dataset.keys())
    counter = {s: 1.0 for s in speakers}
    while True:  # NOTE: Sample speakers uniformly...
        total = sum(counter.values())
        distribution = [total / v for v in counter.values()]
        example = next(generators[random.choices(speakers, distribution)[0]])
        assert example.alignments is not None, "To balance dataset, alignments must be defined."
        seconds = example.alignments[-1][1][1] - example.alignments[0][1][0]
        if seconds < max_seconds and include_example(example):
            yield example
            counter[speaker] += seconds


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
    'ʒ', 'ʔ', 'ˈ', 'ˌ', 'θ', 'ᵻ'
]
# fmt: on
