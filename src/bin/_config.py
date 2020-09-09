from collections import defaultdict

import logging
import pprint
import random
import typing

from hparams import add_config
from hparams import configurable
from hparams import HParams
from torch import nn
from torchnlp.random import fork_rng

import torch

from src import datasets
from src.audio import AudioFileMetadata
from src.audio import iso226_weighting
from src.bin._utils import Dataset
from src.bin._utils import format_audio_filter
from src.bin._utils import split_examples
from src.environment import DATABASES_PATH
from src.utils import seconds_to_string

logger = logging.getLogger(__name__)
pprinter = pprint.PrettyPrinter(indent=4)

# TODO: Should some of these items be saved in `HParams` so they can be saved to Comet.ml?

RANDOM_SEED = 1212212
# SOURCE (Tacotron 1):
# We use 24 kHz sampling rate for all experiments.
SAMPLE_RATE = 24000
TRAINING_DB = str(DATABASES_PATH.absolute() / 'training.db')
_LOUD_NORM_AUDIO_FILTER = format_audio_filter(
    'loudnorm', integrated_loudness=-21, loudness_range=4, true_peak=-6.1, print_format='summary')
_SPEAKER_AUDIO_FILERS: typing.Dict[datasets.Speaker, typing.List[str]] = defaultdict(list)
_SPEAKER_AUDIO_FILERS.update({
    datasets.HILARY_NORIEGA: [
        format_audio_filter(
            'acompressor',
            threshold='0.032',
            ratio=12,
            attack=325,
            release=390,
            knee=6,
            detection='rms',
            makeup=4),
        format_audio_filter('equalizer', frequency=200, width_type='q', width=0.6, gain=-2.4)
    ],
    datasets.SAM_SCHOLL: [
        format_audio_filter(
            'acompressor',
            threshold='0.05',
            ratio=12,
            attack=325,
            release=390,
            knee=6,
            detection='rms',
            makeup=4),
        format_audio_filter('equalizer', frequency=200, width_type='q', width=1.3, gain=-4.5),
        format_audio_filter('equalizer', frequency=3420, width_type='q', width=0.76, gain=2.8)
    ],
    datasets.ALICIA_HARRIS: [
        format_audio_filter(
            'acompressor',
            threshold='0.013',
            ratio=12,
            attack=325,
            release=390,
            knee=6,
            detection='rms',
            makeup=4),
        format_audio_filter('equalizer', frequency=200, width_type='q', width=0.58, gain=-2.0),
        format_audio_filter('equalizer', frequency=3560, width_type='q', width=0.58, gain=0.9)
    ],
    datasets.MARK_ATHERLAY: [
        format_audio_filter(
            'acompressor',
            threshold='0.05',
            ratio=12,
            attack=325,
            release=390,
            knee=6,
            detection='rms',
            makeup=4),
        format_audio_filter('equalizer', frequency=144, width_type='q', width=0.94, gain=-1.7),
        format_audio_filter('equalizer', frequency=2000, width_type='q', width=0.95, gain=2.3)
    ]
})
PHONEME_SEPARATOR = '|'
# SOURCE (Tacotron 2):
# We transform the STFT magnitude to the mel scale using an 80 channel mel
# filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
# compression.
# SOURCE (Tacotron 2 Author):
# Google mentioned they settled on [20, 12000] with 128 filters in Google Chat.
FRAME_CHANNELS = 128


def _get_window(window: str, window_length: int, window_hop: int):
    # NOTE: `torch.hann_window` is different than the `scipy` window used by `librosa`.
    # Learn more here: https://github.com/pytorch/audio/issues/452
    window_tensor = None

    try:
        import librosa
        window = librosa.filters.get_window(window, window_length)
        window_tensor = torch.tensor(window).float()
    except ImportError:
        logger.info('Ignoring optional `librosa` configurations.')

    try:
        import scipy
        assert scipy.signal.check_COLA(window, window_length, window_length - window_hop)
    except ImportError:
        logger.info('Ignoring optional `scipy` configurations.')

    return window_tensor


def configure_audio_processing():
    sox_encoding = '32-bit Floating Point PCM'
    metadata = AudioFileMetadata(
        sample_rate=SAMPLE_RATE,
        # NOTE: The prior work like Tacotrn-2 only considers mono audio.
        channels=1,
        # NOTE: The signal model supports 32-bit audio.
        encoding=sox_encoding)
    ffmpeg_encoding = 'pcm_f32le'
    ffmpeg_format = 'f32le'

    # SOURCE (Tacotron 2):
    # mel spectrograms are computed through a shorttime Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
    # TODO: Parameterizing frame sizes in milliseconds can help ensure that your code is invariant
    # to the sample rate; however, we would need to assure that we're still using powers of two
    # for performance.
    # TODO: Verify that `frame_hop` is equal to the signal model's upsample property, ensuring that
    # that relationship holds.
    # TODO: 50ms / 12.5ms spectrogram is not typical spectrogram parameterization, a more typical
    # parameterization is 25ms / 10ms. Learn more:
    # https://www.dsprelated.com/freebooks/sasp/Classic_Spectrograms.html
    # https://github.com/pytorch/audio/issues/384#issuecomment-597020705
    # https://pytorch.org/audio/compliance.kaldi.html
    frame_size = 2048  # NOTE: Frame size in samples
    fft_length = 2048
    assert frame_size % 4 == 0
    frame_hop = frame_size // 4
    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    window = _get_window('hann', frame_size, frame_hop)

    # NOTE: The human range is commonly given as 20 to 20,000 Hz
    # (https://en.wikipedia.org/wiki/Hearing_range).
    hertz_bounds = {'lower_hertz': 20, 'upper_hertz': 20000}

    try:
        import librosa
        librosa.effects.trim = configurable(librosa.effects.trim)
        add_config({
            librosa.effects.trim: HParams(frame_length=frame_size, hop_length=frame_hop),
        })
    except ImportError:
        logger.info('Ignoring optional `librosa` configurations.')

    try:
        import IPython
        IPython.display.Audio.__init__ = configurable(IPython.display.Audio.__init__)
        add_config({IPython.display.Audio.__init__: HParams(rate=metadata.sample_rate)})
    except ImportError:
        logger.info('Ignoring optional `IPython` configurations.')

    add_config({
        'src.audio': {
            'framed_rms_from_power_spectrogram': HParams(window=window),
            'framed_rms_from_signal': HParams(frame_length=frame_size, hop_length=frame_hop),
            'pad_remainder': HParams(multiple=frame_hop, mode='constant', constant_values=0.0),
            'read_audio': HParams(expected_metadata=metadata),
            'write_audio': HParams(sample_rate=metadata.sample_rate),
        },
        'src.service.worker.stream_text_to_speech_synthesis': HParams(sample_rate=SAMPLE_RATE),
        'src.visualize': {
            'plot_waveform': HParams(sample_rate=SAMPLE_RATE),
            'plot_spectrogram': HParams(sample_rate=SAMPLE_RATE, frame_hop=frame_hop),
            'plot_mel_spectrogram': HParams(**hertz_bounds)
        }
    })

    add_config({
        'SignalTodBMelSpectrogram.__init__':
            HParams(
                sample_rate=SAMPLE_RATE,
                frame_hop=frame_hop,
                window=window,
                fft_length=fft_length,
                num_mel_bins=FRAME_CHANNELS,
                # SOURCE (Tacotron 2):
                # Prior to log compression, the filterbank output magnitudes are clipped to a
                # minimum value of 0.01 in order to limit dynamic range in the logarithmic
                # domain.
                # NOTE: The `min_decibel` is set to ensure there is around 100 dB of
                # dynamic range, allowing us to make the most use of the maximum 96 dB dynamic
                # range a 16-bit audio file can provide. This is assuming that a full-scale
                # 997 Hz sine wave is the maximum dB which would be around ~47 dB. Tacotron 2's
                # equivalent  of 0.01 (~ -40 dB).
                min_decibel=-50.0,
                # NOTE: ISO226 is one of the latest standards for determining loudness:
                # https://www.iso.org/standard/34222.html. It does have some issues though:
                # http://www.lindos.co.uk/cgi-bin/FlexiData.cgi?SOURCE=Articles&VIEW=full&id=2
                get_weighting=iso226_weighting,
                **hertz_bounds),
        'griffin_lim':
            HParams(
                frame_size=frame_size,
                frame_hop=frame_hop,
                fft_length=fft_length,
                window=window.numpy(),
                sample_rate=SAMPLE_RATE,
                # SOURCE (Tacotron 1):
                # We found that raising the predicted magnitudes by a power of 1.2 before
                # feeding to Griffin-Lim reduces artifacts
                power=1.20,
                # SOURCE (Tacotron 1):
                # We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
                # iterations seems to be enough), which is reasonably fast.
                iterations=30,
                get_weighting=iso226_weighting,
                **hertz_bounds),
        'src.spectrogram_model.model.SpectrogramModel.__init__':
            HParams(
                # NOTE: This is based on one of the slowest legitimate example in the
                # dataset:
                # "rate(WSL_SMurphyScript34-39,24000)/script_52_chunk_9.wav"
                # NOTE: This configuration is related to the dataset preprocessing step:
                # `_filter_too_much_audio_per_character`
                # NOTE: This number was configured with the help of:
                # `QA_Datasets/Sample_Dataset.ipynb`
                max_frames_per_token=0.16 / (frame_hop / SAMPLE_RATE),),
        'src.bin._utils.normalize_audio':
            HParams(
                encoding=ffmpeg_encoding,
                sample_rate=SAMPLE_RATE,
                channels=metadata.channels,
                get_audio_filters=lambda s: ','.join([_LOUD_NORM_AUDIO_FILTER] +
                                                     _SPEAKER_AUDIO_FILERS(s))),
        'src.bin._utils.get_spectrogram_example':
            HParams(
                format_=ffmpeg_format,
                encoding=ffmpeg_encoding,
                sample_rate=SAMPLE_RATE,
                channels=metadata.channels,
                loudness_implementation='DeMan',
                max_loudness_annotations=5,
                loudness_precision=0,
                max_speed_annotations=5,
                speed_precision=0,
                # NOTE: We approximated the uncertainty in the stop token by viewing
                # the stop token predictions by a fully trained model without
                # this smoothing. We found that a fully trained model would
                # learn a similar curve over 4 - 8 frames in January 2020, on Comet.
                # NOTE: This was rounded up to 10 after the spectrograms got
                # 17% larger.
                # TODO: In July 2020, the spectrogram size was decreased by 2x, we
                # should test decreasing `length` by 2x, also.
                stop_token_range=10,
                stop_token_standard_deviation=2,
            )
    })


def configure_models():
    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    encoder_output_size = 128

    # SOURCE (Tacotron 2):
    # Specifically, generation completes at the first frame for which this
    # probability exceeds a threshold of 0.5.
    stop_threshold = 0.5

    add_config({
        'src': {
            'spectrogram_model': {
                'encoder.Encoder.__init__':
                    HParams(
                        # SOURCE (Tacotron 2):
                        # Input characters are represented using a learned 512-dimensional character
                        # embedding
                        # ...
                        # which are passed through a stack of 3 convolutional layers each containing
                        # 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters
                        # ...
                        # The output of the final convolutional layer is passed into a single
                        # bi-directional [19] LSTM [20] layer containing 512 units (256) in each
                        # direction) to generate the encoded features.
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
                        out_dim=encoder_output_size),
                'attention.LocationSensitiveAttention.__init__':
                    HParams(
                        # SOURCE (Tacotron 2):
                        # Location features are computed using 32 1-D convolution filters of length
                        # 31.
                        # SOURCE (Tacotron 2):
                        # Attention probabilities are computed after projecting inputs and location
                        # features to 128-dimensional hidden representations.
                        hidden_size=128,
                        convolution_filter_size=31,

                        # NOTE: The text speech alignment is monotonic; therefore, there is no need
                        # to pay attention to any text outside of a narrow band of a couple
                        # characters on either side.
                        # NOTE: In Comet, we report the metric "attention_std". The standard
                        # deviation for the attention alignment is helpful to set this metric in
                        # such a way that it doesn't affect model performance.
                        window_length=9,
                    ),
                'decoder.AutoregressiveDecoder.__init__':
                    HParams(
                        encoder_output_size=encoder_output_size,

                        # SOURCE (Tacotron 2):
                        # The prediction from the previous time step is first passed through a small
                        # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
                        pre_net_hidden_size=256,

                        # SOURCE (Tacotron 2):
                        # The prenet output and attention context vector are concatenated and
                        # passed through a stack of 2 uni-directional LSTM layers with 1024 units.
                        lstm_hidden_size=1024,
                    ),
                'pre_net.PreNet.__init__':
                    HParams(
                        # SOURCE (Tacotron 2):
                        # The prediction from the previous time step is first passed through a small
                        # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
                        num_layers=2),
                'model.SpectrogramModel.__init__':
                    HParams(
                        frame_channels=FRAME_CHANNELS,

                        # NOTE: The spectrogram input ranges from around -50 to 50. This scaler
                        # puts the input and output in a more reasonable range for the model of
                        # -5 to 5.
                        output_scalar=10.0,

                        # SOURCE (Transfer Learning from Speaker Verification to Multispeaker
                        #         Text-To-Speech Synthesis):
                        # The paper mentions their proposed model uses a 256 dimension
                        # embedding.
                        # NOTE: See https://github.com/wellsaid-labs/Text-to-Speech/pull/258 to
                        # learn more about this parameter.
                        speaker_embedding_dim=128),
            },
            'signal_model': {
                'SignalModel.__init__':
                    HParams(
                        input_size=FRAME_CHANNELS,
                        hidden_size=32,
                        max_channel_size=512,
                        ratios=[2] * 9,
                        # SOURCE https://en.wikipedia.org/wiki/%CE%9C-law_algorithm:
                        # For a given input x, the equation for μ-law encoding is where μ = 255 in
                        # the North American and Japanese standards.
                        mu=255),
                # NOTE: We found this hidden size to be effective on Comet in April 2020.
                'SpectrogramDiscriminator.__init__':
                    HParams(hidden_size=512),
            },
            # TODO: Update this?
            'bin.train.spectrogram_model.trainer.Trainer._do_loss_and_maybe_backwards':
                HParams(stop_threshold=stop_threshold),
        }
    })

    add_config({
        'src.spectrogram_model': {
            # SOURCE (Tacotron 2):
            # In order to introduce output variation at inference time, dropout with
            # probability 0.5 is applied only to layers in the pre-net of the
            # autoregressive decoder.
            'pre_net.PreNet.__init__': HParams(dropout=0.5),
            # NOTE: This dropout approach proved effective in Comet in March 2020.
            'model.SpectrogramModel': {
                '__init__': {HParams(speaker_embed_dropout=0.1)},
                'forward': {HParams(stop_threshold=stop_threshold)},
            },
            # NOTE: This below dropout approach proved effective in Comet in March 2020.
            'attention.LocationSensitiveAttention.__init__': HParams(dropout=0.1),
            'decoder.AutoregressiveDecoder.__init__': HParams(stop_net_dropout=0.5),
            'encoder.Encoder.__init__': HParams(dropout=0.1)
        }
    })

    nn.modules.batchnorm._BatchNorm.__init__ = configurable(
        nn.modules.batchnorm._BatchNorm.__init__)
    nn.LayerNorm.__init__ = configurable(nn.LayerNorm.__init__)
    add_config({
        # NOTE: `momentum=0.01` to match Tensorflow defaults
        'torch.nn.modules.batchnorm._BatchNorm.__init__': HParams(momentum=0.01),
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        'torch.nn.LayerNorm.__init__': HParams(eps=1e-12),
    })

    add_config({
        # NOTE: Window size smoothing parameter is not super sensative.
        'src.optimizers.AutoOptimizer.__init__': HParams(window_size=128),
        # NOTE: The `beta` parameter is not super sensative.
        'src.optimizers.ExponentialMovingParameterAverage.__init__': HParams(beta=0.9999),
    })


def get_dataset(dev_size=60 * 60) -> typing.Tuple[Dataset, Dataset]:
    """ Define the dataset to train and evaluate the TTS models on.

    Args:
        dev_size (optional): Number of seconds per speaker in the development dataset.

    Returns:
        train (dict[Speaker, list[Example]]): Dictionary that maps a speaker to the data.
        dev (dict[Speaker, list[Example]]): Dictionary that maps a speaker to the data.
    """
    # NOTE: This `seed` should never change; otherwise, the training and dev datasets may be
    # different from experiment to experiment.
    with fork_rng(seed=123):
        logger.info('Loading dataset...')
        dev = {}
        train = {}
        iterator = [
            (datasets.HILARY_NORIEGA, datasets.hilary_noriega_speech_dataset()),
        ]
        for speaker, examples in iterator:
            train[speaker], dev[speaker] = split_examples(examples, dev_size)

        # NOTE: Elliot Miller is not included due to his unannotated character portrayals.
        train[datasets.LINDA_JOHNSON] = datasets.lj_speech_dataset()
        train[datasets.JUDY_BIEBER] = datasets.m_ailabs_en_us_judy_bieber_speech_dataset()
        train[datasets.MARY_ANN] = datasets.m_ailabs_en_us_mary_ann_speech_dataset()
        train[datasets.ELIZABETH_KLETT] = datasets.m_ailabs_en_uk_elizabeth_klett_speech_dataset()

        # NOTE: This assumes that a negligible amount of data is unusable in each example.
        get_distribution = lambda d: pprint.pformat({
            k: seconds_to_string(sum(e.alignments[-1][1][1] - e.alignments[0][1][0] for e in v))
            for k, v in d.items()
        })
        logger.info('Training dataset speaker distribution:\n%s', get_distribution(train))
        logger.info('Development dataset speaker distribution:\n%s', get_distribution(train))
        return train, dev


def _include_example(example: datasets.Example) -> bool:
    """ Return `True` iff `example` should be included in the dataset.

    TODO: Consider using phoneme or grapheme for calculating speed.
    TODO: Develop the filter further to remove problematic data, for example:
      - Gaps in alignments
          - Has text?
          - Has long pause?
          - Is loud?
      - Starting or ending alignment
      - Audio length after removing silence
      - etc.
    """
    assert example.alignments is not None

    text = example.text[example.alignments[0].text[0]:example.alignments[-1].text[-1]]

    if len(text) == 0:
        return False

    if any(c.isdigit() for c in text):
        return False

    if not example.audio_path.is_file():
        return False

    if example.alignments[0].audio[0] == example.alignments[-1].audio[-1]:
        return False

    # NOTE: Filter our particular books from M-AILABS dataset due:
    # - Inconsistent acoustic setup compared to other samples from the same speaker
    # NOTE: Filter out the North & South book because it uses English in a way that's not consistent
    # with editor usage, for example: "To-morrow, you will-- Come back to-night, John!"
    if (example.metadata is not None and
        (example.metadata['books'] == datasets.m_ailabs.MIDNIGHT_PASSENGER or
         example.metadata['books'] == datasets.m_ailabs.NORTH_AND_SOUTH)):
        return False

    return True


def get_dataset_generator(dataset: Dataset,
                          include_example: typing.Callable[[datasets.Example],
                                                           bool] = _include_example,
                          max_seconds=15) -> typing.Generator[datasets.Example, None, None]:
    """ Define the dataset generator to train and evaluate the TTS models on.

    Args:
        dataset
        include_example
        max_seconds: The maximum seconds delimited by an `Example`.
    """
    generators = {}
    for speaker, examples in dataset.items():
        # NOTE: Some datasets were pre-cut, and this preserves their distribution.
        is_singles = all([e.alignments is None or len(e.alignments) == 1 for e in examples])
        generators[speaker] = datasets.dataset_generator(
            examples, max_seconds=max_seconds if is_singles else float('inf'))

    speakers = list(dataset.keys())
    counter = {s: 1.0 for s in speakers}
    while True:
        # Sample speakers uniformly...
        total = sum(counter.values())
        distribution = [total / v for v in counter.values()]
        example = next(generators[random.choices(speakers, distribution)[0]])
        assert example.alignments is not None, 'To balance dataset, alignments must be defined.'
        seconds = example.alignments[-1][1][1] - example.alignments[0][1][0]
        if seconds < max_seconds and include_example(example):
            yield example
            counter[speaker] += seconds
