from collections import Counter
from functools import partial

import itertools
import logging
import pprint
import random

from hparams import add_config
from hparams import configurable
from hparams import HParams
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss
from torch.optim import Adam
from torchnlp.random import fork_rng

import torch
import torchnlp

from src import datasets
from src.audio import get_num_seconds
from src.audio import iso226_weighting
from src.audio import WAVE_FORMAT_IEEE_FLOAT
from src.audio import WavFileMetadata
from src.bin.train.signal_model.trainer import SpectrogramLoss
from src.datasets import filter_
from src.datasets import normalize_audio_column
from src.utils import log_runtime
from src.utils import seconds_to_string
from src.utils import slice_by_cumulative_sum

logger = logging.getLogger(__name__)
pprint = pprint.PrettyPrinter(indent=4)


def _get_window(window, window_length):
    # NOTE: `torch.hann_window` is different than the `scipy` window used by `librosa`.
    # Learn more here: https://github.com/pytorch/audio/issues/452
    window_tensor = None

    try:
        import librosa
        window_tensor = torch.tensor(librosa.filters.get_window(window, window_length)).float()
    except ImportError:
        logger.info('Ignoring optional `librosa` configurations.')

    return window_tensor


def _set_audio_processing():
    # SOURCE (Tacotron 1):
    # We use 24 kHz sampling rate for all experiments.
    sample_rate = 24000

    # SOURCE (Tacotron 2):
    # mel spectrograms are computed through a shorttime Fourier transform (STFT)
    # using a 50 ms frame size, 12.5 ms frame hop, and a Hann window function.
    # TODO: Parameterizing frame sizes in milliseconds can help ensure that your code is invariant
    # to the sample rate.
    # TODO: 50ms / 12.5ms spectrogram is not typical spectrogram parameterization, a more typical
    # parameterization is 25ms / 10ms. Learn more:
    # https://www.dsprelated.com/freebooks/sasp/Classic_Spectrograms.html
    # https://github.com/pytorch/audio/issues/384#issuecomment-597020705
    # https://pytorch.org/audio/compliance.kaldi.html
    frame_size = 1024  # NOTE: Frame size in samples
    fft_length = 2048
    assert frame_size % 4 == 0
    frame_hop = frame_size // 4
    # NOTE: A "hann window" is standard for calculating an FFT, it's even mentioned as a "popular
    # window" on Wikipedia (https://en.wikipedia.org/wiki/Window_function).
    window = 'hann'
    window_tensor = _get_window(window, frame_size)

    # SOURCE (Tacotron 2):
    # We transform the STFT magnitude to the mel scale using an 80 channel mel
    # filterbank spanning 125 Hz to 7.6 kHz, followed by log dynamic range
    # compression.
    # SOURCE (Tacotron 2 Author):
    # Google mentioned they settled on [20, 12000] with 128 filters in Google Chat.
    frame_channels = 128
    # NOTE: The human range is commonly given as 20 to 20,000 Hz
    # (https://en.wikipedia.org/wiki/Hearing_range).
    hertz_bounds = {'lower_hertz': 20, 'upper_hertz': 20000}

    # NOTE: The prior work only considers mono audio.
    channels = 1
    # Based on SoX this encoding is commonly used with a 16 or 24−bit encoding size. Learn more:
    # http://sox.sourceforge.net/sox.html
    encoding = 'signed-integer'
    # TODO: Now that the model predicts audio in 32-bits, consider normalizing to 32-bits.
    bits = 16

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
        add_config({IPython.display.Audio.__init__: HParams(rate=sample_rate)})
    except ImportError:
        logger.info('Ignoring optional `IPython` configurations.')

    add_config({
        'src.bin.evaluate._get_sample_rate': HParams(sample_rate=sample_rate),
        'src.audio': {
            'framed_rms_from_power_spectrogram':
                HParams(window=window_tensor),
            'framed_rms_from_signal':
                HParams(frame_length=frame_size, hop_length=frame_hop),
            '_db_mel_spectrogram_to_spectrogram':
                HParams(get_weighting=iso226_weighting),
            'pad_remainder':
                HParams(multiple=frame_hop, mode='constant', constant_values=0.0),
            'read_audio':
                HParams(
                    assert_metadata=WavFileMetadata(
                        sample_rate=sample_rate,
                        bits=bits,
                        channels=channels,
                        encoding=encoding,
                    )),
            'write_audio':
                HParams(sample_rate=sample_rate),
            # NOTE: Practically, `frame_rate` is equal to `sample_rate`. However, the terminology is
            # more appropriate because `sample_rate` is ambiguous. In a multi-channel scenario, each
            # channel has its own set of samples. It's unclear if `sample_rate` depends on the
            # number of channels, scaling linearly per channel.
            'build_wav_header':
                HParams(frame_rate=sample_rate),
            'SignalTodBMelSpectrogram.__init__':
                HParams(
                    sample_rate=sample_rate,
                    frame_hop=frame_hop,
                    window=window_tensor,
                    fft_length=fft_length,
                    num_mel_bins=frame_channels,
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
                ),
            'griffin_lim':
                HParams(
                    frame_size=frame_size,
                    frame_hop=frame_hop,
                    fft_length=fft_length,
                    window=window,
                    sample_rate=sample_rate,
                    # SOURCE (Tacotron 1):
                    # We found that raising the predicted magnitudes by a power of 1.2 before
                    # feeding to Griffin-Lim reduces artifacts
                    power=1.20,
                    # SOURCE (Tacotron 1):
                    # We observed that Griffin-Lim converges after 50 iterations (in fact, about 30
                    # iterations seems to be enough), which is reasonably fast.
                    iterations=30,
                ),
            '_mel_filters':
                HParams(**hertz_bounds),
            'normalize_audio':
                HParams(bits=bits, sample_rate=sample_rate, channels=channels, encoding=encoding)
        },
        'src.visualize': {
            'plot_waveform': HParams(sample_rate=sample_rate),
            'plot_spectrogram': HParams(sample_rate=sample_rate, frame_hop=frame_hop),
            'plot_mel_spectrogram': HParams(**hertz_bounds)
        },
        'src.bin.chunk_wav_and_text': {
            'seconds_to_samples': HParams(sample_rate=sample_rate),
            'samples_to_seconds': HParams(sample_rate=sample_rate),
            'chunk_alignments': HParams(sample_rate=sample_rate),
            'align_wav_and_scripts': HParams(sample_rate=sample_rate),
        },
    })

    return frame_channels, frame_hop, sample_rate


def _set_model_size(frame_channels):

    # SOURCE (Tacotron 2):
    # The prediction from the previous time step is first passed through a small
    # pre-net containing 2 fully connected layers of 256 hidden ReLU units.
    pre_net_hidden_size = 256

    # SOURCE (Tacotron 2):
    # Attention probabilities are computed after projecting inputs and location
    # features to 128-dimensional hidden representations.
    attention_hidden_size = 128

    # SOURCE (Tacotron 2):
    # Specifically, generation completes at the first frame for which this
    # probability exceeds a threshold of 0.5.
    stop_threshold = 0.5

    # NOTE: This parameter is determined manually by running `signal_model` tests until they pass.
    # There needs to be enough padding such that the output shape is valid.
    signal_model_padding = 14

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

                        # SOURCE (Tacotron 2)
                        # Attention probabilities are computed after projecting inputs and location
                        # features to 128-dimensional hidden representations.
                        out_dim=attention_hidden_size),
                'attention.LocationSensitiveAttention.__init__':
                    HParams(
                        # SOURCE (Tacotron 2):
                        # Location features are computed using 32 1-D convolution filters of length
                        # 31.
                        num_convolution_filters=32,
                        convolution_filter_size=31,
                    ),
                'decoder.AutoregressiveDecoder.__init__':
                    HParams(
                        pre_net_hidden_size=pre_net_hidden_size,
                        attention_hidden_size=attention_hidden_size,

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
                'post_net.PostNet.__init__':
                    HParams(
                        # SOURCE (Tacotron 2):
                        # Finally, the predicted mel spectrogram is passed
                        # through a 5-layer convolutional post-net which predicts a residual
                        # to add to the prediction to improve the overall reconstruction
                        num_convolution_layers=5,

                        # SOURCE (Tacotron 2):
                        # Each post-net layer is comprised of 512 filters with shape 5 × 1 with
                        # batch normalization, followed by tanh activations on all but the final
                        # layer
                        num_convolution_filters=512,
                        convolution_filter_size=5,
                    ),
                'model.SpectrogramModel': {
                    '__init__':
                        HParams(
                            frame_channels=frame_channels,

                            # SOURCE (Transfer Learning from Speaker Verification to Multispeaker
                            #         Text-To-Speech Synthesis):
                            # The paper mentions their proposed model uses a 256 dimension
                            # embedding.
                            # NOTE: See https://github.com/wellsaid-labs/Text-to-Speech/pull/258 to
                            # learn more about this parameter.
                            speaker_embedding_dim=128),
                    '_infer':
                        HParams(stop_threshold=stop_threshold)
                }  # noqa: E122
            },
            'signal_model': {
                'SignalModel.__init__':
                    HParams(
                        input_size=frame_channels,
                        hidden_size=32,
                        max_channel_size=512,
                        padding=signal_model_padding,
                        ratios=[2] * 8,
                        # SOURCE https://en.wikipedia.org/wiki/%CE%9C-law_algorithm:
                        # For a given input x, the equation for μ-law encoding is where μ = 255 in
                        # the North American and Japanese standards.
                        mu=255),
                'SpectrogramDiscriminator.__init__':
                    HParams(hidden_size=128),
            },
            'bin.train.spectrogram_model.trainer.Trainer._do_loss_and_maybe_backwards':
                HParams(stop_threshold=stop_threshold),
        }
    })

    return signal_model_padding


def _filter_audio_path_not_found(example):
    """ Filter out examples with no audio file.
    """
    if not example.audio_path.is_file():
        logger.warning('[%s] Not found audio file, skipping: %s', example.speaker,
                       example.audio_path)
        return False

    return True


def _filter_too_little_audio_per_character(example, min_seconds_per_character=0.0375):
    """ Filter out examples with too little audio per character.

    MOTIVATION: In October 2019, Rhyan and Michael observed that actors typically cannot speak
    more than 300 characters in 12 seconds; therefore, there is likely a dataset error if
    more than 300 characters are paired with 12 seconds of audio. For example, the speaker
    may have no read some of the text.

    NOTE: With `min_seconds_per_character=0.04`, then 300 characters must have at least 12 seconds
    of audio.

    NOTE: We adjusted `min_seconds_per_character` to 0.0375 because some of our speakers spoke a
    little faster than 0.04.

    Args:
        example (TextSpeechRow)
        min_seconds_per_character (float)

    Returns:
        (bool)
    """
    num_seconds = get_num_seconds(example.audio_path)
    if len(example.text) * min_seconds_per_character > num_seconds:
        logger.warning(('[%s] Likely some text was not spoken; ' +
                        'therefore, this example with %f seconds per character ' +
                        '[%f second(s) / %d character(s)] at `%s` was removed.'),
                       example.speaker, num_seconds / len(example.text), num_seconds,
                       len(example.text), example.audio_path)
        return False
    return True


def _filter_too_little_characters(example, min_characters=3):
    """ There was likely an error in the dataset creation if an example contains 2 or 1 characters.

    NOTE: This rule was developed with the help of `QA_Datasets/Sample_Dataset.ipynb`.
    """
    if min_characters > len(example.text):
        logger.warning('[%s] Hardly any text "%s" in this example, skipping %s', example.speaker,
                       example.text, example.audio_path)
        return False
    return True


def _filter_too_little_audio(example, min_audio=0.25):
    """ There was likely an error in the dataset creation if an example contains less than 0.25
    seconds of audio.

    NOTE: This rule was developed with the help of `QA_Datasets/Sample_Dataset.ipynb`.
    """
    num_seconds = get_num_seconds(example.audio_path)
    if min_audio > num_seconds:
        logger.warning('[%s] Hardly any audio (%ss) in this example, skipping %s', example.speaker,
                       num_seconds, example.audio_path)
        return False
    return True


def _filter_too_much_audio_per_character(example, min_seconds=3.0, max_seconds_per_character=0.11):
    """ Filter out examples with too much audio per character.

    NOTE: This rule was developed with the help of `QA_Datasets/Sample_Dataset.ipynb`.

    While `max_seconds_per_character` cutoff isn't perfect, it does remove more bad data than
    good data based on the dataset as of Feburary 2020. This cutoff relies on the
    "Law of large numbers" for that reason it doesn't apply well to clips under 3 seconds in length
    which can go up to 0.20 seconds per character.

    Args:
        example (TextSpeechRow)
        min_seconds (float): The mininum number of seconds before this rule goes into affect.
        max_seconds_per_character (float): The maximum number of seconds per character before
          filtering out the example.

    Returns:
        (bool)
    """
    num_seconds = get_num_seconds(example.audio_path)
    if len(example.text) * max_seconds_per_character < num_seconds and min_seconds < num_seconds:
        logger.warning(('[%s] Likely some text was repeated or the actor took a break; Example' +
                        ' "%s" with %f seconds per character ' +
                        '[%f second(s) / %d character(s)] at `%s` was removed.'),
                       example.speaker, example.text, num_seconds / len(example.text), num_seconds,
                       len(example.text), example.audio_path)
        return False
    return True


def _filter_no_text(example):
    """ Filter out examples with no text.
    """
    if len(example.text) == 0:
        logger.warning('[%s] Text is absent, skipping: %s', example.speaker, example.audio_path)
        return False
    return True


def _filter_books(example):
    """ Filter out examples originating from various audiobooks.
    """
    from src import datasets  # NOTE: Prevent circular dependency

    # Filter our particular books from M-AILABS dataset due:
    # - Inconsistent acoustic setup compared to other samples from the same speaker
    # - Audible noise in the background
    if (example.speaker == datasets.m_ailabs.MIDNIGHT_PASSENGER.speaker and
            datasets.m_ailabs.MIDNIGHT_PASSENGER.title in str(example.audio_path)):
        return False

    return True


def _filter_elliot_miller(example):
    """ Filter examples with the actor Elliot Miller due to his unannotated character portrayals.
    """
    from src import datasets  # NOTE: Prevent circular dependency
    if example.speaker == datasets.m_ailabs.ELLIOT_MILLER:
        return False
    return True


def _filter_no_numbers(example):
    """ Filter examples with numbers inside the text instead of attempting to verbalize them.
    """
    if len(set(example.text).intersection(set('0123456789'))) > 0:
        return False
    return True


def _preprocess_dataset(dataset):
    """ Preprocess a dataset.

    Args:
        dataset (iterable of TextSpeechRow)

    Return:
        dataset (iterable of TextSpeechRow)
    """
    dataset = filter_(_filter_audio_path_not_found, dataset)
    dataset = filter_(_filter_no_text, dataset)
    dataset = filter_(_filter_elliot_miller, dataset)
    dataset = filter_(_filter_no_numbers, dataset)
    dataset = filter_(_filter_books, dataset)
    dataset = filter_(_filter_too_little_characters, dataset)
    # TODO: Investigate trimming audio silences during normalization to assist the following rules
    # that dependent on the length of the audio.
    # NOTE: `normalize_audio_column` run before audio length filters because it caches the audio
    # metadata needed to execute the below filters quickly.
    dataset = normalize_audio_column(dataset)
    dataset = filter_(_filter_too_little_audio, dataset)
    dataset = filter_(_filter_too_little_audio_per_character, dataset)
    dataset = filter_(_filter_too_much_audio_per_character, dataset)
    random.shuffle(dataset)
    return dataset


def _split_dataset(dataset, num_second_dev_set=60 * 60):
    """ Split a dataset into a development and train set.

    Args:
        dataset (iterable of TextSpeechRow)
        num_second_dev_set (int, optional): Number of seconds of audio data in the development set.

    Return:
        train (iterable of TextSpeechRow): The rest of the data.
        dev (iterable of TextSpeechRow): Dataset with `num_second_dev_set` of data.
    """
    dev = slice_by_cumulative_sum(
        dataset,
        max_total_value=num_second_dev_set,
        get_value=lambda e: get_num_seconds(e.audio_path))
    train = dataset[len(dev):]
    assert len(dev) > 0, 'The dev dataset has no examples.'
    assert len(train) > 0, 'The train dataset has no examples.'
    assert len(dev) + len(train) == len(dataset), 'The `_split_dataset` lost at least 1 example.'
    return train, dev


def get_dataset():
    """ Define the dataset to train the text-to-speech models on.

    Returns:
        train (iterable)
        dev (iterable)
    """
    # NOTE: This `seed` should never change; otherwise, the training and dev datasets may be
    # different from experiment to experiment.
    with fork_rng(seed=123):
        logger.info('Loading dataset...')
        train_dev_splits = [
            _split_dataset(_preprocess_dataset(d)) for d in [
                datasets.hilary_speech_dataset(),
                datasets.beth_speech_dataset(),
                datasets.heather_speech_dataset(),
                datasets.susan_speech_dataset(),
                datasets.sam_speech_dataset(),
                datasets.frank_speech_dataset(),
                datasets.adrienne_speech_dataset(),
                datasets.alicia_speech_dataset(),
                datasets.george_speech_dataset(),
                datasets.megan_speech_dataset(),
                datasets.elise_speech_dataset(),
                datasets.hanuman_speech_dataset(),
                datasets.jack_speech_dataset(),
                datasets.mark_speech_dataset(),
                datasets.steven_speech_dataset()
            ]
        ]
        train = [s[0] for s in train_dev_splits]
        train += [
            _preprocess_dataset(d) for d in [
                datasets.lj_speech_dataset(),
                datasets.m_ailabs_en_us_speech_dataset(),
                datasets.beth_custom_speech_dataset(),
            ]
        ]
        train = list(itertools.chain.from_iterable(train))
        random.shuffle(train)
        logger.info('Loaded %d train dataset examples (%s) with a speaker distribution of:\n%s',
                    len(train),
                    seconds_to_string(sum([get_num_seconds(e.audio_path) for e in train])),
                    pprint.pformat(Counter([e.speaker for e in train])))

        dev = list(itertools.chain.from_iterable([s[1] for s in train_dev_splits]))
        random.shuffle(dev)
        logger.info('Loaded %d dev dataset examples (%s) with a speaker distribution of:\n%s',
                    len(dev), seconds_to_string(sum([get_num_seconds(e.audio_path) for e in dev])),
                    pprint.pformat(Counter([e.speaker for e in dev])))

        return train, dev


def spectrogram_model_lr_multiplier_schedule(step, warmup=500):
    """ Learning rate multiplier schedule.

    Args:
        step (int): The current step.
        warmup (int): The number of warmup steps.

    Returns:
        (float): Multiplier on the base learning rate.
    """
    if step < warmup:
        return step / warmup

    return 1.0


def signal_model_lr_multiplier_schedule(step, warmup=500):
    """ Learning rate multiplier schedule.

    Args:
        step (int): The current step.
        warmup (int): The number of warmup steps.

    Returns:
        (float): Multiplier on the base learning rate.
    """
    if step < warmup:
        return step / warmup

    return 1.0


@log_runtime
def set_hparams():
    """ Using the ``configurable`` module set the hyperparameters for the source code.
    """
    # NOTE: Prevent circular dependency
    from src.signal_model import SignalModel
    from src.signal_model import SpectrogramDiscriminator
    from src.spectrogram_model import SpectrogramModel

    frame_channels, frame_hop, sample_rate = _set_audio_processing()
    signal_model_padding = _set_model_size(frame_channels)

    Adam.__init__ = configurable(Adam.__init__)
    nn.modules.batchnorm._BatchNorm.__init__ = configurable(
        nn.modules.batchnorm._BatchNorm.__init__)
    nn.LayerNorm.__init__ = configurable(nn.LayerNorm.__init__)
    add_config({
        # NOTE: `momentum=0.01` to match Tensorflow defaults
        'torch.nn.modules.batchnorm._BatchNorm.__init__': HParams(momentum=0.01),
        # NOTE: BERT uses `eps=1e-12` for `LayerNorm`, see here:
        # https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py
        'torch.nn.LayerNorm.__init__': HParams(eps=1e-12),
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
        'torch.optim.adam.Adam.__init__': HParams(
            betas=(0.9, 0.999),
            amsgrad=False,
            lr=10**-3,
        )
    })

    spectrogram_model_dev_batch_size = 224

    seed = 1212212

    torchnlp.samplers.DeterministicSampler.__init__ = configurable(
        torchnlp.samplers.DeterministicSampler.__init__)
    add_config({
        'torchnlp.samplers.DeterministicSampler.__init__': HParams(random_seed=seed),
    })

    add_config({
        'src': {
            'spectrogram_model': {
                # SOURCE (Tacotron 2):
                # In order to introduce output variation at inference time, dropout with
                # probability 0.5 is applied only to layers in the pre-net of the
                # autoregressive decoder.
                'pre_net.PreNet.__init__':
                    HParams(dropout=0.5),
                'model.SpectrogramModel.__init__':
                    HParams(
                        # NOTE: This is based on one of the slowest legitimate example in the
                        # dataset:
                        # "rate(WSL_SMurphyScript34-39,24000)/script_52_chunk_9.wav"
                        # NOTE: This configuration is related to the dataset preprocessing step:
                        # `_filter_too_much_audio_per_character`
                        # NOTE: This number was configured with the help of:
                        # `QA_Datasets/Sample_Dataset.ipynb`
                        max_frames_per_token=0.2 / (frame_hop / sample_rate),

                        # NOTE: The spectrogram input ranges from around -50 to 50. This scaler
                        # puts the input and output in a more reasonable range for the model of
                        # -5 to 5.
                        output_scalar=10.0)
            },
            # NOTE: Parameters set after experimentation on a 1 Px100 GPU.
            'datasets.utils.add_predicted_spectrogram_column':
                HParams(batch_size=(spectrogram_model_dev_batch_size // 2)),
            'bin': {
                'evaluate._get_dev_dataset': HParams(dataset=get_dataset),
                'train': {
                    'spectrogram_model': {
                        '__main__._get_dataset':
                            HParams(dataset=get_dataset),
                        'trainer.Trainer.__init__':
                            HParams(
                                lr_multiplier_schedule=spectrogram_model_lr_multiplier_schedule,
                                # SOURCE: Tacotron 2
                                # To train the feature prediction network, we apply the standard
                                # maximum-likelihood training procedure (feeding in the correct
                                # output instead of the predicted output on the decoder side, also
                                # referred to as teacher-forcing) with a batch size of 64 on a
                                # single GPU.
                                # NOTE: Parameters set after experimentation on a 2
                                # Px100 GPU.
                                train_batch_size=56,
                                dev_batch_size=spectrogram_model_dev_batch_size,

                                # SOURCE (Tacotron 2):
                                # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
                                optimizer=Adam,

                                # SOURCE (Tacotron 2 Author):
                                # The author confirmed they used BCE loss in Google Chat.
                                criterion_stop_token=BCEWithLogitsLoss,

                                # SOURCE: Tacotron 2
                                # We minimize the summed mean squared error (MSE) from before and
                                # after the post-net to aid convergence.
                                criterion_spectrogram=MSELoss,

                                # Tacotron 2 like model with any changes documented via Comet.ml.
                                model=SpectrogramModel),
                        'data_loader.get_normalized_half_gaussian':
                            HParams(
                                # NOTE: We approximated the uncertainty in the stop token by viewing
                                # the stop token predictions by a fully trained model without
                                # this smoothing. We found that a fully trained model would
                                # learn a similar curve over 4 - 8 frames in January 2020, on Comet.
                                # NOTE: This was rounded up to 10 after the spectrograms got
                                # 17% larger.
                                length=10,
                                standard_deviation=2),
                    },
                    'signal_model': {
                        '__main__._get_dataset':
                            HParams(dataset=get_dataset),
                        'trainer.Trainer.__init__':
                            HParams(
                                # SOURCE (Tacotron 2):
                                # We train with a batch size of 128 distributed across 32 GPUs with
                                # synchronous updates, using the Adam optimizer with β1 = 0.9, β2 =
                                # 0.999, eps = 10−8 and a fixed learning rate of 10−4
                                # NOTE: Parameters set after experimentation on a 8 V100 GPUs.
                                train_batch_size=128,
                                # SOURCE: Efficient Neural Audio Synthesis
                                # The WaveRNN models are trained on sequences of 960 audio samples.
                                # SOURCE: Parallel WaveNet: Fast High-Fidelity Speech Synthesis
                                # The teacher WaveNet network was trained for 1,000,000 steps with
                                # the ADAM optimiser [14] with a minibatch size of 32 audio clips,
                                # each containing 7,680 timesteps (roughly 320ms).
                                # NOTE: The `spectrogram_slice_size` must be larger than the
                                # `fft_length - frame_hop` of the largest `SpectrogramLoss`;
                                # otherwise, the loss can't be computed.
                                train_spectrogram_slice_size=int(8192 / frame_hop),
                                dev_batch_size=16,
                                dev_spectrogram_slice_size=int(32768 / frame_hop),
                                optimizer=Adam,

                                # NOTE: We employ a small warmup because the model can be unstable
                                # at the start of it's training.
                                lr_multiplier_schedule=signal_model_lr_multiplier_schedule,
                                model=SignalModel,
                                # NOTE: The `num_mel_bins` must be proportional to `fft_length`,
                                # learn more:
                                # https://stackoverflow.com/questions/56929874/what-is-the-warning-empty-filters-detected-in-mel-frequency-basis-about
                                criterions=[
                                    partial(
                                        SpectrogramLoss,
                                        fft_length=2048,
                                        frame_hop=256,
                                        window=_get_window('hann', 1024),
                                        num_mel_bins=128),
                                    partial(
                                        SpectrogramLoss,
                                        fft_length=1024,
                                        frame_hop=128,
                                        window=_get_window('hann', 512),
                                        num_mel_bins=64),
                                    partial(
                                        SpectrogramLoss,
                                        fft_length=512,
                                        frame_hop=64,
                                        window=_get_window('hann', 256),
                                        num_mel_bins=32),
                                ]),
                        'trainer.SpectrogramLoss.__init__':
                            HParams(
                                criterion=torch.nn.L1Loss,
                                # NOTE: This discriminator approach is largely based on:
                                # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
                                # NOTE: This approach proved successful in Comet experiment
                                # f590fe3c51a04130ad65736f8aa5fd81 run in February 2020.
                                discriminator=SpectrogramDiscriminator,
                                discriminator_optimizer=torch.optim.Adam,
                                discriminator_criterion=torch.nn.BCEWithLogitsLoss,
                            ),
                        # NOTE: The `DataLoader` pads the data before hand so that the model
                        # trains on real padding instead of zero padding.
                        'data_loader.DataLoader.__init__':
                            HParams(spectrogram_slice_pad=signal_model_padding),
                    }
                },
            },
            # NOTE: The expected signal model output is 32-bit float.
            'audio.build_wav_header':
                HParams(wav_format=WAVE_FORMAT_IEEE_FLOAT, num_channels=1, sample_width=4),
            # NOTE: Window size smoothing parameter is not super sensative.
            'optimizers.AutoOptimizer.__init__':
                HParams(window_size=128),
            # NOTE: The `beta` parameter is not super sensative.
            'optimizers.ExponentialMovingParameterAverage.__init__':
                HParams(beta=0.999),
            'environment.set_seed':
                HParams(seed=seed),
        }
    })
