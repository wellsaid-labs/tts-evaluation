from src.audio import get_num_seconds
from src.datasets import LINDA_JOHNSON
from src.datasets import TextSpeechRow
from src.environment import ROOT_PATH
from src.hparams import _filter_audio_path_not_found
from src.hparams import _filter_no_numbers
from src.hparams import _filter_no_text
from src.hparams import _filter_too_little_audio
from src.hparams import _filter_too_little_audio_per_character
from src.hparams import _filter_too_little_characters
from src.hparams import _filter_too_much_audio_per_character
from src.hparams import _preprocess_dataset
from src.hparams import _split_dataset
from src.hparams import signal_model_lr_multiplier_schedule
from src.hparams import spectrogram_model_lr_multiplier_schedule
from tests._utils import get_tts_mocks


def test_signal_model_lr_multiplier_schedule():
    assert signal_model_lr_multiplier_schedule(0, 500) == 0.0
    assert signal_model_lr_multiplier_schedule(250, 500) == 0.5
    assert signal_model_lr_multiplier_schedule(500, 500) == 1.0
    assert signal_model_lr_multiplier_schedule(750, 500) == 1.0


def test_spectrogram_model_lr_multiplier_schedule():
    assert spectrogram_model_lr_multiplier_schedule(0, 500) == 0.0
    assert spectrogram_model_lr_multiplier_schedule(250, 500) == 0.5
    assert spectrogram_model_lr_multiplier_schedule(500, 500) == 1.0
    assert spectrogram_model_lr_multiplier_schedule(750, 500) == 1.0


def test__filter_audio_path_not_found():
    example = TextSpeechRow(
        text='', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH / 'tests/_test_data/abcdefghij.wav')
    assert not _filter_audio_path_not_found(example)

    example = TextSpeechRow(
        text='',
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert _filter_audio_path_not_found(example)


def test__filter_no_numbers():
    example = TextSpeechRow(text='123', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert not _filter_no_numbers(example)

    example = TextSpeechRow(text='Hi There!', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert _filter_no_numbers(example)


def test__filter_no_text():
    example = TextSpeechRow(text='', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert not _filter_no_text(example)

    example = TextSpeechRow(text='Hi There!', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert _filter_no_text(example)


def test__filter_too_little_audio():
    example = TextSpeechRow(
        text='',
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert not _filter_too_little_audio(example, 3.0)
    assert _filter_too_little_audio(example, 1.0)


def test__filter_too_little_characters():
    example = TextSpeechRow(text='Hi!', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert not _filter_too_little_characters(example, 5)

    example = TextSpeechRow(text='Hi There!', speaker=LINDA_JOHNSON, audio_path=ROOT_PATH)
    assert _filter_too_little_characters(example)


def test__filter_too_much_audio_per_character():
    example = TextSpeechRow(
        text='A.',
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert not _filter_too_much_audio_per_character(
        example, min_seconds=1.0, max_seconds_per_character=0.1)
    assert _filter_too_much_audio_per_character(
        example, min_seconds=3.0, max_seconds_per_character=0.1)
    assert _filter_too_much_audio_per_character(
        example, min_seconds=1.0, max_seconds_per_character=1.5)


def test__filter_too_little_audio_per_character():
    example = TextSpeechRow(
        text=('Lorem ipsum dolor sit amet, consectetur adipiscing elit. ' +
              'Nulla molestie est vitae lorem suscipit, eget vulputate urna pharetra. ' +
              'Cras nec velit augue. ' + 'Praesent malesuada tempus tristique. ' +
              'Suspendisse consequat eros in quam iaculis faucibus eu et tellus. ' +
              'Sed vestibulum faucibus elit. ' + 'Proin vehicula ipsum ac nibh volutpat auctor. ' +
              'Aliquam ac accumsan ante. Vivamus et nisl eu tortor aliquet cursus eget a ex. ' +
              'Phasellus orci ex, hendrerit hendrerit scelerisque vel, rutrum ac odio. ' +
              'Vestibulum mollis pharetra ipsum at fermentum.'),
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert not _filter_too_little_audio_per_character(example, min_seconds_per_character=0.04)

    example = TextSpeechRow(
        text=('At Walsall, in Staffordshire,'),
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert _filter_too_little_audio_per_character(example, min_seconds_per_character=0.04)


def test__split_dataset():
    """
    Ensure that `_split_dataset` splits the dev dataset to be less than `num_second_dev_set`
    in length. So much so, that if the next training example was added to the dev dataset, it'd be
    over `num_second_dev_set`.
    """
    num_second_dev_set = 5
    train, dev = _split_dataset(get_tts_mocks()['dataset'], num_second_dev_set=num_second_dev_set)
    assert sum([get_num_seconds(d.audio_path) for d in dev]) < num_second_dev_set
    assert sum([get_num_seconds(d.audio_path)
                for d in dev]) + get_num_seconds(train[0].audio_path) > num_second_dev_set


def test__preprocess_dataset():
    """ Smoke test to ensure that `_preprocess_dataset` doesn't fail. """
    _preprocess_dataset(get_tts_mocks()['dataset'])
