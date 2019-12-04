from src.audio import get_num_seconds
from src.datasets import LINDA_JOHNSON
from src.datasets import TextSpeechRow
from src.environment import ROOT_PATH
from src.hparams import _filter_too_little_audio
from src.hparams import _split_dataset
from src.hparams import _preprocess_dataset
from tests._utils import get_tts_mocks


def test__filter_too_little_audio():
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
    assert not _filter_too_little_audio(example, min_seconds_per_character=0.04)

    example = TextSpeechRow(
        text=('At Walsall, in Staffordshire,'),
        speaker=LINDA_JOHNSON,
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert _filter_too_little_audio(example, min_seconds_per_character=0.04)


def test__split_dataset():
    """
    Ensure that `_split_dataset` splits the dev dataset to be less than `num_second_dev_set`
    in length. So much so, that if the next training example was added to the dev dataset, it'd be over
    `num_second_dev_set`.
    """
    num_second_dev_set = 5
    train, dev = _split_dataset(get_tts_mocks()['dataset'], num_second_dev_set=num_second_dev_set)
    assert sum([get_num_seconds(d.audio_path) for d in dev]) < num_second_dev_set
    assert sum([get_num_seconds(d.audio_path)
                for d in dev]) + get_num_seconds(train[0].audio_path) >= num_second_dev_set


def test__preprocess_dataset():
    """ Smoke test to ensure that `_preprocess_dataset` doesn't fail. """
    _preprocess_dataset(get_tts_mocks()['dataset'])
