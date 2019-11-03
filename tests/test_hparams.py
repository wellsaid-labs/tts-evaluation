from src.environment import ROOT_PATH
from src.datasets import TextSpeechRow
from src.datasets import Speaker
from src.datasets import Gender
from src.hparams import _filter_too_little_audio


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
        speaker=Speaker('Linda Johnson', Gender.FEMALE),
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert not _filter_too_little_audio(
        example, min_seconds_per_character=0.04, sample_rate=24000, bits=16)

    example = TextSpeechRow(
        text=('At Walsall, in Staffordshire,'),
        speaker=Speaker('Linda Johnson', Gender.FEMALE),
        audio_path=ROOT_PATH / 'tests/_test_data/_disk/data/LJSpeech-1.1/wavs/LJ005-0210.wav')
    assert _filter_too_little_audio(
        example, min_seconds_per_character=0.04, sample_rate=24000, bits=16)
