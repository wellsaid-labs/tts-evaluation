import numpy as np

from src.bin.evaluate import _save
from src.bin.evaluate import main
from src.datasets import Gender
from src.datasets import Speaker
from src.environment import TEMP_PATH
from src.environment import SAMPLES_PATH
from tests._utils import get_tts_mocks


def test__save():
    filename = _save(TEMP_PATH, ['tag_1', 'tag_2'], Speaker('First Last', Gender.FEMALE),
                     np.array([0, 0.5, 0, -0.5]))
    expected_filename = 'speaker=first_last,tag_1,tag_2.wav'
    assert filename == expected_filename
    assert (TEMP_PATH / expected_filename).exists()


def test__save_obscure():
    filename = _save(
        TEMP_PATH, ['tag_1', 'tag_2'],
        Speaker('First Last', Gender.FEMALE),
        np.array([0, 0.5, 0, -0.5]),
        obscure=True)
    expected_filename = 'speaker=first_last,tag_1,tag_2'
    assert filename == '%x.wav' % hash(expected_filename)


def test_main(capsys):
    """ Test evaluation of the TTS pipeline. """
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        mocks = get_tts_mocks()
        num_samples = 2
        metadata_filename = 'metadata.csv'
        main(
            dataset=mocks['dev_dataset'],
            signal_model_checkpoint=mocks['signal_model_checkpoint'],
            spectrogram_model_checkpoint=mocks['spectrogram_model_checkpoint'],
            num_samples=num_samples,
            metadata_filename=metadata_filename,
            spectrogram_model_device=mocks['device'])
        assert (SAMPLES_PATH / metadata_filename).exists()
        assert len(list(SAMPLES_PATH.glob('*.log'))) == 2
        assert len(list(SAMPLES_PATH.glob('*.wav'))) == num_samples * 3


def test_main__no_checkpoints(capsys):
    """ Test evaluation of the target dataset without any generation. """
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        mocks = get_tts_mocks()
        num_samples = 2
        metadata_filename = 'metadata.csv'
        main(
            dataset=mocks['dev_dataset'],
            signal_model_checkpoint=None,
            spectrogram_model_checkpoint=None,
            obscure=True,
            num_samples=num_samples,
            metadata_filename=metadata_filename)
        assert (SAMPLES_PATH / metadata_filename).exists()
        assert len(list(SAMPLES_PATH.glob('*.log'))) == 2
        assert len(list(SAMPLES_PATH.glob('*.wav'))) == num_samples


def test_main__text_only_dataset(capsys):
    """ Test evaluation of the custom text without the corresponding audio. """
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        mocks = get_tts_mocks()
        num_samples = 2
        metadata_filename = 'metadata.csv'
        dataset = [e._replace(audio_path=None) for e in mocks['dev_dataset']]
        main(
            dataset=dataset,
            signal_model_checkpoint=mocks['signal_model_checkpoint'],
            spectrogram_model_checkpoint=mocks['spectrogram_model_checkpoint'],
            num_samples=num_samples,
            metadata_filename=metadata_filename)
        assert (SAMPLES_PATH / metadata_filename).exists()
        assert len(list(SAMPLES_PATH.glob('*.log'))) == 2
        assert len(list(SAMPLES_PATH.glob('*.wav'))) == num_samples * 2
