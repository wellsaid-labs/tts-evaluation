from pathlib import Path

from unittest import mock
import shutil
import pytest

from src.datasets import m_ailabs_speech_dataset
from src.datasets.m_ailabs import Gender
from src.datasets.m_ailabs import Speaker
from src.datasets.m_ailabs import THE_SEA_FAIRIES
from src.utils import Checkpoint

from tests.datasets.utils import urlretrieve_side_effect

M_AILABS_DIRECTORY = Path('tests/_test_data/M-AILABS')


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = M_AILABS_DIRECTORY / 'en_US'
    print('Clean up: removing {}'.format(cleanup_dir))
    if cleanup_dir.is_dir():
        shutil.rmtree(str(cleanup_dir))


@mock.patch('pathlib.Path.is_file')
@mock.patch('src.utils.Checkpoint.from_path')
@mock.patch('urllib.request.urlretrieve')
@pytest.mark.usefixtures('cleanup')
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_from_path, mock_is_file):
    mock_is_file.return_value = True
    mock_urlretrieve.side_effect = urlretrieve_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    train, dev = m_ailabs_speech_dataset(
        directory=M_AILABS_DIRECTORY,
        resample=None,
        norm=False,
        guard=False,
        lower_hertz=None,
        upper_hertz=None,
        loudness=False,
        splits=(0.8, 0.2))
    assert len(train) == round(2046 * 0.8)
    assert len(dev) == round(2046 * 0.2)

    # Check sum to ensure its the same exact split
    assert sum([len(r.text) for r in dev]) == 45205
    assert sum([len(r.text) for r in train]) == 181444

    # Test deterministic shuffle
    assert train[0].text == ('Dorothy sprang forward and caught the fluffy fowl in her arms, '
                             'uttering at the same time a glad cry.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_15_f000101.wav') in str(
                train[0].audio_path)
    assert dev[0].text == (
        'This, said the man, taking up a box and handling it gently, contains twelve '
        'dozen rustles enough to last any lady a year. Will you buy it, my dear? he asked, '
        'addressing Dorothy.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_10_f000048.wav') in str(
                dev[0].audio_path)


@mock.patch('src.utils.Checkpoint.from_path')
@mock.patch('urllib.request.urlretrieve')
@pytest.mark.usefixtures('cleanup')
def test_m_ailabs_speech_dataset_pickers(mock_urlretrieve, mock_from_path):
    mock_urlretrieve.side_effect = urlretrieve_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Smoke test pickers
    kwargs = {
        'directory': M_AILABS_DIRECTORY,
        'resample': None,
        'norm': False,
        'guard': False,
        'lower_hertz': None,
        'upper_hertz': None,
        'loudness': False,
        'splits': (0.8, 0.2)
    }
    train, dev = m_ailabs_speech_dataset(picker=Gender.FEMALE, **kwargs)
    train, dev = m_ailabs_speech_dataset(picker=Speaker.ELLIOT_MILLER, **kwargs)
    train, dev = m_ailabs_speech_dataset(picker=THE_SEA_FAIRIES, **kwargs)
