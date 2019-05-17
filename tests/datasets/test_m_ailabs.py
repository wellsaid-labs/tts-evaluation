from pathlib import Path

from unittest import mock
import shutil
import pytest

from src.datasets import m_ailabs_speech_dataset
from src.datasets.m_ailabs import ELLIOT_MILLER
from src.datasets.m_ailabs import Gender
from src.datasets.m_ailabs import THE_SEA_FAIRIES
from src.utils import Checkpoint

from tests.datasets.utils import url_first_side_effect

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
    mock_urlretrieve.side_effect = url_first_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    data = m_ailabs_speech_dataset(directory=M_AILABS_DIRECTORY)

    assert len(data) == 2046
    assert sum([len(r.text) for r in data]) == 226649
    assert data[0].text == ('To My Readers.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav') in str(
                data[0].audio_path)


@mock.patch('src.utils.Checkpoint.from_path')
@mock.patch('urllib.request.urlretrieve')
@pytest.mark.usefixtures('cleanup')
def test_m_ailabs_speech_dataset_pickers(mock_urlretrieve, mock_from_path):
    mock_urlretrieve.side_effect = url_first_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Smoke test pickers
    _ = m_ailabs_speech_dataset(picker=Gender.FEMALE, directory=M_AILABS_DIRECTORY)
    _ = m_ailabs_speech_dataset(picker=ELLIOT_MILLER, directory=M_AILABS_DIRECTORY)
    _ = m_ailabs_speech_dataset(picker=THE_SEA_FAIRIES, directory=M_AILABS_DIRECTORY)
