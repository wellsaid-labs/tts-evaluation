from pathlib import Path
from unittest import mock

from src.datasets import m_ailabs_en_us_speech_dataset
from src.utils import Checkpoint

from tests.datasets.utils import url_first_side_effect
from tests._utils import create_disk_garbage_collection_fixture

M_AILABS_DIRECTORY = Path('tests/_test_data/M-AILABS')

gc_fixture_data = create_disk_garbage_collection_fixture(M_AILABS_DIRECTORY / 'en_US', autouse=True)


@mock.patch('pathlib.Path.is_file')
@mock.patch('src.utils.Checkpoint.from_path')
@mock.patch('urllib.request.urlretrieve')
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_from_path, mock_is_file):
    mock_is_file.return_value = True
    mock_urlretrieve.side_effect = url_first_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    data = m_ailabs_en_us_speech_dataset(directory=M_AILABS_DIRECTORY)

    assert len(data) == 2046
    assert sum([len(r.text) for r in data]) == 226649
    assert data[0].text == ('To My Readers.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav') in str(
                data[0].audio_path)
