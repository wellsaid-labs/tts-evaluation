from unittest import mock

from src.datasets import m_ailabs_en_us_speech_dataset
from src.environment import TEST_DATA_PATH
from tests._utils import url_first_side_effect

M_AILABS_DIRECTORY = TEST_DATA_PATH / 'datasets/M-AILABS'


@mock.patch('pathlib.Path.is_file')
@mock.patch('urllib.request.urlretrieve')
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_is_file):
    mock_is_file.return_value = True
    mock_urlretrieve.side_effect = url_first_side_effect

    # Check a row are parsed correctly
    data = m_ailabs_en_us_speech_dataset(directory=M_AILABS_DIRECTORY)

    assert len(data) == 2046
    assert sum([len(r.text) for r in data]) == 226649
    assert data[0].text == ('To My Readers.')
    assert str(M_AILABS_DIRECTORY / 'en_US/by_book/female/judy_bieber' /
               'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav') in str(
                   data[0].audio_path)
