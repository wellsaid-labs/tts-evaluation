from pathlib import Path
from unittest import mock

import re

from src.datasets import lj_speech_dataset
from src.utils import Checkpoint

from tests.datasets.utils import url_first_side_effect
from tests._utils import create_disk_garbage_collection_fixture

lj_directory = Path('tests/_test_data/')

verbalize_test_cases = {
    'LJ044-0055': 'five four four Camp Street New',  # Test special case
    'LJ032-0036': 'Number two two zero two one three zero four six two',  # Test special case
    # Test time
    'LJ036-0167': 'he would have entered the cab at twelve forty-seven or twelve forty-eight p.m.',
    'LJ002-0215': 'the Star Chamber in the sixteenth Charles',  # Test Ordinals
    'LJ013-0088': 'England notes for one thousand pounds each,',  # Test currency
    'LJ037-0204': 'Post Office Box two nine one five,',  # Test PO Box
    'LJ032-0025': 'bearing serial number C two seven six six',  # Test Serial
    'LJ028-0257': 'five twenty-one Nebuchadnezzar the third',  # Test Year
    'LJ028-0363': 'June thirteen, three twenty-three B.C.,',  # Test Year
    'LJ047-0127': 'On August twenty-one, nineteen sixty-three, Bureau',  # Test Year
    'LJ016-0090': 'them towards Number one, Newgate Street',  # Test Numero
    'LJ037-0252': 'Commission Exhibit Number one sixty-two',  # Test Numero
    # Test Number
    'LJ039-0063': 'rifle, at a range of one hundred seventy-seven to two hundred sixty-six feet',
    'LJ004-0049':
        'Mister Gurney, Mister Fry, Messrs Forster, and Mister T. F. Buxton',  # Test Abbreviations
    'LJ011-0064': 'Reverend Mister Springett',  # Test Abbreviations,
    'LJ047-0160':
        'found it to be four one one Elm Street. End quote.',  # Test Normalized White Space
    'LJ017-0007': 'Henry the eighth a new',  # Test Roman Numbers
    'LJ016-0257': 'd\'etre',  # Test Remove Accents
    'LJ018-0029': 'Muller',  # Test Remove Accents
    'LJ018-0396': 'celebre',  # Test Remove Accents
    'LJ020-0106': 'three hours\'',  # Test Quotation Normalization
    'LJ020-0002': '"sponge,"',  # Test Quotation Normalization
}

gc_fixture_data = create_disk_garbage_collection_fixture(
    lj_directory / 'LJSpeech-1.1', autouse=True)


@mock.patch("src.utils.Checkpoint.from_path")
@mock.patch("urllib.request.urlretrieve")
def test_lj_speech_dataset(mock_urlretrieve, mock_from_path):
    mock_urlretrieve.side_effect = url_first_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    data = lj_speech_dataset(directory=lj_directory)

    assert len(data) == 13100
    assert sum([len(r.text) for r in data]) == 1310332
    assert data[0].text == (
        'Printing, in the only sense with which we are at present concerned, differs from most if '
        'not from all the arts and crafts represented in the Exhibition')
    assert 'tests/_test_data/LJSpeech-1.1/wavs/LJ001-0001.wav' in str(data[0].audio_path)

    _re_filename = re.compile('LJ[0-9]{3}-[0-9]{4}')

    # Test verbilization
    seen = 0
    for row in data:
        basename = row.audio_path.name[:10]
        assert _re_filename.match(basename)
        if basename in verbalize_test_cases:
            seen += 1
            assert verbalize_test_cases[basename] in row.text
    assert seen == len(verbalize_test_cases)
