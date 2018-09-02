from pathlib import Path

import mock
import re
import shutil
import pytest
import os

from src.datasets import lj_speech_dataset
from tests.datasets.utils import urlretrieve_side_effect

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


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = os.path.join(lj_directory, 'LJSpeech-1.1')
    print("Clean up: removing {}".format(cleanup_dir))
    shutil.rmtree(cleanup_dir)


@mock.patch("urllib.request.urlretrieve")
@pytest.mark.usefixtures("cleanup")
def test_lj_speech_dataset(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev = lj_speech_dataset(
        directory=lj_directory,
        verbalize=True,
        resample=None,
        norm=False,
        guard=False,
        lower_hertz=None,
        upper_hertz=None,
        loudness=False,
        splits=(0.8, 0.2),
        check_wavfiles=False)
    assert len(train) == 13100 * 0.8
    assert len(dev) == 13100 * 0.2

    # Check sum to ensure its the same exact split
    assert sum([len(r['text']) for r in dev]) == 258045
    assert sum([len(r['text']) for r in train]) == 1052287

    # Test deterministic shuffle
    assert train[0]['text'] == (
        'Once a warrant-holder sent down a clerk to view certain goods, and the clerk found that '
        'these goods had already a "stop" upon them, or were pledged.')
    assert 'tests/_test_data/LJSpeech-1.1/wavs/LJ014-0331.wav' in str(train[0]['wav_filename'])
    assert dev[0]['text'] == (
        'Mister Mullay went, and a second interview was agreed upon, when a third person, '
        'Mister Owen,')
    assert 'tests/_test_data/LJSpeech-1.1/wavs/LJ011-0243.wav' in str(dev[0]['wav_filename'])

    _re_filename = re.compile('LJ[0-9]{3}-[0-9]{4}')

    # Test verbilization
    seen = 0
    for data in [train, dev]:
        for row in data:
            basename = row['wav_filename'].name[:10]
            assert _re_filename.match(basename)
            if basename in verbalize_test_cases:
                seen += 1
                assert verbalize_test_cases[basename] in row['text']
    assert seen == len(verbalize_test_cases)


@mock.patch("src.datasets.process.os.system")
@mock.patch("urllib.request.urlretrieve")
@pytest.mark.usefixtures("cleanup")
def test_lj_speech_dataset_audio_processing(mock_urlretrieve, mock_os_system):
    mock_urlretrieve.side_effect = urlretrieve_side_effect
    mock_os_system.return_value = None

    # Check a row are parsed correctly
    train, dev = lj_speech_dataset(
        directory=lj_directory,
        verbalize=False,
        resample=24000,
        norm=True,
        guard=True,
        lower_hertz=100,
        upper_hertz=200,
        loudness=True,
        splits=(0.8, 0.2),
        check_wavfiles=False)

    # Ensure that every argument loudness, upper_hertz, lower_hertz, guard, norm and resample
    # is run
    assert 'norm' in mock_os_system.call_args[0][0]
    assert 'guard' in mock_os_system.call_args[0][0]
    assert 'rate 24000' in mock_os_system.call_args[0][0]
    assert 'sinc 100-200' in mock_os_system.call_args[0][0]
    assert 'loudness' in mock_os_system.call_args[0][0]
