from pathlib import Path

import mock
import shutil
import pytest

from src.datasets import m_ailabs_speech_dataset
from src.datasets.m_ailabs import Gender
from src.datasets.m_ailabs import Speaker
from src.datasets.m_ailabs import THE_SEA_FAIRIES
from tests.datasets.utils import urlretrieve_side_effect

M_AILABS_DIRECTORY = Path('tests/_test_data/M-AILABS')


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = M_AILABS_DIRECTORY / 'en_US'
    print("Clean up: removing {}".format(cleanup_dir))
    shutil.rmtree(str(cleanup_dir))


@mock.patch("urllib.request.urlretrieve")
@pytest.mark.usefixtures("cleanup")
def test_m_ailabs_speech_dataset(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Check a row are parsed correctly
    train, dev = m_ailabs_speech_dataset(
        directory=M_AILABS_DIRECTORY,
        resample=None,
        norm=False,
        guard=False,
        lower_hertz=None,
        upper_hertz=None,
        loudness=False,
        splits=(0.8, 0.2),
        check_wavfiles=False)
    assert len(train) == round(2046 * 0.8)
    assert len(dev) == round(2046 * 0.2)

    # Check sum to ensure its the same exact split
    assert sum([len(r['text']) for r in dev]) == 45205
    assert sum([len(r['text']) for r in train]) == 181444

    # Test deterministic shuffle
    assert train[0]['text'] == ('Dorothy sprang forward and caught the fluffy fowl in her arms, '
                                'uttering at the same time a glad cry.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_15_f000101.wav') in str(
                train[0]['wav_filename'])
    assert dev[0]['text'] == (
        'This, said the man, taking up a box and handling it gently, contains twelve '
        'dozen rustles enough to last any lady a year. Will you buy it, my dear? he asked, '
        'addressing Dorothy.')
    assert ('tests/_test_data/M-AILABS/en_US/by_book/female/judy_bieber/'
            'dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_10_f000048.wav') in str(
                dev[0]['wav_filename'])


@mock.patch("urllib.request.urlretrieve")
@pytest.mark.usefixtures("cleanup")
def test_m_ailabs_speech_dataset_pickers(mock_urlretrieve):
    mock_urlretrieve.side_effect = urlretrieve_side_effect

    # Smoke test pickers
    kwargs = {
        'directory': M_AILABS_DIRECTORY,
        'resample': None,
        'norm': False,
        'guard': False,
        'lower_hertz': None,
        'upper_hertz': None,
        'loudness': False,
        'splits': (0.8, 0.2),
        'check_wavfiles': False
    }
    train, dev = m_ailabs_speech_dataset(picker=Gender.FEMALE, **kwargs)
    train, dev = m_ailabs_speech_dataset(picker=Speaker.ELLIOT_MILLER, **kwargs)
    train, dev = m_ailabs_speech_dataset(picker=THE_SEA_FAIRIES, **kwargs)


@mock.patch("src.datasets._process.os.system")
@mock.patch("urllib.request.urlretrieve")
@pytest.mark.usefixtures("cleanup")
def test_m_ailabs_speech_dataset_audio_processing(mock_urlretrieve, mock_os_system):
    mock_urlretrieve.side_effect = urlretrieve_side_effect
    mock_os_system.return_value = None

    # Check a row are parsed correctly
    train, dev = m_ailabs_speech_dataset(
        directory=M_AILABS_DIRECTORY,
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
