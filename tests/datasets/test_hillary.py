from pathlib import Path

import mock
import shutil
import pytest

from src.datasets import hillary_dataset
from tests.datasets.utils import _download_file_from_drive_side_effect

data_directory = Path('tests/_test_data/')


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = data_directory / 'Hillary'
    print("Clean up: removing {}".format(cleanup_dir))
    shutil.rmtree(str(cleanup_dir))


@mock.patch("torchnlp.download._download_file_from_drive")
@pytest.mark.usefixtures("cleanup")
def test_hillary_dataset(mock_download_file_from_drive):
    mock_download_file_from_drive.side_effect = _download_file_from_drive_side_effect

    # Check a row are parsed correctly
    train, dev = hillary_dataset(
        directory=data_directory,
        resample=None,
        norm=False,
        guard=False,
        splits=(0.8, 0.2),
        check_wavfiles=False)
    assert len(train) == 94
    assert len(dev) == 24


@mock.patch("src.datasets._process.os.system")
@mock.patch("torchnlp.download._download_file_from_drive")
@pytest.mark.usefixtures("cleanup")
def test_hillary_dataset_audio_processing(mock_download_file_from_drive, mock_os_system):
    mock_download_file_from_drive.side_effect = _download_file_from_drive_side_effect
    mock_os_system.return_value = None

    # Check a row are parsed correctly
    train, dev = hillary_dataset(
        directory=data_directory,
        resample=24000,
        norm=True,
        guard=True,
        splits=(0.8, 0.2),
        check_wavfiles=False)

    # Ensure that every argument loudness, upper_hertz, lower_hertz, guard, norm and resample
    # is run
    assert 'norm' in mock_os_system.call_args[0][0]
    assert 'guard' in mock_os_system.call_args[0][0]
    assert 'rate 24000' in mock_os_system.call_args[0][0]
