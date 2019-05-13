from pathlib import Path

from unittest import mock
import shutil
import pytest

from src.datasets import hilary_speech_dataset
from src.utils import Checkpoint

from tests.datasets.utils import _download_file_from_drive_side_effect

data_directory = Path('tests/_test_data/')


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = data_directory / 'Hilary Noriega'
    print("Clean up: removing {}".format(cleanup_dir))
    if cleanup_dir.is_dir():
        shutil.rmtree(str(cleanup_dir))


@mock.patch("src.utils.Checkpoint.from_path")
@mock.patch("torchnlp.download._download_file_from_drive")
@pytest.mark.usefixtures("cleanup")
def test_hilary_speech_dataset(mock_download_file_from_drive, mock_from_path):
    mock_download_file_from_drive.side_effect = _download_file_from_drive_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    data = hilary_speech_dataset(directory=data_directory)
    assert len(data) == 118
