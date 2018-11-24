from pathlib import Path

from unittest import mock
import shutil
import pytest

from src.datasets import hilary_dataset
from src.utils import Checkpoint

from tests.datasets.utils import _download_file_from_drive_side_effect
from tests.datasets.utils import compute_spectrogram_side_effect

data_directory = Path('tests/_test_data/')


@pytest.fixture
def cleanup():
    yield
    cleanup_dir = data_directory / 'Hilary'
    print("Clean up: removing {}".format(cleanup_dir))
    shutil.rmtree(str(cleanup_dir))


@mock.patch("src.utils.Checkpoint.from_path")
@mock.patch("src.datasets.hilary.compute_spectrogram")
@mock.patch("torchnlp.download._download_file_from_drive")
@pytest.mark.usefixtures("cleanup")
def test_hilary_dataset(mock_download_file_from_drive, mock_compute_spectrogram, mock_from_path):
    mock_download_file_from_drive.side_effect = _download_file_from_drive_side_effect
    mock_compute_spectrogram.side_effect = compute_spectrogram_side_effect
    mock_from_path.return_value = Checkpoint(directory='.', model=lambda x: x, step=0)

    # Check a row are parsed correctly
    train, dev = hilary_dataset(
        directory=data_directory, resample=None, norm=False, guard=False, splits=(0.8, 0.2))
    assert len(train) == 94
    assert len(dev) == 24
