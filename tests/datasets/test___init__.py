from pathlib import Path
from unittest import mock

import pandas

from src.datasets import beth_custom_speech_dataset
from src.datasets import beth_speech_dataset
from src.datasets import heather_speech_dataset
from src.datasets import hilary_speech_dataset
from src.datasets import sam_speech_dataset
from src.datasets import susan_speech_dataset

from tests.datasets.utils import url_second_side_effect
from tests.datasets.utils import url_first_side_effect

data_directory = Path('tests/_test_data/')


@mock.patch("torchnlp.download._download_file_from_drive")
def test_hilary_speech_dataset(mock_download_file_from_drive):
    mock_download_file_from_drive.side_effect = url_second_side_effect

    # Check a row are parsed correctly
    data = hilary_speech_dataset(directory=data_directory)
    assert len(data) == 118


@mock.patch("src.datasets.utils.download_file_maybe_extract")
@mock.patch("pandas.read_csv")
def test_datasets(mock_read_csv, mock_download_file_maybe_extract):
    mock_download_file_maybe_extract.side_effect = url_first_side_effect
    mock_read_csv.return_value = pandas.DataFrame([])

    speech_datasets = [
        sam_speech_dataset, heather_speech_dataset, beth_custom_speech_dataset, beth_speech_dataset,
        susan_speech_dataset
    ]
    for dataset in speech_datasets:
        # Check a row are parsed correctly
        data = dataset(directory=data_directory)
        assert len(data) == 0
