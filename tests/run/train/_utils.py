import pathlib
import shutil
import tempfile
from unittest import mock

import torch
import torch.distributed

import lib
import run
from lib.datasets import JUDY_BIEBER, LINDA_JOHNSON, m_ailabs_en_us_judy_bieber_speech_dataset
from run._config import Cadence, DatasetType
from run._utils import split_dataset
from run.train._utils import CometMLExperiment, _get_dataset_stats
from tests import _utils


@mock.patch("urllib.request.urlretrieve")
def setup_experiment(mock_urlretrieve):
    """Setup basic experiment for testing."""
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect

    run._config.configure()

    # Test loading data
    directory = _utils.TEST_DATA_PATH / "datasets"
    temp_directory = pathlib.Path(tempfile.TemporaryDirectory().name)
    shutil.copytree(directory, temp_directory)
    books = [lib.datasets.m_ailabs.DOROTHY_AND_WIZARD_OZ]
    dataset = {
        JUDY_BIEBER: m_ailabs_en_us_judy_bieber_speech_dataset(temp_directory, books=books),
        LINDA_JOHNSON: lib.datasets.lj_speech_dataset(temp_directory),
    }

    # Test splitting data
    dataset = run._utils.normalize_audio(dataset)
    dev_speakers = set([JUDY_BIEBER])
    train_dataset, dev_dataset = split_dataset(dataset, dev_speakers, 3)

    # Check dataset statistics are correct
    stats = _get_dataset_stats(train_dataset, dev_dataset)
    get_dataset_label = lambda n, t, s=None: run._config.get_dataset_label(
        n, cadence=Cadence.STATIC, type_=t, speaker=s
    )
    assert stats == {
        get_dataset_label("num_passages", DatasetType.TRAIN): 3,
        get_dataset_label("num_characters", DatasetType.TRAIN): 58,
        get_dataset_label("num_seconds", DatasetType.TRAIN): "5s 777ms",
        get_dataset_label("num_audio_files", DatasetType.TRAIN): 3,
        get_dataset_label("num_passages", DatasetType.TRAIN, JUDY_BIEBER): 2,
        get_dataset_label("num_characters", DatasetType.TRAIN, JUDY_BIEBER): 29,
        get_dataset_label("num_seconds", DatasetType.TRAIN, JUDY_BIEBER): "3s 820ms",
        get_dataset_label("num_audio_files", DatasetType.TRAIN, JUDY_BIEBER): 2,
        get_dataset_label("num_passages", DatasetType.TRAIN, LINDA_JOHNSON): 1,
        get_dataset_label("num_characters", DatasetType.TRAIN, LINDA_JOHNSON): 29,
        get_dataset_label("num_seconds", DatasetType.TRAIN, LINDA_JOHNSON): "1s 958ms",
        get_dataset_label("num_audio_files", DatasetType.TRAIN, LINDA_JOHNSON): 1,
        get_dataset_label("num_passages", DatasetType.DEV): 1,
        get_dataset_label("num_characters", DatasetType.DEV): 34,
        get_dataset_label("num_seconds", DatasetType.DEV): "2s 650ms",
        get_dataset_label("num_audio_files", DatasetType.DEV): 1,
        get_dataset_label("num_passages", DatasetType.DEV, JUDY_BIEBER): 1,
        get_dataset_label("num_characters", DatasetType.DEV, JUDY_BIEBER): 34,
        get_dataset_label("num_seconds", DatasetType.DEV, JUDY_BIEBER): "2s 650ms",
        get_dataset_label("num_audio_files", DatasetType.DEV, JUDY_BIEBER): 1,
    }

    # Create training state
    comet = CometMLExperiment(disabled=True, project_name="project name")
    device = torch.device("cpu")
    store = torch.distributed.TCPStore("127.0.0.1", 29500, 0, True)
    return train_dataset, dev_dataset, comet, device, store


def mock_distributed_data_parallel(module, *_, **__):
    # NOTE: `module.module = module` would cause the `named_children` property to error, so
    # instead we set a `property`, learn more:
    # https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    module.__class__.module = property(lambda self: module)
    return module
