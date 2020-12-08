import pathlib
import shutil
import tempfile
from unittest import mock

import torch

import lib
import run
from lib.datasets import JUDY_BIEBER, LINDA_JOHNSON, m_ailabs_en_us_judy_bieber_speech_dataset
from run._config import Cadence, DatasetType
from run._utils import Context
from run.train.spectrogram_model import (
    _configure,
    _DistributedMetrics,
    _get_data_loaders,
    _run_inference,
    _run_step,
    _State,
)
from tests import _utils


def test__configure():
    """ Test `spectrogram_model._configure` finds and configures modules. """
    _configure({})


def _mock_distributed_data_parallel(module, *_, **__):
    # NOTE: `module.module = module` would cause the `named_children` property to error, so
    # instead we set a `property`, learn more:
    # https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    module.__class__.module = property(lambda self: module)
    return module


@mock.patch("urllib.request.urlretrieve")
def test_integration(mock_urlretrieve):
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect

    _configure({})

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
    run._utils.normalize_audio(dataset)
    dev_speakers = set([JUDY_BIEBER])
    train_dataset, dev_dataset = run._config.split_dataset(dataset, dev_speakers, 3)

    # Check dataset statistics are correct
    stats = run._utils.get_dataset_stats(train_dataset, dev_dataset)
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
    comet = run._utils.CometMLExperiment(disabled=True, project_name="project name")
    device = torch.device("cpu")
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = _mock_distributed_data_parallel
        state = _State.from_dataset(train_dataset, dev_dataset, comet, device)
    assert state.model.module == state.model  # Enusre the mock worked
    # fmt: off
    assert state.input_encoder.grapheme_encoder.vocab == [
        '<pad>', '<unk>', '</s>', '<s>', '<copy>', 't', 'h', 'e', ' ', 'b', 'o', 'y', 'n', 'd', '.',
        'm', 'r', 'a', 's', 'w', 'l', ',', 'i', 'f', 'u', 'k', 'g'
    ]
    # fmt: on
    assert state.input_encoder.speaker_encoder.vocab == list(train_dataset.keys())
    assert state.model.vocab_size == state.input_encoder.grapheme_encoder.vocab_size
    assert state.model.num_speakers == state.input_encoder.speaker_encoder.vocab_size

    batch_size = 1
    train_loader, dev_loader = _get_data_loaders(
        state, train_dataset, dev_dataset, batch_size, batch_size, 100, 1
    )

    # Test `_run_step` with `_DistributedMetrics` and `_State`
    with run._utils.set_context(Context.TRAIN, state.model, comet):
        metrics = _DistributedMetrics()
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        _run_step(state, metrics, batch, train_loader, DatasetType.TRAIN, True)
        assert state.step.item() == 1
        assert metrics.batch_size == [batch.length]
        assert metrics.num_frames == [batch.spectrogram.lengths[0].item()]
        assert metrics.num_spans_per_text_length == {
            len(batch.text[0]) // metrics.text_length_bucket_size: 1.0
        }
        assert metrics.num_frames_per_speaker == {
            batch.speaker[0]: batch.spectrogram.lengths[0].item()
        }
        assert list(metrics.num_skips_per_speaker.keys()) == batch.speaker
        assert metrics.num_tokens_per_speaker == {
            batch.speaker[0]: batch.encoded_text.lengths[0].item()
        }
        assert len(metrics.predicted_frame_alignment_std) == 1
        assert len(metrics.predicted_frame_alignment_norm) == 1
        assert len(metrics.stop_token_num_correct) == 1

        metrics.log(comet, lambda l: l[-1], DatasetType.TRAIN, Cadence.STEP)
        metrics.log(comet, sum, DatasetType.TRAIN, Cadence.MULTI_STEP)

    # Test `_run_inference` with `_DistributedMetrics` and `_State`
    with run._utils.set_context(Context.EVALUATE_INFERENCE, state.model, comet):
        metrics = _DistributedMetrics()
        batch = next(iter(dev_loader))
        _run_inference(state, metrics, batch, dev_loader, DatasetType.DEV, True)
        assert state.step.item() == 1
        assert metrics.num_reached_max_frames[0] + metrics.batch_size[0] == 1

        metrics.log(comet, lambda l: l[-1], DatasetType.DEV, Cadence.STEP)
        metrics.log(comet, sum, DatasetType.DEV, Cadence.MULTI_STEP)

    # Test loading and saving a checkpoint
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = _mock_distributed_data_parallel
        loaded = state.from_checkpoint(
            state.to_checkpoint(checkpoints_directory=pathlib.Path(".")), comet, device
        )
    assert state.step == loaded.step
