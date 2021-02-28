import pathlib
import shutil
import tempfile
from unittest import mock

import torch
import torch.distributed
from hparams import add_config

import lib
import run
from lib.datasets import JUDY_BIEBER, LINDA_JOHNSON, m_ailabs_en_us_judy_bieber_speech_dataset
from run._config import Cadence, DatasetType
from run._utils import split_dataset
from run.train._utils import CometMLExperiment, Context, Timer, _get_dataset_stats, set_context
from run.train.spectrogram_model.__main__ import _make_configuration
from run.train.spectrogram_model._metrics import Metrics
from run.train.spectrogram_model._worker import _get_data_loaders, _run_inference, _run_step, _State
from tests import _utils


def _mock_distributed_data_parallel(module, *_, **__):
    # NOTE: `module.module = module` would cause the `named_children` property to error, so
    # instead we set a `property`, learn more:
    # https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    module.__class__.module = property(lambda self: module)
    return module


@mock.patch("urllib.request.urlretrieve")
def test_integration(mock_urlretrieve):
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

    add_config(_make_configuration(train_dataset, dev_dataset, True))

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
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = _mock_distributed_data_parallel
        state = _State.from_dataset(train_dataset, dev_dataset, comet, device)
    assert state.model.module == state.model  # Enusre the mock worked
    # fmt: off
    assert state.input_encoder.grapheme_encoder.vocab == [
        '<pad>', '<unk>', '</s>', '<s>', '<copy>', 'a', 't', ' ', 'w', 'l', 's', ',', 'i', 'n', 'f',
        'o', 'r', 'd', 'h', 'e', 'b', 'y', '.', 'm', 'u', 'k', 'g'
    ]
    # fmt: on
    speakers = state.input_encoder.speaker_encoder.vocab
    assert speakers == list(train_dataset.keys())
    assert state.model.vocab_size == state.input_encoder.phoneme_encoder.vocab_size
    assert state.model.num_speakers == state.input_encoder.speaker_encoder.vocab_size

    batch_size = 1
    train_loader, dev_loader = _get_data_loaders(
        state, train_dataset, dev_dataset, batch_size, batch_size, 1, 1, 0, 2
    )

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, state.model, comet):
        timer = Timer()
        metrics = Metrics(store, comet, speakers)
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        _run_step(state, metrics, batch, train_loader, DatasetType.TRAIN, timer, True)
        assert state.step.item() == 1

        # fmt: off
        keys = [
            metrics.ALIGNMENT_NUM_SKIPS, metrics.ALIGNMENT_STD_SUM, metrics.ALIGNMENT_NORM_SUM,
            metrics.NUM_REACHED_MAX, metrics.RMS_SUM_PREDICTED, metrics.RMS_SUM
        ]
        # fmt: on
        for key in keys:
            assert len(metrics.all[key]) == 1
            assert len(metrics.all[f"{key}/{batch.spans[0].speaker.label}"]) == 1
        assert all(metrics.all[metrics.NUM_CORRECT_STOP_TOKEN]) == 1

        num_frames = [[batch.spectrogram.lengths[0].item()]]
        num_tokens = [[batch.encoded_phonemes.lengths[0].item()]]
        num_seconds = [[batch.spans[0].audio_length]]
        bucket = len(batch.spans[0].script) // metrics.TEXT_LENGTH_BUCKET_SIZE
        values = {
            metrics.NUM_FRAMES_MAX: num_frames,
            metrics.NUM_FRAMES_PREDICTED: num_frames,
            f"{metrics.NUM_FRAMES_PREDICTED}/{JUDY_BIEBER.label}": num_frames,
            metrics.NUM_FRAMES: num_frames,
            f"{metrics.NUM_FRAMES}/{JUDY_BIEBER.label}": num_frames,
            metrics.NUM_SECONDS: num_seconds,
            f"{metrics.NUM_SECONDS}/{JUDY_BIEBER.label}": num_seconds,
            f"{metrics.NUM_SPANS_PER_TEXT_LENGTH}/{bucket}": [[batch_size]],
            metrics.NUM_SPANS: [[batch.length]],
            f"{metrics.NUM_SPANS}/{JUDY_BIEBER.label}": [[batch.length]],
            metrics.NUM_TOKENS: num_tokens,
            f"{metrics.NUM_TOKENS}/{JUDY_BIEBER.label}": num_tokens,
        }
        for key, value in values.items():
            assert metrics.all[key] == value

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test `_run_inference` with `Metrics` and `_State`
    with set_context(Context.EVALUATE_INFERENCE, state.model, comet):
        timer = Timer()
        metrics = Metrics(store, comet, speakers)
        batch = next(iter(train_loader))
        _run_inference(state, metrics, batch, dev_loader, DatasetType.DEV, timer, True)
        assert state.step.item() == 1
        total = metrics.all[metrics.NUM_REACHED_MAX][0][0] + metrics.all[metrics.NUM_SPANS][0][0]
        assert total == 1

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test loading and saving a checkpoint
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = _mock_distributed_data_parallel
        loaded = state.from_checkpoint(
            state.to_checkpoint(checkpoints_directory=pathlib.Path(".")), comet, device
        )
    assert state.step == loaded.step
    assert metrics._store.num_keys() == 1
