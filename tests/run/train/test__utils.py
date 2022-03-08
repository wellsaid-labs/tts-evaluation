import tempfile
import time
from pathlib import Path

import hparams
import pytest
import torch
import torch.distributed
import torch.nn
from matplotlib import pyplot

import lib
import run
from run._config import Cadence, DatasetType, get_dataset_label
from run.data._loader import Alignment, make_en_speaker
from tests._utils import TEST_DATA_PATH
from tests.run._utils import make_passage

TEST_DATA_PATH = TEST_DATA_PATH / "audio"
TEST_DATA_LJ = TEST_DATA_PATH / "bit(rate(lj_speech,24000),32).wav"


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    hparams.clear_config()


def test__maybe_make_experiment_directories(capsys):
    """Test `maybe_make_experiment_directories` creates a directory structure."""
    with tempfile.TemporaryDirectory() as directory:
        with capsys.disabled():  # NOTE: Disable capsys because it messes with `sys.stdout`
            path = Path(directory)
            recorder = lib.environment.RecordStandardStreams()
            run_name = "run_name"
            checkpoints_directory_name = "checkpoints"
            run_log_filename = "run.log"
            run_root, checkpoints_directory = run.train._utils._maybe_make_experiment_directories(
                path,
                recorder,
                run_name=run_name,
                checkpoints_directory_name=checkpoints_directory_name,
                run_log_filename=run_log_filename,
            )
            assert run_root.is_dir()
            assert run_root.parent == path
            assert run_root.name == run_name
            assert checkpoints_directory.is_dir()
            assert checkpoints_directory.parent == run_root
            assert checkpoints_directory.name == checkpoints_directory_name
            assert recorder.log_path.parent == run_root
            assert recorder.log_path.name == run_log_filename


def test__maybe_make_experiment_directories_from_checkpoint(capsys):
    """Test `maybe_make_experiment_directories_from_checkpoint` creates a directory structure."""
    with tempfile.TemporaryDirectory() as directory:
        with capsys.disabled():  # NOTE: Disable capsys because it messes with `sys.stdout`
            path = Path(directory)
            recorder = lib.environment.RecordStandardStreams()
            run_name = "run"
            checkpoints_directory_name = "checkpoints"
            run_log_filename = "run.log"
            run_root, checkpoints_directory = run.train._utils._maybe_make_experiment_directories(
                path,
                recorder,
                run_name=run_name,
                checkpoints_directory_name=checkpoints_directory_name,
                run_log_filename=run_log_filename,
            )
            run_suffix = "_1"
            run_root, _ = run.train._utils._maybe_make_experiment_directories_from_checkpoint(
                checkpoints_directory / "checkpoint.pt",
                recorder,
                run_prefix=run_name,
                run_suffix=run_suffix,
            )
            assert run_root.parent == path
            assert run_root.name == run_name + run_suffix


def test__get_dataset_stats():
    """Test `run.train._utils.get_dataset_stats` measures dataset statistics correctly."""
    _alignment = lambda a, b: Alignment((a, b), (a * 10, b * 10), (a, b))
    _passage = lambda a, b, s: make_passage([_alignment(a, b)], s)
    a = make_en_speaker("a")
    b = make_en_speaker("b")
    train = {a: [_passage(0, 2, a), _passage(0, 2, a)], b: [_passage(0, 1, a)]}
    stats = run.train._utils._get_dataset_stats(train, {})
    static = Cadence.STATIC
    get_label = lambda n, t, s=None: get_dataset_label(n, cadence=static, type_=t, speaker=s)
    assert stats == {
        get_label("num_passages", DatasetType.TRAIN): 3,
        get_label("num_characters", DatasetType.TRAIN): 5,
        get_label("num_seconds", DatasetType.TRAIN): "50s 0ms",
        get_label("num_audio_files", DatasetType.TRAIN): 1,  # NOTE: This counts unique audio files.
        get_label("num_passages", DatasetType.TRAIN, a): 2,
        get_label("num_characters", DatasetType.TRAIN, a): 4,
        get_label("num_seconds", DatasetType.TRAIN, a): "40s 0ms",
        get_label("num_audio_files", DatasetType.TRAIN, a): 1,
        get_label("num_passages", DatasetType.TRAIN, b): 1,
        get_label("num_characters", DatasetType.TRAIN, b): 1,
        get_label("num_seconds", DatasetType.TRAIN, b): "10s 0ms",
        get_label("num_audio_files", DatasetType.TRAIN, b): 1,
        get_label("num_passages", DatasetType.DEV): 0,
        get_label("num_characters", DatasetType.DEV): 0,
        get_label("num_seconds", DatasetType.DEV): "0ms",
        get_label("num_audio_files", DatasetType.DEV): 0,
    }


def test_comet_ml_experiment():
    """Test if `run.train._utils.CometMLExperimentw` initializes, and the patched functions
    execute."""
    comet = run.train._utils.CometMLExperiment(disabled=True)
    with comet.context_manager(run.train._utils.Context.TRAIN):
        assert comet.context == str(run.train._utils.Context.TRAIN.value)
        comet.set_step(None)
        comet.set_step(0)
        comet.set_step(0)
        comet.set_step(1)
        comet.log_html_audio(
            metadata="random metadata",
            audio={"predicted_audio": torch.rand(100), "gold_audio": torch.rand(100)},
            speaker=make_en_speaker(""),
        )
        comet.log_npy("random", make_en_speaker(""), torch.rand(100))
        figure = pyplot.figure()
        pyplot.close(figure)
        comet.log_figures({run._config.Label("figure"): figure})
        comet.log_current_epoch(0)
        comet.log_epoch_end(0)
        comet.set_name("name")
        comet.add_tags(["tag"])


def test_set_context():
    """Test `run.train._utils.set_context` updates comet, module, and grad context."""
    comet = run.train._utils.CometMLExperiment(disabled=True)
    rnn = torch.nn.LSTM(10, 20, 2).eval()
    ema = lib.optimizers.ExponentialMovingParameterAverage(rnn.parameters())

    assert not rnn.training
    with run.train._utils.set_context(run.train._utils.Context.TRAIN, comet, rnn, ema=ema):
        assert rnn.training
        assert comet.context == run.train._utils.Context.TRAIN.value
        output, _ = rnn(torch.randn(5, 3, 10))
        assert output.requires_grad
        assert len(ema.backup) == 0
    assert not rnn.training

    with run.train._utils.set_context(run.train._utils.Context.EVALUATE, comet, rnn, ema=ema):
        assert not rnn.training
        assert comet.context == run.train._utils.Context.EVALUATE.value
        output, _ = rnn(torch.randn(5, 3, 10))
        assert not output.requires_grad
        assert len(ema.backup) > 0


def test__nested_to_flat_config():
    """Test `_nested_to_flat_config` flattens nested dicts, including edge cases with
    an empty dict."""
    assert (
        run.train._utils._nested_to_flat_config(
            {
                "a": {
                    "b": "c",
                    "d": {
                        "e": "f",
                    },
                },
                "g": "h",
                "i": {},
                "j": [],
            },
            delimitator=".",
        )
        == {"a.b": "c", "a.d.e": "f", "g": "h", "j": []}
    )


def test_timer():
    """Test `run.train._utils.Timer` times an event from start to end."""
    wait = 0.05
    name = "start"
    prefix = "prefix/"
    timer = run.train._utils.Timer(prefix=prefix)
    timer.record_event(name)
    time.sleep(wait)
    timers = timer.get_timers()
    assert len(timers) == 1
    assert timers[run._config.get_timer_label(f"{prefix}{name}")] > wait
