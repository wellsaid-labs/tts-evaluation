import pathlib
import tempfile
from unittest import mock

import config as cf
import pytest

import lib
from run._config import (
    Cadence,
    DatasetType,
    make_signal_model_train_config,
    make_spectrogram_model_train_config,
)
from run.data._loader.english.lj_speech import LINDA_JOHNSON
from run.data._loader.english.m_ailabs import JUDY_BIEBER
from run.data._loader.structures import Language
from run.train._utils import Context, Timer, save_checkpoint, set_context
from run.train.signal_model._metrics import Metrics
from run.train.signal_model._worker import (
    _get_data_loaders,
    _HandleBatchArgs,
    _run_step,
    _visualize_inferred,
    _visualize_select_cases,
)
from tests.run._utils import make_spec_and_sig_worker_state, mock_distributed_data_parallel
from tests.run.train._utils import setup_experiment


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    yield
    cf.purge()


def test_integration():
    train_dataset, dev_dataset, comet, device = setup_experiment()
    cf.add(make_spectrogram_model_train_config(train_dataset, dev_dataset, True))
    cf.add(make_signal_model_train_config(train_dataset, dev_dataset, True))
    _, state, __ = make_spec_and_sig_worker_state(comet, device)

    batch_size = 1
    train_loader, dev_loader = _get_data_loaders(
        state=state,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        train_batch_size=batch_size,
        dev_batch_size=batch_size,
        train_slice_size=4,
        dev_slice_size=4,
        train_span_bucket_size=1,
        dev_span_bucket_size=1,
        train_steps_per_epoch=1,
        dev_steps_per_epoch=1,
        num_workers=0,
        prefetch_factor=2,
    )

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, comet, *state.models, ema=state.ema):
        timer = Timer()
        metrics = Metrics(comet)
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        args = (state, train_loader, Context.TRAIN, DatasetType.TRAIN, metrics, timer, batch)
        _run_step(_HandleBatchArgs(*args))
        assert state.step.item() == 1

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test inference visualizations
    with set_context(Context.EVALUATE_INFERENCE, comet, *state.models, ema=state.ema):
        _visualize_inferred(state, dev_loader, DatasetType.DEV)
        _visualize_select_cases(
            state,
            DatasetType.TEST,
            Cadence.MULTI_STEP,
            cases=[(Language.ENGLISH, "Hi There")],
            speakers={JUDY_BIEBER, LINDA_JOHNSON},
            num_cases=1,
        )

    # Test loading and saving a checkpoint
    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        checkpoint = state.to_checkpoint()
        temp_dir = tempfile.TemporaryDirectory()
        checkpoint_path = save_checkpoint(checkpoint, pathlib.Path(temp_dir.name), "ckpt")
        checkpoint = lib.environment.load(checkpoint_path)
        loaded = state.from_checkpoint(checkpoint, comet, device)

    assert state.step == loaded.step
