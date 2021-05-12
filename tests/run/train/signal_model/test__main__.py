import pathlib
import tempfile
from unittest import mock

from hparams import add_config

import lib
from run._config import Cadence, DatasetType
from run.train import spectrogram_model
from run.train._utils import Context, Timer, save_checkpoint, set_context
from run.train.signal_model.__main__ import _make_configuration
from run.train.signal_model._metrics import Metrics
from run.train.signal_model._worker import (
    _get_data_loaders,
    _HandleBatchArgs,
    _run_step,
    _State,
    _visualize_inferred,
    _visualize_inferred_end_to_end,
)
from tests.run.train._utils import mock_distributed_data_parallel, setup_experiment


def _make_checkpoint(temp_dir, train_dataset, dev_dataset, comet, device, name="spectrogram"):
    add_config(spectrogram_model.__main__._make_configuration(train_dataset, dev_dataset, True))
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        from_dataset = spectrogram_model._worker._State.from_dataset
        checkpoint = from_dataset(train_dataset, dev_dataset, comet, device).to_checkpoint()
        checkpoint_path = save_checkpoint(checkpoint, pathlib.Path(temp_dir.name), name)
    return checkpoint_path


def test_integration():
    train_dataset, dev_dataset, comet, device = setup_experiment()
    temp_dir = tempfile.TemporaryDirectory()
    checkpoint_path = _make_checkpoint(temp_dir, train_dataset, dev_dataset, comet, device)
    add_config(_make_configuration(train_dataset, dev_dataset, True))

    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        state = _State.make(checkpoint_path, comet, device)
    speakers = state.spectrogram_model_input_encoder.speaker_encoder.vocab

    batch_size = 1
    train_loader, dev_loader = _get_data_loaders(
        state, train_dataset, dev_dataset, batch_size, batch_size, 4, 4, 1, 1, 1, 1, 0, 2
    )

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, comet, *state.models, ema=state.ema):
        timer = Timer()
        metrics = Metrics(comet, speakers)
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
        _visualize_inferred_end_to_end(state, dev_loader, DatasetType.DEV)

    # Test loading and saving a checkpoint
    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        checkpoint = state.to_checkpoint()
        checkpoint_path = save_checkpoint(checkpoint, pathlib.Path(temp_dir.name), "signal")
        checkpoint = lib.environment.load(checkpoint_path)
        loaded = state.from_checkpoint(checkpoint, comet, device)

    assert state.step == loaded.step
