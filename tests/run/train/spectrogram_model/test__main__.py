import typing
from unittest import mock

from hparams import add_config

from run._config import Cadence, DatasetType
from run.data._loader.english import JUDY_BIEBER
from run.train._utils import Context, Timer, set_context
from run.train.spectrogram_model.__main__ import _make_configuration
from run.train.spectrogram_model._metrics import Metrics, MetricsKey
from run.train.spectrogram_model._model import SpectrogramModel
from run.train.spectrogram_model._worker import (
    _get_data_loaders,
    _HandleBatchArgs,
    _log_vocab,
    _run_inference,
    _run_step,
)
from tests.run._utils import make_spec_worker_state, mock_distributed_data_parallel
from tests.run.train._utils import setup_experiment


def test_integration():
    train_dataset, dev_dataset, comet, device = setup_experiment()
    add_config(_make_configuration(train_dataset, dev_dataset, True))
    state = make_spec_worker_state(comet, device)

    assert state.model.module == state.model  # Ensure the mock worked
    model = typing.cast(SpectrogramModel, state.model)

    batch_size = 2
    train_loader, dev_loader = _get_data_loaders(
        state, train_dataset, dev_dataset, batch_size, batch_size, 1, 1, False, True, 0, 2
    )

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, comet, state.model):
        timer = Timer()
        metrics = Metrics(comet)
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        args = (state, train_loader, Context.TRAIN, DatasetType.TRAIN, metrics, timer, batch, True)
        _run_step(_HandleBatchArgs(*args))
        assert state.step.item() == 1

        is_not_diff = lambda b, v: len(set(b) - set(v.keys())) == 0
        characters = [c for s in batch.inputs.tokens for c in s]
        assert is_not_diff(characters, model.token_vocab)
        assert is_not_diff((s.speaker for s in batch.spans), model.speaker_vocab)
        assert is_not_diff((s.session for s in batch.spans), model.session_vocab)

        # fmt: off
        keys = [
            metrics.ALIGNMENT_NUM_SKIPS, metrics.ALIGNMENT_STD_SUM, metrics.ALIGNMENT_NORM_SUM,
            metrics.NUM_REACHED_MAX, metrics.RMS_SUM_PREDICTED, metrics.RMS_SUM
        ]
        # fmt: on
        for key in keys:
            assert len(metrics.data[MetricsKey(key)]) == 1
            assert len(metrics.data[MetricsKey(key, batch.spans[0].speaker)]) == 1
        assert all(metrics.data[MetricsKey(metrics.NUM_CORRECT_STOP_TOKEN)]) == 1

        max_frames = [(batch.spectrogram.lengths.max().item(),)]
        num_frames = [(batch.spectrogram.lengths.sum().item(),)]
        num_tokens = [(sum(len(t) for t in batch.inputs.tokens),)]
        num_seconds = [(sum(s.audio_length for s in batch.spans),)]
        bucket = len(batch.spans[0].script) // metrics.TEXT_LENGTH_BUCKET_SIZE
        values = {
            MetricsKey(metrics.NUM_FRAMES_MAX): max_frames,
            MetricsKey(metrics.NUM_FRAMES_PREDICTED): num_frames,
            MetricsKey(metrics.NUM_FRAMES_PREDICTED, JUDY_BIEBER): num_frames,
            MetricsKey(metrics.NUM_FRAMES): num_frames,
            MetricsKey(metrics.NUM_FRAMES, JUDY_BIEBER): num_frames,
            MetricsKey(metrics.NUM_SECONDS): num_seconds,
            MetricsKey(metrics.NUM_SECONDS, JUDY_BIEBER): num_seconds,
            MetricsKey(metrics.NUM_SPANS_PER_TEXT_LENGTH, None, bucket): [(batch_size,)],
            MetricsKey(metrics.NUM_SPANS): [(len(batch),)],
            MetricsKey(metrics.NUM_SPANS, JUDY_BIEBER): [(len(batch),)],
            MetricsKey(metrics.NUM_TOKENS): num_tokens,
            MetricsKey(metrics.NUM_TOKENS, JUDY_BIEBER): num_tokens,
        }
        for key, value in values.items():
            assert metrics.data[key] == value, str(key)

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)
        _log_vocab(state, DatasetType.TRAIN)

    # Test `_run_inference` with `Metrics` and `_State`
    with set_context(Context.EVALUATE_INFERENCE, comet, state.model):
        timer = Timer()
        metrics = Metrics(comet)
        batch = next(iter(train_loader))
        args = (state, dev_loader, Context.EVALUATE, DatasetType.DEV, metrics, timer, batch, True)
        _run_inference(_HandleBatchArgs(*args))
        assert state.step.item() == 1
        total = sum(metrics.data[MetricsKey(metrics.NUM_REACHED_MAX)][0])
        total += sum(metrics.data[MetricsKey(metrics.NUM_SPANS)][0])
        assert total == batch_size

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test loading and saving a checkpoint
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        loaded = state.from_checkpoint(state.to_checkpoint(), comet, device)
    assert state.step == loaded.step
