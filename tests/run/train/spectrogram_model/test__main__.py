import typing
from unittest import mock

import config as cf
import pytest

from run._config import Cadence, DatasetType, config_spec_model_training_from_datasets
from run._config.train import exclude_from_decay
from run._models.spectrogram_model import SpectrogramModel
from run.data._loader.english.m_ailabs import JUDY_BIEBER
from run.train._utils import Context, Timer, set_context
from run.train.spectrogram_model._metrics import Metrics, MetricsKey
from run.train.spectrogram_model._worker import (
    _get_data_loaders,
    _HandleBatchArgs,
    _log_vocab,
    _run_inference,
    _run_step,
)
from tests.run._utils import make_spec_worker_state, mock_distributed_data_parallel
from tests.run.train._utils import setup_experiment


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    yield
    cf.purge()


def test_integration():
    train_dataset, dev_dataset, comet, device = setup_experiment()
    config_spec_model_training_from_datasets(train_dataset, dev_dataset, True)
    state = make_spec_worker_state(comet, device)

    assert state.model.module == state.model  # Ensure the mock worked
    model = typing.cast(SpectrogramModel, state.model)

    batch_size = 2
    train_loader, dev_loader = _get_data_loaders(
        state=state,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        train_batch_size=batch_size,
        dev_batch_size=batch_size,
        train_steps_per_epoch=1,
        dev_steps_per_epoch=1,
        train_get_weight=lambda _, f: f,
        dev_get_weight=lambda *_: 1.0,
        num_workers=0,
        prefetch_factor=2,
    )

    # Test `exclude_from_decay` excluded the right parameters
    no_decay, decay, groups = state._make_optimizer_groups(state.model, exclude_from_decay)
    assert len(no_decay) == len(groups[0]["params"])
    assert groups[0]["weight_decay"] == 0.0
    assert len(decay) == len(groups[1]["params"])
    assert "encoder.norm_embed.weight" in no_decay
    assert "encoder.blocks.0.norm.weight" in no_decay
    assert "decoder.linear_stop_token.1.bias" in no_decay
    assert "decoder.linear_out.0.weight" in no_decay

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, comet, state.model):
        timer = Timer()
        metrics = Metrics(comet)
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        args = (state, train_loader, Context.TRAIN, DatasetType.TRAIN, metrics, timer, batch, True)
        cf.partial(_run_step)(_HandleBatchArgs(*args))
        assert state.step.item() == 1

        is_not_diff = lambda b, v: len(set(b) - set(v.keys())) == 0
        characters = [c for s in batch.processed.tokens for c in s]
        assert is_not_diff(characters, model.token_embed.vocab)
        assert is_not_diff((s.speaker.label for s in batch.spans), model.speaker_embed.vocab)
        assert is_not_diff((s.session for s in batch.spans), model.session_embed.vocab)

        # fmt: off
        keys = [
            metrics.ALIGNMENT_STD_SUM, metrics.ALIGNMENT_NORM_SUM,
            metrics.NUM_REACHED_MAX, metrics.RMS_SUM_PRED, metrics.RMS_SUM
        ]
        # fmt: on
        for key in keys:
            assert len(metrics.data[MetricsKey(key)]) == 1
            assert len(metrics.data[MetricsKey(key, batch.spans[0].speaker)]) == 1
        assert all(metrics.data[MetricsKey(metrics.NUM_CORRECT_STOP_TOKEN)]) == 1

        max_frames = [[batch.spectrogram.lengths.max().item()]]
        num_frames = [[batch.spectrogram.lengths.sum().item()]]
        num_tokens = [[sum(len(t) for t in batch.processed.tokens)]]
        num_seconds = [[sum(s.audio_length for s in batch.spans)]]
        bucket = len(batch.spans[0].script) // metrics.TEXT_LENGTH_BUCKET_SIZE
        values = {
            MetricsKey(metrics.NUM_FRAMES_MAX): max_frames,
            MetricsKey(metrics.NUM_FRAMES_PRED): num_frames,
            MetricsKey(metrics.NUM_FRAMES_PRED, JUDY_BIEBER): num_frames,
            MetricsKey(metrics.NUM_FRAMES): num_frames,
            MetricsKey(metrics.NUM_FRAMES, JUDY_BIEBER): num_frames,
            MetricsKey(metrics.NUM_SECONDS): num_seconds,
            MetricsKey(metrics.NUM_SECONDS, JUDY_BIEBER): num_seconds,
            MetricsKey(metrics.NUM_SPANS_PER_TEXT_LENGTH, None, bucket): [[batch_size]],
            MetricsKey(metrics.NUM_SPANS): [[len(batch)]],
            MetricsKey(metrics.NUM_SPANS, JUDY_BIEBER): [[len(batch)]],
            MetricsKey(metrics.NUM_TOKENS): num_tokens,
            MetricsKey(metrics.NUM_TOKENS, JUDY_BIEBER): num_tokens,
        }
        for key, value in values.items():
            assert metrics.data[key] == value, str(key)

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)
        _log_vocab(state, DatasetType.TRAIN)

    # Test `_run_inference` with `Metrics` and `_State`, along with `_visualize_select_cases`
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
