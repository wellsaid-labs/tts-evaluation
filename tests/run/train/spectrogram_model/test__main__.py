from unittest import mock

from hparams import add_config

from run._config import Cadence, DatasetType
from run.data._loader import JUDY_BIEBER
from run.train._utils import Context, Timer, set_context
from run.train.spectrogram_model.__main__ import _make_configuration
from run.train.spectrogram_model._metrics import Metrics
from run.train.spectrogram_model._worker import (
    _get_data_loaders,
    _HandleBatchArgs,
    _run_inference,
    _run_step,
    _State,
)
from tests._utils import mock_distributed_data_parallel
from tests.run.train._utils import setup_experiment


def test_integration():
    train_dataset, dev_dataset, comet, device = setup_experiment()

    add_config(_make_configuration(train_dataset, dev_dataset, True))

    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        state = _State.from_dataset(train_dataset, dev_dataset, comet, device)
    assert state.model.module == state.model  # Enusre the mock worked
    # fmt: off
    assert sorted(state.input_encoder.grapheme_encoder.vocab) == sorted([
        '<pad>', '<unk>', '</s>', '<s>', '<copy>', 'a', 't', ' ', 'w', 'l', 's', ',', 'i', 'n', 'f',
        'o', 'r', 'd', 'h', 'e', 'b', 'y', '.', 'm', 'u', 'k', 'g'
    ])
    # fmt: on
    speakers = state.input_encoder.speaker_encoder.vocab
    assert speakers == list(train_dataset.keys())
    assert state.model.vocab_size == state.input_encoder.phoneme_encoder.vocab_size
    assert state.model.num_speakers == state.input_encoder.speaker_encoder.vocab_size
    assert state.model.num_sessions == state.input_encoder.session_encoder.vocab_size

    batch_size = 1
    train_loader, dev_loader = _get_data_loaders(
        state, train_dataset, dev_dataset, batch_size, batch_size, 1, 1, False, True, 0, 2
    )

    # Test `_run_step` with `Metrics` and `_State`
    with set_context(Context.TRAIN, comet, state.model):
        timer = Timer()
        metrics = Metrics(comet, speakers)
        batch = next(iter(train_loader))
        assert state.step.item() == 0

        args = (state, train_loader, Context.TRAIN, DatasetType.TRAIN, metrics, timer, batch, True)
        _run_step(_HandleBatchArgs(*args))
        assert state.step.item() == 1

        # fmt: off
        keys = [
            metrics.ALIGNMENT_NUM_SKIPS, metrics.ALIGNMENT_STD_SUM, metrics.ALIGNMENT_NORM_SUM,
            metrics.NUM_REACHED_MAX, metrics.RMS_SUM_PREDICTED, metrics.RMS_SUM
        ]
        # fmt: on
        for key in keys:
            assert len(metrics.data[key]) == 1
            assert len(metrics.data[f"{key}/{batch.spans[0].speaker.label}"]) == 1
        assert all(metrics.data[metrics.NUM_CORRECT_STOP_TOKEN]) == 1

        num_frames = [(batch.spectrogram.lengths[0].item(),)]
        num_tokens = [(batch.encoded_phonemes.lengths[0].item(),)]
        num_seconds = [(batch.spans[0].audio_length,)]
        bucket = len(batch.spans[0].script) // metrics.TEXT_LENGTH_BUCKET_SIZE
        values = {
            metrics.NUM_FRAMES_MAX: num_frames,
            metrics.NUM_FRAMES_PREDICTED: num_frames,
            f"{metrics.NUM_FRAMES_PREDICTED}/{JUDY_BIEBER.label}": num_frames,
            metrics.NUM_FRAMES: num_frames,
            f"{metrics.NUM_FRAMES}/{JUDY_BIEBER.label}": num_frames,
            metrics.NUM_SECONDS: num_seconds,
            f"{metrics.NUM_SECONDS}/{JUDY_BIEBER.label}": num_seconds,
            f"{metrics.NUM_SPANS_PER_TEXT_LENGTH}/{bucket}": [(batch_size,)],
            metrics.NUM_SPANS: [(len(batch),)],
            f"{metrics.NUM_SPANS}/{JUDY_BIEBER.label}": [(len(batch),)],
            metrics.NUM_TOKENS: num_tokens,
            f"{metrics.NUM_TOKENS}/{JUDY_BIEBER.label}": num_tokens,
        }
        for key, value in values.items():
            assert metrics.data[key] == value

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test `_run_inference` with `Metrics` and `_State`
    with set_context(Context.EVALUATE_INFERENCE, comet, state.model):
        timer = Timer()
        metrics = Metrics(comet, speakers)
        batch = next(iter(train_loader))
        args = (state, dev_loader, Context.EVALUATE, DatasetType.DEV, metrics, timer, batch, True)
        _run_inference(_HandleBatchArgs(*args))
        assert state.step.item() == 1
        total = metrics.data[metrics.NUM_REACHED_MAX][0][0] + metrics.data[metrics.NUM_SPANS][0][0]
        assert total == 1

        metrics.log(lambda l: l[-1:], type_=DatasetType.TRAIN, cadence=Cadence.STEP)
        metrics.log(is_verbose=True, type_=DatasetType.TRAIN, cadence=Cadence.MULTI_STEP)

    # Test loading and saving a checkpoint
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        loaded = state.from_checkpoint(state.to_checkpoint(), comet, device)
    assert state.step == loaded.step
