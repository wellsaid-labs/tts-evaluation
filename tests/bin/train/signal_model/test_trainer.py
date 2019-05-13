from contextlib import ExitStack
from unittest import mock

import os

import pytest
import torch

from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import Trainer
from src.utils import AnomalyDetector
from src.utils import Checkpoint

from tests.utils import get_example_spectrogram_text_speech_rows
from tests.utils import MockCometML


@mock.patch('src.bin.train.signal_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.add_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.add_predicted_spectrogram_column')
def get_trainer(add_predicted_spectrogram_column_mock, add_spectrogram_column_mock, comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    add_predicted_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    add_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    return Trainer(
        device=torch.device('cpu'),
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows(),
        comet_ml_project_name='',
        train_batch_size=1,
        dev_batch_size=1)


@mock.patch('src.bin.train.signal_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.add_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.add_predicted_spectrogram_column')
def test_checkpoint(add_predicted_spectrogram_column_mock, add_spectrogram_column_mock,
                    comet_ml_mock):
    trainer = get_trainer()
    comet_ml_mock.return_value = MockCometML()
    add_predicted_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    add_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()

    checkpoint_path = trainer.save_checkpoint('tests/_test_data/')
    trainer.from_checkpoint(
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows())

    # Clean up
    os.unlink(checkpoint_path)


def test__do_loss_and_maybe_backwards():
    trainer = get_trainer()
    batch = SignalModelTrainingRow(
        input_signal=None,
        input_spectrogram=None,
        target_signal_coarse=torch.LongTensor([[0, 0, 1]]),
        target_signal_fine=torch.LongTensor([[1, 1, 1]]),
        signal_mask=torch.ByteTensor([[1, 1, 0]]))
    predicted_coarse = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])
    predicted_fine = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])

    (coarse_loss, fine_loss, num_predictions) = trainer._do_loss_and_maybe_backwards(
        batch, (predicted_coarse, predicted_fine, None), False)
    assert coarse_loss.item() == pytest.approx(0.3132616)
    assert fine_loss.item() == pytest.approx(1.3132616)
    assert num_predictions == 2


def test__maybe_rollback():
    trainer = get_trainer()
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_BOTH)
    trainer.anomaly_detector = anomaly_detector
    for i in range(min_steps):
        trainer._maybe_rollback(1)
        assert len(trainer.rollback) == min(i + 2, trainer.rollback.maxlen)
    trainer._maybe_rollback(2)
    assert len(trainer.rollback) == 1
    trainer._maybe_rollback(0)
    assert len(trainer.rollback) == 1


def _get_example_batched_training_row(batch_size=2,
                                      signal_length=16,
                                      num_frames=4,
                                      frame_channels=8,
                                      bits=16):
    """ Get an example training row. """
    bins = int(2**(bits / 2))
    return SignalModelTrainingRow(
        input_signal=torch.rand(batch_size, signal_length),
        input_spectrogram=torch.rand(batch_size, num_frames, frame_channels),
        target_signal_coarse=torch.randint(bins, (batch_size, signal_length), dtype=torch.long),
        target_signal_fine=torch.randint(bins, (batch_size, signal_length), dtype=torch.long),
        signal_mask=torch.ByteTensor(batch_size, signal_length).fill_(1))


def test_visualize_inferred():
    signal_length = 16
    trainer = get_trainer()
    infer_pass_return = (torch.LongTensor(signal_length).zero_(),
                         torch.LongTensor(signal_length).zero_(), None)

    with mock.patch('src.signal_model.wave_rnn._WaveRNNInferrer.forward') as mock_forward:
        mock_forward.return_value = infer_pass_return
        trainer.visualize_inferred()

        mock_forward.assert_called()


def test_run_epoch():
    batch_size = 2
    signal_length = 16
    bins = 2
    trainer = get_trainer()
    loaded_data = [
        _get_example_batched_training_row(batch_size=batch_size, signal_length=signal_length),
        _get_example_batched_training_row(batch_size=batch_size, signal_length=signal_length)
    ]
    forward_pass_return = (torch.FloatTensor(batch_size, signal_length, bins),
                           torch.FloatTensor(batch_size, signal_length, bins), None)
    with ExitStack() as stack:
        (MockDataLoader, mock_data_parallel, mock_backward, mock_optimizer_step,
         mock_auto_optimizer_step) = tuple([
             stack.enter_context(mock.patch(arg)) for arg in [
                 'src.bin.train.signal_model.trainer.DataLoader',
                 'src.bin.train.signal_model.trainer.torch.nn.parallel.data_parallel',
                 'torch.Tensor.backward', 'src.optimizers.Optimizer.step',
                 'src.optimizers.AutoOptimizer.step'
             ]
         ])
        MockDataLoader.return_value = loaded_data
        mock_data_parallel.return_value = forward_pass_return
        mock_backward.return_value = None
        mock_optimizer_step.return_value = None
        mock_auto_optimizer_step.return_value = None

        trainer.run_epoch(train=False)
        trainer.run_epoch(train=False, trial_run=True)
        trainer.run_epoch(train=True)

        mock_data_parallel.assert_called()
