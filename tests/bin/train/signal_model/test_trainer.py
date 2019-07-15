from contextlib import ExitStack
from unittest import mock

import pytest
import torch

from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import Trainer
from src.utils import Checkpoint

from tests._utils import get_example_spectrogram_text_speech_rows
from tests._utils import MockCometML


@mock.patch('src.bin.train.signal_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.add_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.add_predicted_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def get_trainer(register_mock, add_predicted_spectrogram_column_mock, add_spectrogram_column_mock,
                comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    add_predicted_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    add_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    register_mock.return_value = None
    return Trainer(
        device=torch.device('cpu'),
        checkpoints_directory='tests/_test_data/',
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows(),
        comet_ml_project_name='',
        train_batch_size=1,
        dev_batch_size=1)


@mock.patch('src.bin.train.signal_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.add_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.add_predicted_spectrogram_column')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def test_checkpoint(register_mock, add_predicted_spectrogram_column_mock,
                    add_spectrogram_column_mock, comet_ml_mock):
    trainer = get_trainer()
    comet_ml_mock.return_value = MockCometML()
    add_predicted_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    add_spectrogram_column_mock.return_value = get_example_spectrogram_text_speech_rows()
    register_mock.return_value = None

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory='tests/_test_data/',
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows())


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


def test__get_gru_orthogonal_loss():
    trainer = get_trainer()
    assert trainer._get_gru_orthogonal_loss().item() > 0


def test__partial_rollback():
    trainer = get_trainer()
    assert trainer.step == 0
    trainer.step += 1
    trainer._partial_rollback()
    assert trainer.step == 0
    assert trainer.num_rollbacks == 1


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
        mock_optimizer_step.return_value = 2.0
        mock_auto_optimizer_step.return_value = 2.0

        trainer.run_epoch(train=False)
        trainer.run_epoch(train=False, trial_run=True)
        assert trainer.epoch == 0
        trainer.run_epoch(train=True, num_epochs=2)
        assert trainer.epoch == 2

        mock_data_parallel.assert_called()
