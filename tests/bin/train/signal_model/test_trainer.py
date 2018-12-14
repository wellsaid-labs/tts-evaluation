from contextlib import ExitStack
from unittest import mock

import torch

from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import Trainer
from src.utils import AnomalyDetector

from tests.utils import MockCometML
from tests.utils import get_example_spectrogram_text_speech_rows


@mock.patch('src.bin.train.signal_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.compute_spectrograms')
def get_trainer(compute_spectrograms_mock, comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    compute_spectrograms_mock.return_value = get_example_spectrogram_text_speech_rows()
    return Trainer(
        device=torch.device('cpu'),
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows(),
        comet_ml_project_name='',
        train_batch_size=1,
        dev_batch_size=1)


def test__do_loss_and_maybe_backwards():
    trainer = get_trainer()
    batch = SignalModelTrainingRow(
        input_signal=None,
        input_spectrogram=None,
        target_signal_coarse=(torch.LongTensor([[0, 0, 1]]), [3]),
        target_signal_fine=(torch.LongTensor([[1, 1, 1]]), [3]),
        signal_mask=(torch.FloatTensor([[1, 1, 0]]), [3]))
    predicted_coarse = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])
    predicted_fine = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])

    (coarse_loss, fine_loss, num_predictions) = trainer._do_loss_and_maybe_backwards(
        batch, (predicted_coarse, predicted_fine, None), False)
    assert coarse_loss.item() == 0.31326165795326233
    assert fine_loss.item() == 1.31326162815094
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
    signal_lengths = [signal_length] * batch_size
    bins = int(2**(bits / 2))
    return SignalModelTrainingRow(
        input_signal=(torch.rand(batch_size, signal_length), signal_lengths),
        input_spectrogram=(torch.rand(batch_size, num_frames, frame_channels),
                           [num_frames] * batch_size),
        target_signal_coarse=(torch.randint(bins, (batch_size, signal_length), dtype=torch.long),
                              signal_lengths),
        target_signal_fine=(torch.randint(bins, (batch_size, signal_length), dtype=torch.long),
                            signal_lengths),
        signal_mask=(torch.FloatTensor(batch_size, signal_length).fill_(1), signal_lengths))


def test_visualize_infered():
    signal_length = 16
    trainer = get_trainer()
    infer_pass_return = (torch.LongTensor(signal_length).zero_(),
                         torch.LongTensor(signal_length).zero_(), None)

    with mock.patch('src.bin.train.signal_model.trainer.WaveRNN.infer') as mock_infer:
        mock_infer.return_value = infer_pass_return
        trainer.visualize_infered()

        mock_infer.assert_called()


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
                 'torch.Tensor.backward',
                 'src.optimizer.Optimizer.step', 'src.optimizer.AutoOptimizer.step'
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
