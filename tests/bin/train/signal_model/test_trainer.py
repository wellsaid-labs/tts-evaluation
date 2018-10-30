from contextlib import contextmanager
from contextlib import ExitStack

from unittest import mock

import pytest
import torch

from src.bin.train.signal_model.trainer import Trainer
from src.utils import AnomalyDetector


class MockTensorboard():

    def __init__(*args, **kwargs):
        pass

    @contextmanager
    def set_step(self, *args, **kwargs):
        yield self

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self


@pytest.fixture
def trainer_fixture():
    trainer = Trainer(
        device=torch.device('cpu'),
        train_dataset=[],
        dev_dataset=[],
        train_tensorboard=MockTensorboard(),
        dev_tensorboard=MockTensorboard(),
        train_batch_size=1,
        dev_batch_size=1)

    trainer.tensorboard = trainer.train_tensorboard
    return trainer


def test__compute_loss(trainer_fixture):
    batch = {
        'slice': {
            'signal_mask': torch.FloatTensor([[1, 1, 0]]),
            'target_signal_coarse': torch.LongTensor([[0, 0, 1]]),
            'target_signal_fine': torch.LongTensor([[1, 1, 1]])
        }
    }
    predicted_coarse = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])
    predicted_fine = torch.FloatTensor([[[1, 0], [1, 0], [1, 0]]])

    (coarse_loss, fine_loss, num_predictions) = trainer_fixture._compute_loss(
        batch, predicted_coarse, predicted_fine)
    assert coarse_loss.item() == 0.31326162815093994
    assert fine_loss.item() == 1.31326162815094
    assert num_predictions == 2


def test__sample_infered(trainer_fixture):
    batch_size = 2
    batch = {
        'signal': torch.LongTensor(batch_size, 16),
        'spectrogram': torch.FloatTensor(batch_size, 16, 8)
    }
    trainer_fixture.model = mock.Mock()
    trainer_fixture.model.infer.return_value = (torch.LongTensor(16).zero_(),
                                                torch.LongTensor(16).zero_(), None)
    trainer_fixture._sample_infered(batch)
    trainer_fixture.model.infer.assert_called()


def test__sample_predicted(trainer_fixture):
    batch_size = 2
    max_signal_length = 8
    bins = 4
    batch = {
        'slice': {
            'signal_lengths': [4, max_signal_length],
            'target_signal_coarse': torch.LongTensor(batch_size, max_signal_length).zero_(),
            'target_signal_fine': torch.LongTensor(batch_size, max_signal_length).zero_(),
        }
    }
    predicted_coarse = torch.FloatTensor(batch_size, max_signal_length, bins)
    predicted_fine = torch.FloatTensor(batch_size, max_signal_length, bins)
    trainer_fixture._sample_predicted(batch, predicted_coarse, predicted_fine)


def test__maybe_rollback(trainer_fixture):
    min_steps = 10
    anomaly_detector = AnomalyDetector(min_steps=min_steps, type_=AnomalyDetector.TYPE_BOTH)
    trainer_fixture.anomaly_detector = anomaly_detector
    for i in range(min_steps):
        trainer_fixture._maybe_rollback(1)
        assert len(trainer_fixture.rollback) == min(i + 2, trainer_fixture.rollback.maxlen)
    trainer_fixture._maybe_rollback(2)
    assert len(trainer_fixture.rollback) == 1
    trainer_fixture._maybe_rollback(0)
    assert len(trainer_fixture.rollback) == 1


def test_run_epoch(trainer_fixture):
    batch_size = 2
    signal_length = 16
    num_frames = 4
    frame_channels = 8
    bins = 2
    batch = {
        'slice': {
            'input_signal': torch.FloatTensor(batch_size, signal_length),
            'target_signal_coarse': torch.LongTensor(batch_size, signal_length).zero_(),
            'target_signal_fine': torch.LongTensor(batch_size, signal_length).zero_(),
            'spectrogram': torch.FloatTensor(batch_size, num_frames, frame_channels),
            'spectrogram_lengths': [4, frame_channels],
            'signal_mask': torch.FloatTensor(batch_size, signal_length),
            'signal_lengths': [8, signal_length],
        },
        'spectrogram': [
            torch.FloatTensor(batch_size, 4, frame_channels),
            torch.FloatTensor(batch_size, num_frames, frame_channels)
        ],
        'signal': [torch.FloatTensor(batch_size, 8),
                   torch.FloatTensor(batch_size, signal_length)]
    }
    with ExitStack() as stack:
        MockDataBatchIterator, mock_data_parallel, mock_infer = tuple([
            stack.enter_context(mock.patch(arg)) for arg in [
                'src.bin.train.signal_model.trainer.DataBatchIterator',
                'src.bin.train.signal_model.trainer.torch.nn.parallel.data_parallel',
                'src.bin.train.signal_model.trainer.WaveRNN.infer'
            ]
        ])
        MockDataBatchIterator.return_value = [batch, batch]
        mock_data_parallel.return_value = (torch.FloatTensor(batch_size, signal_length, bins),
                                           torch.FloatTensor(batch_size, signal_length, bins), None)
        mock_infer.return_value = (torch.LongTensor(signal_length).zero_(),
                                   torch.LongTensor(signal_length).zero_(), None)

        trainer_fixture.run_epoch(train=False)
        trainer_fixture.run_epoch(train=False, trial_run=True)
        mock_infer.assert_called()
        mock_data_parallel.assert_called()
        torch.set_grad_enabled(True)  # Reset
