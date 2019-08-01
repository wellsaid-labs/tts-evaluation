from unittest import mock

import pytest
import torch

from src.bin.train.spectrogram_model.data_loader import SpectrogramModelTrainingRow
from src.bin.train.spectrogram_model.trainer import Trainer
from src.environment import TEMP_PATH
from src.utils import Checkpoint
from tests._utils import get_tts_mocks
from tests._utils import MockCometML


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def get_trainer(register_mock, comet_ml_mock, load_data=True):
    """
    Args:
        load_data (bool, optional): If `False` do not load any data for faster instantiation.
    """
    comet_ml_mock.return_value = MockCometML()
    register_mock.return_value = None

    mocks = get_tts_mocks(add_spectrogram=True)
    trainer = Trainer(
        device=mocks['device'],
        train_dataset=mocks['train_dataset'] if load_data else [],
        dev_dataset=mocks['dev_dataset'] if load_data else [],
        checkpoints_directory=TEMP_PATH,
        train_batch_size=1,
        dev_batch_size=1)

    return trainer


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def test_checkpoint(register_mock, comet_ml_mock):
    """ Ensure checkpoint can be saved and loaded from. """
    comet_ml_mock.return_value = MockCometML()
    register_mock.return_value = None

    trainer = get_trainer(load_data=False)

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory=TEMP_PATH,
        train_dataset=trainer.train_dataset,
        dev_dataset=trainer.dev_dataset)


def test_visualize_inferred():
    """ Smoke test to ensure that `visualize_inferred` runs without failure. """
    trainer = get_trainer()
    trainer.visualize_inferred()


def test__do_loss_and_maybe_backwards():
    """ Test that the correct loss values are computed and back propagated. """
    trainer = get_trainer(load_data=False)

    batch = SpectrogramModelTrainingRow(
        text=None,
        speaker=None,
        spectrogram=(torch.FloatTensor([[1, 1], [1, 1], [3, 3]]), [3]),
        stop_token=(torch.FloatTensor([0, 1, 1]), [3]),
        spectrogram_mask=(torch.ByteTensor([1, 1, 0]), [3]),
        spectrogram_expanded_mask=(torch.ByteTensor([[1, 1], [1, 1], [0, 0]]), [3]))
    predicted_pre_spectrogram = torch.FloatTensor([[1, 1], [1, 1], [1, 1]])
    predicted_post_spectrogram = torch.FloatTensor([[0.5, 0.5], [0.5, 0.5], [1, 1]])
    predicted_stop_tokens = torch.FloatTensor([0, 0.5, 0.5])
    predicted_alignments = torch.FloatTensor(3, 1, 5).fill_(0.0)
    predicted_alignments[:, 0, 0].fill_(1.0)

    predictions = (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
                   predicted_alignments)
    (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, num_spectrogram_values,
     num_frames) = trainer._do_loss_and_maybe_backwards(batch, predictions, False)

    assert pre_spectrogram_loss.item() == pytest.approx(0.0)
    assert post_spectrogram_loss.item() == pytest.approx(1.0 / 4)
    assert stop_token_loss.item() == pytest.approx(0.5836120843887329)
    assert num_spectrogram_values == 4
    assert num_frames == 2


def test_run_epoch():
    """ Smoke test to ensure that `run_epoch` runs without failure. """
    trainer = get_trainer()
    trainer.run_epoch(train=False, trial_run=False)
    trainer.run_epoch(train=False, trial_run=True)
    trainer.run_epoch(train=False, infer=True, trial_run=False)
