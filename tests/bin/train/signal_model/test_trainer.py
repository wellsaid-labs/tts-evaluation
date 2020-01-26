from unittest import mock

import pytest
import torch

from src.audio import read_audio
from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import Trainer
from src.environment import TEMP_PATH
from src.utils import Checkpoint
from tests._utils import get_tts_mocks
from tests._utils import MockCometML


@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
@mock.patch('src.datasets.utils.read_audio')
def get_trainer(read_audio_mock, register_mock, load_data=True):
    """
    Args:
        load_data (bool, optional): If `False` do not load any data for faster instantiation.
    """
    mocks = get_tts_mocks(add_predicted_spectrogram=load_data, add_spectrogram=load_data)

    register_mock.return_value = None
    # NOTE: Decrease the audio size for test performance.
    read_audio_mock.side_effect = lambda *args, **kwargs: read_audio(*args, **kwargs)[:900]

    return Trainer(
        comet_ml=MockCometML(),
        device=mocks['device'],
        checkpoints_directory=TEMP_PATH,
        train_dataset=mocks['train_dataset'] if load_data else [],
        dev_dataset=mocks['dev_dataset'] if load_data else [],
        spectrogram_model_checkpoint_path=mocks['spectrogram_model_checkpoint'].path,
        train_batch_size=1,
        dev_batch_size=1)


def test__get_sample_density_gap():
    trainer = get_trainer(load_data=False)
    assert 0.0 == trainer._get_sample_density_gap(
        torch.tensor([10, 0, 10]), torch.tensor([10, 0, 10]), 0.0)
    assert 0.0 == trainer._get_sample_density_gap(
        torch.tensor([10, 0, 10]), torch.tensor([10, 0, 10]), 1.0)
    assert 0.25 == trainer._get_sample_density_gap(
        torch.tensor([-120, 0, 120, 0], dtype=torch.int8),
        torch.tensor([32760, 0, 0, 0], dtype=torch.int16), 0.9)


@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def test_checkpoint(register_mock):
    """ Ensure checkpoint can be saved and loaded from. """
    trainer = get_trainer(load_data=False)

    register_mock.return_value = None

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        comet_ml=MockCometML(),
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory=trainer.checkpoints_directory,
        train_dataset=trainer.train_dataset,
        dev_dataset=trainer.dev_dataset)


def test__do_loss_and_maybe_backwards():
    """ Test that the correct loss values are computed and back propagated. """
    trainer = get_trainer(load_data=False)
    batch = SignalModelTrainingRow(
        spectrogram_mask=None,
        spectrogram=None,
        target_signal=torch.FloatTensor([[0, 1, 0, -1, 0, 1, 0, -1, 0]]),
        source_signal=None,
        signal_mask=torch.BoolTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]]))
    predicted_signal = torch.FloatTensor([[0, 1, 0, -1, 0, 1, 0, -1, 0]])

    (spectral_convergence_loss, log_mel_spectrogram_magnitude_loss,
     num_predictions) = trainer._do_loss_and_maybe_backwards(batch, (predicted_signal, None), False)
    assert spectral_convergence_loss.item() == pytest.approx(0.3132616)
    assert log_mel_spectrogram_magnitude_loss.item() == pytest.approx(1.3132616)
    assert num_predictions == 2


def test__get_gru_orthogonal_loss():
    trainer = get_trainer(load_data=False)
    assert trainer._get_gru_orthogonal_loss().item() >= 0


def test__partial_rollback():
    trainer = get_trainer(load_data=False)
    assert trainer.step == 0
    assert trainer.num_rollbacks == 0
    trainer.step += 1
    trainer._partial_rollback()
    assert trainer.step == 0
    assert trainer.num_rollbacks == 1


def test_visualize_inferred():
    """ Smoke test to ensure that `visualize_inferred` runs without failure. """
    signal_length = 16
    trainer = get_trainer()
    infer_pass_return = (torch.LongTensor(signal_length).zero_(),
                         torch.LongTensor(signal_length).zero_(), None)

    with mock.patch('src.signal_model.wave_rnn._WaveRNNInferrer.forward') as mock_forward:
        mock_forward.return_value = infer_pass_return
        trainer.visualize_inferred()


def test_run_epoch():
    """ Smoke test to ensure that `run_epoch` runs without failure. """
    trainer = get_trainer()
    trainer.run_epoch(train=False, trial_run=False)
    trainer.run_epoch(train=False, trial_run=True)
