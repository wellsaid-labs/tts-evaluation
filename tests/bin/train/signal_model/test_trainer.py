from unittest import mock

import pytest
import torch

from src.audio import full_scale_sine_wave
from src.audio import full_scale_square_wave
from src.audio import read_audio
from src.bin.train.signal_model.data_loader import SignalModelTrainingRow
from src.bin.train.signal_model.trainer import SpectrogramLoss
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
    read_audio_mock.side_effect = lambda *args, **kwargs: read_audio(*args, **kwargs)[:4096]

    return Trainer(
        comet_ml=MockCometML(),
        device=mocks['device'],
        checkpoints_directory=TEMP_PATH,
        train_dataset=mocks['train_dataset'] if load_data else [],
        dev_dataset=mocks['dev_dataset'] if load_data else [],
        spectrogram_model_checkpoint_path=mocks['spectrogram_model_checkpoint'].path,
        train_batch_size=1,
        dev_batch_size=1)


def test_spectrogram_loss__smoke_test():
    spectrogram_loss = SpectrogramLoss()
    predicted = torch.randn(1, 24000)
    target = torch.randn(1, 24000)
    loss, other_loss, accuracy = spectrogram_loss(
        predicted, target, comet_ml=MockCometML(disabled=True))
    assert loss.shape == tuple([])
    assert loss.dtype == torch.float32
    assert other_loss.shape == tuple([])
    assert other_loss.dtype == torch.float32
    assert accuracy.shape == tuple([])
    assert accuracy.dtype == torch.float32


def test_spectrogram_loss__get_name():
    fft_length = 2048
    frame_hop = 512
    spectrogram_loss = SpectrogramLoss(fft_length=fft_length, frame_hop=frame_hop)

    assert spectrogram_loss.get_name(
        is_mel_scale=True, is_decibels=True,
        is_magnitude=True) == 'mel(db(abs(spectrogram(fft_length=%d,frame_hop=%d))))' % (fft_length,
                                                                                         frame_hop)

    assert spectrogram_loss.get_name(
        'target', is_mel_scale=True, is_decibels=True,
        is_magnitude=True) == 'mel(db(abs(spectrogram(target,fft_length=%d,frame_hop=%d))))' % (
            fft_length, frame_hop)

    assert spectrogram_loss.get_name(
        'target', is_mel_scale=False, is_decibels=True,
        is_magnitude=True) == 'db(abs(spectrogram(target,fft_length=%d,frame_hop=%d)))' % (
            fft_length, frame_hop)

    assert spectrogram_loss.get_name(
        'target', is_mel_scale=True, is_decibels=False,
        is_magnitude=True) == 'mel(abs(spectrogram(target,fft_length=%d,frame_hop=%d)))' % (
            fft_length, frame_hop)

    assert spectrogram_loss.get_name(
        'target', is_mel_scale=True, is_decibels=True,
        is_magnitude=False) == 'mel(db(spectrogram(target,fft_length=%d,frame_hop=%d)))' % (
            fft_length, frame_hop)


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
        target_signal=torch.zeros(2, 4096),
        signal_mask=torch.ones(2, 4096))
    predicted_signal = torch.zeros(2, 4096)

    (db_mel_spectrogram_magnitude_loss, discriminator_loss,
     num_predictions) = trainer._do_loss_and_maybe_backwards(batch, predicted_signal, False, True)
    assert db_mel_spectrogram_magnitude_loss.item() == pytest.approx(0.0)
    assert isinstance(discriminator_loss.item(), float)
    assert num_predictions == 8192


@pytest.mark.skip(reason="TODO")
def test__do_loss_and_maybe_backwards__masking():
    """ Test that the correct loss values are computed and back propagated.

    TODO: Finish writting a padding invariant implementation of our loss function and model.
    """
    trainer = get_trainer(load_data=False)

    target_signal = torch.tensor(full_scale_square_wave(4000))
    predicted_signal = torch.tensor(full_scale_sine_wave(4000))
    mask = torch.ones(4000)

    # TODO: Look into why an offset that's not a multiple of the frame size causes a big problem?
    padded_target_signal = torch.nn.functional.pad(target_signal, (256, 256))
    padded_predicted_signal = torch.nn.functional.pad(predicted_signal, (256, 256))
    padded_mask = torch.nn.functional.pad(mask, (256, 256))

    batch = SignalModelTrainingRow(None, None, target_signal, mask)
    padded_batch = SignalModelTrainingRow(None, None, padded_target_signal, padded_mask)

    (db_mel_spectrogram_magnitude_loss, discriminator_loss,
     num_predictions) = trainer._do_loss_and_maybe_backwards(batch, predicted_signal, False, True)

    (padded_db_mel_spectrogram_magnitude_loss, padded_discriminator_loss,
     padded_num_predictions) = trainer._do_loss_and_maybe_backwards(padded_batch,
                                                                    padded_predicted_signal, False,
                                                                    True)

    assert pytest.approx(db_mel_spectrogram_magnitude_loss.item()) == pytest.approx(
        padded_db_mel_spectrogram_magnitude_loss.item())
    assert discriminator_loss == padded_discriminator_loss
    assert num_predictions == padded_num_predictions


def test_visualize_inferred():
    """ Smoke test to ensure that `visualize_inferred` runs without failure. """
    trainer = get_trainer()
    trainer.visualize_inferred()


def test_run_epoch():
    """ Smoke test to ensure that `run_epoch` runs without failure. """
    trainer = get_trainer()
    trainer.run_epoch(train=False, trial_run=False)
    trainer.run_epoch(train=False, trial_run=True)
