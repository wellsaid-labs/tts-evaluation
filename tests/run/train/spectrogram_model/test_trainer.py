from unittest import mock

from torchnlp.encoders.text import BatchedSequences

import numpy as np
import pytest
import torch

from src.audio import amplitude_to_db
from src.audio import full_scale_sine_wave
from src.audio import full_scale_square_wave
from src.audio import power_to_db
from src.bin.train.spectrogram_model.data_loader import SpectrogramModelTrainingRow
from src.bin.train.spectrogram_model.trainer import Trainer
from src.environment import TEMP_PATH
from src.utils import Checkpoint
from tests._utils import get_tts_mocks
from tests._utils import MockCometML


@mock.patch('src.bin.train.spectrogram_model.trainer.atexit.register')
def get_trainer(register_mock, load_data=True):
    """
    Args:
        load_data (bool, optional): If `False` do not load any data for faster instantiation.
    """
    register_mock.return_value = None

    mocks = get_tts_mocks(add_spectrogram=True)
    trainer = Trainer(
        comet_ml=MockCometML(disabled=True),
        device=mocks['device'],
        train_dataset=mocks['train_dataset'] if load_data else [],
        dev_dataset=mocks['dev_dataset'] if load_data else [],
        checkpoints_directory=TEMP_PATH,
        train_batch_size=1,
        dev_batch_size=1)

    return trainer


@mock.patch('src.bin.train.spectrogram_model.trainer.atexit.register')
def test_checkpoint(register_mock):
    """ Ensure checkpoint can be saved and loaded from. """
    register_mock.return_value = None

    trainer = get_trainer(load_data=False)

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        comet_ml=MockCometML(disabled=True),
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory=TEMP_PATH,
        train_dataset=trainer.train_dataset,
        dev_dataset=trainer.dev_dataset)


def test_visualize_inferred():
    """ Smoke test to ensure that `visualize_inferred` runs without failure. """
    trainer = get_trainer()
    trainer.visualize_inferred()


def test__update_loudness_metrics():
    """ Test if loudness metrics are averaged correctly for one iteration. """
    trainer = get_trainer(load_data=False)

    frame_length = 1024
    frame_hop = frame_length // 4
    window = torch.ones(frame_length)

    def get_db_spectrogram(signal):
        spectrogram = torch.stft(
            signal.view(1, -1),  # Add batch dimension
            n_fft=frame_length,
            hop_length=frame_hop,
            win_length=frame_length,
            window=window,
            center=False)
        spectrogram = torch.norm(spectrogram, dim=-1)
        return amplitude_to_db(spectrogram).permute(2, 0, 1)

    square_wave_spectrogram = get_db_spectrogram(torch.tensor(full_scale_square_wave()))
    sine_wave_spectrogram = get_db_spectrogram(torch.tensor(full_scale_sine_wave()))

    trainer._update_loudness_metrics(square_wave_spectrogram, sine_wave_spectrogram, window=window)

    target_loudness = trainer.loudness_metrics['average_target_loudness'].last_update()
    assert power_to_db(torch.tensor(target_loudness)) == 0.0

    predicted_loudness = trainer.loudness_metrics['average_predicted_loudness'].last_update()
    # TODO: This should be equal to `-3.0103001594543457`, fix this discrepancy.
    assert power_to_db(torch.tensor(predicted_loudness)).item() == pytest.approx(-3.0102469)


def test__update_loudness_metrics__masking():
    """ Test if loudness metrics are averaged correctly over multiple iterations. """
    trainer = get_trainer(load_data=False)

    frame_length = 1024
    window = torch.ones(frame_length)
    frame_hop = frame_length // 4

    def get_db_spectrogram(signal):
        spectrogram = torch.stft(
            signal.view(1, -1),  # Add batch dimension
            n_fft=frame_length,
            hop_length=frame_hop,
            win_length=frame_length,
            window=window,
            center=False)
        spectrogram = torch.norm(spectrogram, dim=-1)
        return amplitude_to_db(spectrogram).permute(2, 0, 1)

    num_frames = 4
    target = np.concatenate([full_scale_square_wave(frame_length)] * num_frames * 2)
    predicted = np.concatenate([full_scale_sine_wave(frame_length)] * num_frames +
                               [full_scale_square_wave(frame_length)] * num_frames)

    target_spectrogram = get_db_spectrogram(torch.tensor(target))
    predicted_spectrogram = get_db_spectrogram(torch.tensor(predicted))
    # NOTE: There are some frames in the middle that combine both the square and sine wave, we
    # avoid those.
    spectrogram_mask = torch.cat(
        [torch.ones(num_frames, 1),
         torch.zeros(target_spectrogram.shape[0] - num_frames, 1)])

    trainer._update_loudness_metrics(
        target_spectrogram,
        predicted_spectrogram,
        spectrogram_mask,
        spectrogram_mask,
        window=window)

    target_loudness = trainer.loudness_metrics['average_target_loudness'].last_update()
    assert power_to_db(torch.tensor(target_loudness)) == pytest.approx(0.0, abs=1e-3)

    predicted_loudness = trainer.loudness_metrics['average_predicted_loudness'].last_update()
    assert power_to_db(torch.tensor(predicted_loudness)).item() == pytest.approx(-3.01, abs=1e-3)


def test__update_loudness_metrics__multiple_updates():
    """ Test if loudness metrics are averaged correctly over multiple iterations. """
    trainer = get_trainer(load_data=False)

    frame_length = 1024
    window = torch.ones(frame_length)

    def get_db_spectrogram(signal):
        spectrogram = torch.stft(
            signal.view(1, -1),  # Add batch dimension
            n_fft=frame_length,
            hop_length=frame_length // 4,
            win_length=frame_length,
            window=window,
            center=False)
        spectrogram = torch.norm(spectrogram, dim=-1)
        return amplitude_to_db(spectrogram).permute(2, 0, 1)

    target = np.concatenate([full_scale_square_wave()] * 2)
    predicted = np.concatenate([full_scale_square_wave(), full_scale_sine_wave()])

    # Run as a single spectrogram
    target_spectrogram = get_db_spectrogram(torch.tensor(target))
    predicted_spectrogram = get_db_spectrogram(torch.tensor(predicted))
    trainer._update_loudness_metrics(target_spectrogram, predicted_spectrogram, window=window)
    average_target_loudness = trainer.loudness_metrics['average_target_loudness'].sync().reset()
    average_predicted_loudness = trainer.loudness_metrics['average_predicted_loudness'].sync(
    ).reset()

    # Run as at least 4 spectrograms
    split_size = 4
    assert target_spectrogram.shape[0] > 4 * split_size
    target_splits = target_spectrogram.split(split_size)
    predicted_splits = predicted_spectrogram.split(split_size)
    for target_split, predicted_split in zip(target_splits, predicted_splits):
        trainer._update_loudness_metrics(target_split, predicted_split, window=window)

    assert trainer.loudness_metrics['average_target_loudness'].sync().reset() == pytest.approx(
        average_target_loudness)
    assert trainer.loudness_metrics['average_predicted_loudness'].sync().reset() == pytest.approx(
        average_predicted_loudness)


def test__do_loss_and_maybe_backwards():
    """ Test that the correct loss values are computed and back propagated. """
    trainer = get_trainer(load_data=False)

    batch = SpectrogramModelTrainingRow(
        text=None,
        speaker=None,
        spectrogram=BatchedSequences(torch.FloatTensor([[[1, 1]], [[1, 1]], [[3, 3]]]), [[3]]),
        stop_token=BatchedSequences(torch.FloatTensor([[0], [1], [1]]), [[3]]),
        spectrogram_mask=BatchedSequences(torch.FloatTensor([[1], [1], [0]]), [[3]]),
        spectrogram_expanded_mask=BatchedSequences(
            torch.FloatTensor([[[1, 1]], [[1, 1]], [[0, 0]]]), [[3]]))
    predicted_spectrogram = torch.FloatTensor([[[1, 1]], [[1, 1]], [[1, 1]]])
    predicted_stop_tokens = torch.FloatTensor([[0], [0.5], [0.5]])
    predicted_alignments = torch.zeros(3, 1, 5)
    predicted_alignments[:, 0, 0].fill_(1.0)

    predictions = (predicted_spectrogram, predicted_stop_tokens, predicted_alignments)
    (spectrogram_loss, stop_token_loss, num_spectrogram_values,
     num_frames) = trainer._do_loss_and_maybe_backwards(batch, predictions, False)

    assert spectrogram_loss.item() == pytest.approx(0.0)
    assert stop_token_loss.item() == pytest.approx(1.1672241687774658)
    assert num_spectrogram_values == 4
    assert num_frames == 2


def test_run_epoch():
    """ Smoke test to ensure that `run_epoch` runs without failure. """
    trainer = get_trainer()
    trainer.run_epoch(train=False, trial_run=False)
    trainer.run_epoch(train=False, trial_run=True)
    trainer.run_epoch(train=False, infer=True, trial_run=False)
    trainer.run_epoch(train=True, trial_run=False)
