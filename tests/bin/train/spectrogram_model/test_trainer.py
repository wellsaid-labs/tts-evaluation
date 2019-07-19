import math

from contextlib import ExitStack
from unittest import mock

import pytest
import torch

from src.bin.train.spectrogram_model.data_loader import SpectrogramModelTrainingRow
from src.bin.train.spectrogram_model.trainer import Trainer
from src.utils import Checkpoint

from tests._utils import get_example_spectrogram_text_speech_rows
from tests._utils import MockCometML


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def get_trainer(register_mock, comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    register_mock.return_value = None
    examples = get_example_spectrogram_text_speech_rows()
    trainer = Trainer(
        device=torch.device('cpu'),
        train_dataset=examples,
        dev_dataset=examples,
        checkpoints_directory='tests/_test_data/',
        train_batch_size=1,
        dev_batch_size=1)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token.bias, -math.inf)
    return trainer


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
@mock.patch('src.bin.train.signal_model.trainer.atexit.register')
def test_checkpoint(register_mock, comet_ml_mock):
    trainer = get_trainer()
    comet_ml_mock.return_value = MockCometML()
    register_mock.return_value = None

    checkpoint_path = trainer.save_checkpoint()
    Trainer.from_checkpoint(
        checkpoint=Checkpoint.from_path(checkpoint_path),
        device=torch.device('cpu'),
        checkpoints_directory='tests/_test_data/',
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows())


def test_visualize_inferred():
    num_frames = 8
    num_tokens = 8
    frame_channels = 16
    frame_lengths = torch.full((1,), num_frames)
    trainer = get_trainer()
    infer_pass_return = (torch.FloatTensor(num_frames, frame_channels).fill_(1),
                         torch.FloatTensor(num_frames, frame_channels).fill_(1),
                         torch.FloatTensor(num_frames).fill_(1),
                         torch.FloatTensor(num_frames, num_tokens).fill_(1.0 / num_tokens),
                         frame_lengths)

    with mock.patch('src.spectrogram_model.model.SpectrogramModel._infer') as mock_forward:
        mock_forward.return_value = infer_pass_return
        trainer.visualize_inferred()
        mock_forward.assert_called()


def test__do_loss_and_maybe_backwards():
    trainer = get_trainer()

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
    trainer = get_trainer()
    batch_size = 2
    num_frames = 8
    num_tokens = 8
    frame_channels = 16
    text_vocab_size = trainer.input_encoder.text_encoder.vocab_size
    speaker_vocab_size = trainer.input_encoder.speaker_encoder.vocab_size
    frame_lengths = torch.full((
        1,
        batch_size,
    ), num_frames)
    loaded_data = [
        SpectrogramModelTrainingRow(
            text=(torch.randint(text_vocab_size, (num_tokens, batch_size), dtype=torch.long),
                  torch.full((
                      1,
                      batch_size,
                  ), num_tokens)),
            speaker=(torch.randint(speaker_vocab_size, (1, batch_size), dtype=torch.long),
                     torch.full((
                         1,
                         batch_size,
                     ), 1)),
            spectrogram=(torch.rand(num_frames, batch_size, frame_channels), frame_lengths),
            stop_token=(torch.rand(num_frames, batch_size), frame_lengths),
            spectrogram_mask=(torch.ones(num_frames, batch_size, dtype=torch.float).byte(),
                              frame_lengths),
            spectrogram_expanded_mask=(torch.ones(
                num_frames, batch_size, frame_channels, dtype=torch.float).byte(), frame_lengths))
    ]
    forward_pass_return = (torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                           torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                           torch.FloatTensor(num_frames, batch_size).fill_(1),
                           torch.FloatTensor(num_frames, batch_size,
                                             num_tokens).fill_(1.0 / num_tokens))
    infer_pass_return = (torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                         torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                         torch.FloatTensor(num_frames, batch_size).fill_(1),
                         torch.FloatTensor(num_frames, batch_size,
                                           num_tokens).fill_(1.0 / num_tokens), frame_lengths)
    with ExitStack() as stack:
        (MockDataLoader, mock_aligned, mock_infer, mock_backward, mock_optimizer_step,
         mock_auto_optimizer_step) = tuple([
             stack.enter_context(mock.patch(arg)) for arg in [
                 'src.bin.train.spectrogram_model.trainer.DataLoader',
                 'src.spectrogram_model.model.SpectrogramModel._aligned',
                 'src.spectrogram_model.model.SpectrogramModel._infer', 'torch.Tensor.backward',
                 'src.optimizers.Optimizer.step', 'src.optimizers.AutoOptimizer.step'
             ]
         ])
        MockDataLoader.return_value = loaded_data
        mock_aligned.return_value = forward_pass_return
        mock_infer.return_value = infer_pass_return
        mock_backward.return_value = None
        mock_optimizer_step.return_value = None
        mock_auto_optimizer_step.return_value = None

        trainer.run_epoch(train=False)
        trainer.run_epoch(train=False, trial_run=True)
        trainer.run_epoch(train=True)
        trainer.run_epoch(train=False, infer=True)

        mock_aligned.assert_called()
        mock_infer.assert_called()
