import math

from contextlib import ExitStack
from unittest import mock

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import pytest
import torch

from src.bin.train.spectrogram_model.trainer import Trainer
from src.bin.train.spectrogram_model.data_loader import SpectrogramModelTrainingRow
from src.datasets import Speaker

from tests.utils import get_example_spectrogram_text_speech_rows
from tests.utils import MockCometML


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
@mock.patch('src.bin.train.spectrogram_model.trainer.compute_spectrograms')
def get_trainer(compute_spectrograms_mock, comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    compute_spectrograms_mock.return_value = get_example_spectrogram_text_speech_rows()
    trainer = Trainer(
        device=torch.device('cpu'),
        train_dataset=get_example_spectrogram_text_speech_rows(),
        dev_dataset=get_example_spectrogram_text_speech_rows(),
        comet_ml_project_name='',
        text_encoder=CharacterEncoder(['text encoder']),
        speaker_encoder=IdentityEncoder([Speaker.LINDA_JOHNSON]),
        train_batch_size=1,
        dev_batch_size=1)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token.bias, -math.inf)
    return trainer


def test__do_loss_and_maybe_backwards():
    trainer = get_trainer()

    batch = SpectrogramModelTrainingRow(
        text=None,
        speaker=None,
        spectrogram=(torch.FloatTensor([[1, 1], [1, 1], [3, 3]]), [3]),
        stop_token=(torch.FloatTensor([0, 1, 1]), [3]),
        spectrogram_mask=(torch.FloatTensor([1, 1, 0]), [3]),
        spectrogram_expanded_mask=(torch.FloatTensor([[1, 1], [1, 1], [0, 0]]), [3]))
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
    text_vocab_size = 10
    speaker_vocab_size = 4
    frame_lengths = [num_frames] * batch_size
    loaded_data = [
        SpectrogramModelTrainingRow(
            text=(torch.randint(text_vocab_size, (num_tokens, batch_size), dtype=torch.long),
                  [num_tokens] * batch_size),
            speaker=(torch.randint(speaker_vocab_size, (1, batch_size), dtype=torch.long),
                     [1] * batch_size),
            spectrogram=(torch.rand(num_frames, batch_size, frame_channels), frame_lengths),
            stop_token=(torch.rand(num_frames, batch_size), frame_lengths),
            spectrogram_mask=(torch.ones(num_frames, batch_size, dtype=torch.float), frame_lengths),
            spectrogram_expanded_mask=(torch.ones(
                num_frames, batch_size, frame_channels, dtype=torch.float), frame_lengths))
    ]
    forward_pass_return = (torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                           torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
                           torch.FloatTensor(num_frames, batch_size).fill_(1),
                           torch.FloatTensor(num_frames, batch_size,
                                             num_tokens).fill_(1.0 / num_tokens))
    infer_pass_return = (torch.FloatTensor(num_frames, frame_channels).fill_(2),
                         torch.FloatTensor(num_frames, frame_channels).fill_(1),
                         torch.FloatTensor(num_frames).fill_(1),
                         torch.FloatTensor(num_frames, num_tokens).fill_(1.0 / num_tokens),
                         [[num_frames]])
    with ExitStack() as stack:
        (MockDataLoader, mock_call, mock_infer, mock_backward, mock_optimizer_step,
         mock_auto_optimizer_step) = tuple([
             stack.enter_context(mock.patch(arg)) for arg in [
                 'src.bin.train.spectrogram_model.trainer.DataLoader',
                 'src.bin.train.spectrogram_model.trainer.SpectrogramModel.__call__',
                 'src.bin.train.spectrogram_model.trainer.SpectrogramModel.infer',
                 'torch.Tensor.backward', 'src.optimizer.Optimizer.step',
                 'src.optimizer.AutoOptimizer.step'
             ]
         ])
        MockDataLoader.return_value = loaded_data
        mock_call.return_value = forward_pass_return
        mock_infer.return_value = infer_pass_return
        mock_backward.return_value = None
        mock_optimizer_step.return_value = None
        mock_auto_optimizer_step.return_value = None

        trainer.run_epoch(train=False)
        trainer.run_epoch(train=False, trial_run=True)
        trainer.run_epoch(train=True)

        mock_call.assert_called()
        mock_infer.assert_called()
