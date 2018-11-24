from contextlib import contextmanager
from contextlib import ExitStack

from unittest import mock

from torchnlp.text_encoders import CharacterEncoder
from torchnlp.text_encoders import IdentityEncoder

import pytest
import torch

from src.bin.train.spectrogram_model.trainer import Trainer
from src.datasets import Speaker


class MockCometML():

    def __init__(*args, **kwargs):
        pass

    @contextmanager
    def train(self, *args, **kwargs):
        yield self

    @contextmanager
    def validate(self, *args, **kwargs):
        yield self

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self


def get_trainer():
    trainer = Trainer(
        device=torch.device('cpu'),
        train_dataset=[],
        dev_dataset=[],
        comet_ml_project_name='',
        text_encoder=CharacterEncoder(['text encoder']),
        speaker_encoder=IdentityEncoder([Speaker.LINDA_JOHNSON]),
        train_batch_size=1,
        dev_batch_size=1)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token[0].weight, float('-inf'))
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token[0].bias, float('-inf'))
    return trainer


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
def test__compute_loss(comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    trainer = get_trainer()
    batch = {
        'spectrogram_expanded_mask': torch.FloatTensor([[1, 1], [1, 1], [0, 0]]),
        'spectrogram_mask': torch.FloatTensor([1, 1, 0]),
        'spectrogram': torch.FloatTensor([[1, 1], [1, 1], [3, 3]]),
        'stop_token': torch.FloatTensor([0, 1, 1])
    }
    predicted_pre_spectrogram = torch.FloatTensor([[1, 1], [1, 1], [1, 1]])
    predicted_post_spectrogram = torch.FloatTensor([[0.5, 0.5], [0.5, 0.5], [1, 1]])
    predicted_stop_tokens = torch.FloatTensor([0, 0.5, 0.5])
    predictions = (predicted_pre_spectrogram, predicted_post_spectrogram, predicted_stop_tokens,
                   None)
    (pre_spectrogram_loss, post_spectrogram_loss, stop_token_loss, num_spectrogram_values,
     num_frames) = trainer._do_loss_and_maybe_backwards(batch, predictions, False)
    assert pre_spectrogram_loss.item() == pytest.approx(0.0)
    assert post_spectrogram_loss.item() == pytest.approx(1.0 / 4)
    assert stop_token_loss.item() == pytest.approx(0.6931471824645996 / 2)
    assert num_spectrogram_values == 4
    assert num_frames == 2


@mock.patch('src.bin.train.spectrogram_model.trainer.CometML')
def test_run_epoch(comet_ml_mock):
    comet_ml_mock.return_value = MockCometML()
    trainer = get_trainer()
    batch_size = 2
    num_frames = 8
    num_tokens = 8
    frame_channels = 16
    batch = {
        'text':
            torch.LongTensor(num_tokens, batch_size).fill_(1),
        'speaker':
            torch.LongTensor(1, batch_size).fill_(0),
        'text_lengths': [4, num_tokens],
        'spectrogram':
            torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
        'spectrogram_lengths': [4, num_frames],
        'spectrogram_expanded_mask':
            torch.FloatTensor(num_frames, batch_size, frame_channels).fill_(1),
        'spectrogram_mask':
            torch.FloatTensor(num_frames, batch_size).fill_(1),
        'stop_token':
            torch.FloatTensor(num_frames, batch_size).fill_(1),
    }
    with ExitStack() as stack:
        MockDataBatchIterator, mock_call, mock_infer = tuple([
            stack.enter_context(mock.patch(arg)) for arg in [
                'src.bin.train.spectrogram_model.trainer.DataBatchIterator',
                'src.bin.train.spectrogram_model.trainer.SpectrogramModel.__call__',
                'src.bin.train.spectrogram_model.trainer.SpectrogramModel.infer'
            ]
        ])
        MockDataBatchIterator.return_value = [batch, batch]
        mock_call.return_value = (torch.FloatTensor(num_frames, batch_size,
                                                    frame_channels).fill_(1),
                                  torch.FloatTensor(num_frames, batch_size,
                                                    frame_channels).fill_(1),
                                  torch.FloatTensor(num_frames, batch_size).fill_(1),
                                  torch.FloatTensor(num_frames, batch_size,
                                                    num_tokens).fill_(1.0 / num_tokens))
        mock_infer.return_value = (torch.FloatTensor(num_frames, frame_channels).fill_(2),
                                   torch.FloatTensor(num_frames, frame_channels).fill_(1),
                                   torch.FloatTensor(num_frames).fill_(1),
                                   torch.FloatTensor(num_frames, num_tokens).fill_(
                                       1.0 / num_tokens), [[num_frames]])

        trainer.run_epoch(train=False)
        trainer.run_epoch(train=False, trial_run=True)
        mock_call.assert_called()
        mock_infer.assert_called()
        torch.set_grad_enabled(True)  # Reset
