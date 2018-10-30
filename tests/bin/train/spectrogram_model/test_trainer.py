from contextlib import contextmanager
from contextlib import ExitStack

from unittest import mock

from torchnlp.text_encoders import CharacterEncoder

import pytest
import torch

from src.bin.train.spectrogram_model.trainer import Trainer


class MockTensorboard():

    def __init__(*args, **kwargs):
        pass

    @contextmanager
    def set_step(self, *args, **kwargs):
        yield self

    def __getattr__(self, attr):
        return lambda *args, **kwargs: self


@pytest.fixture
def text_encoder_fixture():
    return CharacterEncoder(['text encoder'])


@pytest.fixture
def trainer_fixture(text_encoder_fixture):
    trainer = Trainer(
        device=torch.device('cpu'),
        train_dataset=[],
        dev_dataset=[],
        train_tensorboard=MockTensorboard(),
        dev_tensorboard=MockTensorboard(),
        text_encoder=text_encoder_fixture,
        train_batch_size=1,
        dev_batch_size=1)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token[0].weight, float('-inf'))
    torch.nn.init.constant_(trainer.model.decoder.linear_stop_token[0].bias, float('-inf'))
    trainer.tensorboard = trainer.train_tensorboard
    return trainer


def test__compute_loss(trainer_fixture):
    batch = {
        'spectrogram_expanded_mask': torch.FloatTensor([[1, 1], [1, 1], [0, 0]]),
        'spectrogram_mask': torch.FloatTensor([1, 1, 0]),
        'spectrogram': torch.FloatTensor([[1, 1], [1, 1], [3, 3]]),
        'stop_token': torch.FloatTensor([0, 1, 1])
    }
    predicted_pre_frames = torch.FloatTensor([[1, 1], [1, 1], [1, 1]])
    predicted_post_frames = torch.FloatTensor([[0.5, 0.5], [0.5, 0.5], [1, 1]])
    predicted_stop_tokens = torch.FloatTensor([0, 0.5, 0.5])
    (pre_frames_loss, post_frames_loss, stop_token_loss,
     num_frame_predictions, num_frames) = trainer_fixture._compute_loss(
         batch, predicted_pre_frames, predicted_post_frames, predicted_stop_tokens)
    assert pre_frames_loss.item() == pytest.approx(0.0)
    assert post_frames_loss.item() == pytest.approx(1.0 / 4)
    assert stop_token_loss.item() == pytest.approx(0.6931471824645996 / 2)
    assert num_frame_predictions == 4
    assert num_frames == 2


def test__sample_infered(trainer_fixture):
    batch_size = 2
    batch = {
        'text': torch.LongTensor(8, batch_size).fill_(1),
        'spectrogram': torch.FloatTensor(8, batch_size, 16).fill_(1),
        'text_lengths': [4, 8],
        'spectrogram_lengths': [4, 8]
    }
    trainer_fixture._sample_infered(batch, max_infer_frames=2)


def test__sample_predicted(trainer_fixture):
    batch_size = 2
    num_frames = 8
    num_tokens = 8
    batch = {
        'text': torch.LongTensor(num_tokens, batch_size).fill_(1),
        'spectrogram': torch.FloatTensor(num_frames, batch_size, 16).fill_(1),
        'text_lengths': [4, num_tokens],
        'spectrogram_lengths': [4, num_frames]
    }
    predicted_pre_spectrogram = torch.FloatTensor(num_frames, batch_size, 16)
    predicted_post_spectrogram = torch.FloatTensor(num_frames, batch_size, 16)
    predicted_stop_tokens = torch.FloatTensor(num_frames, batch_size)
    predicted_alignments = torch.FloatTensor(num_frames, batch_size, num_tokens)
    trainer_fixture._sample_predicted(batch, predicted_pre_spectrogram, predicted_post_spectrogram,
                                      predicted_alignments, predicted_stop_tokens)


def test_run_epoch(trainer_fixture):
    batch_size = 2
    num_frames = 8
    num_tokens = 8
    frame_channels = 16
    batch = {
        'text':
            torch.LongTensor(num_tokens, batch_size).fill_(1),
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

        trainer_fixture.run_epoch(train=False)
        trainer_fixture.run_epoch(train=False, trial_run=True)
        mock_call.assert_called()
        mock_infer.assert_called()
        torch.set_grad_enabled(True)  # Reset
