import torch

from src.feature_model.decoder import AutoregressiveDecoder
from src.feature_model.decoder import AutoregressiveDecoderHiddenState


def test_autoregressive_decoder():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    decoder = AutoregressiveDecoder(
        encoder_hidden_size=encoder_hidden_size, frame_channels=frame_channels)
    for param in decoder.parameters():
        param.data.uniform_(-0.1, 0.1)

    encoded_tokens = torch.autograd.Variable(
        torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1))

    hidden_state = None
    for _ in range(3):
        frames, frames_with_residual, stop_token, hidden_state = decoder(
            encoded_tokens=encoded_tokens, hidden_state=hidden_state)

        assert frames.data.type() == 'torch.FloatTensor'
        assert frames.shape == (1, batch_size, frame_channels)

        assert frames_with_residual.data.type() == 'torch.FloatTensor'
        assert frames_with_residual.shape == (1, batch_size, frame_channels)

        assert stop_token.data.type() == 'torch.FloatTensor'
        assert stop_token.shape == (1, batch_size)

        assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)


def test_autoregressive_decoder_ground_truth():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    num_frames = 10
    decoder = AutoregressiveDecoder(
        encoder_hidden_size=encoder_hidden_size, frame_channels=frame_channels)
    for param in decoder.parameters():
        param.data.uniform_(-0.1, 0.1)

    encoded_tokens = torch.autograd.Variable(
        torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1))
    ground_truth_frames = torch.autograd.Variable(
        torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1))

    frames, frames_with_residual, stop_token, hidden_state = decoder(
        encoded_tokens=encoded_tokens, ground_truth_frames=ground_truth_frames)

    assert frames.data.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.data.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.data.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)
