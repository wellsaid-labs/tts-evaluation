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

    encoded_tokens = torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1)
    tokens_mask = torch.ByteTensor(batch_size, num_tokens).zero_()

    hidden_state = None
    for _ in range(3):
        frames, stop_token, hidden_state, alignment = decoder(
            encoded_tokens=encoded_tokens, tokens_mask=tokens_mask, hidden_state=hidden_state)

        assert frames.type() == 'torch.FloatTensor'
        assert frames.shape == (1, batch_size, frame_channels)

        assert stop_token.type() == 'torch.FloatTensor'
        assert stop_token.shape == (1, batch_size)

        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (1, batch_size, num_tokens)

        assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)


def test_autoregressive_decoder_ground_truth():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    num_frames = 10
    decoder = AutoregressiveDecoder(
        encoder_hidden_size=encoder_hidden_size, frame_channels=frame_channels)

    encoded_tokens = torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1)
    tokens_mask = torch.ByteTensor(batch_size, num_tokens).zero_()
    ground_truth_frames = torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1)

    frames, stop_token, hidden_state, alignment = decoder(
        encoded_tokens=encoded_tokens,
        tokens_mask=tokens_mask,
        ground_truth_frames=ground_truth_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)
