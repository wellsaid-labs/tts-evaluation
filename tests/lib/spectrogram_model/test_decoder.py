import torch

from src.spectrogram_model.decoder import AutoregressiveDecoder
from src.spectrogram_model.decoder import AutoregressiveDecoderHiddenState


def test_autoregressive_decoder():
    encoder_output_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    speaker_embedding_dim = 16
    decoder = AutoregressiveDecoder(
        speaker_embedding_dim=speaker_embedding_dim,
        frame_channels=frame_channels,
        encoder_output_size=encoder_output_size)

    encoded_tokens = torch.rand(num_tokens, batch_size, encoder_output_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    speaker = torch.zeros(batch_size, speaker_embedding_dim)

    hidden_state = None
    for _ in range(3):
        frames, stop_token, hidden_state, alignment = decoder(
            encoded_tokens=encoded_tokens,
            tokens_mask=tokens_mask,
            speaker=speaker,
            hidden_state=hidden_state,
            num_tokens=tokens_mask.long().sum(dim=1))

        assert frames.type() == 'torch.FloatTensor'
        assert frames.shape == (1, batch_size, frame_channels)

        assert stop_token.type() == 'torch.FloatTensor'
        assert stop_token.shape == (1, batch_size)

        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (1, batch_size, num_tokens)

        assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)


def test_autoregressive_decoder_target():
    encoder_output_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    num_frames = 10
    speaker_embedding_dim = 16
    decoder = AutoregressiveDecoder(
        speaker_embedding_dim=speaker_embedding_dim,
        encoder_output_size=encoder_output_size,
        frame_channels=frame_channels)

    encoded_tokens = torch.rand(num_tokens, batch_size, encoder_output_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    target_frames = torch.rand(num_frames, batch_size, frame_channels)
    speaker = torch.zeros(batch_size, speaker_embedding_dim)

    frames, stop_token, hidden_state, alignment = decoder(
        encoded_tokens=encoded_tokens,
        tokens_mask=tokens_mask,
        speaker=speaker,
        target_frames=target_frames,
        num_tokens=tokens_mask.long().sum(dim=1))

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    assert isinstance(hidden_state, AutoregressiveDecoderHiddenState)
