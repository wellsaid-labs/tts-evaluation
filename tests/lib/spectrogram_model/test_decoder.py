import hparams
import pytest
import torch
from hparams import HParams

import lib
from lib.spectrogram_model.attention import AttentionHiddenState
from lib.spectrogram_model.decoder import Decoder, DecoderHiddenState


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    hparams.clear_config()


def _make_decoder(
    num_frame_channels=16,
    speaker_embedding_size=8,
    pre_net_size=3,
    lstm_hidden_size=4,
    encoder_output_size=5,
    stop_net_dropout=0.5,
) -> Decoder:
    """Make `decoder.Decoder` for testing."""
    config = {
        lib.spectrogram_model.pre_net.PreNet.__init__: HParams(num_layers=1, dropout=0.5),
        lib.spectrogram_model.attention.Attention.__init__: HParams(
            hidden_size=4,
            convolution_filter_size=3,
            dropout=0.1,
            window_length=5,
            avg_frames_per_token=1.0,
        ),
    }
    hparams.add_config(config)
    return Decoder(
        num_frame_channels=num_frame_channels,
        speaker_embedding_size=speaker_embedding_size,
        pre_net_size=pre_net_size,
        lstm_hidden_size=lstm_hidden_size,
        encoder_output_size=encoder_output_size,
        stop_net_dropout=stop_net_dropout,
    )


def test_autoregressive_decoder():
    """Test `decoder.Decoder` handles a basic case."""
    batch_size = 5
    num_tokens = 6
    module = _make_decoder()
    tokens = torch.rand(num_tokens, batch_size, module.encoder_output_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    speaker = torch.zeros(batch_size, module.speaker_embedding_size)
    hidden_state = None
    frames = torch.empty(0)
    stop_token = torch.empty(0)
    for _ in range(3):
        frames, stop_token, alignment, window_starts, hidden_state = module(
            tokens=tokens,
            tokens_mask=tokens_mask,
            num_tokens=tokens_mask.long().sum(dim=1),
            speaker=speaker,
            hidden_state=hidden_state,
        )

        assert frames.dtype == torch.float
        assert frames.shape == (1, batch_size, module.num_frame_channels)

        assert stop_token.dtype == torch.float
        assert stop_token.shape == (1, batch_size)

        assert alignment.dtype == torch.float
        assert alignment.shape == (1, batch_size, num_tokens)

        assert window_starts.dtype == torch.long
        assert window_starts.shape == (1, batch_size)

        assert isinstance(hidden_state, DecoderHiddenState)

        assert hidden_state.last_frame.dtype == torch.float
        assert hidden_state.last_frame.shape == (
            1,
            batch_size,
            module.num_frame_channels,
        )

        assert hidden_state.last_attention_context.dtype == torch.float
        assert hidden_state.last_attention_context.shape == (
            batch_size,
            module.encoder_output_size,
        )

        assert isinstance(hidden_state.attention_hidden_state, AttentionHiddenState)
        assert isinstance(hidden_state.lstm_one_hidden_state, tuple)
        assert isinstance(hidden_state.lstm_two_hidden_state, tuple)

    (frames.sum() + stop_token.sum()).backward()


def test_autoregressive_decoder__target():
    """Test `decoder.Decoder` handles `target_frames` inputs."""
    batch_size = 5
    num_frames = 3
    num_tokens = 6
    module = _make_decoder()
    tokens = torch.rand(num_tokens, batch_size, module.encoder_output_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    target_frames = torch.rand(num_frames, batch_size, module.num_frame_channels)
    speaker = torch.zeros(batch_size, module.speaker_embedding_size)

    frames, stop_token, alignment, window_starts, hidden_state = module(
        tokens=tokens,
        tokens_mask=tokens_mask,
        num_tokens=tokens_mask.long().sum(dim=1),
        speaker=speaker,
        target_frames=target_frames,
    )

    assert frames.dtype == torch.float
    assert frames.shape == (num_frames, batch_size, module.num_frame_channels)

    assert stop_token.dtype == torch.float
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.dtype == torch.float
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    assert window_starts.dtype == torch.long
    assert window_starts.shape == (num_frames, batch_size)

    assert hidden_state.last_frame.dtype == torch.float
    assert hidden_state.last_frame.shape == (1, batch_size, module.num_frame_channels)

    assert hidden_state.last_attention_context.dtype == torch.float
    assert hidden_state.last_attention_context.shape == (
        batch_size,
        module.encoder_output_size,
    )

    assert isinstance(hidden_state.attention_hidden_state, AttentionHiddenState)
    assert isinstance(hidden_state.lstm_one_hidden_state, tuple)
    assert isinstance(hidden_state.lstm_two_hidden_state, tuple)

    (frames.sum() + stop_token.sum()).backward()
