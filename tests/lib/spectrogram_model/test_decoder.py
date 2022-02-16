import typing

import hparams
import pytest
import torch
from hparams import HParams

import lib
from lib.spectrogram_model.containers import AttentionHiddenState, DecoderHiddenState, Encoded
from lib.spectrogram_model.decoder import Decoder


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    hparams.clear_config()


def _make_decoder(
    num_frame_channels=16,
    seq_meta_embed_size=8,
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
        seq_meta_embed_size=seq_meta_embed_size,
        pre_net_size=pre_net_size,
        lstm_hidden_size=lstm_hidden_size,
        encoder_output_size=encoder_output_size,
        stop_net_dropout=stop_net_dropout,
    )


def _make_encoded(
    module: Decoder, batch_size: int = 5, num_tokens: int = 6
) -> typing.Tuple[Encoded, typing.Tuple[int, int]]:
    """Make `Encoded` for testing."""
    tokens = torch.rand(num_tokens, batch_size, module.encoder_output_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    seq_metadata = torch.zeros(batch_size, module.seq_meta_embed_size)
    encoded = Encoded(tokens, tokens_mask, tokens_mask.sum(dim=1), seq_metadata)
    return encoded, (batch_size, num_tokens)


def test_autoregressive_decoder():
    """Test `decoder.Decoder` handles a basic case."""
    module = _make_decoder()
    encoded, (batch_size, num_tokens) = _make_encoded(module)
    hidden_state = None
    decoded = None
    for _ in range(3):
        decoded = module(encoded, hidden_state=hidden_state)

        assert decoded.frames.dtype == torch.float
        assert decoded.frames.shape == (1, batch_size, module.num_frame_channels)

        assert decoded.stop_tokens.dtype == torch.float
        assert decoded.stop_tokens.shape == (1, batch_size)

        assert decoded.alignments.dtype == torch.float
        assert decoded.alignments.shape == (1, batch_size, num_tokens)

        assert decoded.window_starts.dtype == torch.long
        assert decoded.window_starts.shape == (1, batch_size)

        assert isinstance(decoded.hidden_state, DecoderHiddenState)

        assert decoded.hidden_state.last_frame.dtype == torch.float
        expected = (1, batch_size, module.num_frame_channels)
        assert decoded.hidden_state.last_frame.shape == expected

        assert decoded.hidden_state.last_attention_context.dtype == torch.float
        expected = (batch_size, module.encoder_output_size)
        assert decoded.hidden_state.last_attention_context.shape == expected

        assert isinstance(decoded.hidden_state.attention_hidden_state, AttentionHiddenState)
        assert isinstance(decoded.hidden_state.lstm_one_hidden_state, tuple)
        assert isinstance(decoded.hidden_state.lstm_two_hidden_state, tuple)

    assert decoded is not None
    (decoded.frames.sum() + decoded.stop_tokens.sum()).backward()


def test_autoregressive_decoder__target():
    """Test `decoder.Decoder` handles `target_frames` inputs."""
    num_frames = 3
    module = _make_decoder()
    encoded, (batch_size, num_tokens) = _make_encoded(module)
    target_frames = torch.rand(num_frames, batch_size, module.num_frame_channels)

    decoded = module(encoded, target_frames=target_frames)

    assert decoded.frames.dtype == torch.float
    assert decoded.frames.shape == (num_frames, batch_size, module.num_frame_channels)

    assert decoded.stop_tokens.dtype == torch.float
    assert decoded.stop_tokens.shape == (num_frames, batch_size)

    assert decoded.alignments.dtype == torch.float
    assert decoded.alignments.shape == (num_frames, batch_size, num_tokens)

    assert decoded.window_starts.dtype == torch.long
    assert decoded.window_starts.shape == (num_frames, batch_size)

    assert decoded.hidden_state.last_frame.dtype == torch.float
    assert decoded.hidden_state.last_frame.shape == (1, batch_size, module.num_frame_channels)

    assert decoded.hidden_state.last_attention_context.dtype == torch.float
    expected = (batch_size, module.encoder_output_size)
    assert decoded.hidden_state.last_attention_context.shape == expected

    assert isinstance(decoded.hidden_state.attention_hidden_state, AttentionHiddenState)
    assert isinstance(decoded.hidden_state.lstm_one_hidden_state, tuple)
    assert isinstance(decoded.hidden_state.lstm_two_hidden_state, tuple)

    (decoded.frames.sum() + decoded.stop_tokens.sum()).backward()
