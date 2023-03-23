import random
import typing

import config as cf
import torch

import run
from lib.utils import lengths_to_mask
from run._models.spectrogram_model.containers import (
    AttentionRNNHiddenState,
    DecoderHiddenState,
    Encoded,
)
from run._models.spectrogram_model.decoder import Decoder


def _make_decoder(
    num_frame_channels=16,
    hidden_size=4,
    attn_size=5,
    stop_net_dropout=0.5,
) -> Decoder:
    """Make `decoder.Decoder` for testing."""
    _config = {
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(num_layers=1, dropout=0.5),
        run._models.spectrogram_model.attention.Attention: cf.Args(
            conv_filter_size=3,
            window_len=5,
            avg_frames_per_token=1.0,
        ),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(
            hidden_size=hidden_size,
            attn_size=attn_size,
            stop_net_dropout=stop_net_dropout,
        ),
    }
    cf.add(_config, overwrite=True)
    return cf.partial(Decoder)(num_frame_channels=num_frame_channels)


def _make_encoded(
    module: Decoder, batch_size: int = 5, max_num_tokens: int = 6
) -> typing.Tuple[Encoded, typing.Tuple[int, int]]:
    """Make `Encoded` for testing."""
    tokens = torch.randn(batch_size, max_num_tokens, module.attn_size)
    token_keys = torch.randn(batch_size, module.attn_size, max_num_tokens)
    num_tokens = [random.randint(1, max_num_tokens) for _ in range(batch_size)]
    num_tokens[-1] = max_num_tokens
    num_tokens = torch.tensor(num_tokens)
    tokens_mask = lengths_to_mask(num_tokens)
    tokens = tokens * tokens_mask.unsqueeze(2)
    token_keys = token_keys * tokens_mask.unsqueeze(1)
    encoded = Encoded(tokens, token_keys, tokens_mask, num_tokens)
    return encoded, (batch_size, max_num_tokens)


def test_decoder():
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

        assert decoded.hidden_state.attn_rnn_hidden_state.last_attn_context.dtype == torch.float
        expected = (batch_size, module.attn_size)
        assert decoded.hidden_state.attn_rnn_hidden_state.last_attn_context.shape == expected

        assert isinstance(decoded.hidden_state.attn_rnn_hidden_state, AttentionRNNHiddenState)
        assert isinstance(decoded.hidden_state.pre_net_hidden_state, tuple)
        assert isinstance(decoded.hidden_state.lstm_hidden_state, tuple)

    assert decoded is not None
    (decoded.frames.sum() + decoded.stop_tokens.sum()).backward()


def test_decoder__target():
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

    assert decoded.hidden_state.attn_rnn_hidden_state.last_attn_context.dtype == torch.float
    expected = (batch_size, module.attn_size)
    assert decoded.hidden_state.attn_rnn_hidden_state.last_attn_context.shape == expected

    assert isinstance(decoded.hidden_state.attn_rnn_hidden_state, AttentionRNNHiddenState)
    assert isinstance(decoded.hidden_state.lstm_hidden_state, tuple)

    (decoded.frames.sum() + decoded.stop_tokens.sum()).backward()


def test_decoder__pad_encoded():
    """Test `decoder.Decoder` pads encoded correctly."""
    module = _make_decoder()
    encoded, (batch_size, num_tokens) = _make_encoded(module)
    window_len = module.attn_rnn.attn.window_len - 1
    attn_size = module.attn_size
    beg_pad_token = torch.rand(batch_size, module.attn_size)
    end_pad_token = torch.rand(batch_size, module.attn_size)
    beg_pad_key = torch.rand(batch_size, module.attn_size)
    end_pad_key = torch.rand(batch_size, module.attn_size)
    padded = module._pad_encoded(encoded, beg_pad_token, end_pad_token, beg_pad_key, end_pad_key)

    assert padded.tokens.dtype == torch.float
    assert padded.tokens.shape == (batch_size, num_tokens + window_len, attn_size)

    assert padded.token_keys.dtype == torch.float
    assert padded.token_keys.shape == (batch_size, attn_size, num_tokens + window_len)

    assert padded.tokens_mask.dtype == torch.bool
    assert padded.tokens_mask.shape == (batch_size, num_tokens + window_len)

    assert padded.num_tokens.dtype == torch.long
    assert padded.num_tokens.shape == (batch_size,)

    pad = window_len // 2
    for i in range(pad):
        idx = torch.arange(batch_size)
        assert padded.tokens_mask[idx, i].all()
        assert padded.tokens_mask[idx, encoded.num_tokens + i + pad].all()
        assert torch.equal(padded.tokens[idx, i], beg_pad_token)
        assert torch.equal(padded.token_keys[idx, :, i], beg_pad_key)
        assert torch.equal(padded.tokens[idx, encoded.num_tokens + i + pad], end_pad_token)
        assert torch.equal(padded.token_keys[idx, :, encoded.num_tokens + i + pad], end_pad_key)

    assert padded.tokens.masked_select(~padded.tokens_mask.unsqueeze(2)).sum() == 0
    assert padded.token_keys.masked_select(~padded.tokens_mask.unsqueeze(1)).sum() == 0
    assert torch.equal(padded.tokens_mask.sum(dim=1), padded.num_tokens)
    assert torch.equal(encoded.num_tokens + window_len, padded.num_tokens)
