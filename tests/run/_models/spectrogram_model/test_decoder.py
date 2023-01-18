import random
import typing

import config as cf
import torch

import run
from lib.utils import lengths_to_mask
from run._models.spectrogram_model.containers import (
    AttentionHiddenState,
    DecoderHiddenState,
    Encoded,
)
from run._models.spectrogram_model.decoder import Decoder


def _make_decoder(
    num_frame_channels=16,
    seq_embed_size=8,
    pre_net_size=3,
    lstm_hidden_size=4,
    encoder_out_size=5,
    stop_net_dropout=0.5,
    stop_net_hidden_size=3,
) -> Decoder:
    """Make `decoder.Decoder` for testing."""
    _config = {
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(num_layers=1, dropout=0.5),
        run._models.spectrogram_model.attention.Attention: cf.Args(
            hidden_size=4,
            conv_filter_size=3,
            dropout=0.1,
            window_length=5,
            avg_frames_per_token=1.0,
        ),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(
            pre_net_size=pre_net_size,
            lstm_hidden_size=lstm_hidden_size,
            encoder_out_size=encoder_out_size,
            stop_net_dropout=stop_net_dropout,
            stop_net_hidden_size=stop_net_hidden_size,
        ),
    }
    cf.add(_config, overwrite=True)
    return cf.partial(Decoder)(
        num_frame_channels=num_frame_channels,
        seq_embed_size=seq_embed_size,
    )


def _make_encoded(
    module: Decoder, batch_size: int = 5, max_num_tokens: int = 6
) -> typing.Tuple[Encoded, typing.Tuple[int, int]]:
    """Make `Encoded` for testing."""
    tokens = torch.randn(max_num_tokens, batch_size, module.encoder_out_size)
    num_tokens = [random.randint(1, max_num_tokens) for _ in range(batch_size)]
    num_tokens[-1] = max_num_tokens
    num_tokens = torch.tensor(num_tokens)
    tokens_mask = lengths_to_mask(num_tokens)
    tokens = tokens * tokens_mask.transpose(0, 1).unsqueeze(-1)
    seq_metadata = torch.randn(batch_size, module.seq_embed_size)
    encoded = Encoded(tokens, tokens_mask, num_tokens, seq_metadata)
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

        assert decoded.hidden_state.last_attention_context.dtype == torch.float
        expected = (batch_size, module.encoder_out_size)
        assert decoded.hidden_state.last_attention_context.shape == expected

        assert isinstance(decoded.hidden_state.attention_hidden_state, AttentionHiddenState)
        assert isinstance(decoded.hidden_state.lstm_one_hidden_state, tuple)
        assert isinstance(decoded.hidden_state.lstm_two_hidden_state, tuple)

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

    assert decoded.hidden_state.last_attention_context.dtype == torch.float
    expected = (batch_size, module.encoder_out_size)
    assert decoded.hidden_state.last_attention_context.shape == expected

    assert isinstance(decoded.hidden_state.attention_hidden_state, AttentionHiddenState)
    assert isinstance(decoded.hidden_state.lstm_one_hidden_state, tuple)
    assert isinstance(decoded.hidden_state.lstm_two_hidden_state, tuple)

    (decoded.frames.sum() + decoded.stop_tokens.sum()).backward()


def test_decoder__pad_encoded():
    """Test `decoder.Decoder` pads encoded correctly."""
    module = _make_decoder()
    encoded, (batch_size, num_tokens) = _make_encoded(module)
    window_length, encoder_size = module.attention.window_length - 1, module.encoder_out_size
    beg_pad_token = torch.rand(batch_size, module.encoder_out_size)
    end_pad_token = torch.rand(batch_size, module.encoder_out_size)
    padded = module._pad_encoded(encoded, beg_pad_token, end_pad_token)

    assert padded.tokens.dtype == torch.float
    assert padded.tokens.shape == (num_tokens + window_length, batch_size, encoder_size)

    assert padded.tokens_mask.dtype == torch.bool
    assert padded.tokens_mask.shape == (batch_size, num_tokens + window_length)

    assert padded.num_tokens.dtype == torch.long
    assert padded.num_tokens.shape == (batch_size,)

    pad = window_length // 2
    for i in range(pad):
        arange = torch.arange(batch_size)
        assert padded.tokens_mask[arange, i].all()
        assert padded.tokens_mask[arange, encoded.num_tokens + i + pad].all()
        assert torch.equal(padded.tokens[i, arange], beg_pad_token)
        assert torch.equal(padded.tokens[encoded.num_tokens + i + pad, arange], end_pad_token)

    assert padded.tokens.masked_select(~padded.tokens_mask.transpose(0, 1).unsqueeze(-1)).sum() == 0
    assert torch.equal(padded.tokens_mask.sum(dim=1), padded.num_tokens)
    assert torch.equal(encoded.num_tokens + window_length, padded.num_tokens)
