import dataclasses
import typing
from functools import partial

import config as cf
import torch
import torch.nn

from run._models import spectrogram_model
from run._models.spectrogram_model.wrapper import Inputs
from tests import _utils

assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def _make_encoder(
    max_tokens=10,
    max_seq_meta_vals=(11, 12),
    max_token_meta_vals=(13,),
    max_word_vector_size=8,
    max_seq_vector_size=2,
    max_anno_vector_size=1,
    hidden_size=8,
    num_layers=2,
    conv_filter_size=5,
    batch_size=4,
    num_tokens=5,
    context=3,
    num_token_meta=1,
    max_frames_per_token=4.5,
    dropout=0.1,
):
    """Make `encoder.Encoder` and it's inputs for testing."""
    annos = ("anno_embed", "anno_mask")
    token_embed_idx = {
        annos[0]: slice(0, max_anno_vector_size),
        annos[1]: slice(max_anno_vector_size, max_anno_vector_size + 1),
        "word_vector": slice(
            max_anno_vector_size + 1, max_anno_vector_size + 1 + max_word_vector_size
        ),
    }
    encoder = cf.partial(spectrogram_model.encoder.Encoder)(
        max_tokens=max_tokens,
        max_seq_meta_vals=max_seq_meta_vals,
        max_token_meta_vals=max_token_meta_vals,
        max_word_vector_size=max_word_vector_size,
        max_seq_vector_size=max_seq_vector_size,
        max_anno_vector_size=max_anno_vector_size,
        annos=[annos],
        hidden_size=hidden_size,
        num_layers=num_layers,
        conv_filter_size=conv_filter_size,
        dropout=dropout,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    [torch.nn.init.normal_(p) for p in encoder.parameters() if p.std() == 0]

    num_tokens_pad = num_tokens + context * 2
    speakers = torch.randint(1, max_seq_meta_vals[0], (batch_size,))
    sessions = torch.randint(1, max_seq_meta_vals[1], (batch_size,))
    tokens = torch.randint(1, max_tokens, (batch_size, num_tokens_pad))
    token_meta = torch.randint(1, max_tokens, (batch_size, num_token_meta, num_tokens_pad))
    word_vector = torch.randn(batch_size, num_tokens_pad, max_word_vector_size)
    anno_vector = torch.randn(batch_size, num_tokens_pad, max_anno_vector_size)
    anno_mask = torch.ones(batch_size, num_tokens_pad, 1)
    token_vectors = torch.cat((anno_vector, anno_mask, word_vector), dim=2)
    seq_vectors = torch.randn(batch_size, max_seq_vector_size)
    max_audio_len = torch.full((batch_size,), max_frames_per_token * num_tokens)
    inputs = Inputs(
        tokens=tokens.tolist(),
        seq_meta=list(zip(speakers.tolist(), sessions.tolist())),
        token_meta=token_meta.tolist(),
        seq_vectors=seq_vectors,
        token_vector_idx=token_embed_idx,
        token_vectors=token_vectors,
        slices=[slice(context, context + num_tokens) for _ in range(batch_size)],
        max_audio_len=max_audio_len,
    )
    return encoder, inputs, (num_tokens, batch_size, hidden_size)


def test_encoder():
    """Test `encoder.Encoder` handles a basic case."""
    module, arg, (num_tokens, batch_size, hidden_size) = _make_encoder()
    encoded = module(arg)

    assert encoded.tokens.dtype == torch.float
    assert encoded.tokens.shape == (batch_size, num_tokens, hidden_size)

    assert encoded.token_keys.dtype == torch.float
    assert encoded.token_keys.shape == (batch_size, hidden_size, num_tokens)

    assert encoded.tokens_mask.dtype == torch.bool
    assert encoded.tokens_mask.shape == (batch_size, num_tokens)

    assert encoded.num_tokens.dtype == torch.long
    assert encoded.num_tokens.shape == (batch_size,)

    mask_ = ~encoded.tokens_mask
    assert encoded.tokens.masked_select(mask_.unsqueeze(2)).sum() == 0
    assert encoded.token_keys.masked_select(mask_.unsqueeze(1)).sum() == 0
    assert torch.equal(encoded.tokens_mask.sum(dim=1), encoded.num_tokens)
    assert encoded.num_tokens.tolist(), [len(t) for t in arg.tokens]

    encoded.tokens.sum().backward()


def test_encoder__filter_size():
    """Test `encoder.Encoder` handles different filter sizes."""
    for filter_size in [1, 3, 5]:
        kwargs = dict(conv_filter_size=filter_size)
        module, arg, (num_tokens, batch_size, hidden_size) = _make_encoder(**kwargs)
        encoded = module(arg)
        assert encoded.tokens.shape == (batch_size, num_tokens, hidden_size)
        assert encoded.token_keys.shape == (batch_size, hidden_size, num_tokens)
        assert encoded.tokens_mask.shape == (batch_size, num_tokens)
        assert encoded.num_tokens.shape == (batch_size,)
        encoded.tokens.sum().backward()


def test_encoder__padding_invariance():
    """Test `encoder.Encoder` is consistent regardless of the padding."""
    module, arg, (_, batch_size, _) = _make_encoder(dropout=0)
    expected = module(arg)
    expected.tokens.sum().backward()
    expected_grad = [p.grad for p in module.parameters() if p.grad is not None]
    module.zero_grad()
    for pad_len in range(1, 10):
        pad_token: typing.List[typing.Hashable] = [module.embed_token.pad_token] * pad_len
        pad_meta: typing.List[typing.Hashable] = [module.embed_token_meta[0].pad_token] * pad_len
        pad_zeros = torch.zeros(batch_size, pad_len, arg.token_vectors.shape[2])
        inp = dataclasses.replace(
            arg,
            tokens=[t + pad_token for t in arg.tokens],
            token_meta=[[s + pad_meta for s in m] for m in arg.token_meta],
            token_vectors=torch.cat([arg.token_vectors, pad_zeros], dim=1),
        )
        result = module(inp)
        result.tokens.sum().backward()
        result_grad = [p.grad for p in module.parameters() if p.grad is not None]
        module.zero_grad()
        assert_almost_equal(result.tokens, expected.tokens, decimal=5)
        [assert_almost_equal(r, e) for r, e in zip(result_grad, expected_grad)]
