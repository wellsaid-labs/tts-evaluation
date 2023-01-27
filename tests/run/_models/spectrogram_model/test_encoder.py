import dataclasses
import typing
from functools import partial

import config as cf
import torch
import torch.nn
from torchnlp.random import fork_rng

from run._models import spectrogram_model
from run._models.spectrogram_model.wrapper import Inputs
from tests import _utils

assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def test__roll():
    """Test `spectrogram_model.encoder._roll` to roll given a simple tensor."""
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    result = spectrogram_model.encoder._roll(tensor, shift=torch.tensor(4), dim=0)
    assert torch.equal(result, torch.tensor([3, 4, 5, 6, 1, 2]))


def test__roll__2d():
    """Test `spectrogram_model.encoder._roll` to roll given a 2d `tensor` with variable
    `shift`. Furthermore, this tests a negative `dim`."""
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = spectrogram_model.encoder._roll(tensor, shift=torch.tensor([0, 1, 2]), dim=-1)
    assert torch.equal(result, torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test__roll__transpose():
    """Test `spectrogram_model.encoder._roll` to roll given a transposed 2d `tensor`.
     `spectrogram_model.encoder._roll` should return consistent results regardless of the
    dimension and ordering of the data."""
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]).transpose(0, 1)
    result = spectrogram_model.encoder._roll(
        tensor, shift=torch.tensor([0, 1, 2]), dim=0
    ).transpose(0, 1)
    assert torch.equal(result, torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test__roll__3d():
    """Test `spectrogram_model.encoder._roll` to roll given a 3d `tensor` and 2d `start`."""
    tensor = torch.tensor([1, 2, 3]).view(1, 3, 1).expand(4, 3, 4)
    shift = torch.arange(0, 16).view(4, 4)
    result = spectrogram_model.encoder._roll(tensor, shift=shift, dim=1)
    assert shift[0, 0] == 0
    assert torch.equal(result[0, :, 0], torch.tensor([1, 2, 3]))
    assert shift[0, 1] == 1
    assert torch.equal(result[0, :, 1], torch.tensor([3, 1, 2]))
    assert shift[0, 2] == 2
    assert torch.equal(result[0, :, 2], torch.tensor([2, 3, 1]))
    assert shift[0, 3] == 3
    assert torch.equal(result[0, :, 3], torch.tensor([1, 2, 3]))
    assert shift[1, 0] == 4
    assert torch.equal(result[1, :, 0], torch.tensor([3, 1, 2]))
    assert shift[1, 1] == 5
    assert torch.equal(result[1, :, 1], torch.tensor([2, 3, 1]))


def _make_rnn(
    input_size: int = 4, hidden_size: int = 5, num_layers: int = 2, **kwargs
) -> spectrogram_model.encoder._RightMaskedBiRNN:
    """Make `encoder._RightMaskedBiRNN` for testing."""
    return spectrogram_model.encoder._RightMaskedBiRNN(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs
    )


def _make_rnn_inputs(
    module: spectrogram_model.encoder._RightMaskedBiRNN,
    batch_size: int = 2,
    seq_len: int = 3,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Make `encoder._RightMaskedBiRNN` inputs for testing."""
    tokens = torch.randn(seq_len, batch_size, module.input_size)
    tokens_mask = torch.ones(seq_len, batch_size, dtype=torch.bool)
    return tokens, tokens_mask


def test__right_masked_bi_rnn__lstm():
    """Test `encoder._RightMaskedBiRNN` is consistent with `torch.nn.LSTM`."""
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(rnn_class=torch.nn.LSTM)
    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    result = masked_bi_rnn(tokens, tokens_mask, num_tokens)
    assert torch.equal(lstm(tokens)[0], masked_bi_rnn(tokens, tokens_mask, num_tokens))
    result.sum().backward()


def test__right_masked_bi_rnn__gru():
    """Test `encoder._RightMaskedBiRNN` is consistent with `torch.nn.GRU`."""
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(rnn_class=torch.nn.GRU)
    with fork_rng(123):
        gru = torch.nn.GRU(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    assert torch.equal(gru(tokens)[0], masked_bi_rnn(tokens, tokens_mask, num_tokens))


def test__right_masked_bi_rnn__jagged_mask():
    """Test `encoder._RightMaskedBiRNN` ignores the jagged padding and is consistent with
    `torch.nn.LSTM`."""
    input_size = 1
    hidden_size = 5

    # tokens [batch_size, seq_len] → [seq_len, batch_size, input_size]
    tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=torch.float32)
    tokens = tokens.transpose(0, 1).unsqueeze(2)
    # tokens_mask [batch_size, seq_len] → [seq_len, batch_size]
    tokens_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool).transpose(0, 1)
    num_tokens = tokens_mask.sum(dim=0)

    masked_bi_rnn = spectrogram_model.encoder._RightMaskedBiRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        rnn_class=torch.nn.LSTM,
    )
    result = masked_bi_rnn(tokens, tokens_mask, num_tokens)

    forward_lstm, backward_lstm = typing.cast(
        typing.Tuple[torch.nn.LSTM, torch.nn.LSTM], masked_bi_rnn.rnn_layers[0]
    )
    expected = torch.zeros(tokens.shape[0], tokens.shape[1], hidden_size * 2)
    expected[:, :, :hidden_size] = forward_lstm(tokens)[0].masked_fill(~tokens_mask.unsqueeze(2), 0)
    expected[:3, 0:1, hidden_size:] = backward_lstm(tokens[:3, 0:1].flip(0))[0].flip(0)
    expected[:2, 1:2, hidden_size:] = backward_lstm(tokens[:2, 1:2].flip(0))[0].flip(0)
    assert_almost_equal(result, expected)


def test__right_masked_bi_rnn__multilayer_mask():
    """Test `encoder._RightMaskedBiRNN` ignores the multiple layers of padding and is consistent
    with  `torch.nn.LSTM`."""
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(num_layers=3, rnn_class=torch.nn.LSTM)
    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    pad_len = 2
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    padded_tokens = torch.cat([tokens, torch.zeros(pad_len, *tokens.shape[1:3])], dim=0)
    padded_tokens_mask = [tokens_mask, torch.zeros(pad_len, tokens_mask.shape[1], dtype=torch.bool)]
    padded_tokens_mask = torch.cat(padded_tokens_mask, dim=0)

    expected = lstm(tokens)[0]
    result = masked_bi_rnn(padded_tokens, padded_tokens_mask, num_tokens)
    assert_almost_equal(expected, result[:-pad_len])
    assert result[-pad_len:].sum().item() == 0
    assert result[:-pad_len].sum().item() != 0


def test__grouped_embedder():
    """Test `spectrogram_model.encoder._GroupedEmbedder` in a basic test."""
    num_groups = 3
    input_size = 3
    num_tokens = 6
    batch_size = 2
    hidden_size = 7
    tokens = [torch.randn(batch_size, num_tokens, input_size) for _ in range(num_groups)]
    mask = [torch.ones(batch_size, num_tokens, 1, dtype=torch.float32) for _ in range(num_groups)]
    embed = spectrogram_model.encoder._GroupedEmbedder(input_size, hidden_size, num_groups, 2)
    out = embed(tokens, mask)
    assert out.shape == (batch_size, num_tokens, hidden_size)


def test__grouped_embedder__group_invariance():
    """Test `spectrogram_model.encoder._GroupedEmbedder` is invariant to how many groups are
    processed together."""
    tokens_a = torch.randn(1, 4, 1)
    tokens_b = torch.randn(1, 4, 1)
    mask_a = torch.tensor([[1, 0, 1, 0]])
    mask_b = torch.tensor([[0, 1, 1, 0]])
    null_mask = torch.tensor([[0, 0, 0, 0]])
    embed = spectrogram_model.encoder._GroupedEmbedder(1, 7, 2, 2)

    result_a = embed([tokens_a, tokens_b], [mask_a, null_mask])[0]
    result_b = embed([tokens_a, tokens_b], [null_mask, mask_b])[0]
    result_combined = embed([tokens_a, tokens_b], [mask_a, mask_b])[0]
    assert_almost_equal(result_a[0], result_combined[0])
    assert_almost_equal(result_b[1], result_combined[1])
    assert_almost_equal((result_a[2] + result_b[2]) / 2, result_combined[2])
    assert_almost_equal(result_a[3], result_b[3])


def _make_encoder(
    max_tokens=10,
    max_seq_meta_vals=(11, 12),
    max_token_meta_vals=(13,),
    max_word_vector_size=8,
    max_seq_vector_size=2,
    seq_embed_size=6,
    token_meta_embed_size=12,
    anno_embed_size=3,
    seq_meta_embed_dropout=0.1,
    max_anno_vector_size=1,
    out_size=8,
    hidden_size=8,
    num_conv_layers=2,
    conv_filter_size=5,
    lstm_layers=2,
    dropout=0.5,
    batch_size=4,
    num_tokens=5,
    context=3,
    num_token_meta=1,
    max_frames_per_token=4.5,
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
        seq_embed_size=seq_embed_size,
        token_meta_embed_size=token_meta_embed_size,
        anno_embed_size=anno_embed_size,
        seq_meta_embed_dropout=seq_meta_embed_dropout,
        max_anno_vector_size=max_anno_vector_size,
        annos=[annos],
        out_size=out_size,
        hidden_size=hidden_size,
        num_conv_layers=num_conv_layers,
        conv_filter_size=conv_filter_size,
        lstm_layers=lstm_layers,
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
    return encoder, inputs, (num_tokens, batch_size, out_size)


def test_encoder():
    """Test `encoder.Encoder` handles a basic case."""
    module, arg, (num_tokens, batch_size, out_size) = _make_encoder()
    encoded = module(arg)

    assert encoded.tokens.dtype == torch.float
    assert encoded.tokens.shape == (num_tokens, batch_size, out_size)

    assert encoded.tokens_mask.dtype == torch.bool
    assert encoded.tokens_mask.shape == (batch_size, num_tokens)

    assert encoded.num_tokens.dtype == torch.long
    assert encoded.num_tokens.shape == (batch_size,)

    mask_ = ~encoded.tokens_mask.transpose(0, 1).unsqueeze(-1)
    assert encoded.tokens.masked_select(mask_).sum() == 0
    assert torch.equal(encoded.tokens_mask.sum(dim=1), encoded.num_tokens)
    assert encoded.num_tokens.tolist(), [len(t) for t in arg.tokens]

    encoded.tokens.sum().backward()


def test_encoder__filter_size():
    """Test `encoder.Encoder` handles different filter sizes."""
    for filter_size in [1, 3, 5]:
        kwargs = dict(conv_filter_size=filter_size)
        module, arg, (num_tokens, batch_size, out_size) = _make_encoder(**kwargs)
        encoded = module(arg)
        assert encoded.tokens.shape == (num_tokens, batch_size, out_size)
        assert encoded.tokens_mask.shape == (batch_size, num_tokens)
        assert encoded.num_tokens.shape == (batch_size,)
        encoded.tokens.sum().backward()


def test_encoder__padding_invariance():
    """Test `encoder.Encoder` is consistent regardless of the padding."""
    module, arg, (_, batch_size, _) = _make_encoder(dropout=0, seq_meta_embed_dropout=0)
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
