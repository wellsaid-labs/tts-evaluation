import random
import typing
from functools import partial

import pytest
import torch

import lib
from run._models.spectrogram_model.attention import Attention, _window
from run._models.spectrogram_model.containers import AttentionHiddenState, Encoded
from tests import _utils


def test__window():
    """Test `_window` to window given a simple tensor."""
    window = _window(torch.tensor([1, 2, 3]), start=torch.tensor(1), length=2, dim=0)[0]
    assert torch.equal(window, torch.tensor([2, 3]))


def test__window__identity():
    """Test `_window` to compute an identity if `length` is equal
    to the dimension size."""
    window = _window(torch.tensor([1, 2, 3]), start=torch.tensor(0), length=3, dim=0)[0]
    assert torch.equal(window, torch.tensor([1, 2, 3]))


def test__window__length_to_small():
    """Test `_window` fails if `length` is too small."""
    with pytest.raises(RuntimeError):
        _window(torch.tensor([1, 2, 3]), start=torch.tensor(0), length=-1, dim=0)


def test__window__length_to_long():
    """Test `_window` fails if `length` is too long."""
    with pytest.raises(AssertionError):
        _window(torch.tensor([1, 2, 3]), start=torch.tensor(0), length=4, dim=0)


def test__window__start_to_small():
    """Test `_window` fails if `start` is out of range."""
    with pytest.raises(AssertionError):
        _window(torch.tensor([1, 2, 3]), start=torch.tensor(-1), length=3, dim=0)


def test__window__start_to_large():
    """Test `_window` fails if `start` is out of range."""
    with pytest.raises(AssertionError):
        _window(torch.tensor([1, 2, 3]), start=torch.tensor(4), length=1, dim=0)


def test__window__window_out_of_range():
    """Test `_window` fails if the window is out of range."""
    with pytest.raises(AssertionError):
        _window(torch.tensor([1, 2, 3]), start=torch.tensor(1), length=3, dim=0)


def test__window__2d():
    """Test `_window` to window given a 2d `tensor` with variable
    `start`."""
    window = _window(
        torch.tensor([[1, 2, 3], [1, 2, 3]]),
        start=torch.tensor([1, 0]),
        length=2,
        dim=1,
    )[0]
    assert torch.equal(window, torch.tensor([[2, 3], [1, 2]]))


def test__window__3d():
    """Test `_window` to window given a 3d `tensor` and 2d `start`.
    Furthermore, this tests a negative `dim`."""
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    window = _window(tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-1)[0]
    assert torch.equal(window, torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def test__window__transpose_invariance():
    """Test `_window` to window given a transposed 3d `tensor`.
    `_window` should return consistent results regardless of the
    dimension and ordering of the data."""
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    tensor = tensor.transpose(-2, -1)
    window = _window(tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-2)[0]
    assert torch.equal(window.transpose(-1, -2), torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def _make_attention(
    query_hidden_size=16,
    attention_hidden_size=8,
    batch_size=3,
    max_num_tokens=12,
    conv_filter_size=5,
    window_length=7,
    avg_frames_per_token=1.0,
) -> typing.Tuple[
    Attention, typing.Tuple[Encoded, torch.Tensor, AttentionHiddenState], typing.Tuple[int, int]
]:
    """Make `attention.Attention` and it's inputs for testing."""
    module = Attention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        conv_filter_size=conv_filter_size,
        window_length=window_length,
        avg_frames_per_token=avg_frames_per_token,
    )
    tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    query = torch.randn(1, batch_size, query_hidden_size)
    padding = (module.cum_alignment_padding, module.cum_alignment_padding)
    cum_alignment = torch.zeros(batch_size, max_num_tokens)
    hidden_state = AttentionHiddenState(
        cum_alignment=lib.utils.pad_tensor(cum_alignment, padding, 1, value=1.0),
        window_start=torch.zeros(batch_size, dtype=torch.long),
    )
    encoded = Encoded(tokens, tokens_mask, tokens_mask.sum(dim=1))
    return module, (encoded, query, hidden_state), (batch_size, max_num_tokens)


def _add_padding(
    amount: int, encoded: Encoded, hidden_state: AttentionHiddenState
) -> typing.Tuple[Encoded, AttentionHiddenState]:
    """Add zero padding to `tokens`, `tokens_mask` and `hidden_state`."""
    tokens_padding = torch.randn(amount, *encoded.tokens.shape[1:])
    padded_tokens = torch.cat([encoded.tokens, tokens_padding], dim=0)
    tokens_mask_padding = torch.zeros(encoded.tokens_mask.shape[0], amount, dtype=torch.bool)
    padded_tokens_mask = torch.cat([encoded.tokens_mask, tokens_mask_padding], dim=1)
    alignment_padding = torch.randn(hidden_state.cum_alignment.shape[0], amount)
    cum_alignment = torch.cat([hidden_state.cum_alignment, alignment_padding], 1)
    padded_hidden_state = hidden_state._replace(cum_alignment=cum_alignment)
    padded_encoded = encoded._replace(tokens=padded_tokens, tokens_mask=padded_tokens_mask)
    return padded_encoded, padded_hidden_state


assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def test_attention():
    """Test `attention.Attention` handles a basic case."""
    module, (encoded, query, hidden_state), (batch_size, max_num_tokens) = _make_attention()
    encoded.tokens_mask[:, -1].fill_(0)
    last_hidden_state = hidden_state
    context = torch.empty(0)
    alignment = torch.empty(0)
    for j in range(3):
        context, alignment, hidden_state = module(encoded, query, last_hidden_state)

        assert context.dtype == torch.float
        assert context.shape == (batch_size, module.hidden_size)
        assert alignment.dtype == torch.float
        assert alignment.shape == (batch_size, max_num_tokens)
        assert hidden_state.cum_alignment.dtype == torch.float
        assert hidden_state.cum_alignment.shape == (
            batch_size,
            max_num_tokens + 2 * module.cum_alignment_padding,
        )
        assert hidden_state.window_start.dtype == torch.long
        assert hidden_state.window_start.shape == (batch_size,)

        # NOTE: Check the mask computation was applied correctly.
        assert alignment.sum(dim=0)[-1].sum() == 0  # Masked
        for i in range(max_num_tokens - 1):
            for k in range(batch_size):
                if (
                    i >= last_hidden_state.window_start[k]
                    and i < last_hidden_state.window_start[k] + module.window_length
                ):
                    assert alignment[k, i] != 0  # Not Masked
                else:
                    assert alignment[k, i] == 0  # Masked

        # NOTE: Check the softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # NOTE: Check the softmax computation was applied correctly.
        padding = module.cum_alignment_padding
        alignment_sum = hidden_state.cum_alignment[:, padding:-padding].sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(j + 1)

        last_hidden_state = hidden_state

    (context.sum() + hidden_state.cum_alignment.sum() + alignment.sum()).backward()


def test_attention__batch_invariance():
    """Test `attention.Attention` is consistent regardless of the batch size."""
    module, (encoded, query, hidden_state), (batch_size, _) = _make_attention()

    index = random.randint(0, batch_size - 1)
    slice_ = slice(index, index + 1)

    args = (
        encoded._replace(
            tokens=encoded.tokens[:, slice_],
            tokens_mask=encoded.tokens_mask[slice_],
            num_tokens=encoded.num_tokens[slice_],
        ),
        query[:, slice_],
        hidden_state._replace(
            cum_alignment=hidden_state.cum_alignment[slice_],
            window_start=hidden_state.window_start[slice_],
        ),
    )
    context, alignment, new_hidden_state = module(*args)
    batch_context, batch_alignment, batch_new_hidden_state = module(encoded, query, hidden_state)

    assert_almost_equal(batch_context[slice_], context)
    assert_almost_equal(batch_alignment[slice_], alignment)
    assert_almost_equal(
        batch_new_hidden_state.cum_alignment[slice_],
        new_hidden_state.cum_alignment,
    )
    assert_almost_equal(batch_new_hidden_state.window_start[slice_], new_hidden_state.window_start)


def test_attention__padding_invariance():
    """Test `attention.Attention` is consistent regardless of the padding."""
    module, (encoded, query, hidden_state), _ = _make_attention()
    num_padding = 4
    padded_encoded, padded_hidden_state = _add_padding(num_padding, encoded, hidden_state)

    context, alignment, hidden_state = module(encoded, query, hidden_state)
    padded_args = (padded_encoded, query, padded_hidden_state)
    padded_context, padded_alignment, padded_hidden_state = module(*padded_args)

    assert_almost_equal(padded_context, context)
    assert_almost_equal(padded_alignment[:, :-num_padding], alignment)
    assert_almost_equal(
        padded_hidden_state.cum_alignment[:, :-num_padding],
        hidden_state.cum_alignment,
    )
    assert_almost_equal(padded_hidden_state.window_start, hidden_state.window_start)


def test_attention__zero():
    """Test `attention.Attention` doesn't have a discontinuity at zero."""
    module, (encoded, query, hidden_state), (batch_size, max_num_tokens) = _make_attention()
    tokens_mask = torch.randn(batch_size, max_num_tokens) < 0.5
    tokens_mask[:, 0] = True  # NOTE: Softmax will fail unless one token is present.
    encoded.tokens.zero_()
    query.zero_()
    context, alignment, hidden_state = module(encoded, query, hidden_state)
    (context.sum() + hidden_state.cum_alignment.sum() + alignment.sum()).backward()


def test_attention__window_invariance():
    """Test `attention.Attention` is consistent regardless of the window size, if
    the window size is larger than the number of tokens."""
    max_num_tokens = 6
    num_padding = 5
    window_length = max_num_tokens + num_padding // 2 + 1
    kwargs = dict(window_length=window_length, max_num_tokens=max_num_tokens)
    module, (encoded, query, hidden_state), _ = _make_attention(**kwargs)
    encoded, hidden_state = _add_padding(num_padding, encoded, hidden_state)

    context, alignment, hidden_state_ = module(encoded, query, hidden_state)
    module.window_length = max_num_tokens + num_padding + 3
    other_context, other_alignment, other_hidden_state = module(encoded, query, hidden_state)

    # NOTE: If `window_length` is larger than `num_tokens`, then `window_start` shouldn't move.
    assert hidden_state_.window_start.sum() == 0
    assert other_hidden_state.window_start.sum() == 0
    assert_almost_equal(other_context, context)
    assert_almost_equal(other_alignment, alignment)
    assert_almost_equal(other_hidden_state.cum_alignment, hidden_state_.cum_alignment)
    assert_almost_equal(other_hidden_state.window_start, hidden_state_.window_start)
