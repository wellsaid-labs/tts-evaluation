import random
import typing
from functools import partial

import pytest
import torch

import lib
import lib.spectrogram_model.attention
from lib.spectrogram_model.attention import Attention, AttentionHiddenState, _window
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
    convolution_filter_size=5,
    dropout=0.5,
    window_length=7,
    avg_frames_per_token=1.0,
) -> typing.Tuple[
    Attention,
    typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionHiddenState],
    typing.Tuple[int, int],
]:
    """Make `attention.Attention` and it's inputs for testing."""
    module = Attention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        convolution_filter_size=convolution_filter_size,
        dropout=dropout,
        window_length=window_length,
        avg_frames_per_token=avg_frames_per_token,
    )
    tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    query = torch.randn(1, batch_size, query_hidden_size)
    padding = (module.cumulative_alignment_padding, module.cumulative_alignment_padding)
    cumulative_alignment = torch.zeros(batch_size, max_num_tokens)
    hidden_state = AttentionHiddenState(
        cumulative_alignment=lib.utils.pad_tensor(cumulative_alignment, padding, 1, value=1.0),
        window_start=torch.zeros(batch_size, dtype=torch.long),
    )
    return (
        module,
        (tokens, tokens_mask, query, hidden_state),
        (batch_size, max_num_tokens),
    )


def _make_num_tokens(tokens_mask):
    """Create `num_tokens` input for `attention.Attention`."""
    return tokens_mask.sum(dim=1)


def _add_padding(
    amount: int,
    tokens: torch.Tensor,
    tokens_mask: torch.Tensor,
    hidden_state: AttentionHiddenState,
) -> typing.Tuple[torch.Tensor, torch.Tensor, AttentionHiddenState]:
    """Add zero padding to `tokens`, `tokens_mask` and `hidden_state`."""
    tokens_padding = torch.randn(amount, tokens.shape[1], tokens.shape[2])
    padded_tokens = torch.cat([tokens, tokens_padding], dim=0)
    tokens_mask_padding = torch.zeros(tokens_mask.shape[0], amount, dtype=torch.bool)
    padded_tokens_mask = torch.cat([tokens_mask, tokens_mask_padding], dim=1)
    alignment_padding = torch.randn(hidden_state.cumulative_alignment.shape[0], amount)
    padded_hidden_state = hidden_state._replace(
        cumulative_alignment=torch.cat([hidden_state.cumulative_alignment, alignment_padding], 1)
    )
    return padded_tokens, padded_tokens_mask, padded_hidden_state


assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def test_location_relative_attention():
    """Test `attention.Attention` handles a basic case."""
    (
        module,
        (tokens, tokens_mask, query, hidden_state),
        (batch_size, max_num_tokens),
    ) = _make_attention()
    tokens_mask[:, -1].fill_(0)
    num_tokens = _make_num_tokens(tokens_mask)
    last_hidden_state = hidden_state
    context = torch.empty(0)
    alignment = torch.empty(0)
    for j in range(3):
        context, alignment, hidden_state = module(
            tokens, tokens_mask, num_tokens, query, last_hidden_state
        )

        assert context.dtype == torch.float
        assert context.shape == (batch_size, module.hidden_size)
        assert alignment.dtype == torch.float
        assert alignment.shape == (batch_size, max_num_tokens)
        assert hidden_state.cumulative_alignment.dtype == torch.float
        assert hidden_state.cumulative_alignment.shape == (
            batch_size,
            max_num_tokens + 2 * module.cumulative_alignment_padding,
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
        padding = module.cumulative_alignment_padding
        alignment_sum = hidden_state.cumulative_alignment[:, padding:-padding].sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(j + 1)

        last_hidden_state = hidden_state

    (context.sum() + hidden_state.cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_relative_attention__batch_invariance():
    """Test `attention.Attention` is consistent regardless of the batch size."""
    (
        module,
        (tokens, tokens_mask, query, hidden_state),
        (batch_size, _),
    ) = _make_attention(dropout=0)
    num_tokens = _make_num_tokens(tokens_mask)

    index = random.randint(0, batch_size - 1)
    slice_ = slice(index, index + 1)

    args = (
        tokens[:, slice_],
        tokens_mask[slice_],
        num_tokens[slice_],
        query[:, slice_],
        hidden_state._replace(
            cumulative_alignment=hidden_state.cumulative_alignment[slice_],
            window_start=hidden_state.window_start[slice_],
        ),
    )
    context, alignment, new_hidden_state = module(*args)
    batch_context, batch_alignment, batch_new_hidden_state = module(
        tokens, tokens_mask, num_tokens, query, hidden_state
    )

    assert_almost_equal(batch_context[slice_], context)
    assert_almost_equal(batch_alignment[slice_], alignment)
    assert_almost_equal(
        batch_new_hidden_state.cumulative_alignment[slice_],
        new_hidden_state.cumulative_alignment,
    )
    assert_almost_equal(batch_new_hidden_state.window_start[slice_], new_hidden_state.window_start)


def test_location_relative_attention__padding_invariance():
    """Test `attention.Attention` is consistent regardless of the padding."""
    (module, (tokens, tokens_mask, query, hidden_state), _) = _make_attention(dropout=0)
    num_tokens = _make_num_tokens(tokens_mask)
    num_padding = 4
    padded_tokens, padded_tokens_mask, padded_hidden_state = _add_padding(
        num_padding, tokens, tokens_mask, hidden_state
    )

    args = (tokens, tokens_mask, num_tokens, query, hidden_state)
    context, alignment, hidden_state = module(*args)
    padded_args = (
        padded_tokens,
        padded_tokens_mask,
        num_tokens,
        query,
        padded_hidden_state,
    )
    padded_context, padded_alignment, padded_hidden_state = module(*padded_args)

    assert_almost_equal(padded_context, context)
    assert_almost_equal(padded_alignment[:, :-num_padding], alignment)
    assert_almost_equal(
        padded_hidden_state.cumulative_alignment[:, :-num_padding],
        hidden_state.cumulative_alignment,
    )
    assert_almost_equal(padded_hidden_state.window_start, hidden_state.window_start)


def test_location_relative_attention__zero():
    """Test `attention.Attention` doesn't have a discontinuity at zero."""
    (module, (tokens, _, query, hidden_state), (batch_size, max_num_tokens)) = _make_attention()
    tokens_mask = torch.randn(batch_size, max_num_tokens) < 0.5
    tokens_mask[:, 0] = True  # NOTE: Softmax will fail unless one token is present.
    num_tokens = _make_num_tokens(tokens_mask)
    tokens.zero_()
    query.zero_()
    context, alignment, hidden_state = module(tokens, tokens_mask, num_tokens, query, hidden_state)
    (context.sum() + hidden_state.cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_relative_attention__window_invariance():
    """Test `attention.Attention` is consistent regardless of the window size, if
    the window size is larger than the number of tokens."""
    max_num_tokens = 6
    num_padding = 5
    module, (tokens, tokens_mask, query, hidden_state), _ = _make_attention(
        window_length=max_num_tokens + num_padding // 2,
        max_num_tokens=max_num_tokens,
        dropout=0,
    )
    tokens, tokens_mask, hidden_state = _add_padding(num_padding, tokens, tokens_mask, hidden_state)
    num_tokens = _make_num_tokens(tokens_mask)
    args = (tokens, tokens_mask, num_tokens, query, hidden_state)

    context, alignment, hidden_state = module(*args)
    module.window_length = max_num_tokens + num_padding + 3
    other_context, other_alignment, other_hidden_state = module(*args)

    # NOTE: If `window_length` is larger than `num_tokens`, then `window_start` shouldn't move.
    assert hidden_state.window_start.sum() == 0
    assert other_hidden_state.window_start.sum() == 0
    assert_almost_equal(other_context, context)
    assert_almost_equal(other_alignment, alignment)
    assert_almost_equal(other_hidden_state.cumulative_alignment, hidden_state.cumulative_alignment)
    assert_almost_equal(other_hidden_state.window_start, hidden_state.window_start)
