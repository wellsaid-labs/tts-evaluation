from functools import partial

import pytest
import random
import torch
import typing

from tests import _utils

import lib.spectrogram_model.attention


def test_window():
    """ Test `lib.spectrogram_model.attention._window` to window given a simple tensor. """
    window = lib.spectrogram_model.attention._window(
        torch.tensor([1, 2, 3]), start=torch.tensor(1), length=2, dim=0)[0]
    assert torch.equal(window, torch.tensor([2, 3]))


def test_window__identity():
    """ Test `lib.spectrogram_model.attention._window` to compute an identity if `length` is equal
    to the dimension size. """
    window = lib.spectrogram_model.attention._window(
        torch.tensor([1, 2, 3]), start=torch.tensor(0), length=3, dim=0)[0]
    assert torch.equal(window, torch.tensor([1, 2, 3]))


def test_window__length_to_small():
    """ Test `lib.spectrogram_model.attention._window` fails if `length` is too small. """
    with pytest.raises(RuntimeError):
        lib.spectrogram_model.attention._window(
            torch.tensor([1, 2, 3]), start=torch.tensor(0), length=-1, dim=0)


def test_window__length_to_long():
    """ Test `lib.spectrogram_model.attention._window` fails if `length` is too long. """
    with pytest.raises(AssertionError):
        lib.spectrogram_model.attention._window(
            torch.tensor([1, 2, 3]), start=torch.tensor(0), length=4, dim=0)


def test_window__start_to_small():
    """ Test `lib.spectrogram_model.attention._window` fails if `start` is out of range.  """
    with pytest.raises(AssertionError):
        lib.spectrogram_model.attention._window(
            torch.tensor([1, 2, 3]), start=torch.tensor(-1), length=3, dim=0)


def test_window__start_to_large():
    """ Test `lib.spectrogram_model.attention._window` fails if `start` is out of range.  """
    with pytest.raises(AssertionError):
        lib.spectrogram_model.attention._window(
            torch.tensor([1, 2, 3]), start=torch.tensor(4), length=1, dim=0)


def test_window__window_out_of_range():
    """ Test `lib.spectrogram_model.attention._window` fails if the window is out of range.  """
    with pytest.raises(AssertionError):
        lib.spectrogram_model.attention._window(
            torch.tensor([1, 2, 3]), start=torch.tensor(1), length=3, dim=0)


def test_window__2d():
    """ Test `lib.spectrogram_model.attention._window` to window given a 2d `tensor` with variable
    `start`. """
    window = lib.spectrogram_model.attention._window(
        torch.tensor([[1, 2, 3], [1, 2, 3]]), start=torch.tensor([1, 0]), length=2, dim=1)[0]
    assert torch.equal(window, torch.tensor([[2, 3], [1, 2]]))


def test_window__3d():
    """ Test `lib.spectrogram_model.attention._window` to window given a 3d `tensor` and 2d `start`.
    Furthermore, this tests a negative `dim`. """
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    window = lib.spectrogram_model.attention._window(
        tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-1)[0]
    assert torch.equal(window, torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def test_window__transpose_invariance():
    """ Test `lib.spectrogram_model.attention._window` to window given a transposed 3d `tensor`.
    `lib.spectrogram_model.attention._window` should return consistent results regardless of the
    dimension and ordering of the data. """
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    tensor = tensor.transpose(-2, -1)
    window = lib.spectrogram_model.attention._window(
        tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-2)[0]
    assert torch.equal(window.transpose(-1, -2), torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def _make_attention(
    query_hidden_size=16,
    attention_hidden_size=8,
    batch_size=3,
    max_num_tokens=12,
    convolution_filter_size=5,
    dropout=0.5,
    window_length=7
) -> typing.Tuple[lib.spectrogram_model.attention.LocationRelativeAttention, typing.Tuple[
        torch.Tensor, ...], typing.Tuple[int, int]]:
    """ Make `attention.LocationRelativeAttention` and it's inputs for testing."""
    module = lib.spectrogram_model.attention.LocationRelativeAttention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        convolution_filter_size=convolution_filter_size,
        dropout=dropout,
        window_length=window_length)
    tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    query = torch.randn(1, batch_size, query_hidden_size)
    cumulative_alignment = torch.abs(torch.randn(batch_size, max_num_tokens))
    initial_cumulative_alignment = torch.randn(batch_size, 1)
    window_start = torch.randint(0, max(max_num_tokens - module.window_length, 1), (batch_size,))
    return module, (tokens, tokens_mask, query, cumulative_alignment, initial_cumulative_alignment,
                    window_start), (batch_size, max_num_tokens)


def _make_num_tokens(tokens_mask):
    """ Create `num_tokens` input for `attention.LocationRelativeAttention`. """
    return tokens_mask.sum(dim=1)


def _add_padding(amount, tokens, tokens_mask, cumulative_alignment):
    """ Add zero padding to `tokens`, `tokens_mask` and `cumulative_alignment`. """
    tokens_padding = torch.randn(amount, tokens.shape[1], tokens.shape[2])
    padded_tokens = torch.cat([tokens, tokens_padding], dim=0)
    tokens_mask_padding = torch.zeros(tokens_mask.shape[0], amount, dtype=torch.bool)
    padded_tokens_mask = torch.cat([tokens_mask, tokens_mask_padding], dim=1)
    alignment_padding = torch.randn(cumulative_alignment.shape[0], amount)
    padded_cumulative_alignment = torch.cat([cumulative_alignment, alignment_padding], dim=1)
    return padded_tokens, padded_tokens_mask, padded_cumulative_alignment


assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def test_location_relative_attention():
    """ Test `lib.spectrogram_model.attention.LocationRelativeAttention` handles a basic case. """
    module, (tokens, tokens_mask, query, *_), (batch_size, max_num_tokens) = _make_attention()
    tokens_mask[:, -1].fill_(0)
    num_tokens = _make_num_tokens(tokens_mask)
    cumulative_alignment = None
    last_window_start = torch.zeros(batch_size, dtype=torch.long)
    for j in range(3):
        context, cumulative_alignment, alignment, window_start = module(
            tokens,
            tokens_mask,
            num_tokens,
            query,
            cumulative_alignment,
            window_start=last_window_start)

        assert context.type() == 'torch.FloatTensor'
        assert context.shape == (batch_size, module.hidden_size)
        assert cumulative_alignment.type() == 'torch.FloatTensor'
        assert cumulative_alignment.shape == (batch_size, max_num_tokens)
        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (batch_size, max_num_tokens)
        assert window_start.type() == 'torch.LongTensor'
        assert window_start.shape == (batch_size,)

        # NOTE: Check the mask computation was applied correctly.
        assert alignment.sum(dim=0)[-1].sum() == 0  # Masked
        for i in range(max_num_tokens - 1):
            for k in range(batch_size):
                if i >= last_window_start[k] and i < last_window_start[k] + module.window_length:
                    assert alignment[k, i] != 0  # Not Masked
                else:
                    assert alignment[k, i] == 0  # Masked

        # NOTE: Check the softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # NOTE: Check the softmax computation was applied correctly.
        cumulative_alignment_sum = cumulative_alignment.sum(dim=1)
        for i in range(batch_size):
            assert cumulative_alignment_sum[i].item() == pytest.approx(j + 1)

        last_window_start = window_start

    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_relative_attention__batch_invariance():
    """ Test `attention.LocationRelativeAttention` is consistent regardless of the batch size. """
    module, (tokens, tokens_mask, query, *other), (batch_size, _) = _make_attention(dropout=0)
    num_tokens = _make_num_tokens(tokens_mask)

    index = random.randint(0, batch_size - 1)
    slice_ = slice(index, index + 1)

    args = (tokens[:, slice_], tokens_mask[slice_], num_tokens[slice_], query[:, slice_],
            *(t[slice_] for t in other))
    context, cumulative_alignment, alignment, window_start = module(*args)
    batch_context, batch_cumulative_alignment, batch_alignment, batch_window_start = module(
        tokens, tokens_mask, num_tokens, query, *other)

    assert_almost_equal(batch_context[slice_], context)
    assert_almost_equal(batch_cumulative_alignment[slice_], cumulative_alignment)
    assert_almost_equal(batch_alignment[slice_], alignment)
    assert_almost_equal(batch_window_start[slice_], window_start)


def test_location_relative_attention__padding_invariance():
    """ Test `attention.LocationRelativeAttention` is consistent regardless of the padding. """
    module, (tokens, tokens_mask, query, cumulative_alignment,
             *other), (batch_size, _) = _make_attention(dropout=0)
    num_tokens = _make_num_tokens(tokens_mask)
    num_padding = 4
    padded_tokens, padded_tokens_mask, padded_cumulative_alignment = _add_padding(
        num_padding, tokens, tokens_mask, cumulative_alignment)

    args = (tokens, tokens_mask, num_tokens, query, cumulative_alignment, *other)
    context, cumulative_alignment, alignment, window_start = module(*args)
    padded_context, padded_cumulative_alignment, padded_alignment, padded_window_start = module(
        padded_tokens, padded_tokens_mask, num_tokens, query, padded_cumulative_alignment, *other)

    assert_almost_equal(padded_context, context)
    assert_almost_equal(padded_cumulative_alignment[:, :-num_padding], cumulative_alignment)
    assert_almost_equal(padded_alignment[:, :-num_padding], alignment)
    assert_almost_equal(window_start, padded_window_start)


def test_location_relative_attention__zero():
    """ Test `attention.LocationRelativeAttention` doesn't have a discontinuity at zero. """
    module, (tokens, _, query, *_), (batch_size, max_num_tokens) = _make_attention()
    tokens_mask = torch.randn(batch_size, max_num_tokens) < 0.5
    tokens_mask[:, 0] = True  # NOTE: Softmax will fail unless one token is present.
    num_tokens = _make_num_tokens(tokens_mask)
    tokens.zero_()
    query.zero_()
    context, cumulative_alignment, alignment, _ = module(tokens, tokens_mask, num_tokens, query)
    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_relative_attention__window_invariance():
    """ Test `attention.LocationRelativeAttention` is consistent regardless of the window size, if
    the window size is larger than the number of tokens. """
    max_num_tokens = 6
    num_padding = 5
    module, (tokens, tokens_mask, query, cumulative_alignment, *other), _ = _make_attention(
        window_length=max_num_tokens + num_padding // 2, max_num_tokens=max_num_tokens, dropout=0)
    tokens, tokens_mask, cumulative_alignment = _add_padding(num_padding, tokens, tokens_mask,
                                                             cumulative_alignment)
    num_tokens = _make_num_tokens(tokens_mask)
    args = (tokens, tokens_mask, num_tokens, query, cumulative_alignment, *other)

    context, cumulative_alignment, alignment, window_start = module(*args)
    module.window_length = max_num_tokens + num_padding + 3
    other_context, other_cumulative_alignment, other_alignment, other_window_start = module(*args)

    # NOTE: If `window_length` is larger than `num_tokens`, then `window_start` shouldn't move.
    assert window_start.sum() == 0
    assert other_window_start.sum() == 0
    assert_almost_equal(other_context, context)
    assert_almost_equal(other_cumulative_alignment, cumulative_alignment)
    assert_almost_equal(other_alignment, alignment)
    assert_almost_equal(other_window_start, window_start)
