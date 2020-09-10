import pytest
import random
import torch

from src.spectrogram_model.attention import LocationSensitiveAttention
from src.spectrogram_model.attention import window
from tests._utils import assert_almost_equal


def test_window():
    assert torch.equal(
        window(torch.tensor([1, 2, 3]), start=torch.tensor(1), length=2, dim=0)[0],
        torch.tensor([2, 3]))


def test_window__large_window():
    assert torch.equal(
        window(torch.tensor([1, 2, 3]), start=torch.tensor(0), length=3, dim=0)[0],
        torch.tensor([1, 2, 3]))


def test_window__length_to_long():
    with pytest.raises(AssertionError):
        window(torch.tensor([1, 2, 3]), start=torch.tensor(0), length=4, dim=0)


def test_window__start_and_length_to_long():
    with pytest.raises(AssertionError):
        window(torch.tensor([1, 2, 3]), start=torch.tensor(1), length=3, dim=0)


def test_window__negative_start():
    with pytest.raises(AssertionError):
        window(torch.tensor([1, 2, 3]), start=torch.tensor(-1), length=3, dim=0)


def test_window__2d():
    assert torch.equal(
        window(torch.tensor([[1, 2, 3], [1, 2, 3]]), start=torch.tensor([1, 0]), length=2,
               dim=1)[0], torch.tensor([[2, 3], [1, 2]]))


def test_window__3d():
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    assert torch.equal(
        window(tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-1)[0],
        torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def test_window__transpose():
    """ Test that the `window` function works with a dimension that's not the last dimension. """
    tensor = torch.tensor([[[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]])
    tensor = tensor.transpose(-2, -1)
    assert torch.equal(
        window(tensor, start=torch.tensor([[0, 1], [2, 3]]), length=2, dim=-2)[0].transpose(-1, -2),
        torch.tensor([[[1, 2], [2, 3]], [[3, 4], [4, 5]]]))


def test_location_sensative_attention__large_window():
    """ Test a window larger than the number of tokens. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16
    window_length = 21

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        window_length=window_length).eval()

    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    num_tokens = torch.full((batch_size,), max_num_tokens, dtype=torch.long)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)
    cumulative_alignment = torch.abs(torch.randn(batch_size, max_num_tokens))

    (windowed_context, windowed_cumulative_alignment, windowed_alignment,
     windowed_window_start) = attention(encoded_tokens, tokens_mask, num_tokens, query)
    assert windowed_window_start.sum() == 0

    attention.train()
    (context, cumulative_alignment, alignment, _) = attention(encoded_tokens, tokens_mask,
                                                              num_tokens, query)

    assert_almost_equal(windowed_context, context)
    assert_almost_equal(windowed_cumulative_alignment, cumulative_alignment)
    assert_almost_equal(windowed_alignment, alignment)


def test_location_sensative_attention__large_window__padding_invariant():
    """ Test a window larger than the number of tokens of non-padded tokens. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16
    window_length = 13
    padding = 5

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        window_length=window_length).eval()

    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    tokens_mask[:, -padding:] = 0.0
    num_tokens = torch.full((batch_size,), max_num_tokens - padding, dtype=torch.long)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)
    cumulative_alignment = torch.abs(torch.randn(batch_size, max_num_tokens))

    (windowed_context, windowed_cumulative_alignment, windowed_alignment,
     windowed_window_start) = attention(encoded_tokens, tokens_mask, num_tokens, query)
    # NOTE: Ensure the window does not move past 0.0; otherwise, it'll exclude any earlier tokens
    # from the window. This is not ideal if the `window_length` is already larger than the number of
    # tokens.
    assert windowed_window_start.sum() == 0

    attention.train()
    (context, cumulative_alignment, alignment, _) = attention(encoded_tokens, tokens_mask,
                                                              num_tokens, query)

    assert_almost_equal(windowed_context, context)
    assert_almost_equal(windowed_cumulative_alignment, cumulative_alignment)
    assert_almost_equal(windowed_alignment, alignment)


def test_location_sensative_attention__window():
    query_hidden_size = 16
    attention_hidden_size = 8
    batch_size = 5
    max_num_tokens = 6
    window_length = 3
    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size,
        window_length=window_length).eval()
    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    tokens_mask[:, -1].fill_(0)
    num_tokens = torch.full((batch_size,), max_num_tokens - 1, dtype=torch.long)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)
    last_window_start = torch.zeros(batch_size, dtype=torch.long)

    cumulative_alignment = None
    for j in range(3):
        context, cumulative_alignment, alignment, window_start = attention(
            encoded_tokens,
            tokens_mask,
            num_tokens,
            query,
            cumulative_alignment=cumulative_alignment,
            window_start=last_window_start)

        assert window_start.type() == 'torch.LongTensor'
        assert window_start.shape == (batch_size,)

        assert context.type() == 'torch.FloatTensor'
        assert context.shape == (batch_size, attention_hidden_size)

        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (batch_size, max_num_tokens)

        assert cumulative_alignment.type() == 'torch.FloatTensor'
        assert cumulative_alignment.shape == (batch_size, max_num_tokens)

        # Check the mask computation was applied correctly.
        assert alignment.sum(dim=0)[-1].sum() == 0  # Masked
        for i in range(max_num_tokens - 1):
            for k in range(batch_size):
                if i >= last_window_start[k] and i < last_window_start[k] + window_length:
                    assert alignment[k, i] != 0  # Not Masked
                else:
                    assert alignment[k, i] == 0  # Masked

        # Check the softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # Check the softmax computation was applied correctly.
        cumulative_alignment_sum = cumulative_alignment.sum(dim=1)
        for i in range(batch_size):
            assert cumulative_alignment_sum[i].item() == pytest.approx(j + 1)

        last_window_start = window_start


def test_location_sensative_attention__zeros():
    """ Test to ensure the attention module doesn't have any zero discontinuities. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    tokens_mask = torch.randn(batch_size, max_num_tokens) < 0.5
    tokens_mask[0, :] = True  # NOTE: Softmax will fail unless one token is present.
    num_tokens = tokens_mask.sum(dim=1)
    encoded_tokens = torch.zeros(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.zeros(batch_size, query_hidden_size)
    cumulative_alignment = torch.zeros(batch_size, max_num_tokens)
    initial_cumulative_alignment = torch.zeros(batch_size, 1)

    context, cumulative_alignment, alignment, _ = attention(encoded_tokens, tokens_mask, num_tokens,
                                                            query, cumulative_alignment,
                                                            initial_cumulative_alignment)

    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_sensative_attention__backward():
    """ Test to ensure the attention module can backprop. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    num_tokens = tokens_mask.sum(dim=1)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)
    cumulative_alignment = torch.randn(batch_size, max_num_tokens)
    initial_cumulative_alignment = torch.randn(batch_size, 1)

    context, cumulative_alignment, alignment, _ = attention(encoded_tokens, tokens_mask, num_tokens,
                                                            query, cumulative_alignment,
                                                            initial_cumulative_alignment)

    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_sensative_attention():
    query_hidden_size = 16
    attention_hidden_size = 8
    batch_size = 5
    max_num_tokens = 6
    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)
    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    tokens_mask[:, -1].fill_(0)
    num_tokens = tokens_mask.sum(dim=1)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)

    cumulative_alignment = None
    for j in range(3):
        context, cumulative_alignment, alignment, window_start = attention(
            encoded_tokens,
            tokens_mask,
            num_tokens,
            query,
            cumulative_alignment=cumulative_alignment)

        assert window_start.type() == 'torch.LongTensor'
        assert window_start.shape == (batch_size,)

        assert context.type() == 'torch.FloatTensor'
        assert context.shape == (batch_size, attention_hidden_size)

        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (batch_size, max_num_tokens)

        assert cumulative_alignment.type() == 'torch.FloatTensor'
        assert cumulative_alignment.shape == (batch_size, max_num_tokens)

        # Check the mask computation was applied correctly.
        tokens_sum = alignment.sum(dim=0)
        assert tokens_sum[-1].sum() == 0  # Masked
        for i in range(max_num_tokens - 1):
            assert tokens_sum[i].sum() != 0  # Not Masked

        # Check the softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # Check the softmax computation was applied correctly.
        cumulative_alignment_sum = cumulative_alignment.sum(dim=1)
        for i in range(batch_size):
            assert cumulative_alignment_sum[i].item() == pytest.approx(j + 1)


def test_location_sensative_attention__batch_invariant():
    """ Test to ensure that batch size doesn't affect the output. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16
    dropout = 0

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size, dropout=dropout)

    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    num_tokens = tokens_mask.sum(dim=1)
    encoded_tokens = torch.randn(max_num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(1, batch_size, query_hidden_size)
    cumulative_alignment = torch.randn(batch_size, max_num_tokens)
    initial_cumulative_alignment = torch.randn(batch_size, 1)

    index = random.randint(0, batch_size - 1)
    slice_ = slice(index, index + 1)

    single_context, single_cumulative_alignment, single_alignment, _ = attention(
        encoded_tokens[:, slice_], tokens_mask[slice_], num_tokens[slice_], query[:, slice_],
        cumulative_alignment[slice_], initial_cumulative_alignment[slice_])

    batched_context, batched_cumulative_alignment, batched_alignment, _ = attention(
        encoded_tokens, tokens_mask, num_tokens, query, cumulative_alignment,
        initial_cumulative_alignment)

    assert_almost_equal(batched_context[slice_], single_context, decimal=5)
    assert_almost_equal(
        batched_cumulative_alignment[slice_], single_cumulative_alignment, decimal=5)
    assert_almost_equal(batched_alignment[slice_], single_alignment, decimal=5)


def test_location_sensative_attention__padding_invariant():
    """ Test to ensure that padding doesn't affect the output. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    max_num_tokens = 16
    num_padding = 4
    dropout = 0

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size, dropout=dropout)

    initial_cumulative_alignment = torch.randn(batch_size, 1)
    query = torch.randn(1, batch_size, query_hidden_size)

    tokens_mask = torch.ones(batch_size, max_num_tokens, dtype=torch.bool)
    num_tokens = tokens_mask.sum(dim=1)
    padded_tokens_mask = torch.cat(
        [tokens_mask, torch.zeros(batch_size, num_padding, dtype=torch.bool)], dim=1)

    padded_encoded_tokens = torch.randn(max_num_tokens + num_padding, batch_size,
                                        attention_hidden_size)
    padded_cumulative_alignment = torch.randn(batch_size, max_num_tokens + num_padding)

    encoded_tokens = padded_encoded_tokens[:-num_padding]
    cumulative_alignment = padded_cumulative_alignment[:, :-num_padding]

    single_context, single_cumulative_alignment, single_alignment, _ = attention(
        encoded_tokens, tokens_mask, num_tokens, query, cumulative_alignment,
        initial_cumulative_alignment)

    padded_context, padded_cumulative_alignment, padded_alignment, _ = attention(
        padded_encoded_tokens, padded_tokens_mask, num_tokens, query, padded_cumulative_alignment,
        initial_cumulative_alignment)

    assert_almost_equal(
        padded_cumulative_alignment[:, :-num_padding], single_cumulative_alignment, decimal=5)
    assert_almost_equal(padded_alignment[:, :-num_padding], single_alignment, decimal=5)
    assert_almost_equal(padded_context, single_context, decimal=5)
