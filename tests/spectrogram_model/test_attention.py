import numpy
import pytest
import random
import torch

from src.spectrogram_model.attention import LocationSensitiveAttention


def test_location_sensative_attention_zeros():
    """ Test to ensure the attention module doesn't have any zero discontinuities. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    num_tokens = 16

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    tokens_mask = torch.rand(batch_size, num_tokens) < 0.5
    tokens_mask[0, :] = True  # NOTE: Softmax will fail unless one token is present.
    encoded_tokens = torch.zeros(num_tokens, batch_size, attention_hidden_size)
    query = torch.zeros(batch_size, query_hidden_size)
    cumulative_alignment = torch.zeros(batch_size, num_tokens)
    initial_cumulative_alignment = torch.zeros(batch_size, 1)

    context, cumulative_alignment, alignment = attention(encoded_tokens, tokens_mask, query,
                                                         cumulative_alignment,
                                                         initial_cumulative_alignment)

    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_sensative_attention_backward():
    """ Test to ensure the attention module can backprop. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    num_tokens = 16

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    encoded_tokens = torch.randn(num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(batch_size, query_hidden_size)
    cumulative_alignment = torch.randn(batch_size, num_tokens)
    initial_cumulative_alignment = torch.randn(batch_size, 1)

    context, cumulative_alignment, alignment = attention(encoded_tokens, tokens_mask, query,
                                                         cumulative_alignment,
                                                         initial_cumulative_alignment)

    (context.sum() + cumulative_alignment.sum() + alignment.sum()).backward()


def test_location_sensative_attention():
    query_hidden_size = 16
    attention_hidden_size = 8
    batch_size = 5
    num_tokens = 6
    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    tokens_mask[:, -1].fill_(0)
    encoded_tokens = torch.rand(num_tokens, batch_size, attention_hidden_size)
    query = torch.rand(batch_size, query_hidden_size)

    cumulative_alignment = None
    for j in range(3):
        context, cumulative_alignment, alignment = attention(
            encoded_tokens, tokens_mask, query, cumulative_alignment=cumulative_alignment)

        assert context.type() == 'torch.FloatTensor'
        assert context.shape == (batch_size, attention_hidden_size)

        assert alignment.type() == 'torch.FloatTensor'
        assert alignment.shape == (batch_size, num_tokens)

        assert cumulative_alignment.type() == 'torch.FloatTensor'
        assert cumulative_alignment.shape == (batch_size, num_tokens)

        # Check the mask computation was applied correctly.
        tokens_sum = alignment.sum(dim=0)
        assert tokens_sum[-1].sum() == 0  # Masked
        for i in range(num_tokens - 1):
            assert tokens_sum[i].sum() != 0  # Not Masked

        # Check the softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # Check the softmax computation was applied correctly.
        cumulative_alignment_sum = cumulative_alignment.sum(dim=1)
        for i in range(batch_size):
            assert cumulative_alignment_sum[i].item() <= j + 1


def test_location_sensative_attention__batch_invariant():
    """ Test to ensure that batch size doesn't affect the output. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    num_tokens = 16

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    encoded_tokens = torch.randn(num_tokens, batch_size, attention_hidden_size)
    query = torch.randn(1, batch_size, query_hidden_size)
    cumulative_alignment = torch.randn(batch_size, num_tokens)
    initial_cumulative_alignment = torch.randn(batch_size, 1)

    index = random.randint(0, batch_size - 1)
    slice_ = slice(index, index + 1)

    single_context, single_cumulative_alignment, single_alignment = attention(
        encoded_tokens[:, slice_], tokens_mask[slice_], query[:, slice_],
        cumulative_alignment[slice_], initial_cumulative_alignment[slice_])

    batched_context, batched_cumulative_alignment, batched_alignment = attention(
        encoded_tokens, tokens_mask, query, cumulative_alignment, initial_cumulative_alignment)

    numpy.testing.assert_almost_equal(
        batched_context.detach().numpy()[slice_], single_context.detach().numpy(), decimal=5)

    numpy.testing.assert_almost_equal(
        batched_cumulative_alignment.detach().numpy()[slice_],
        single_cumulative_alignment.detach().numpy(),
        decimal=5)

    numpy.testing.assert_almost_equal(
        batched_alignment.detach().numpy()[slice_], single_alignment.detach().numpy(), decimal=5)


def test_location_sensative_attention__padding_invariant():
    """ Test to ensure that padding doesn't affect the output. """
    query_hidden_size = 32
    attention_hidden_size = 24
    batch_size = 8
    num_tokens = 16
    num_padding = 4

    attention = LocationSensitiveAttention(
        query_hidden_size=query_hidden_size, hidden_size=attention_hidden_size)

    initial_cumulative_alignment = torch.randn(batch_size, 1)
    query = torch.randn(1, batch_size, query_hidden_size)

    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    padded_tokens_mask = torch.cat(
        [tokens_mask, torch.zeros(batch_size, num_padding, dtype=torch.bool)], dim=1)

    padded_encoded_tokens = torch.randn(num_tokens + num_padding, batch_size, attention_hidden_size)
    padded_cumulative_alignment = torch.randn(batch_size, num_tokens + num_padding)

    encoded_tokens = padded_encoded_tokens[:-num_padding]
    cumulative_alignment = padded_cumulative_alignment[:, :-num_padding]

    single_context, single_cumulative_alignment, single_alignment = attention(
        encoded_tokens, tokens_mask, query, cumulative_alignment, initial_cumulative_alignment)

    padded_context, padded_cumulative_alignment, padded_alignment = attention(
        padded_encoded_tokens, padded_tokens_mask, query, padded_cumulative_alignment,
        initial_cumulative_alignment)

    numpy.testing.assert_almost_equal(
        padded_cumulative_alignment[:, :-num_padding].detach().numpy(),
        single_cumulative_alignment.detach().numpy(),
        decimal=5)

    numpy.testing.assert_almost_equal(
        padded_alignment[:, :-num_padding].detach().numpy(),
        single_alignment.detach().numpy(),
        decimal=5)

    numpy.testing.assert_almost_equal(
        padded_context.detach().numpy(), single_context.detach().numpy(), decimal=5)
