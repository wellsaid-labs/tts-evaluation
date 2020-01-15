import torch
import pytest

from src.spectrogram_model.attention import LocationSensitiveAttention


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

        # Check the Softmax computation was applied correctly.
        alignment_sum = alignment.sum(dim=1)
        for i in range(batch_size):
            assert alignment_sum[i].item() == pytest.approx(1, 0.0001)

        # Check the Softmax computation was applied correctly.
        cumulative_alignment_sum = cumulative_alignment.sum(dim=1)
        for i in range(batch_size):
            assert cumulative_alignment_sum[i].item() == pytest.approx(j + 1, 0.0001)
