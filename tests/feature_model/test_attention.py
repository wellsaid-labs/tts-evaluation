import torch
import pytest

from src.feature_model.attention import LocationSensitiveAttention


def test_location_sensative_attention():
    encoder_hidden_size = 8
    query_hidden_size = 16
    attention_hidden_size = 8
    batch_size = 5
    num_tokens = 6
    attention = LocationSensitiveAttention(
        encoder_hidden_size=encoder_hidden_size,
        query_hidden_size=query_hidden_size,
        hidden_size=attention_hidden_size)
    encoded_tokens = torch.autograd.Variable(
        torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1))
    query = torch.autograd.Variable(torch.FloatTensor(batch_size, query_hidden_size).uniform_(0, 1))

    alignment = None
    for _ in range(3):
        context, alignment = attention(encoded_tokens, query, last_alignment=alignment)

        assert context.data.type() == 'torch.FloatTensor'
        assert context.shape == (batch_size, attention_hidden_size)

        assert alignment.data.type() == 'torch.FloatTensor'
        assert alignment.shape == (batch_size, num_tokens)

        # Check the Softmax computation was applied correctly.
        sum_ = alignment.sum(dim=1)
        for i in range(batch_size):
            assert sum_[i].item() == pytest.approx(1, 0.0001)
