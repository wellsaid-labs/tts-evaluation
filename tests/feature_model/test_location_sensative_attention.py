import torch

from src.feature_model.location_sensative_attention import LocationSensitiveAttention


def test_location_sensative_attention_initial():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    attention = LocationSensitiveAttention(encoder_hidden_size=encoder_hidden_size)

    encoded_tokens = torch.autograd.Variable(
        torch.FloatTensor(num_tokens, batch_size, encoder_hidden_size).uniform_(0, 1))
    query = torch.autograd.Variable(
        torch.FloatTensor(batch_size, encoder_hidden_size).uniform_(0, 1))
    context, alignment = attention(encoded_tokens, query)

    assert context.data.type() == 'torch.FloatTensor'
    assert context.shape == (batch_size, encoder_hidden_size)

    assert alignment.data.type() == 'torch.FloatTensor'
    assert alignment.shape == (batch_size, num_tokens)

    context, alignment = attention(encoded_tokens, query, alignment)

    assert context.data.type() == 'torch.FloatTensor'
    assert context.shape == (batch_size, encoder_hidden_size)

    assert alignment.data.type() == 'torch.FloatTensor'
    assert alignment.shape == (batch_size, num_tokens)
