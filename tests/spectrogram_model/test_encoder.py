import torch

from src.spectrogram_model.encoder import Encoder

encoder_params = {
    'batch_size': 4,
    'num_tokens': 5,
    'vocab_size': 10,
    'lstm_hidden_size': 64,
    'embedding_dim': 32,
    'lstm_bidirectional': True,
}


def test_encoder():
    encoder = Encoder(
        encoder_params['vocab_size'],
        lstm_hidden_size=encoder_params['lstm_hidden_size'],
        lstm_bidirectional=encoder_params['lstm_bidirectional'],
        embedding_dim=encoder_params['embedding_dim'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(encoder_params['batch_size'], encoder_params['num_tokens']).random_(
        1, encoder_params['vocab_size'])
    output = encoder(input_)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (
        encoder_params['num_tokens'], encoder_params['batch_size'],
        (encoder_params['lstm_hidden_size'] / 2) * (2
                                                    if encoder_params['lstm_bidirectional'] else 1))

    # Smoke test backward
    output.sum().backward()


def test_encoder_filter_size():
    for filter_size in [1, 3, 5]:
        encoder = Encoder(
            encoder_params['vocab_size'],
            lstm_hidden_size=encoder_params['lstm_hidden_size'],
            lstm_bidirectional=encoder_params['lstm_bidirectional'],
            embedding_dim=encoder_params['embedding_dim'],
            convolution_filter_size=filter_size)

        # NOTE: 1-index to avoid using 0 typically associated with padding
        input_ = torch.LongTensor(encoder_params['batch_size'],
                                  encoder_params['num_tokens']).random_(
                                      1, encoder_params['vocab_size'])
        output = encoder(input_)

        assert output.type() == 'torch.FloatTensor'
        assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                                (encoder_params['lstm_hidden_size'] / 2) *
                                (2 if encoder_params['lstm_bidirectional'] else 1))
