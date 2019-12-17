import numpy
import torch

from torchnlp.random import fork_rng

from src.spectrogram_model.encoder import Encoder
from src.spectrogram_model.encoder import MaskedBackwardLSTM


def test_masked_backward_lstm_uneven_mask():
    input_size = 1
    lstm_hidden_size = 5
    num_layers = 2

    transform = lambda t: t.transpose(0, 1).unsqueeze(2)

    tokens = transform(torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])).float()
    tokens_mask = transform(torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])).bool()

    with fork_rng(123):
        backward_lstm = MaskedBackwardLSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)

    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)

    one = transform(torch.tensor([[1, 2, 3]])).float()
    two = transform(torch.tensor([[1, 2]])).float()
    expected_one = lstm(one.flip(0))[0].flip(0)
    expected_two = lstm(two.flip(0))[0].flip(0)

    result = backward_lstm(tokens, tokens_mask)

    numpy.testing.assert_almost_equal((expected_one.sum() + expected_two.sum()).detach().numpy(),
                                      result.sum().detach().numpy())


def test_masked_backward_lstm():
    batch_size = 2
    seq_len = 3
    input_size = 4
    lstm_hidden_size = 5
    padding_len = 2
    tokens = torch.randn(seq_len, batch_size, input_size)
    padding = torch.zeros(padding_len, batch_size, input_size)
    padded_tokens = torch.cat([tokens, padding], dim=0)
    tokens_mask = torch.cat(
        [torch.ones(seq_len, batch_size, 1),
         torch.zeros(padding_len, batch_size, 1)], dim=0).bool()

    with fork_rng(123):
        backward_lstm = MaskedBackwardLSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)

    with fork_rng(123):
        lstm = torch.nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)

    expected = lstm(tokens.flip(0))[0].flip(0)
    result = backward_lstm(padded_tokens, tokens_mask)
    assert torch.equal(expected, result[:-padding_len])
    assert result[-padding_len:].sum().item() == 0
    numpy.testing.assert_almost_equal(
        expected.sum().detach().numpy(), result.sum().detach().numpy(), decimal=4)
    assert result.sum().item() != 0


def test_masked_backward_lstm_backwards():
    """ Test if `MaskedBackwardLSTM` can compute a gradient. """
    batch_size = 2
    seq_len = 3
    input_size = 4
    lstm_hidden_size = 5
    tokens = torch.randn(seq_len, batch_size, input_size)
    tokens_mask = torch.ones(seq_len, batch_size, 1).bool()

    backward_lstm = MaskedBackwardLSTM(
        input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)

    backward_lstm(tokens, tokens_mask).sum().backward()


encoder_params = {
    'num_speakers': 2,
    'batch_size': 4,
    'num_tokens': 5,
    'vocab_size': 10,
    'lstm_hidden_size': 64,
    'token_embedding_dim': 32,
    'speaker_embedding_dim': 16,
    'out_dim': 8,
}


def test_encoder():
    encoder = Encoder(
        encoder_params['vocab_size'],
        encoder_params['num_speakers'],
        out_dim=encoder_params['out_dim'],
        lstm_hidden_size=encoder_params['lstm_hidden_size'],
        speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
        token_embedding_dim=encoder_params['token_embedding_dim'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(encoder_params['batch_size'],
                              encoder_params['num_tokens']).random_(1, encoder_params['vocab_size'])
    speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
    tokens_mask = torch.full((encoder_params['batch_size'], encoder_params['num_tokens']), 1).bool()
    output = encoder(input_, tokens_mask, speaker)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                            encoder_params['out_dim'])

    # Smoke test backward
    output.sum().backward()


def test_encoder_one_speaker():
    encoder = Encoder(
        encoder_params['vocab_size'],
        1,
        out_dim=encoder_params['out_dim'],
        lstm_hidden_size=encoder_params['lstm_hidden_size'],
        speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
        token_embedding_dim=encoder_params['token_embedding_dim'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(encoder_params['batch_size'],
                              encoder_params['num_tokens']).random_(1, encoder_params['vocab_size'])
    speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
    tokens_mask = torch.full((encoder_params['batch_size'], encoder_params['num_tokens']), 1).bool()
    output = encoder(input_, tokens_mask, speaker)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                            encoder_params['out_dim'])

    # Smoke test backward
    output.sum().backward()


def test_encoder_filter_size():
    for filter_size in [1, 3, 5]:
        encoder = Encoder(
            encoder_params['vocab_size'],
            encoder_params['num_speakers'],
            out_dim=encoder_params['out_dim'],
            lstm_hidden_size=encoder_params['lstm_hidden_size'],
            speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
            token_embedding_dim=encoder_params['token_embedding_dim'],
            convolution_filter_size=filter_size)

        # NOTE: 1-index to avoid using 0 typically associated with padding
        input_ = torch.LongTensor(encoder_params['batch_size'],
                                  encoder_params['num_tokens']).random_(
                                      1, encoder_params['vocab_size'])
        speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
        tokens_mask = torch.full((encoder_params['batch_size'], encoder_params['num_tokens']),
                                 1).bool()
        output = encoder(input_, tokens_mask, speaker)

        assert output.type() == 'torch.FloatTensor'
        assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                                encoder_params['out_dim'])
