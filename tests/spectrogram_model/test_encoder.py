import torch

from torchnlp.text_encoders.reserved_tokens import RESERVED_ITOS

from src.spectrogram_model.encoder import Encoder

encoder_params = {
    'num_speakers': 2 + len(RESERVED_ITOS),
    'batch_size': 4,
    'num_tokens': 5,
    'vocab_size': 10,
    'lstm_hidden_size': 64,
    'token_embedding_dim': 32,
    'speaker_embedding_dim': 16,
    'lstm_bidirectional': True,
    'out_dim': 8,
}


def test_encoder():
    encoder = Encoder(
        encoder_params['vocab_size'],
        encoder_params['num_speakers'],
        out_dim=encoder_params['out_dim'],
        lstm_hidden_size=encoder_params['lstm_hidden_size'],
        lstm_bidirectional=encoder_params['lstm_bidirectional'],
        speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
        token_embedding_dim=encoder_params['token_embedding_dim'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(encoder_params['batch_size'], encoder_params['num_tokens']).random_(
        1, encoder_params['vocab_size'])
    speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
    output = encoder(input_, speaker)

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
        lstm_bidirectional=encoder_params['lstm_bidirectional'],
        speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
        token_embedding_dim=encoder_params['token_embedding_dim'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(encoder_params['batch_size'], encoder_params['num_tokens']).random_(
        1, encoder_params['vocab_size'])
    speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
    output = encoder(input_, speaker)

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
            lstm_bidirectional=encoder_params['lstm_bidirectional'],
            speaker_embedding_dim=encoder_params['speaker_embedding_dim'],
            token_embedding_dim=encoder_params['token_embedding_dim'],
            convolution_filter_size=filter_size)

        # NOTE: 1-index to avoid using 0 typically associated with padding
        input_ = torch.LongTensor(encoder_params['batch_size'],
                                  encoder_params['num_tokens']).random_(
                                      1, encoder_params['vocab_size'])
        speaker = torch.LongTensor(encoder_params['batch_size']).fill_(0)
        output = encoder(input_, speaker)

        assert output.type() == 'torch.FloatTensor'
        assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                                encoder_params['out_dim'])
