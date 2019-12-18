import torch
from src.spectrogram_model.encoder import Encoder
from torchnlp.random import set_seed
import warnings

warnings.filterwarnings('ignore', message='@configurable: Overwriting configured argument')
warnings.filterwarnings('ignore', message='@configurable: No config for')

set_seed(123)

vocab_size = 6
out_dim = 5
token_embedding_dim = 6
num_convolution_layers = 2
num_convolution_filters = token_embedding_dim
convolution_filter_size = 3
convolution_dropout = 0.0
lstm_hidden_size = token_embedding_dim
lstm_layers = 1
lstm_bidirectional = True
lstm_dropout = 0.0
num_speakers = 1
speaker_embedding_dim = token_embedding_dim
speaker_embedding_dropout = 0.0

tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0, 0]])
mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0, 0]]).bool()
speaker = torch.tensor([0])
encoder = Encoder(
    num_speakers=1,
    speaker_embedding_dim=speaker_embedding_dim,
    speaker_embedding_dropout=speaker_embedding_dropout,
    vocab_size=vocab_size,
    out_dim=out_dim,
    token_embedding_dim=token_embedding_dim,
    num_convolution_layers=num_convolution_layers,
    num_convolution_filters=num_convolution_filters,
    convolution_filter_size=convolution_filter_size,
    convolution_dropout=convolution_dropout,
    lstm_hidden_size=lstm_hidden_size,
    lstm_layers=lstm_layers,
    lstm_dropout=lstm_dropout)

print(encoder(tokens, mask, speaker).sum(), tokens.numel())
