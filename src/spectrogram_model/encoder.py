import torch

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.encoders.text import DEFAULT_PADDING_INDEX


class MaskedBackwardLSTM(nn.Module):
    """ An LSTM that processes it's input backwards accounting for any padding on the ends of
    the input.

    TODO: Expand this module to have the same interface as a regular LSTM.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.backward_lstm = nn.LSTM(**kwargs)

    def forward(self, tokens, tokens_mask):
        """
        Args:
            tokens (torch.FloatTensor [seq_len, batch_size, input_size]): Batched set of sequences.
            tokens_mask (torch.BoolTensor [seq_len, batch_size, 1 or input_size]): Binary mask
                applied on tokens.

        Returns:
            encoded_tokens (torch.FloatTensor [seq_len, batch_size, input_size])
        """
        tokens = tokens.masked_fill(~tokens_mask, 0)
        # Ex. Assume we are dealing with a one dimensional input, like this:
        # tokens = [1, 2, 3, 0, 0]
        # length = 3
        # tokens.shape[0] = 5
        reversed_tokens = torch.zeros(tokens.shape, dtype=tokens.dtype, device=tokens.device)
        lengths = tokens_mask.int().sum(0)
        iterator = lambda: zip(range(tokens.shape[1]), list(lengths))
        for i, length in iterator():
            # Ex. [1, 2, 3, 0, 0] → [0, 0, 1, 2, 3]
            reversed_tokens[-length:, i] = tokens[:length, i]

        # Ex. [0, 0, 1, 2, 3] → [3, 2, 1, 0, 0]
        reversed_tokens = reversed_tokens.flip(0)

        lstm_results, _ = self.backward_lstm(reversed_tokens)

        # Ex. [3, 2, 1, 0, 0] → [0, 0, 1, 2, 3]
        lstm_results = lstm_results.flip(0)

        results = torch.zeros(lstm_results.shape, dtype=tokens.dtype, device=tokens.device)
        for i, length in iterator():
            # Ex. [0, 0, 1, 2, 3] → [1, 2, 3, 0, 0]
            results[:length, i] = lstm_results[-length:, i]

        return results.masked_fill(~tokens_mask, 0)


class Encoder(nn.Module):
    """ Encodes sequence as a hidden feature representation.

    SOURCE (Tacotron 2):
        The encoder converts a character sequence into a hidden feature representation. Input
        characters are represented using a learned 512-dimensional character embedding, which are
        passed through a stack of 3 convolutional layers each containing 512 filters with shape 5 ×
        1, i.e., where each filter spans 5 characters, followed by batch normalization [18] and ReLU
        activations. As in Tacotron, these convolutional layers model longer-term context (e.g.,
        N-grams) in the input character sequence. The output of the final convolutional layer is
        passed into a single bi-directional [19] LSTM [20] layer containing 512 units (256 in each
        direction) to generate the encoded features.

        ...

        The convolutional layers in the network are regularized using dropout [25] with probability
        0.5, and LSTM layers are regularized using zoneout [26] with probability 0.1. In order to
        introduce output variation at inference time, dropout with probability 0.5 is applied only
        to layers in the pre-net of the autoregressive decoder.

    Reference:
        * PyTorch BatchNorm vs Tensorflow parameterization possible source of error...
          https://stackoverflow.com/questions/48345857/batchnorm-momentum-convention-pytorch?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa

    Args:
        vocab_size (int): Maximum size of the vocabulary used to encode ``tokens``.
        out_dim (int): Number of dimensions to output.
        token_embedding_dim (int): Size of the token embedding dimensions.
        num_convolution_layers (int): Number of convolution layers to apply.
        num_convolution_filters (odd :clas:`int`): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int): Size of the convolving kernel.
        lstm_hidden_size (int): The number of features in the LSTM hidden state. Must be
            an even integer if ``lstm_bidirectional`` is ``True``. The hidden size of the final
            hidden feature representation.
        lstm_layers (int): Number of recurrent LSTM layers.
    """

    @configurable
    def __init__(self,
                 vocab_size,
                 out_dim=HParam(),
                 token_embedding_dim=HParam(),
                 num_convolution_layers=HParam(),
                 num_convolution_filters=HParam(),
                 convolution_filter_size=HParam(),
                 convolution_dropout=HParam(),
                 lstm_hidden_size=HParam(),
                 lstm_layers=HParam(),
                 lstm_dropout=HParam()):

        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

        self.embed_token = nn.Embedding(
            vocab_size, token_embedding_dim, padding_idx=DEFAULT_PADDING_INDEX)
        self.embed_layer_norm = nn.LayerNorm(token_embedding_dim)
        self.project = nn.Sequential(nn.Linear(lstm_hidden_size, out_dim), nn.LayerNorm(out_dim))

        self.convolution_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=num_convolution_filters if i != 0 else token_embedding_dim,
                    out_channels=num_convolution_filters,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)), nn.ReLU(inplace=True),
                nn.Dropout(p=convolution_dropout)) for i in range(num_convolution_layers)
        ])

        self.normalization_layers = nn.ModuleList(
            [nn.LayerNorm(num_convolution_filters) for i in range(num_convolution_layers)])

        assert lstm_hidden_size % 2 == 0, '`lstm_hidden_size` must be divisable by 2'

        self.forward_lstm = nn.LSTM(
            input_size=num_convolution_filters,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers)
        self.backward_lstm = MaskedBackwardLSTM(
            input_size=num_convolution_filters,
            hidden_size=lstm_hidden_size // 2,
            num_layers=lstm_layers)

        # NOTE: Tacotron 2 authors mentioned using Zoneout; unfortunately, Zoneout or any LSTM state
        # dropout in PyTorch forces us to unroll the LSTM and slow down this component x3 to x4. For
        # right now, we will not be using state dropout on the LSTM. We are applying dropout onto
        # the LSTM output instead.
        self.lstm_dropout = nn.Dropout(p=lstm_dropout)

        # Initialize weights
        for module in self.convolution_layers.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, tokens, tokens_mask):
        """
        NOTE: The results for the spectrogram model are different depending on the batch size due
        to the backwards LSTM needing to consider variable padding.

        Args:
            tokens (torch.LongTensor [batch_size, num_tokens]): Batched set of sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask applied on
                tokens.

        Returns:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, out_dim]): Batched set of
                encoded sequences.
        """
        # [batch_size, num_tokens] → [batch_size, num_tokens, token_embedding_dim]
        tokens = self.embed_token(tokens)
        tokens = self.embed_layer_norm(tokens)

        # Our input is expected to have shape `[batch_size, num_tokens, token_embedding_dim]`.  The
        # convolution layers expect input of shape
        # `[batch_size, in_channels (token_embedding_dim), sequence_length (num_tokens)]`. We thus
        # need to transpose the tensor first.
        tokens = tokens.transpose(1, 2)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        tokens_mask = tokens_mask.unsqueeze(1)

        # [batch_size, num_convolution_filters, num_tokens]
        for conv, normalization_layer in zip(self.convolution_layers, self.normalization_layers):
            tokens = tokens.masked_fill(~tokens_mask, 0)
            tokens = conv(tokens)
            tokens = tokens.transpose(1, 2)
            tokens = normalization_layer(tokens)
            tokens = tokens.transpose(1, 2)

        # Our input is expected to have shape `[batch_size, num_convolution_filters, num_tokens]`.
        # The lstm layers expect input of shape
        # `[seq_len (num_tokens), batch_size, input_size (num_convolution_filters)]`. We thus need
        # to permute the tensor first.
        tokens = tokens.permute(2, 0, 1)
        tokens_mask = tokens_mask.permute(2, 0, 1)

        # [num_tokens, batch_size, lstm_hidden_size // 2]
        forward_encoded_tokens = self.forward_lstm(tokens)[0]
        # [num_tokens, batch_size, lstm_hidden_size // 2]
        backward_encoded_tokens = self.backward_lstm(tokens, tokens_mask)
        # [num_tokens, batch_size, lstm_hidden_size]
        encoded_tokens = torch.cat([forward_encoded_tokens, backward_encoded_tokens], dim=2)
        out = self.lstm_dropout(encoded_tokens)
        out = self.lstm_layer_norm(out.masked_fill(~tokens_mask, 0))

        # [num_tokens, batch_size, lstm_hidden_size] →
        # [num_tokens, batch_size, out_dim]
        return self.project(out).masked_fill(~tokens_mask, 0)
