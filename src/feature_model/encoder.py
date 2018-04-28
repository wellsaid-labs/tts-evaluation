import torch
import torch.nn as nn

from torchnlp.text_encoders import PADDING_INDEX

from src.configurable import configurable


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
        embedding_dim (int, optional): Size of the embedding dimensions.
        num_convolution_layers (int, optional): Number of convolution layers to apply.
        num_convolution_filters (odd :clas:`int`, optional): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int, optional): Size of the convolving kernel.
        lstm_hidden_size (int, optional): The number of features in the LSTM hidden state. Must be
            an even integer if ``lstm_bidirectional`` is True. The hidden size of the final
            hidden feature representation.
        lstm_layers (int, optional): Number of recurrent LSTM layers.
        lstm_bidirectional (bool, optional): If True, becomes a bidirectional LSTM.
    """

    @configurable
    def __init__(self,
                 vocab_size,
                 embedding_dim=512,
                 num_convolution_layers=3,
                 num_convolution_filters=512,
                 convolution_filter_size=5,
                 convolution_dropout=0.5,
                 lstm_hidden_size=512,
                 lstm_layers=1,
                 lstm_bidirectional=True,
                 lstm_variational_dropout=0.1):

        super(Encoder, self).__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_INDEX)
        self.convolution_layers = nn.Sequential(*tuple([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=num_convolution_filters if i != 0 else embedding_dim,
                    out_channels=num_convolution_filters,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)),
                nn.BatchNorm1d(num_features=num_convolution_filters),
                nn.ReLU(),
                nn.Dropout(p=convolution_dropout)) for i in range(num_convolution_layers)
        ]))

        if lstm_bidirectional:
            assert lstm_hidden_size % 2 == 0, '`lstm_hidden_size` must be divisable by 2'
            lstm_hidden_size = lstm_hidden_size // 2

        self.lstm = nn.LSTM(
            input_size=num_convolution_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            # NOTE: Tacotron 2 authors mentioned using Zoneout; unfortunatly, Zoneout in PyTorch
            # forces us to unroll the LSTM and slow down this component x3 to x4. For right now, we
            # will by using variational dropout.
            # NOTE: The Zoneout paper also mentions that it fairs worse than variational dropout.
            dropout=lstm_variational_dropout,
        )

    def forward(self, tokens):
        """
        Args:
            tokens (torch.LongTensor [batch_size, num_tokens]): Batched set of sequences.

        Returns:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batched set of
                encoded sequences where:
                ``hidden_size = (lstm_hidden_size / 2) * (2 if lstm_bidirectional else 1)``
        """
        # [batch_size, num_tokens] → [batch_size, num_tokens, embedding_dim]
        tokens = self.embed(tokens)

        # Our input is expected to have shape `[batch_size, num_tokens, embedding_dim]`.  The
        # convolution layers expect input of shape
        # `[batch_size, in_channels (embedding_dim), sequence_length (num_tokens)]`. We thus need to
        # transpose the tensor first.
        tokens = torch.transpose(tokens, 1, 2)

        # [batch_size, num_convolution_filters, num_tokens]
        tokens = self.convolution_layers(tokens)

        # Our input is expected to have shape `[batch_size, num_convolution_filters, num_tokens]`.
        # The lstm layers expect input of shape
        # `[seq_len (num_tokens), batch_size, input_size (num_convolution_filters)]`. We thus need
        # to permute the tensor first.
        tokens = tokens.permute(2, 0, 1)

        # [num_tokens, batch_size, lstm_hidden_size * (2 if lstm_bidirectional else 1) ]
        encoded_tokens, _ = self.lstm(tokens)
        return encoded_tokens
