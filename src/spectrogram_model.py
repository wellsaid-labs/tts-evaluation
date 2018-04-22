import torch
import torch.nn as nn

from torchnlp.text_encoders import PADDING_INDEX

from src.configurable import configurable


class SpectrogramModel(nn.Module):
    """ Character sequence consumed to predict a spectrogram.

    SOURCE (Tacotron 2):
        The network is composed of an encoder and a decoder with attention. The encoder converts a
        character sequence into a hidden feature representation which the decoder consumes to
        predict a spectrogram. Input characters are represented using a learned 512-dimensional
        character embedding, which are passed through a stack of 3 convolutional layers each
        containing 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters,
        followed by batch normalization [18] and ReLU activations. As in Tacotron, these
        convolutional layers model longer-term context (e.g., N-grams) in the input character
        sequence. The output of the final convolutional layer is passed into a single
        bi-directional [19] LSTM [20] layer containing 512 units (256 in each direction) to
        generate the encoded features.

        The encoder output is consumed by an attention network which summarizes the full encoded
        sequence as a fixed-length context vector for each decoder output step. We use the
        location-sensitive attention from [21], which extends the additive attention mechanism
        [22] to use cumulative attention weights from previous decoder time steps as an additional
        feature. This encourages the model to move forward consistently through the input,
        mitigating potential failure modes where some subsequences are repeated or ignored by the
        decoder. Attention probabilities are computed after projecting inputs and location
        features to 128-dimensional hidden representations. Location features are computed using
        32 1-D convolution filters of length 31.

        The decoder is an autoregressive recurrent neural network which predicts a mel spectrogram
        from the encoded input sequence one frame at a time. The prediction from the previous time
        step is first passed through a small pre-net containing 2 fully connected layers of 256
        hidden ReLU units. We found that the pre-net acting as an information bottleneck was
        essential for learning attention. The prenet output and attention context vector are
        concatenated and passed through a stack of 2 uni-directional LSTM layers with 1024 units.
        The concatenation of the LSTM output and the attention context vector is projected through
        a linear transform to predict the target spectrogram frame. Finally, the predicted mel
        spectrogram is passed through a 5-layer convolutional post-net which predicts a residual to
        add to the prediction to improve the overall reconstruction. Each post-net layer is
        comprised of 512 filters with shape 5 × 1 with batch normalization, followed by tanh
        activations on all but the final layer.

        In parallel to spectrogram frame prediction, the concatenation of decoder LSTM output and
        the attention context is projected down to a scalar and passed through a sigmoid activation
        to predict the probability that the output sequence has completed. This “stop token”
        prediction is used during inference to allow the model to dynamically determine when to
        terminate generation instead of always generating for a fixed duration. Specifically,
        generation completes at the first frame for which this probability exceeds a threshold of
        0.5.

        The convolutional layers in the network are regularized using dropout [25] with probability
        0.5, and LSTM layers are regularized using zoneout [26] with probability 0.1. In order to
        introduce output variation at inference time, dropout with probability 0.5 is applied only
        to layers in the pre-net of the autoregressive decoder.

      Reference:
          * Tacotron 2 Paper:
            https://arxiv.org/pdf/1712.05884.pdf
      """

    @configurable
    def __init__(self):

        super(SpectrogramModel, self).__init__()

    def forward(self, tokens):
        """
        Args:
            tokens (torch.LongTensor [batch_size, num_tokens]): Batch of sequences.
        """
        pass


class _Encoder(nn.Module):
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
            an even integer if ``lstm_bidirectional`` is True.
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
                 lstm_hidden_size=512,
                 lstm_layers=1,
                 lstm_bidirectional=True):

        super(_Encoder, self).__init__()

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
                nn.BatchNorm1d(num_features=num_convolution_filters, momentum=0.01), nn.ReLU())
            for i in range(num_convolution_layers)
        ]))

        if lstm_bidirectional:
            assert lstm_hidden_size % 2 == 0, '`lstm_hidden_size` must be divisable by 2'
            lstm_hidden_size = lstm_hidden_size // 2

        self.lstm = nn.LSTM(
            input_size=num_convolution_filters,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
        )

    def forward(self, tokens):
        """
        Args:
            tokens (torch.LongTensor [batch_size, num_tokens]): Batch of sequences.

        Returns:
            tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batch of sequences
                encoded where:
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
        # to transpose the tensor first.

        # [batch_size, num_convolution_filters, num_tokens] →
        # [batch_size, num_tokens, num_convolution_filters]
        tokens = torch.transpose(tokens, 1, 2)
        # [batch_size, num_tokens, num_convolution_filters] →
        # [num_tokens, batch_size, num_convolution_filters]
        tokens = torch.transpose(tokens, 0, 1)

        # [num_tokens, batch_size, lstm_hidden_size * (2 if lstm_bidirectional else 1) ]
        tokens, _ = self.lstm(tokens)
        return tokens
