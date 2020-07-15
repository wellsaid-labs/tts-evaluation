from functools import lru_cache

import torch

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.encoders.text import DEFAULT_PADDING_INDEX
from torchnlp.nn import LockedDropout

from src.utils import LSTM


@lru_cache(maxsize=8)
def _roll_helper(length, device, dimension, num_dimensions):
    """ Helper to ensure `indices` and `dimension` are not recalculated unnecessarily. """
    with torch.no_grad():
        indices = torch.arange(0, length, device=device)
        dimension = num_dimensions + dimension if dimension < 0 else dimension

        # EXAMPLE:
        # indicies.shape == (3,)
        # tensor.shape == (1, 2, 3, 4, 5)
        # indices_shape == [1, 1, 3, 1, 1]
        indices_shape = [1] * dimension + [-1] + [1] * (num_dimensions - dimension - 1)
        indices = indices.view(*tuple(indices_shape))
    return indices, dimension


def roll(tensor, shift, dim=-1):
    """ Shift a tensor along the specified dimension.

    Args:
        tensor (torch.Tensor [*, dim, *]): The tensor to shift.
        shift (torch.Tensor [*]): The number of elements to shift `dim`. This tensor must have one
            less dimensions than `tensor`.
        dim (int): The dimension to shift.

    Returns:
        tensor (torch.Tensor [*, dim, *]): The tensor that was shifted.
    """
    shift = shift.unsqueeze(dim)
    assert shift.dim() == tensor.dim(
    ), 'The `shift` tensor must be the same size as `tensor` without the `dim` dimension.'
    indices, dim = _roll_helper(tensor.shape[dim], tensor.device, dim, tensor.dim())
    indices = indices.detach().expand(*tensor.shape)
    indices = (indices - shift) % tensor.shape[dim]
    return torch.gather(tensor, dim, indices)


class RightMaskedBiRNN(nn.Module):
    """ A bidirectional RNN that ignores any masked input on the right side of the sequence.

    Unfortunatly, this RNN does not return the hidden state due to related performance
    implications. Similarly, it does not properly handle left side masking due to performance
    implications.

    Args:
        rnn_class: The RNN class to instantiate.
        ... See the documentation for `rnn_class`.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, rnn_class=torch.nn.LSTM):
        super().__init__()

        self.rnn_layers = nn.ModuleList([
            nn.ModuleList([
                rnn_class(
                    input_size=input_size if i == 0 else hidden_size * 2,
                    hidden_size=hidden_size,
                    bias=bias),
                rnn_class(
                    input_size=input_size if i == 0 else hidden_size * 2,
                    hidden_size=hidden_size,
                    bias=bias)
            ]) for i in range(num_layers)
        ])

    def _backward_pass(self, backward_rnn, tokens, tokens_mask):
        """ Compute the backwards RNN pass.

        Args:
            backward_rnn (nn.Module)
            tokens (torch.FloatTensor [seq_len, batch_size, backward_rnn.input_size]): Batched set
                of sequences.
            tokens_mask (torch.BoolTensor [seq_len, batch_size]): Binary mask applied on tokens.

        Returns:
            (torch.FloatTensor [seq_len, batch_size, hidden_size]): Output features predicted
                by the RNN.
        """
        # Ex. Assume we are dealing with a one dimensional input, like this:
        # tokens = [1, 2, 3, 0, 0]
        # tokens_mask = [1, 1, 1, 0, 0]
        # lengths = [3]
        # tokens.shape[0] = 5
        lengths = tokens_mask.int().sum(0)  # TODO: Reuse the already computed `num_tokens`
        lengths = lengths.unsqueeze(1)

        # Ex. [1, 2, 3, 0, 0] → [0, 0, 1, 2, 3]
        tokens = roll(tokens, (tokens.shape[0] - lengths), dim=0)

        # Ex. [0, 0, 1, 2, 3] → [3, 2, 1, 0, 0]
        tokens = tokens.flip(0)

        rnn_results, _ = backward_rnn(tokens)

        # Ex. [3, 2, 1, 0, 0] → [0, 0, 1, 2, 3]
        rnn_results = rnn_results.flip(0)

        # Ex. [0, 0, 1, 2, 3] → [1, 2, 3, 0, 0]
        return roll(rnn_results, lengths, dim=0)

    def forward(self, tokens, tokens_mask):
        """ Compute the RNN pass.

        Args:
            tokens (torch.FloatTensor [seq_len, batch_size, input_size]): Batched set of sequences.
            tokens_mask (torch.BoolTensor [seq_len, batch_size] or [seq_len, batch_size, 1]): Binary
                mask applied on tokens.

        Returns:
            (torch.FloatTensor [seq_len, batch_size, hidden_size * 2]): Output features predicted
                by the RNN.
        """
        output = tokens
        tokens_mask_expanded = tokens_mask if len(
            tokens_mask.shape) == 3 else tokens_mask.unsqueeze(2)
        tokens_mask = tokens_mask.view(tokens_mask.shape[0], tokens_mask.shape[1])
        for forward_rnn, backward_rnn in self.rnn_layers:
            # [seq_len, batch_size, input_size or hidden_size * 2] →
            # [seq_len, batch_size, hidden_size * 2]
            forward_output, _ = forward_rnn(output)

            # [seq_len, batch_size, input_size or hidden_size * 2] →
            # [seq_len, batch_size, hidden_size * 2]
            backward_output = self._backward_pass(backward_rnn, output, tokens_mask)

            # [seq_len, batch_size, hidden_size] (concat) [seq_len, batch_size, hidden_size] →
            # [seq_len, batch_size, hidden_size * 2]
            output = torch.cat([forward_output, backward_output],
                               dim=2).masked_fill(~tokens_mask_expanded, 0)
        return output


# TODO: Remove.
class RightMaskedBiLSTM(RightMaskedBiRNN):
    pass


class LayerNorm(nn.LayerNorm):

    def forward(self, tensor):
        return super().forward(tensor.transpose(1, 2)).transpose(1, 2)


class Conv1dLockedDropout(LockedDropout):

    def forward(self, tensor):
        return super().forward(tensor.permute(2, 0, 1)).permute(1, 2, 0)


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
        speaker_embedding_dim (int): The size of the speaker embedding.
        out_dim (int): Number of dimensions to output.
        hidden_size (int): The hidden size for internal RNN, embedding, and convolution features.
            This hidden size must be even.
        num_convolution_layers (int): Number of convolution layers to apply.
        convolution_filter_size (int): Size of the convolving kernel.
        lstm_layers (int): Number of recurrent LSTM layers.
        dropout (float): The dropout probability for hidden encoder features.
    """

    @configurable
    def __init__(self,
                 vocab_size,
                 speaker_embedding_dim,
                 out_dim=HParam(),
                 hidden_size=HParam(),
                 num_convolution_layers=HParam(),
                 convolution_filter_size=HParam(),
                 lstm_layers=HParam(),
                 dropout=HParam()):

        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')
        assert hidden_size % 2 == 0, '`hidden_size` must be divisable by even'
        assert speaker_embedding_dim < hidden_size, (
            'The `hidden_size` must be larger than the `speaker_embedding_dim` to accommodate it.')

        self.embed_token = nn.Sequential(
            nn.Embedding(
                vocab_size, hidden_size - speaker_embedding_dim, padding_idx=DEFAULT_PADDING_INDEX),
            nn.LayerNorm(hidden_size - speaker_embedding_dim))

        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                Conv1dLockedDropout(dropout),
                nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)), nn.ReLU())
            for i in range(num_convolution_layers)
        ])

        for module in self.conv_layers.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

        self.norm_layers = nn.ModuleList(
            [LayerNorm(hidden_size) for i in range(num_convolution_layers)])

        self.lstm = RightMaskedBiRNN(
            rnn_class=LSTM,
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=lstm_layers)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.lstm_dropout = LockedDropout(dropout)

        self.project_out = nn.Sequential(
            LockedDropout(dropout), nn.Linear(hidden_size, out_dim), nn.LayerNorm(out_dim))

    def forward(self, tokens, tokens_mask, speaker):
        """
        Args:
            tokens (torch.LongTensor [batch_size, num_tokens]): Batched set of sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask applied on
                tokens.
            speaker (torch.FloatTensor [batch_size, speaker_embedding_dim])

        Returns:
            output (torch.FloatTensor [num_tokens, batch_size, out_dim]): Batched set of encoded
                sequences.
        """
        # [batch_size, speaker_embedding_dim] → [batch_size, num_tokens, speaker_embedding_dim]
        speaker = speaker.unsqueeze(1).expand(-1, tokens.shape[1], -1)
        # [batch_size, num_tokens] → [batch_size, num_tokens, hidden_size]
        # TODO: The speaker embedding decreases the size of the token embedding. This side-effect
        # is not intuitive. In order to implement this, we recommend that you slice in the
        # residual connection.
        tokens = torch.cat([self.embed_token(tokens), speaker], dim=2)

        # Our input is expected to have shape `[batch_size, num_tokens, hidden_size]`.  The
        # convolution layers expect input of shape
        # `[batch_size, in_channels (hidden_size), sequence_length (num_tokens)]`. We thus
        # need to transpose the tensor first.
        tokens = tokens.transpose(1, 2)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        tokens_mask = tokens_mask.unsqueeze(1)

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            tokens = tokens.masked_fill(~tokens_mask, 0)
            tokens = norm(tokens + conv(tokens))

        # Our input is expected to have shape `[batch_size, hidden_size, num_tokens]`.
        # The lstm layers expect input of shape
        # `[seq_len (num_tokens), batch_size, input_size (hidden_size)]`. We thus need
        # to permute the tensor first.
        tokens = tokens.permute(2, 0, 1)
        tokens_mask = tokens_mask.permute(2, 0, 1)

        tokens = self.lstm_norm(tokens + self.lstm(self.lstm_dropout(tokens), tokens_mask))

        # [num_tokens, batch_size, hidden_size] →
        # [num_tokens, batch_size, out_dim]
        return self.project_out(tokens).masked_fill(~tokens_mask, 0)
