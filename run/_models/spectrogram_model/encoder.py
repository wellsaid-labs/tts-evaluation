import typing
from functools import lru_cache

import config as cf
import torch
import torch.nn
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence
from torchnlp.nn import LockedDropout

from lib.utils import LSTM, NumeralizePadEmbed
from run._models.spectrogram_model.containers import Encoded, Inputs


@lru_cache(maxsize=8)
def _roll_helper(
    length: int, device: torch.device, dimension: int, num_dimensions: int
) -> typing.Tuple[torch.Tensor, int]:
    """Helper to ensure `indices` and `dimension` are not recalculated unnecessarily."""
    indices = torch.arange(0, length, device=device)
    dimension = num_dimensions + dimension if dimension < 0 else dimension
    # EXAMPLE:
    # indicies.shape == (3,)
    # tensor.shape == (1, 2, 3, 4, 5)
    # indices_shape == [1, 1, 3, 1, 1]
    indices_shape = [1] * dimension + [-1] + [1] * (num_dimensions - dimension - 1)
    indices = indices.view(*tuple(indices_shape))
    return indices, dimension


def _roll(tensor: torch.Tensor, shift: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shift a tensor along the specified dimension.

    Args:
        tensor (torch.Tensor [*, dim, *]): The tensor to shift.
        shift (torch.Tensor [*]): The number of elements to shift `dim`. This tensor must have one
            less dimensions than `tensor`.
        dim: The dimension to shift.

    Returns:
        tensor (torch.Tensor [*, dim, *]): The tensor that was shifted.
    """
    shift = shift.unsqueeze(dim)
    assert (
        shift.dim() == tensor.dim()
    ), "The `shift` tensor must be the same size as `tensor` without the `dim` dimension."
    indices, dim = _roll_helper(tensor.shape[dim], tensor.device, dim, tensor.dim())
    indices = indices.detach().expand(*tensor.shape)
    indices = (indices - shift) % tensor.shape[dim]
    return torch.gather(tensor, dim, indices)


_RecurrentNeuralNetwork = typing.Union[torch.nn.LSTM, torch.nn.GRU]


class _RightMaskedBiRNN(torch.nn.Module):
    """A bidirectional RNN that ignores any masked input on the right side of the sequence.

    NOTE: Unfortunatly, this RNN does not return the hidden state due to related performance
    implications. Similarly, it does not properly handle left side masking due to performance
    implications.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        rnn_class: typing.Union[
            typing.Type[torch.nn.LSTM], typing.Type[torch.nn.GRU]
        ] = torch.nn.LSTM,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        _rnn = lambda i: rnn_class(
            input_size=input_size if i == 0 else hidden_size * 2, hidden_size=hidden_size, bias=bias
        )
        self.rnn_layers = ModuleList([ModuleList([_rnn(i), _rnn(i)]) for i in range(num_layers)])

    @staticmethod
    def _backward_pass(
        backward_rnn: typing.Union[torch.nn.LSTM, torch.nn.GRU],
        tokens: torch.Tensor,
        num_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the backwards RNN pass.

        Args:
            backward_rnn
            tokens (torch.FloatTensor [seq_len, batch_size, backward_rnn.input_size])
            num_tokens (torch.LongTensor [batch_size])

        Returns:
            (torch.FloatTensor [seq_len, batch_size, hidden_size]): RNN output.
        """
        # Ex. Assume we are dealing with a one dimensional input, like this:
        # tokens = [1, 2, 3, 0, 0]
        # tokens_mask = [1, 1, 1, 0, 0]
        # num_tokens = [3]
        # tokens.shape[0] = 5
        num_tokens = num_tokens.unsqueeze(1)

        # Ex. [1, 2, 3, 0, 0] → [0, 0, 1, 2, 3]
        tokens = _roll(tokens, (tokens.shape[0] - num_tokens), dim=0)

        # Ex. [0, 0, 1, 2, 3] → [3, 2, 1, 0, 0]
        tokens = tokens.flip(0)

        rnn_results, _ = backward_rnn(tokens)

        # Ex. [3, 2, 1, 0, 0] → [0, 0, 1, 2, 3]
        rnn_results = rnn_results.flip(0)

        # Ex. [0, 0, 1, 2, 3] → [1, 2, 3, 0, 0]
        return _roll(rnn_results, num_tokens, dim=0)

    def forward(self, tokens: torch.Tensor, tokens_mask: torch.Tensor, num_tokens: torch.Tensor):
        """Compute the RNN pass.

        Args:
            tokens (torch.FloatTensor [seq_len, batch_size, input_size])
            tokens_mask (torch.BoolTensor [seq_len, batch_size, 1 (optional)])
            num_tokens (torch.LongTensor [batch_size])

        Returns:
            (torch.FloatTensor [seq_len, batch_size, hidden_size * 2]): Output features predicted
                by the RNN.
        """
        output = tokens
        tokens_mask_expanded = tokens_mask.view(tokens_mask.shape[0], tokens_mask.shape[1], 1)
        tokens_mask = tokens_mask.view(tokens_mask.shape[0], tokens_mask.shape[1])
        Iterable = typing.Iterable[typing.Tuple[_RecurrentNeuralNetwork, _RecurrentNeuralNetwork]]
        for forward_rnn, backward_rnn in typing.cast(Iterable, iter(self.rnn_layers)):
            # [seq_len, batch_size, input_size or hidden_size * 2] →
            # [seq_len, batch_size, hidden_size * 2]
            forward_output, _ = forward_rnn(output)

            # [seq_len, batch_size, input_size or hidden_size * 2] →
            # [seq_len, batch_size, hidden_size * 2]
            backward_output = self._backward_pass(backward_rnn, output, num_tokens)

            # [seq_len, batch_size, hidden_size] (concat) [seq_len, batch_size, hidden_size] →
            # [seq_len, batch_size, hidden_size * 2]
            output = torch.cat([forward_output, backward_output], dim=2)
            output = output.masked_fill(~tokens_mask_expanded, 0)
        return output


class _LayerNorm(torch.nn.LayerNorm):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, **cf.get(func=torch.nn.LayerNorm))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().forward(tensor.transpose(1, 2)).transpose(1, 2)


class _Conv1dLockedDropout(LockedDropout):
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().forward(tensor.permute(2, 0, 1)).permute(1, 2, 0)


class Encoder(torch.nn.Module):
    """Encode a discrete sequence as a sequence of differentiable vector(s).

    Args:
        max_tokens: The maximum number of tokens.
        max_seq_meta_values: The maximum number of sequence metadata values.
        max_token_meta_values: The maximum number of token metadata values.
        max_token_embed_size: The maximum size of the inputted token embedding.
        seq_meta_embed_size: The size of the sequence metadata embedding.
        token_meta_embed_size: The size of the token metadata embedding.
        seq_meta_embed_dropout: The sequence metadata embedding dropout probability.
        out_size: The size of the encoder output.
        hidden_size: The size of the encoders hidden representation. This value must be even.
        num_conv_layers: Number of convolution layers.
        conv_filter_size: Size of the convolving kernel. This value must be odd.
        lstm_layers: Number of recurrent LSTM layers.
        dropout: Dropout probability used to regularize the encoders hidden representation.
    """

    def __init__(
        self,
        max_tokens: int,
        max_seq_meta_values: typing.Tuple[int, ...],
        max_token_meta_values: typing.Tuple[int, ...],
        max_token_embed_size: int,
        seq_meta_embed_size: int,
        token_meta_embed_size: int,
        seq_meta_embed_dropout: float,
        out_size: int,
        hidden_size: int,
        num_conv_layers: int,
        conv_filter_size: int,
        lstm_layers: int,
        dropout: float,
    ):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert conv_filter_size % 2 == 1, "`conv_filter_size` must be odd"
        assert hidden_size % 2 == 0, "`hidden_size` must be even"

        layer_norm = cf.partial(torch.nn.LayerNorm)

        self.max_token_embed_size = max_token_embed_size
        self.embed_seq_metadata = self._make_embeds(seq_meta_embed_size, max_seq_meta_values)
        self.seq_meta_embed_dropout = torch.nn.Dropout(seq_meta_embed_dropout)
        self.embed_token_metadata = self._make_embeds(token_meta_embed_size, max_token_meta_values)
        self.embed_token: NumeralizePadEmbed[typing.Hashable]
        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(
                hidden_size + seq_meta_embed_size + max_token_embed_size + token_meta_embed_size,
                hidden_size,
            ),
            torch.nn.ReLU(),
            layer_norm(hidden_size),
        )

        self.conv_layers = ModuleList(
            torch.nn.Sequential(
                _Conv1dLockedDropout(dropout),
                torch.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=conv_filter_size,
                    padding=int((conv_filter_size - 1) / 2),
                ),
                torch.nn.ReLU(),
            )
            for _ in range(num_conv_layers)
        )

        self.norm_layers = ModuleList(_LayerNorm(hidden_size) for _ in range(num_conv_layers))

        self.lstm = _RightMaskedBiRNN(
            rnn_class=LSTM,
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=lstm_layers,
        )
        self.lstm_norm = layer_norm(hidden_size)
        self.lstm_dropout = LockedDropout(dropout)

        self.project_out = torch.nn.Sequential(
            LockedDropout(dropout),
            torch.nn.Linear(hidden_size, out_size),
            layer_norm(out_size),
        )

        for module in self.conv_layers.modules():
            if isinstance(module, torch.nn.Conv1d):
                gain = torch.nn.init.calculate_gain("relu")
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)

    def _make_embeds(self, embed_size: int, max_values: typing.Tuple[int, ...]):
        message = "`embed_size` must be divisable by the number of metadata attributes."
        assert len(max_values) == 0 or embed_size % len(max_values) == 0, message
        size = embed_size // len(max_values) if len(max_values) > 0 else 0
        return ModuleList(NumeralizePadEmbed(n, embedding_dim=size) for n in max_values)

    def __call__(self, inputs: Inputs) -> Encoded:
        return super().__call__(inputs)

    def forward(self, inputs: Inputs) -> Encoded:
        # [batch_size, num_tokens] →
        # tokens [batch_size, num_tokens, hidden_size]
        # tokens_mask [batch_size, num_tokens]
        tokens, tokens_mask = self.embed_token(inputs.tokens, batch_first=True)

        # [batch_size, num_tokens] → [batch_size]
        num_tokens = tokens_mask.sum(dim=1)
        device = tokens.device

        # [batch_size] → [batch_size, seq_meta_embed_size]
        iter_ = zip(self.embed_seq_metadata, inputs.seq_metadata)
        seq_metadata = [embed(meta, batch_first=True)[0] for embed, meta in iter_]
        seq_metadata = self.seq_meta_embed_dropout(torch.cat(seq_metadata, dim=1))
        # [batch_size, seq_meta_embed_size] → [batch_size, num_tokens, seq_meta_embed_size]
        seq_metadata_expanded = seq_metadata.unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens] → [batch_size, num_tokens, token_meta_embed_size]
        iter_ = zip(self.embed_token_metadata, inputs.token_metadata)
        token_metadata = tuple([embed(meta, batch_first=True)[0] for embed, meta in iter_])

        if isinstance(inputs.token_embeddings, list):
            token_embed: torch.Tensor = pad_sequence(inputs.token_embeddings, batch_first=True)
        else:
            token_embed = inputs.token_embeddings

        if token_embed.shape[2] != self.max_token_embed_size:
            token_embed_ = torch.zeros(*tokens.shape[:2], self.max_token_embed_size, device=device)
            token_embed_[:, :, 0 : token_embed.shape[2]] = token_embed
            token_embed = token_embed_

        # [batch_size, num_tokens, hidden_size] (cat)
        # [batch_size, num_tokens, token_meta_embed_size] (cat)
        # [batch_size, num_tokens, seq_meta_embed_size] (cat)
        # [batch_size, num_tokens, max_token_embed_size] →
        # [batch_size, num_tokens,
        #  hidden_size + seq_meta_embed_size + max_token_embed_size]
        tokens = torch.cat([tokens, seq_metadata_expanded, *token_metadata, token_embed], dim=2)

        # [batch_size, num_tokens,
        #  hidden_size + seq_meta_embed_size + token_meta_embed_size + max_token_embed_size] →
        # [batch_size, num_tokens, hidden_size]
        tokens = self.embed(tokens)

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
        tokens = self.lstm_norm(
            tokens + self.lstm(self.lstm_dropout(tokens), tokens_mask, num_tokens)
        )

        # [num_tokens, batch_size, hidden_size] →
        # [num_tokens, batch_size, out_dim]
        tokens = self.project_out(tokens).masked_fill(~tokens_mask, 0)

        # [num_tokens, batch_size, 1] → [batch_size, num_tokens]
        tokens_mask = tokens_mask.squeeze(2).transpose(0, 1)
        # [num_tokens, batch_size, out_dim] → [batch_size, num_tokens, out_dim]
        tokens = tokens.transpose(0, 1)

        tokens = [tokens[i][s] for i, s in enumerate(inputs.slices)]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens)

        return Encoded(tokens, inputs.tokens_mask, inputs.num_tokens, seq_metadata)
