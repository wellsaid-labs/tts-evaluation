import typing
from functools import lru_cache

import config as cf
import torch
import torch.nn
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence
from torchnlp.nn import LockedDropout

from lib.distributed import NumeralizePadEmbed
from lib.utils import LSTM
from run._models.spectrogram_model.containers import Encoded
from run._models.spectrogram_model.inputs import Inputs


@lru_cache(maxsize=8)
def _roll_helper(
    length: int, device: torch.device, dimension: int, num_dimensions: int
) -> typing.Tuple[torch.Tensor, int]:
    """Helper to ensure `indices` and `dimension` are not recalculated unnecessarily."""
    indices = torch.arange(0, length, device=device)
    dimension = num_dimensions + dimension if dimension < 0 else dimension
    # EXAMPLE:
    # indices.shape == (3,)
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


class _GroupedEmbedder(torch.nn.Module):
    """This embeds a sequence by processing it multiple times, each time with different weights and
    a different mask. Lastly, the multiple versions are recombined.

    Args:
        input_size: The input size of a feature in the group.
        hidden_size: The size of the processing layers per feature.
        num_groups: The number of times to process the sequence.
        num_layers: The number of layers to process the sequence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_groups: int,
        num_layers: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.input_size = input_size
        self.layers = [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=input_size * num_groups if i == 0 else hidden_size * num_groups,
                    out_channels=hidden_size * num_groups,
                    kernel_size=1,
                    groups=num_groups,
                ),
                torch.nn.ReLU(),
            )
            for i in range(num_layers)
        ]
        self.layers = ModuleList(self.layers)

    def _seperate(self, tokens: torch.Tensor, mask: torch.Tensor):
        """Processes `tokens` in seperable groups with seperable masks.

        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, input_size * num_groups])
            mask (torch.FloatTensor [batch_size, num_tokens, num_groups])

        Returns:
            torch.FloatTensor [batch_size, num_tokens, num_groups, hidden_size]
            torch.LongTensor [batch_size, num_tokens, num_groups, hidden_size]
        """
        # [batch_size, num_tokens, num_groups] →
        # [batch_size, num_tokens, num_groups * hidden_size]
        mask = mask.repeat_interleave(self.hidden_size, 2)
        # [batch_size, num_tokens, num_groups * hidden_size] →
        # [batch_size, num_groups * hidden_size, num_tokens]
        tokens, mask = tokens.transpose(1, 2), mask.transpose(1, 2)
        mask_bool = ~mask.bool()
        for layer in self.layers:
            tokens = torch.masked_fill(layer(tokens), mask_bool, 0)
        # [batch_size, num_groups * hidden_size, num_tokens] →
        # [batch_size, num_tokens, num_groups * hidden_size]
        tokens, mask = tokens.transpose(1, 2), mask.transpose(1, 2)
        # [batch_size, num_tokens, num_groups * hidden_size] →
        # [batch_size, num_tokens, num_groups, hidden_size]
        tokens = torch.stack(tokens.tensor_split(self.num_groups, dim=2), dim=2)
        mask = torch.stack(mask.tensor_split(self.num_groups, dim=2), dim=2)
        return tokens, mask

    def __call__(
        self, tokens: typing.List[torch.Tensor], mask: typing.List[torch.Tensor]
    ) -> torch.Tensor:
        return super().__call__(tokens, mask)

    def forward(
        self, tokens: typing.List[torch.Tensor], mask: typing.List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, input_size]): A list of token
                embeddings.
            mask (torch.FloatTensor [batch_size, num_tokens]): A list of masks for each embedding.

        Returns:
            torch.FloatTensor [batch_size, num_tokens, hidden_size]
        """
        assert len(tokens) == self.num_groups
        assert len(mask) == self.num_groups

        # [batch_size, num_tokens, input_size * num_groups]
        tokens_ = torch.cat(tokens, dim=2)
        # [batch_size, num_tokens, num_groups]
        mask_ = torch.cat([m.view(*tokens_.shape[:2], 1) for m in mask], dim=2)

        # [batch_size, num_tokens, input_size * hidden_size] →
        # [batch_size, num_tokens, num_groups, hidden_size]
        tokens_, mask_ = self._seperate(tokens_, mask_)

        # NOTE: Handle padding where each of the masks is zero by masking it to zero.
        # [batch_size, num_tokens, num_groups, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        combined_mask = mask_.sum(dim=2)
        combined = tokens_.sum(dim=2) / combined_mask.clamp(min=1)
        return torch.masked_fill(combined, combined_mask == 0, 0)


class Encoder(torch.nn.Module):
    """Encode a discrete sequence as a sequence of differentiable vector(s).

    Args:
        max_tokens: The maximum number of tokens.
        max_seq_meta_vals: The maximum number of sequence metadata values for each feature.
        max_token_meta_vals: The maximum number of token metadata values for each feature.
        max_word_vector_size: The maximum size of `inputs.anno_embed("word_vector")`.
        max_seq_vector_size: The maximum size of the sequence vector.
        max_anno_vector_size: The maximum size of a annotation vector aside from the word vector.
        annos: The annotations to process.
        num_anno_layers: The number of layers to process annotations with.
        seq_embed_size: The size of the sequence metadata embeddings.
        token_meta_embed_size: The size of the token metadata embeddings.
        anno_embed_size: The size of the annotation embeddings.
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
        max_seq_meta_vals: typing.Tuple[int, ...],
        max_token_meta_vals: typing.Tuple[int, ...],
        max_word_vector_size: int,
        max_seq_vector_size: int,
        max_anno_vector_size: int,
        annos: typing.List[typing.Tuple[str, str]],
        num_anno_layers: int,
        seq_embed_size: int,
        token_meta_embed_size: int,
        anno_embed_size: int,
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

        self.max_word_vector_size = max_word_vector_size
        self.max_anno_vector_size = max_anno_vector_size
        self.max_seq_vector_size = max_seq_vector_size
        self.annos = annos

        self.embed_seq_meta = ModuleList(
            NumeralizePadEmbed(n, embedding_dim=seq_embed_size // len(max_seq_meta_vals))
            for n in max_seq_meta_vals
        )
        self.embed_seq_vector = torch.nn.Sequential(
            torch.nn.Linear(max_seq_vector_size, seq_embed_size),
            torch.nn.ReLU(),
            torch.nn.Linear(seq_embed_size, seq_embed_size),
            torch.nn.ReLU(),
            layer_norm(seq_embed_size),
        )
        self.embed_seq = torch.nn.Sequential(
            torch.nn.Dropout(seq_meta_embed_dropout),
            torch.nn.Linear(seq_embed_size * 2, seq_embed_size),
            torch.nn.ReLU(),
            layer_norm(seq_embed_size),
        )
        self.embed_token_meta = ModuleList(
            NumeralizePadEmbed(n, embedding_dim=token_meta_embed_size // len(max_token_meta_vals))
            for n in max_token_meta_vals
        )
        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        self.embed_anno_separable = _GroupedEmbedder(
            self.max_anno_vector_size,
            anno_embed_size // len(self.annos),
            len(self.annos),
            num_anno_layers,
        )
        self.embed_anno = torch.nn.Sequential(
            torch.nn.Linear(anno_embed_size // len(self.annos), anno_embed_size),
            torch.nn.ReLU(),
            layer_norm(anno_embed_size),
        )
        total_embed_size = (
            hidden_size
            + seq_embed_size
            + token_meta_embed_size
            + max_word_vector_size
            + anno_embed_size
        )
        self.embed = torch.nn.Sequential(
            torch.nn.Linear(total_embed_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
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

        self.lstm_dropout = LockedDropout(dropout)
        self.lstm = _RightMaskedBiRNN(
            rnn_class=LSTM,
            input_size=hidden_size * 2 + anno_embed_size,
            hidden_size=hidden_size // 2,
            num_layers=lstm_layers,
        )
        self.lstm_norm = layer_norm(hidden_size)

        self.project_out = torch.nn.Sequential(
            LockedDropout(dropout),
            torch.nn.Linear(hidden_size * 2 + anno_embed_size, out_size),
            layer_norm(out_size),
        )

        for module in self.conv_layers.modules():
            if isinstance(module, torch.nn.Conv1d):
                gain = torch.nn.init.calculate_gain("relu")
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)

    def __call__(self, inputs: Inputs) -> Encoded:
        return super().__call__(inputs)

    def forward(self, inputs: Inputs) -> Encoded:
        # [batch_size, num_tokens] →
        # tokens [batch_size, num_tokens, hidden_size]
        # tokens_mask [batch_size, num_tokens]
        tokens, tokens_mask = self.embed_token(inputs.tokens, batch_first=True)

        # NOTE: This could be different than `inputs.num_tokens` if `inputs.tokens` contains
        # padding tokens which are ignored by `self.embed_token`.
        # TODO: During testing, we may inject those, let's not do that.
        num_tokens = tokens_mask.sum(dim=1)

        # [batch_size] → [batch_size, seq_embed_size * len(max_seq_meta_vals)]
        seq_meta = [[s[i] for s in inputs.seq_meta] for i in range(inputs.num_seq_meta)]
        iter_ = zip(self.embed_seq_meta, seq_meta)
        seq_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]
        seq_meta_embed = torch.cat(seq_meta, dim=1)
        # [batch_size, max_seq_vector_size] → [batch_size, seq_embed_size]
        seq_vector_embed = self.embed_seq_vector(inputs.get_seq_vec(self.max_seq_vector_size))
        # [batch_size, seq_embed_size * len(max_seq_meta_vals)] (cat)
        # [batch_size, seq_embed_size] →
        # [batch_size, seq_embed_size * (len(max_seq_meta_vals) + 1)]
        seq_embed = torch.cat((seq_meta_embed, seq_vector_embed), dim=1)
        # [batch_size, seq_embed_size * (len(max_seq_meta_vals) + 1)] →
        # [batch_size, seq_embed_size]
        seq_embed: torch.Tensor = self.embed_seq(seq_embed)
        # [batch_size, seq_embed_size] → [batch_size, num_tokens, seq_embed_size]
        seq_embed_expanded = seq_embed.unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens] → [batch_size, num_tokens, token_meta_embed_size]
        token_meta = [[s[i] for s in inputs.token_meta] for i in range(inputs.num_token_meta)]
        iter_ = zip(self.embed_token_meta, token_meta)
        token_meta = tuple([embed(meta, batch_first=True)[0] for embed, meta in iter_])

        # [batch_size, num_tokens, max_anno_vector_size] →
        # [batch_size, num_tokens, anno_embed_size]
        anno_embeds = [inputs.get_token_vec(a, self.max_anno_vector_size) for a, _ in self.annos]
        anno_masks = [inputs.get_token_vec(m) for _, m in self.annos]
        anno_embed = self.embed_anno(self.embed_anno_separable(anno_embeds, anno_masks))

        # [batch_size, num_tokens, max_word_vector_size]
        word_vector = inputs.get_token_vec("word_vector", self.max_word_vector_size)

        # [batch_size, num_tokens, hidden_size] (cat)
        # [batch_size, num_tokens, seq_embed_size] (cat)
        # [batch_size, num_tokens, token_meta_embed_size * len(max_token_meta_vals)] (cat)
        # [batch_size, num_tokens, max_word_vector_size]
        # [batch_size, num_tokens, anno_embed_size] →
        # [batch_size, num_tokens, *]
        embeds = [tokens, seq_embed_expanded, *token_meta, word_vector, anno_embed]
        tokens = torch.cat(embeds, dim=2)

        # [batch_size, num_tokens, *] →
        # [batch_size, num_tokens, hidden_size]
        tokens: torch.Tensor = self.embed(tokens)
        tokens_mask = tokens_mask.unsqueeze(2)
        tokens = tokens.masked_fill(~tokens_mask, 0)
        conditional = tokens.clone()
        # [num_tokens, batch_size, hidden_size] (cat)
        # [num_tokens, batch_size, anno_embed_size] →
        # [num_tokens, batch_size, hidden_size + anno_embed_size] →
        conditional = torch.cat((conditional, anno_embed), dim=2)

        # [batch_size, num_tokens, *] →
        # [batch_size, in_channels (*), sequence_length (num_tokens)]
        tokens = tokens.transpose(1, 2)
        tokens_mask = tokens_mask.transpose(1, 2)
        conditional = conditional.transpose(1, 2)

        for conv, norm in zip(self.conv_layers, self.norm_layers):
            tokens = tokens.masked_fill(~tokens_mask, 0)
            tokens = norm(tokens + conv(tokens))

        # [batch_size, hidden_size, num_tokens] →
        # [seq_len (num_tokens), batch_size, input_size (hidden_size)]
        tokens = tokens.permute(2, 0, 1)
        tokens_mask = tokens_mask.permute(2, 0, 1)
        conditional = conditional.permute(2, 0, 1)

        # [num_tokens, batch_size, hidden_size] (cat)
        # [num_tokens, batch_size, hidden_size + anno_embed_size] →
        # [num_tokens, batch_size, hidden_size * 2 + anno_embed_size]
        juncture = torch.cat((tokens, conditional), dim=2)

        # [num_tokens, batch_size, hidden_size * 2] →
        # [num_tokens, batch_size, hidden_size] →
        update = self.lstm(self.lstm_dropout(juncture), tokens_mask, num_tokens)
        tokens = self.lstm_norm(tokens + update)

        # [num_tokens, batch_size, hidden_size] (cat)
        # [num_tokens, batch_size, hidden_size] →
        # [num_tokens, batch_size, hidden_size * 2]
        tokens = torch.cat((tokens, conditional), dim=2)

        # [num_tokens, batch_size, hidden_size * 2 + anno_embed_size] →
        # [num_tokens, batch_size, out_size]
        tokens = self.project_out(tokens).masked_fill(~tokens_mask, 0)

        # [num_tokens, batch_size, out_size] → [batch_size, num_tokens, out_size]
        tokens = tokens.transpose(0, 1)

        # [num_tokens, batch_size, 1] → [batch_size, num_tokens]
        tokens_mask = tokens_mask.squeeze(2).transpose(0, 1)
        tokens = pad_sequence([tokens[i][s] for i, s in enumerate(inputs.slices)])
        return Encoded(tokens, inputs.sliced_tokens_mask, inputs.num_sliced_tokens, seq_embed)
