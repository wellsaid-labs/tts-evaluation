import functools
import math
import typing

import config as cf
import torch
import torch.nn
import torch.nn.functional
from torch.nn import Mish, ModuleList, Sequential
from torch.nn.utils.rnn import pad_sequence

from lib.distributed import NumeralizePadEmbed
from lib.utils import LSTM
from run._models.spectrogram_model.containers import Encoded
from run._models.spectrogram_model.inputs import Inputs


class _Convs(torch.nn.Module):
    """
    A layer of stacked convolutions and non-linearities.
    """

    def __init__(self, hidden_size: int, kernel_size: int, num_layers: int):
        super().__init__()
        conv = lambda: torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.block = ModuleList(Sequential(Mish(), conv()) for _ in range(num_layers))
        last_op = Sequential(Mish(), torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1))
        self.block.append(last_op)

    def __call__(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().__call__(tokens, mask)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, hidden_size])
        """
        block = tokens.transpose(1, 2)
        mask = ~mask.transpose(1, 2)
        for conv in self.block:
            block = conv(block.masked_fill(mask, 0))
        return block.transpose(1, 2)


class _Highway(torch.nn.Module):
    """A Highway module similar to the one used in ELMo.

    Learn more here:
    - Highway Network: https://arxiv.org/abs/1505.00387
    - ELMo: https://arxiv.org/pdf/1802.05365.pdf
    - ELMo Highway Implementation:
      https://github.com/allenai/allennlp/blob/main/allennlp/modules/highway.py
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.highway = torch.nn.Linear(hidden_size, hidden_size * 2)
        self.act = Mish()
        self.gate_act = torch.nn.Sigmoid()

    def __call__(self, tokens: torch.Tensor):
        return super().__call__(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, hidden_size])
        """
        vals, gate = self.highway(tokens).chunk(2, dim=-1)
        gate = self.gate_act(gate)
        return self.act(vals) * gate + tokens * (1 - gate)


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
        hidden_size: The size of the encoders hidden representation.
        num_layers: Number of layers for processing input.
        conv_filter_size: Size of the convolving kernel. This value must be odd.
        num_conv_block_layers: The number of layers in each convolution block.
    """

    def __init__(
        self,
        max_tokens: int,
        max_seq_meta_vals: typing.Tuple[int, ...],
        max_token_meta_vals: typing.Tuple[int, ...],
        max_word_vector_size: int,
        max_seq_vector_size: int,
        max_anno_vector_size: int,
        annos: typing.Sequence[typing.Tuple[str, str]],
        hidden_size: int,
        num_layers: int,
        conv_filter_size: int,
        num_conv_block_layers: int,
    ):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert conv_filter_size % 2 == 1, "`conv_filter_size` must be odd"

        self.max_word_vector_size = max_word_vector_size
        self.max_anno_vector_size = max_anno_vector_size
        self.max_seq_vector_size = max_seq_vector_size
        self.annos = annos
        self.hidden_size = hidden_size

        embed = functools.partial(NumeralizePadEmbed, embedding_dim=hidden_size)
        self.embed_seq_meta = ModuleList(embed(n) for n in max_seq_meta_vals)
        self.embed_seq_vector = torch.nn.Linear(max_seq_vector_size, hidden_size)
        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        self.embed_token_meta = ModuleList(embed(n) for n in max_token_meta_vals)
        self.embed_word_vec = torch.nn.Sequential(
            torch.nn.Linear(self.max_word_vector_size, hidden_size),
            cf.partial(torch.nn.LayerNorm)(hidden_size),
        )
        self.embed_annos = torch.nn.Conv1d(
            in_channels=max_anno_vector_size * len(self.annos),
            out_channels=hidden_size * len(self.annos),
            kernel_size=1,
            groups=len(self.annos),
        )
        self.embed_anno_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Mish(),
            torch.nn.Linear(hidden_size, hidden_size * 6),
        )
        self.scale_tok = math.sqrt(len(self.embed_token_meta) + len([self.embed_token]))
        self.scale_anno = len(self.annos) + len([self.embed_seq_vector]) + len(self.embed_seq_meta)
        self.scale_anno = math.sqrt(self.scale_anno)

        conv = lambda: _Convs(hidden_size, conv_filter_size, num_conv_block_layers)
        rnn = lambda: LSTM(hidden_size, hidden_size, batch_first=True)
        norm = lambda: cf.partial(torch.nn.LayerNorm)(hidden_size)

        self.convs = ModuleList(conv() for _ in range(num_layers))
        self.highways = ModuleList(_Highway(hidden_size) for _ in range(num_layers))
        self.rnns = ModuleList(rnn() for _ in range(num_layers))
        self.residual = math.sqrt(2)

        num_blocks = len([self.convs, self.rnns, self.embed_token])
        self.norms = ModuleList(norm() for _ in range(num_blocks))

        self.proj_out = torch.nn.Linear(hidden_size, hidden_size * 2)

    def __call__(self, inputs: Inputs) -> Encoded:
        return super().__call__(inputs)

    def forward(self, inputs: Inputs) -> Encoded:
        # [batch_size, num_tokens] →
        # tokens [batch_size, num_tokens, hidden_size]
        # tokens_mask [batch_size, num_tokens]
        tokens, tokens_mask = self.embed_token(inputs.tokens, batch_first=True)

        # [batch_size, num_tokens] → [batch_size, num_tokens, hidden_size]
        iter_ = zip(self.embed_token_meta, inputs.token_meta_transposed)
        token_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]

        # [batch_size, num_tokens, hidden_size]
        word_vector = inputs.get_token_vec("word_vector", self.max_word_vector_size)

        # [batch_size] → [batch_size, hidden_size]
        iter_ = zip(self.embed_seq_meta, inputs.seq_meta_transposed)
        seq_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]
        # [len(max_seq_meta_vals), batch_size, hidden_size] → [batch_size, num_tokens, hidden_size]
        seq_embed = torch.stack(seq_meta).sum(dim=0).unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, max_seq_vector_size] → [batch_size, num_tokens, hidden_size]
        seq_vector = self.embed_seq_vector(inputs.get_seq_vec(self.max_seq_vector_size))
        seq_vector = seq_vector.unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens, max_anno_vector_size] →
        # [batch_size, num_tokens, hidden_size * 2]
        anno_vecs = [inputs.get_token_vec(a, self.max_anno_vector_size) for a, _ in self.annos]
        anno_vecs = torch.cat(anno_vecs, dim=2).transpose(1, 2)
        anno_embeds = self.embed_annos(anno_vecs).transpose(1, 2).chunk(len(self.annos), dim=2)
        anno_masks = [inputs.get_token_vec(m) for _, m in self.annos]
        anno_embeds = [e.masked_fill(~m.bool(), 0) for e, m in zip(anno_embeds, anno_masks)]
        anno_embeds.extend([seq_vector, seq_embed])
        anno_embed = torch.stack(anno_embeds).sum(dim=0) / self.scale_anno
        anno_embeds = self.embed_anno_net(anno_embed).chunk(6, dim=2)

        tokens = torch.stack([tokens] + token_meta).sum(dim=0) / self.scale_tok
        # NOTE: Add together the various embedding networks, annotation embeddings, word vector
        # embeddings, and token embeddings.
        tokens = self.norms[0](tokens) + self.embed_word_vec(word_vector) + anno_embeds[1]
        tokens = tokens * anno_embeds[0]
        tokens_mask = tokens_mask.unsqueeze(2)
        tokens: torch.Tensor = tokens.masked_fill(~tokens_mask, 0)

        # TODO: Much like our attention relies on learned token padding, we could upstream it
        # further to our convolutions.

        # tokens [batch_size, num_tokens, hidden_size]
        for convs, highway in zip(self.convs, self.highways):
            tokens = (tokens + convs(tokens, tokens_mask)) / self.residual
            tokens = highway(tokens)

        tokens = (self.norms[1](tokens) + anno_embeds[3]) * anno_embeds[2]
        for rnn in self.rnns:
            tokens = (tokens + rnn(tokens)[0]) / self.residual

        tokens = (self.norms[2](tokens) + anno_embeds[5]) * anno_embeds[4]
        tokens = self.proj_out(tokens).masked_fill(~tokens_mask, 0)
        tokens = pad_sequence([tokens[i][s] for i, s in enumerate(inputs.slices)])
        tokens, token_keys = tokens.chunk(2, dim=2)

        return Encoded(
            # [num_tokens, batch_size, out_dim] → [batch_size, num_tokens, out_dim]
            tokens=tokens.transpose(0, 1),
            # [num_tokens, batch_size, out_dim] → [batch_size, out_dim, num_tokens]
            token_keys=token_keys.permute(1, 2, 0),
            tokens_mask=inputs.sliced_tokens_mask,
            num_tokens=inputs.num_sliced_tokens,
        )
