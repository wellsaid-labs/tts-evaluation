import functools
import typing

import config as cf
import torch
import torch.nn
import torch.nn.functional
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

from lib.distributed import NumeralizePadEmbed
from lib.utils import LSTM
from run._models.spectrogram_model.containers import Encoded
from run._models.spectrogram_model.inputs import Inputs


class _FeedForward(torch.nn.Module):
    """A feed forward layer as defined for transformers.

    NOTE: This SwiGLU layer is based on:
    https://github.com/facebookresearch/llama/blob/main/llama/model.py
    """

    def __init__(self, in_size: int, out_size: int, in_size_mult: int):
        super().__init__()
        self.proj = torch.nn.Linear(in_size, in_size * in_size_mult * 2)
        self.out = torch.nn.Linear(in_size * in_size_mult, out_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.FloatTensor [*, in_size])

        Returns:
            x (torch.FloatTensor [*, out_size])
        """
        gate, val = self.proj(x).chunk(2, dim=-1)
        return self.out(torch.nn.functional.silu(gate) * val)


class _Block(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.lstm = LSTM(hidden_size, hidden_size, batch_first=True)
        self.ff_norm = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.ff = cf.partial(_FeedForward)(hidden_size, hidden_size)

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        return super().__call__(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, hidden_size])
        """
        tokens = tokens + self.lstm(self.norm(tokens))[0]
        tokens = tokens + self.ff(self.ff_norm(tokens))
        return tokens


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
        dropout: The probability of dropout.
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
        dropout: float,
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

        self.dropout = torch.nn.Dropout(dropout)

        embed = functools.partial(NumeralizePadEmbed, embedding_dim=hidden_size)
        self.embed_seq_meta = ModuleList(embed(n) for n in max_seq_meta_vals)
        self.embed_seq_vector = torch.nn.Linear(max_seq_vector_size, hidden_size)
        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        self.embed_token_meta = ModuleList(embed(n) for n in max_token_meta_vals)
        self.embed_word_vec = torch.nn.Linear(self.max_word_vector_size, hidden_size)
        self.norm_embed = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.embed_annos = torch.nn.Conv1d(
            in_channels=max_anno_vector_size * len(self.annos),
            out_channels=hidden_size * len(self.annos),
            kernel_size=1,
            groups=len(self.annos),
        )

        self.pre_net = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.GELU(),
                torch.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=conv_filter_size,
                    padding=(conv_filter_size - 1) // 2,
                ),
            )
            for _ in range(num_layers)
        )
        self.blocks = ModuleList(_Block(hidden_size) for _ in range(num_layers))
        self.out = torch.nn.Sequential(
            cf.partial(torch.nn.LayerNorm)(hidden_size),
            torch.nn.Linear(hidden_size, hidden_size * 2),
        )

    def __call__(self, inputs: Inputs) -> Encoded:
        return super().__call__(inputs)

    def forward(self, inputs: Inputs) -> Encoded:
        # [batch_size, num_tokens] →
        # tokens [batch_size, num_tokens, hidden_size]
        # tokens_mask [batch_size, num_tokens]
        tokens, tokens_mask = self.embed_token(inputs.tokens, batch_first=True)

        # [batch_size] → [batch_size, hidden_size]
        iter_ = zip(self.embed_seq_meta, inputs.seq_meta_transposed)
        seq_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]
        # [batch_size, max_seq_vector_size] → [batch_size, hidden_size]
        seq_vector = self.embed_seq_vector(inputs.get_seq_vec(self.max_seq_vector_size))
        seq_embed = torch.stack(seq_meta + [seq_vector])
        # [len(max_seq_meta_vals) + 1, batch_size, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        seq_embed = seq_embed.sum(dim=0).unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens] → [batch_size, num_tokens, hidden_size]
        iter_ = zip(self.embed_token_meta, inputs.token_meta_transposed)
        token_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]

        # [batch_size, num_tokens, hidden_size]
        word_vector = inputs.get_token_vec("word_vector", self.max_word_vector_size)
        word_vector = self.embed_word_vec(word_vector)

        # [batch_size, num_tokens, max_anno_vector_size] →
        # [batch_size, num_tokens, hidden_size]
        anno_vecs = [inputs.get_token_vec(a, self.max_anno_vector_size) for a, _ in self.annos]
        anno_vecs = torch.cat(anno_vecs, dim=2).transpose(1, 2)
        anno_embeds = self.embed_annos(anno_vecs).transpose(1, 2).chunk(len(self.annos), dim=2)
        anno_masks = [inputs.get_token_vec(m) for _, m in self.annos]
        anno_embeds = [e.masked_fill(~m.bool(), 0) for e, m in zip(anno_embeds, anno_masks)]

        feats = [tokens, seq_embed, word_vector] + token_meta + anno_embeds
        tokens = self.norm_embed(self.dropout(torch.stack(feats)).sum(dim=0))
        tokens_mask = tokens_mask.unsqueeze(2)
        tokens: torch.Tensor = tokens.masked_fill(~tokens_mask, 0)

        # TODO: Much like our attention relies on learned token padding, we could upstream it
        # further to our convolutions.

        # [batch_size, num_tokens, hidden_size] →
        # [batch_size, hidden_size, num_tokens]
        block, block_mask = tokens.transpose(1, 2), tokens_mask.transpose(1, 2)
        for module in self.pre_net:
            block = module(block).masked_fill(~block_mask, 0)

        tokens = tokens + block.transpose(1, 2)

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.out(tokens).masked_fill(~tokens_mask, 0)
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
