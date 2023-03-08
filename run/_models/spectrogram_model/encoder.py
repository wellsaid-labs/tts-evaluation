import functools
import math
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
    def __init__(self, hidden_size: int, cond_size: int, conv_filter_size: int):
        super().__init__()
        self.norm = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.lstm = LSTM(hidden_size + cond_size, hidden_size)
        self.conv = torch.nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=conv_filter_size,
            padding=int((conv_filter_size - 1) / 2),
        )
        self.ff_norm = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.ff = cf.partial(_FeedForward)(hidden_size + cond_size, hidden_size)

    def __call__(
        self, tokens: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(tokens, mask, cond)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens (torch.FloatTensor [batch_size, num_tokens, hidden_size])
            mask (torch.FloatTensor [batch_size, num_tokens, 1])
            cond (torch.FloatTensor [batch_size, num_tokens, hidden_size])
        """
        # [batch_size, num_tokens] → [batch_size]
        block = self.norm(tokens)
        block = torch.cat([block, cond], dim=2)
        block = self.lstm(block.transpose(0, 1))[0].transpose(0, 1)
        block = block.masked_fill(~mask, 0)
        block = self.conv(block.transpose(1, 2)).transpose(1, 2)
        tokens = (tokens + block) / math.sqrt(2)
        next_block = self.ff_norm(tokens)
        next_block = torch.cat([next_block, cond], dim=2)
        next_block = self.ff(next_block)
        return (tokens + next_block) / math.sqrt(2)


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
        out_size: The size of the encoder output.
        hidden_size: The size of the encoders hidden representation.
        cond_size: The size of the encoder conditional representation.
        num_layers: Number of layers for processing input.
        conv_filter_size: Size of the convolving kernel. This value must be odd.
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
        cond_size: int,
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

        modules = (NumeralizePadEmbed(n, embedding_dim=cond_size) for n in max_seq_meta_vals)
        self.embed_seq_meta = ModuleList(modules)
        self.embed_seq_vector = torch.nn.Linear(max_seq_vector_size, cond_size)
        self.norm_seq_embed = cf.partial(torch.nn.LayerNorm)(cond_size)

        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        modules = (NumeralizePadEmbed(n, embedding_dim=hidden_size) for n in max_token_meta_vals)
        self.embed_token_meta = ModuleList(modules)
        self.embed_word_vec = torch.nn.Linear(self.max_word_vector_size, hidden_size)
        self.norm_embed = cf.partial(torch.nn.LayerNorm)(hidden_size)

        self.embed_annos = torch.nn.Conv1d(
            in_channels=max_anno_vector_size * len(self.annos),
            out_channels=cond_size * len(self.annos),
            kernel_size=1,
            groups=len(self.annos),
        )
        self.norm_cond = cf.partial(torch.nn.LayerNorm)(cond_size)

        blocks = (_Block(hidden_size, cond_size, conv_filter_size) for _ in range(num_layers))
        self.blocks = ModuleList(blocks)
        self.out = cf.partial(torch.nn.LayerNorm)(hidden_size)

    def __call__(self, inputs: Inputs) -> Encoded:
        return super().__call__(inputs)

    def forward(self, inputs: Inputs) -> Encoded:
        # [batch_size, num_tokens] →
        # tokens [batch_size, num_tokens, hidden_size]
        # tokens_mask [batch_size, num_tokens]
        tokens, tokens_mask = self.embed_token(inputs.tokens, batch_first=True)

        # [batch_size] → [batch_size, cond_size]
        iter_ = zip(self.embed_seq_meta, inputs.seq_meta_transposed)
        seq_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]
        # [batch_size, max_seq_vector_size] → [batch_size, cond_size]
        seq_vector = self.embed_seq_vector(inputs.get_seq_vec(self.max_seq_vector_size))
        seq_embed = self.dropout(torch.stack(seq_meta + [seq_vector]))
        # [len(max_seq_meta_vals) + 1, batch_size, cond_size] →
        # [batch_size, cond_size]
        seq_embed = self.norm_seq_embed(seq_embed.sum(dim=0))
        # [batch_size, cond_size] → [batch_size, num_tokens, cond_size]
        seq_embed_expanded = seq_embed.unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens] → [batch_size, num_tokens, hidden_size]
        iter_ = zip(self.embed_token_meta, inputs.token_meta_transposed)
        token_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]

        # [batch_size, num_tokens, hidden_size]
        word_vector = inputs.get_token_vec("word_vector", self.max_word_vector_size)
        word_vector = self.embed_word_vec(word_vector)

        feats = [tokens, word_vector] + token_meta
        tokens = self.norm_embed(torch.stack(feats).sum(dim=0))
        tokens_mask = tokens_mask.unsqueeze(2)
        tokens = tokens.masked_fill(~tokens_mask, 0)

        # [batch_size, num_tokens, max_anno_vector_size] →
        # [batch_size, num_tokens, cond_size]
        anno_vecs = [inputs.get_token_vec(a, self.max_anno_vector_size) for a, _ in self.annos]
        anno_vecs = torch.cat(anno_vecs, dim=2).transpose(1, 2)
        anno_embeds = self.embed_annos(anno_vecs).transpose(1, 2).chunk(len(self.annos), dim=2)
        anno_masks = [inputs.get_token_vec(m) for _, m in self.annos]
        anno_embeds = [e.masked_fill(~m.bool(), 0) for e, m in zip(anno_embeds, anno_masks)]

        # [batch_size, num_tokens, cond_size]
        cond = self.norm_cond(torch.stack(anno_embeds + [seq_embed_expanded]).sum(dim=0))
        cond = cond.masked_fill(~tokens_mask, 0)

        for block in self.blocks:
            tokens = block(tokens, tokens_mask, cond)

        tokens = self.out(tokens).masked_fill(~tokens_mask, 0)
        tokens = pad_sequence([tokens[i][s] for i, s in enumerate(inputs.slices)])
        return Encoded(tokens, inputs.sliced_tokens_mask, inputs.num_sliced_tokens, seq_embed)
