import functools
import typing

import config as cf
import torch
import torch.nn
from torch.nn import ModuleList
from torch.nn.utils.rnn import pad_sequence

from lib.distributed import NumeralizePadEmbed
from lib.utils import LSTM
from run._models.spectrogram_model.containers import Encoded
from run._models.spectrogram_model.inputs import Inputs


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
        tokens = self.norm(tokens)
        tokens = torch.cat([tokens, cond], dim=2)
        tokens = self.lstm(tokens.transpose(0, 1))[0].transpose(0, 1)
        tokens = tokens.masked_fill(~mask, 0)
        tokens = self.conv(tokens.transpose(1, 2)).transpose(1, 2)
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
        num_anno_layers: The number of layers to process annotations with.
        seq_embed_size: The size of the sequence metadata embeddings.
        token_meta_embed_size: The size of the token metadata embeddings.
        anno_embed_size: The size of the annotation embeddings.
        out_size: The size of the encoder output.
        hidden_size: The size of the encoders hidden representation. This value must be even.
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
        num_layers: int,
        conv_filter_size: int,
    ):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert conv_filter_size % 2 == 1, "`conv_filter_size` must be odd"
        assert hidden_size % 2 == 0, "`hidden_size` must be even"

        self.max_word_vector_size = max_word_vector_size
        self.max_anno_vector_size = max_anno_vector_size
        self.max_seq_vector_size = max_seq_vector_size
        self.annos = annos
        self.hidden_size = hidden_size
        embed = functools.partial(NumeralizePadEmbed, embedding_dim=hidden_size)

        self.embed_seq_meta = ModuleList(embed(n) for n in max_seq_meta_vals)
        self.embed_seq_vector = torch.nn.Linear(max_seq_vector_size, hidden_size)
        self.norm_seq_embed = cf.partial(torch.nn.LayerNorm)(hidden_size)

        self.embed_token_meta = ModuleList(embed(n) for n in max_token_meta_vals)
        self.embed_token = NumeralizePadEmbed(max_tokens, hidden_size)
        self.embed_annos = ModuleList(
            torch.nn.Linear(max_anno_vector_size, hidden_size) for _ in range(len(self.annos))
        )

        self.norm_embed = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.blocks = ModuleList(
            _Block(hidden_size, hidden_size, conv_filter_size) for _ in range(num_layers)
        )
        self.mlp_block_norm = cf.partial(torch.nn.LayerNorm)(hidden_size)
        self.mlp_block = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size * 4),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 4, hidden_size),
        )
        self.out = cf.partial(torch.nn.LayerNorm)(hidden_size)

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
        # [batch_size, hidden_size]
        seq_embed = self.norm_seq_embed(seq_embed.sum(dim=0))
        # [batch_size, hidden_size] → [batch_size, num_tokens, hidden_size]
        seq_embed_expanded = seq_embed.unsqueeze(1).expand(-1, tokens.shape[1], -1)

        # [batch_size, num_tokens] → [batch_size, num_tokens, hidden_size]
        iter_ = zip(self.embed_token_meta, inputs.token_meta_transposed)
        token_meta = [embed(meta, batch_first=True)[0] for embed, meta in iter_]

        # [batch_size, num_tokens, max_anno_vector_size] →
        # [batch_size, num_tokens, hidden_size]
        anno_vecs = [inputs.get_token_vec(a, self.max_anno_vector_size) for a, _ in self.annos]
        anno_embeds = [e(v) for v, e in zip(anno_vecs, self.embed_annos)]
        anno_masks = [inputs.get_token_vec(m) for _, m in self.annos]
        anno_embeds = [e.masked_fill(~m.bool(), 0) for e, m in zip(anno_embeds, anno_masks)]

        # [batch_size, num_tokens, hidden_size]
        assert self.hidden_size >= self.max_word_vector_size
        word_vector = inputs.get_token_vec("word_vector", self.hidden_size)

        feats = [tokens, word_vector] + anno_embeds + token_meta
        tokens = self.norm_embed(torch.stack(feats).sum(dim=0))
        tokens_mask = tokens_mask.unsqueeze(2)
        tokens = tokens.masked_fill(~tokens_mask, 0)

        for block in self.blocks:
            tokens = tokens + block(tokens, tokens_mask, seq_embed_expanded)

        # [batch_size, num_tokens, hidden_size * 2 + anno_embed_size] →
        # [batch_size, num_tokens, out_size]
        mlp_block_in = self.mlp_block_norm(tokens)
        mlp_block_in = torch.cat([tokens, seq_embed_expanded], dim=2)
        tokens = tokens + self.mlp_block(mlp_block_in).masked_fill(~tokens_mask, 0)
        tokens = self.out(tokens)

        tokens = pad_sequence([tokens[i][s] for i, s in enumerate(inputs.slices)])
        return Encoded(tokens, inputs.sliced_tokens_mask, inputs.num_sliced_tokens, seq_embed)
