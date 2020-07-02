from functools import lru_cache

import logging
import math

import torch

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.nn import LockedDropout

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _window_helper(length, dimension, num_dimensions, device):
    """ Helper to ensure `indices` and `dimension` are not recalculated unnecessarily. """
    with torch.no_grad():
        dimension = num_dimensions + dimension if dimension < 0 else dimension
        indices = torch.arange(0, length, device=device)
        indices_shape = [1] * dimension + [-1] + [1] * (num_dimensions - dimension - 1)
        indices = indices.view(*tuple(indices_shape))
    return indices, dimension


def window(tensor, start, length, dim, check_invariant=True):
    """ Return a window of `tensor` with variable window offsets.

    Args:
        tensor (torch.Tensor [*, dim_length, *]): The tensor to window.
        start (torch.Tensor [*]): The start of a window.
        length (int): The length of the window.
        dim (int): The `tensor` dimension to window.
        check_invariant (bool, optional): If `True`, then check invariants via asserts.

    Returns:
        tensor (torch.Tensor [*, length, *]): The windowed tensor.
        tensor (torch.Tensor [*, length, *]): The indices used to `gather` the window and that can
            be used with `scatter` to reverse the operation.
    """
    if check_invariant:
        assert start.min() >= 0, 'The window `start` must be positive.'
        assert length <= tensor.shape[dim], 'The `length` is larger than the `tensor`.'
        assert start.max() + length <= tensor.shape[
            dim], 'The window `start` must smaller or equal to than `tensor.shape[dim] - length`.'

    indices, dim = _window_helper(length, dim, tensor.dim(), tensor.device)
    indices_shape = list(tensor.shape)
    indices_shape[dim] = length
    indices = indices.detach().expand(*tuple(indices_shape))
    start = start.unsqueeze(dim)

    if check_invariant:
        assert start.dim() == tensor.dim(
        ), 'The `start` tensor must be the same size as `tensor` without the `dim` dimension.'

    indices = indices + start
    return torch.gather(tensor, dim, indices), indices


class LocationSensitiveAttention(nn.Module):
    """ Query using the Bahdanau attention mechanism with additional location features.

    SOURCE (Tacotron 2):
        The encoder output is consumed by an attention network which summarizes the full encoded
        sequence as a fixed-length context vector for each decoder output step. We use the
        location-sensitive attention from [21], which extends the additive attention mechanism
        [22] to use cumulative attention weights from previous decoder time steps as an additional
        feature. This encourages the model to move forward consistently through the input,
        mitigating potential failure modes where some subsequences are repeated or ignored by the
        decoder. Attention probabilities are computed after projecting inputs and location
        features to 128-dimensional hidden representations. Location features are computed using
        32 1-D convolution filters of length 31.

    SOURCE (Attention-Based Models for Speech Recognition):
        This is achieved by adding as inputs to the attention mechanism auxiliary convolutional
        features which are extracted by convolving the attention weights from the previous step
        with trainable filters.

        ...

        We extend this content-based attention mechanism of the original model to be location-aware
        by making it take into account the alignment produced at the previous step. First, we
        extract k vectors fi,j ∈ R^k for every position j of the previous alignment αi−1 by
        convolving it with a matrix F ∈ R^k×r.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
        * Attention-Based Models for Speech Recognition
          https://arxiv.org/pdf/1506.07503.pdf

    Args:
        query_hidden_size (int): The hidden size of the query input.
        hidden_size (int): The hidden size of the hidden attention features.
        convolution_filter_size (int): Size of the convolving kernel applied to the cumulative
            alignment.
        dropout (float): The dropout probability.
        window_length (int): The size of the attention window applied during inference.
    """

    @configurable
    def __init__(self,
                 query_hidden_size,
                 hidden_size=HParam(),
                 convolution_filter_size=HParam(),
                 dropout=HParam(),
                 window_length=HParam()):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, '`convolution_filter_size` must be odd'

        self.dropout = dropout
        self.hidden_size = hidden_size
        self.window_length = window_length

        self.alignment_conv_padding = int((convolution_filter_size - 1) / 2)
        self.alignment_conv = nn.Conv1d(
            in_channels=1, out_channels=hidden_size, kernel_size=convolution_filter_size, padding=0)

        self.project_query = nn.Linear(query_hidden_size, hidden_size)
        self.project_scores = nn.Sequential(
            LockedDropout(dropout), nn.Linear(hidden_size, 1, bias=False))

    def forward(self,
                encoded_tokens,
                tokens_mask,
                num_tokens,
                query,
                cumulative_alignment=None,
                initial_cumulative_alignment=None,
                window_start=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batched set of
                encoded sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask where zero's
                represent padding in ``encoded_tokens``.
            num_tokens (torch.LongTensor [batch_size]): The number of tokens in each sequence.
            query (torch.FloatTensor [1, batch_size, query_hidden_size]): Query vector used to score
                individual token vectors.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens], optional): Cumlative
                attention alignment from the last queries. If this vector is not included, the
                ``cumulative_alignment`` defaults to a zero vector.
            initial_cumulative_alignment (torch.FloatTensor [batch_size, 1]): The left-side
                padding value for the `alignment_conv`. This can also be interpreted as the
                cumulative alignment for the former tokens.
            window_start (torch.LongTensor [batch_size]): The start of the attention window.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size]): Computed attention
                context vector.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): Cumlative attention
                alignment vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Computed attention alignment
                vector.
            window_start (torch.LongTensor [batch_size] or None): The updated window starting
                location.
        """
        max_num_tokens, batch_size, _ = encoded_tokens.shape
        device = encoded_tokens.device
        window_length = min(self.window_length, max_num_tokens)

        # NOTE: Attention alignment is sometimes refered to as attention weights.
        if cumulative_alignment is None:
            cumulative_alignment = torch.zeros(batch_size, max_num_tokens, device=device)
        if initial_cumulative_alignment is None:
            initial_cumulative_alignment = torch.zeros(batch_size, 1, device=device)
        if window_start is None:
            window_start = torch.zeros(batch_size, device=device, dtype=torch.long)

        cumulative_alignment = cumulative_alignment.masked_fill(~tokens_mask, 0)

        # [batch_size, num_tokens] → [batch_size, 1, num_tokens]
        location_features = cumulative_alignment.unsqueeze(1)

        # NOTE: Add `self.alignment_conv_padding` to both sides.
        initial_cumulative_alignment = initial_cumulative_alignment.unsqueeze(-1).expand(
            -1, -1, self.alignment_conv_padding)
        location_features = torch.cat([initial_cumulative_alignment, location_features], dim=-1)
        location_features = torch.nn.functional.pad(
            location_features, (0, self.alignment_conv_padding), mode='constant', value=0.0)

        tokens_mask, window_indices = window(tokens_mask, window_start, window_length, 1, False)
        encoded_tokens = window(encoded_tokens, window_start.unsqueeze(1), window_length, 0,
                                False)[0]
        location_features = window(location_features, window_start.unsqueeze(1),
                                   window_length + self.alignment_conv_padding * 2, 2, False)[0]

        # [batch_size, 1, num_tokens] → [batch_size, hidden_size, num_tokens]
        location_features = self.alignment_conv(location_features)

        # [1, batch_size, query_hidden_size] → [batch_size, hidden_size, 1]
        query = self.project_query(query).view(batch_size, self.hidden_size, 1)

        # [batch_size, hidden_size, num_tokens]
        location_features = torch.tanh(location_features + query)

        # [batch_size, hidden_size, num_tokens] →
        # [num_tokens, batch_size, hidden_size] →
        # [batch_size, num_tokens]
        score = self.project_scores(location_features.permute(2, 0, 1)).squeeze(2).transpose(0, 1)

        score.data.masked_fill_(~tokens_mask, -math.inf)

        # [batch_size, num_tokens] → [batch_size, num_tokens]
        alignment = torch.softmax(score, dim=1)

        # NOTE: Transpose and unsqueeze to fit the requirements for ``torch.bmm``
        # [num_tokens, batch_size, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        encoded_tokens = encoded_tokens.transpose(0, 1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # encoded_tokens [batch_size (b), num_tokens (m), hidden_size (p)] →
        # [batch_size (b), 1 (n), hidden_size (p)]
        context = torch.bmm(alignment.unsqueeze(1), encoded_tokens)

        # [batch_size, 1, hidden_size] → [batch_size, hidden_size]
        context = context.squeeze(1)

        alignment = torch.zeros(
            batch_size, max_num_tokens, device=device).scatter_(1, window_indices, alignment)
        window_start = alignment.max(dim=1)[1] - window_length // 2
        # TODO: Cache `num_tokens - window_length` clamped at 0 so that we dont need to
        # recompute the `clamp` and subtraction each time.
        window_start = torch.clamp(torch.min(window_start, num_tokens - window_length), min=0)

        # [batch_size, num_tokens] + [batch_size, num_tokens] → [batch_size, num_tokens]
        cumulative_alignment = cumulative_alignment + alignment

        return context, cumulative_alignment, alignment, window_start
