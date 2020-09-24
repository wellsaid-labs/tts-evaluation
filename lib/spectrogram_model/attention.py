from functools import lru_cache

import logging
import math
import typing

import torch

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.nn import LockedDropout

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _window_helper(length: int, dimension: int, num_dimensions: int,
                   device: torch.device) -> typing.Tuple[torch.Tensor, int]:
    """ Helper to ensure `indices` and `dimension` are not recalculated unnecessarily. """
    with torch.no_grad():
        dimension = num_dimensions + dimension if dimension < 0 else dimension
        indices = torch.arange(0, length, device=device)
        indices_shape = [1] * dimension + [-1] + [1] * (num_dimensions - dimension - 1)
        indices = indices.view(*tuple(indices_shape))
    return indices, dimension


def _window(tensor: torch.Tensor,
            start: torch.Tensor,
            length: int,
            dim: int,
            check_invariants: bool = True) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """ Return a window of `tensor` with variable window offsets.

    Args:
        tensor (torch.Tensor [*, dim_length, *]): The tensor to window.
        start (torch.Tensor [*]): The start of a window.
        length: The length of the window.
        dim: The `tensor` dimension to window.
        check_invariants: If `True`, then check invariants via asserts.

    Returns:
        tensor (torch.Tensor [*, length, *]): The windowed tensor.
        tensor (torch.Tensor [*, length, *]): The indices used to `gather` the window and that can
            be used with `scatter` to reverse the operation.
    """
    if check_invariants:
        assert start.min() >= 0, 'The window `start` must be positive.'
        assert length <= tensor.shape[dim], 'The `length` is larger than the `tensor`.'
        assert start.max() + length <= tensor.shape[
            dim], 'The window `start` must smaller or equal to than `tensor.shape[dim] - length`.'

    indices, dim = _window_helper(length, dim, tensor.dim(), tensor.device)
    indices_shape = list(tensor.shape)
    indices_shape[dim] = length
    indices = indices.detach().expand(*tuple(indices_shape))
    start = start.unsqueeze(dim)

    if check_invariants:
        assert start.dim() == tensor.dim(
        ), 'The `start` tensor must be the same size as `tensor` without the `dim` dimension.'

    indices = indices + start
    return torch.gather(tensor, dim, indices), indices


class LocationRelativeAttention(nn.Module):
    """ Query using the Bahdanau attention mechanism given location features.

    Reference:
        - Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
        - Attention-Based Models for Speech Recognition:
          https://arxiv.org/pdf/1506.07503.pdf
        - Location-Relative Attention Mechanisms For Robust Long-Form Speech Synthesis
          https://arxiv.org/abs/1910.10288

    Args:
        query_hidden_size: The hidden size of the query input.
        hidden_size: The hidden size of the hidden attention features.
        convolution_filter_size: Size of the convolving kernel applied to the cumulative alignment.
        dropout: The dropout probability.
        window_length: The size of the attention window applied during inference.
    """

    @configurable
    def __init__(self,
                 query_hidden_size: int,
                 hidden_size: int = HParam(),
                 convolution_filter_size: int = HParam(),
                 dropout: float = HParam(),
                 window_length: int = HParam()):
        super().__init__()
        # Learn more:
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

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        query: torch.Tensor,
        cumulative_alignment: typing.Optional[torch.Tensor] = None,
        initial_cumulative_alignment: typing.Optional[torch.Tensor] = None,
        window_start: typing.Optional[torch.Tensor] = None,
        token_skip_warning: int = 2
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        NOTE: Attention alignment is sometimes refered to as attention weights.

        Args:
            tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batch of sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Sequence mask(s) to deliminate
                padding in `tokens` with zeros.
            num_tokens (torch.LongTensor [batch_size]): Number of tokens in each sequence.
            query (torch.FloatTensor [1, batch_size, query_hidden_size]): Attention query.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens], optional): Cumulation
                of alignments from other queries. If `None` then this defaults to the zero vector.
            initial_cumulative_alignment (torch.FloatTensor [batch_size, 1]): The
                `cumulative_alignment` vector has a non-zero value for every token that is has
                attended to. Assuming the model is attending to tokens from left-to-right
                and the model starts reading at the first token, then any padding to the left
                of the first token should be non-zero to be consistent with `cumulative_alignment`.
                `initial_cumulative_alignment` is the value of the padding on the left-side of
                `cumulative_alignment`.
            window_start (torch.LongTensor [batch_size]): The start of the attention window.
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a warning will be thrown.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size]): Attention context vector.
            cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): Updated
                `cumulative_alignment`.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Attention alignment.
            window_start (torch.LongTensor [batch_size]): Updated `window_start`.
        """
        max_num_tokens, batch_size, _ = tokens.shape
        device = tokens.device
        window_length = min(self.window_length, max_num_tokens)

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
            location_features, [0, self.alignment_conv_padding], mode='constant', value=0.0)

        tokens_mask, window_indices = _window(tokens_mask, window_start, window_length, 1, False)
        tokens = _window(tokens, window_start.unsqueeze(1), window_length, 0, False)[0]
        location_features = _window(location_features, window_start.unsqueeze(1),
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

        # NOTE: Transpose and unsqueeze to fit the requirements for `torch.bmm`
        # [num_tokens, batch_size, hidden_size] →
        # [batch_size, num_tokens, hidden_size]
        tokens = tokens.transpose(0, 1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # tokens [batch_size (b), num_tokens (m), hidden_size (p)] →
        # [batch_size (b), 1 (n), hidden_size (p)]
        context = torch.bmm(alignment.unsqueeze(1), tokens)

        # [batch_size, 1, hidden_size] → [batch_size, hidden_size]
        context = context.squeeze(1)

        alignment = torch.zeros(
            batch_size, max_num_tokens, device=device).scatter_(1, window_indices, alignment)

        last_window_start = window_start
        window_start = alignment.max(dim=1)[1] - window_length // 2
        # TODO: Cache `num_tokens - window_length` clamped at 0 so that we dont need to
        # recompute the `clamp` and subtraction each time.
        window_start = torch.clamp(torch.min(window_start, num_tokens - window_length), min=0)
        max_tokens_skipped = (window_start - last_window_start).max()
        if max_tokens_skipped > token_skip_warning:
            logger.warning('Attention module skipped %d tokens.', max_tokens_skipped)

        # [batch_size, num_tokens] + [batch_size, num_tokens] → [batch_size, num_tokens]
        cumulative_alignment = cumulative_alignment + alignment

        return context, cumulative_alignment, alignment, window_start
