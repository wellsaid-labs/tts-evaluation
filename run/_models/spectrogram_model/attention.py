import logging
import math
import typing
from functools import lru_cache

import torch
import torch.nn
from torchnlp.nn import LockedDropout

from run._models.spectrogram_model.containers import AttentionHiddenState, Encoded

logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _window_helper(
    length: int, dimension: int, num_dimensions: int, device: torch.device
) -> typing.Tuple[torch.Tensor, int]:
    """Helper to ensure `indices` and `dimension` are not recalculated unnecessarily."""
    dimension = num_dimensions + dimension if dimension < 0 else dimension
    indices = torch.arange(0, length, device=device)
    indices_shape = [1] * dimension + [-1] + [1] * (num_dimensions - dimension - 1)
    indices = indices.view(*tuple(indices_shape))
    return indices, dimension


def _window(
    tensor: torch.Tensor,
    start: torch.Tensor,
    length: int,
    dim: int,
    check_invariants: bool = True,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Return a window of `tensor` with variable window offsets.

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
        assert start.min() >= 0, "The window `start` must be positive."
        assert length <= tensor.shape[dim], "The `length` is larger than the `tensor`."
        message = "The window `start` must smaller or equal to than `tensor.shape[dim] - length`."
        assert start.max() + length <= tensor.shape[dim], message

    indices, dim = _window_helper(length, dim, tensor.dim(), tensor.device)
    indices_shape = list(tensor.shape)
    indices_shape[dim] = length
    indices = indices.detach().expand(*tuple(indices_shape))
    start = start.unsqueeze(dim)

    if check_invariants:
        message = "`start` tensor must be the same size as `tensor` without the `dim` dimension."
        assert start.dim() == tensor.dim(), message

    indices = indices + start
    gather = torch.gather(tensor, dim, indices)

    if check_invariants:
        assert gather.shape[dim] == length
        assert indices.shape[dim] == length

    return gather, indices


class Attention(torch.nn.Module):
    """Query using the Bahdanau attention mechanism given location features.

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
        conv_filter_size: Size of the convolving kernel applied to the cumulative alignment.
        dropout: The dropout probability.
        window_length: The size of the attention window applied during inference.
    """

    def __init__(
        self,
        query_hidden_size: int,
        hidden_size: int,
        conv_filter_size: int,
        dropout: float,
        window_length: int,
        avg_frames_per_token: float,
    ):
        super().__init__()
        # Learn more:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert conv_filter_size % 2 == 1, "`conv_filter_size` must be odd"
        assert window_length % 2 == 1, "`window_length` must be odd"
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.window_length = window_length
        self.cum_alignment_padding = int((conv_filter_size - 1) / 2)
        self.alignment_conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=hidden_size,
            kernel_size=conv_filter_size,
            padding=0,
        )
        self.project_query = torch.nn.Linear(query_hidden_size, hidden_size)
        self.project_scores = torch.nn.Sequential(
            LockedDropout(dropout), torch.nn.Linear(hidden_size, 1, bias=False)
        )
        self.avg_frames_per_token = avg_frames_per_token

    def __call__(
        self,
        encoded: Encoded,
        query: torch.Tensor,
        hidden_state: AttentionHiddenState,
        token_skip_warning: float = math.inf,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, AttentionHiddenState]:
        return super().__call__(encoded, query, hidden_state, token_skip_warning)

    def forward(
        self,
        encoded: Encoded,
        query: torch.Tensor,
        hidden_state: AttentionHiddenState,
        token_skip_warning: float,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, AttentionHiddenState]:
        """
        Args:
            ...
            query (torch.FloatTensor [1, batch_size, query_hidden_size]): Attention query.
            hidden_state
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a `logger.warning` will be logged.

        Returns:
            context (torch.FloatTensor [batch_size, token_size]): Attention context vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Attention alignment.
            hidden_state
        """
        max_num_tokens, batch_size, _ = encoded.tokens.shape
        device = encoded.tokens.device
        cum_alignment_padding = self.cum_alignment_padding
        window_length = min(self.window_length, max_num_tokens)

        cum_alignment, window_start = hidden_state

        part = slice(cum_alignment_padding, -cum_alignment_padding)

        tokens_mask = encoded.tokens_mask
        tokens_mask, window_indices = _window(tokens_mask, window_start, window_length, 1, False)
        tokens = _window(encoded.tokens, window_start.unsqueeze(1), window_length, 0, False)[0]
        length = window_length + cum_alignment_padding * 2
        cum_alignment_window = _window(cum_alignment, window_start, length, 1, False)[0]

        # [batch_size, 1, num_tokens] → [batch_size, hidden_size, num_tokens]
        location_features = cum_alignment_window.unsqueeze(1) / self.avg_frames_per_token - 1.0
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
        # [num_tokens, batch_size, token_size] →
        # [batch_size, num_tokens, token_size]
        tokens = tokens.transpose(0, 1)

        # alignment [batch_size (b), 1 (n), num_tokens (m)] (bmm)
        # tokens [batch_size (b), num_tokens (m), token_size (p)] →
        # [batch_size (b), 1 (n), token_size (p)]
        context = torch.bmm(alignment.unsqueeze(1), tokens)

        # [batch_size, 1, token_size] → [batch_size, token_size]
        context = context.squeeze(1)

        length = max_num_tokens + cum_alignment_padding * 2
        indices = window_indices + cum_alignment_padding
        alignment = torch.zeros(batch_size, length, device=device).scatter_(1, indices, alignment)

        last_window_start = window_start
        window_start = alignment.max(dim=1)[1] - window_length // 2 - cum_alignment_padding
        # TODO: Cache `num_tokens - window_length` clamped at 0 so that we dont need to
        # recompute the `clamp` and subtraction each time.
        # NOTE: `torch.clamp` does not prompt a consistent left-to-right progression. This can
        # be addressed with proper padding on the attention input.
        window_start = torch.min(window_start, encoded.num_tokens - window_length)
        window_start = torch.clamp(window_start, min=0)
        window_start = torch.max(last_window_start, window_start)
        if not math.isinf(token_skip_warning):
            assert token_skip_warning >= 0, "The number of tokens skipped is a positive number."
            max_tokens_skipped = (window_start - last_window_start).max()
            if max_tokens_skipped > token_skip_warning:
                logger.warning("Attention module skipped %d tokens.", max_tokens_skipped)

        hidden_state = AttentionHiddenState(cum_alignment + alignment, window_start)

        return context, alignment[:, part], hidden_state