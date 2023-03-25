import logging
import math
import typing
from functools import lru_cache

import torch
import torch.nn

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
    check_invariants: bool = False,
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
        hidden_size: The hidden size of the hidden attention features.
        conv_filter_size: Size of the convolving kernel applied to the cumulative alignment.
        window_len: The size of the attention window applied during inference.
        avg_frames_per_token: A statistic used to normalize the model.
    """

    def __init__(
        self,
        hidden_size: int,
        conv_filter_size: int,
        window_len: int,
        avg_frames_per_token: float,
    ):
        super().__init__()

        # Learn more:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert conv_filter_size % 2 == 1, "`conv_filter_size` must be odd"
        assert window_len % 2 == 1, "`window_len` must be odd"

        self.hidden_size = hidden_size
        self.window_len = window_len
        self.avg_frames_per_token = avg_frames_per_token
        self.padding = conv_filter_size // 2
        self.scale = math.sqrt(self.hidden_size)
        self.alignment_conv = torch.nn.Conv1d(2, hidden_size, kernel_size=conv_filter_size)

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
        check_invariants: bool = False,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, AttentionHiddenState]:
        """
        Args:
            ...
            query (torch.FloatTensor [1, batch_size, hidden_size]): Attention query.
            hidden_state
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a `logger.warning` will be logged.

        Returns:
            context (torch.FloatTensor [batch_size, hidden_size]): Attention context vector.
            alignment (torch.FloatTensor [batch_size, num_tokens]): Attention alignment.
            hidden_state
        """
        batch_size, num_tokens, _ = encoded.tokens.shape
        last_align, cum_alignment = hidden_state.alignment, hidden_state.cum_alignment
        win_start = hidden_state.window_start
        window_len = min(self.window_len, num_tokens)
        padded_window_len = window_len + self.padding * 2

        win_start_ = win_start.unsqueeze(1)
        tokens_mask, window_idx = _window(encoded.tokens_mask, win_start, window_len, 1)
        tokens = _window(encoded.tokens, win_start_, window_len, 1)[0]
        keys = _window(encoded.token_keys, win_start_, window_len, 2)[0]
        last_align_window = _window(last_align, win_start, padded_window_len, 1)[0]
        cum_align_window = _window(cum_alignment, win_start, padded_window_len, 1)[0]

        if check_invariants:
            message = "Only valid tokens are allowed in the window."
            assert torch.all(tokens_mask[encoded.num_tokens >= self.window_len]), message

        # [batch_size, 1, padded_window_len] → [batch_size, hidden_size, window_len]
        cum_align_window = cum_align_window.unsqueeze(1) / self.avg_frames_per_token - 1.0
        last_align_window = last_align_window.unsqueeze(1) * 2.0 - 1.0
        position = torch.cat([cum_align_window, last_align_window], dim=1).detach()
        position = self.alignment_conv(position.detach())

        # [batch_size, hidden_size, window_len]
        keys = position + keys

        # query [batch_size (b), 1 (n), hidden_size (m)] (bmm)
        # keys [batch_size (b), hidden_size (m), window_len (p)] →
        # [batch_size (b), 1 (n), window_len (p)]
        query = query.view(batch_size, 1, self.hidden_size)
        score = (torch.bmm(query, keys) / self.scale).squeeze(1)
        score.data.masked_fill_(~tokens_mask, -math.inf)

        # [batch_size, window_len] → [batch_size, window_len]
        align_window = torch.softmax(score, dim=1)

        # alignment [batch_size (b), 1 (n), window_len (m)] (bmm)
        # tokens [batch_size (b), window_len (m), hidden_size (p)] →
        # [batch_size (b), 1 (n), hidden_size (p)]
        context = torch.bmm(align_window.unsqueeze(1), tokens)
        context = context.squeeze(1)  # [batch_size, 1, hidden_size] → [batch_size, hidden_size]

        window_idx = window_idx + self.padding
        alignment = last_align.new_zeros(*last_align.shape).scatter_(1, window_idx, align_window)
        last_win_start = win_start

        # TODO: Cache `num_tokens - window_len` clamped at 0 so that we dont need to
        # recompute the `clamp` and subtraction each time.
        # NOTE: `torch.min/max` does not prompt a consistent left-to-right progression. This can
        # be addressed with proper padding on the attention input.
        window_start = alignment.max(dim=1)[1] - padded_window_len // 2
        window_start = torch.max(last_win_start, window_start)
        window_start = torch.min(window_start, encoded.num_tokens - window_len)
        window_start = torch.clamp(window_start, min=0)

        if not math.isinf(token_skip_warning):
            assert token_skip_warning >= 0, "This must be a positive number."
            max_tokens_skipped = (window_start - last_win_start).max()
            if max_tokens_skipped > token_skip_warning:
                logger.warning("Attention module skipped %d tokens.", max_tokens_skipped)

        hidden_state = AttentionHiddenState(alignment, cum_alignment + alignment, window_start)

        unpadded_alignment = alignment[:, self.padding : -self.padding]
        if check_invariants:
            message = "The attention module should only align on real tokens."
            assert alignment[:, : self.padding].sum() == 0.0, message
            assert alignment[:, -self.padding :].sum() == 0.0, message
            assert unpadded_alignment.masked_select(~encoded.tokens_mask).sum() == 0.0, message

        return context, unpadded_alignment, hidden_state
