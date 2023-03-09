import math
import typing

import config as cf
import torch
import torch.nn
from torch.nn import functional

from lib.utils import LSTM, LSTMCell, lengths_to_mask
from run._models.spectrogram_model.attention import Attention
from run._models.spectrogram_model.containers import (
    AttentionHiddenState,
    AttentionRNNHiddenState,
    Decoded,
    DecoderHiddenState,
    Encoded,
)
from run._models.spectrogram_model.pre_net import PreNet

_AttentionRNNOut = typing.Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, AttentionRNNHiddenState
]


class _AttentionRNN(torch.nn.Module):
    """An autoregressive attention layer supported by an RNN."""

    def __init__(self, input_size: int, output_size: int, attn_size: int):
        super().__init__()
        self.lstm = LSTMCell(input_size + attn_size, output_size)
        self.attn = cf.partial(Attention)(output_size, attn_size)

    def __call__(
        self, query: torch.Tensor, encoded: Encoded, hidden_state: AttentionRNNHiddenState, **kwargs
    ) -> _AttentionRNNOut:
        return super().__call__(query, encoded, hidden_state, **kwargs)

    def forward(
        self, query: torch.Tensor, encoded: Encoded, hidden_state: AttentionRNNHiddenState, **kwargs
    ) -> _AttentionRNNOut:
        """
        Args:
            query (torch.FloatTensor [seq_length, batch_size, input_size])
            ...

        Returns:
            rnn_outs (torch.FloatTensor [seq_length, batch_size, output_size]): The output of the
                RNN layer.
            attn_contexts (torch.FloatTensor [seq_length, batch_size, attn_size]): The output of the
                attention layer.
            alignments (torch.FloatTensor [seq_length, batch_size, num_tokens]): The attention
                alignments.
            window_starts (torch.FloatTensor [seq_length, batch_size]): The attention window
                indicies.
            hidden_state: The hidden state for attention rnn.
        """
        # NOTE: Iterate over all frames for incase teacher-forcing; in sequential prediction,
        # iterates over a single frame.
        lstm_hidden_state, last_attn_context, attn_hidden_state = hidden_state
        lstm_out_list: typing.List[torch.Tensor] = []
        attn_cntxts_list: typing.List[torch.Tensor] = []
        alignments_list: typing.List[torch.Tensor] = []
        window_start_list: typing.List[torch.Tensor] = []
        for split in query.split(1, dim=0):
            split = split.squeeze(0)

            # [batch_size, input_size] (concat) [batch_size, attn_size] →
            # [batch_size, input_size + attn_size]
            split = torch.cat([split, last_attn_context], dim=1)

            # [batch_size, input_size + attn_size] → [batch_size, output_size]
            lstm_hidden_state = self.lstm(split, lstm_hidden_state)
            assert lstm_hidden_state is not None
            split = lstm_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attn_context [batch_size, attn_size]
            query = split.unsqueeze(0)
            last_attn_context, alignment, attn_hidden_state = self.attn(
                encoded, query, attn_hidden_state, **kwargs
            )

            lstm_out_list.append(split)
            attn_cntxts_list.append(last_attn_context)
            alignments_list.append(alignment)
            window_start_list.append(attn_hidden_state.window_start)

        lstm_out = torch.stack(lstm_out_list, dim=0)  # [seq_length, batch_size, output_size]
        attn_cntxts = torch.stack(attn_cntxts_list, dim=0)  # [seq_length, batch_size, attn_size]
        alignments = torch.stack(alignments_list, dim=0)  # [seq_length, batch_size, num_tokens]
        window_starts = torch.stack(window_start_list, dim=0)  # [seq_length, batch_size]
        hidden_state = AttentionRNNHiddenState(
            lstm_hidden_state, last_attn_context, attn_hidden_state
        )

        return lstm_out, attn_cntxts, alignments, window_starts, hidden_state


class Decoder(torch.nn.Module):
    """Predicts the next spectrogram frame, given the previous spectrogram frame.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf

    Args:
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands")
        hidden_size: The hidden size the decoder layers.
        attn_size: The size of the attention hidden state.
        stop_net_dropout: The dropout probability of the stop net.
    """

    def __init__(
        self, num_frame_channels: int, hidden_size: int, attn_size: int, stop_net_dropout: float
    ):
        super().__init__()

        self.num_frame_channels = num_frame_channels
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.init_state_segments = [self.num_frame_channels, 1, attn_size, attn_size, attn_size]
        self.init_state = torch.nn.Sequential(
            torch.nn.Linear(attn_size * 2, attn_size),
            torch.nn.GELU(),
            torch.nn.Linear(attn_size, sum(self.init_state_segments)),
        )
        self.pre_net = cf.partial(PreNet)(num_frame_channels, hidden_size)
        self.attn_rnn = _AttentionRNN(hidden_size, hidden_size, attn_size)
        self.linear_stop_token = torch.nn.Sequential(
            torch.nn.Dropout(stop_net_dropout),
            torch.nn.Linear(hidden_size + attn_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.lstm_out = LSTM(hidden_size + attn_size, hidden_size)
        self.linear_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, num_frame_channels),
        )

    def _pad_encoded(
        self, encoded: Encoded, beg_pad_token: torch.Tensor, end_pad_token: torch.Tensor
    ) -> Encoded:
        """Add padding to `encoded` so that the attention module window has space at the beginning
        and end of the sequence.

        Args:
            beg_pad_token (torch.FloatTensor [batch_size, attn_size]): Pad token to
                add to the beginning of each sequence.
            end_pad_token (torch.FloatTensor [batch_size, attn_size]): Pad token to
                add to the end of each sequence.
        """
        device, pad_length = encoded.tokens.device, self.attn_rnn.attn.window_length // 2
        batch_size, attn_size = encoded.tokens_mask.shape[0], self.attn_size

        mask_padding = torch.zeros(batch_size, pad_length, device=device, dtype=torch.bool)
        # [batch_size, num_tokens] → [batch_size, num_tokens + window_length - 1]
        tokens_mask = torch.cat([mask_padding, encoded.tokens_mask, mask_padding], dim=1)
        tokens_padding = torch.zeros(pad_length, batch_size, attn_size, device=device)
        # [num_tokens, batch_size, out_dim] → [num_tokens + window_length - 1, batch_size, out_dim]
        tokens = torch.cat([tokens_padding, encoded.tokens, tokens_padding], dim=0)

        new_num_tokens = encoded.num_tokens + pad_length * 2
        new_mask = lengths_to_mask(new_num_tokens)

        # [batch_size, num_tokens] → [num_tokens, batch_size]
        indices = tokens_mask.logical_xor(new_mask).transpose(0, 1)
        # [batch_size, attn_size] → [num_frames, batch_size, attn_size]
        end_pad_token = end_pad_token.unsqueeze(0).expand(*tokens.shape)
        tokens[indices] = end_pad_token[indices]
        tokens[:pad_length] = beg_pad_token.unsqueeze(0).expand(pad_length, *tokens.shape[1:])

        return encoded._replace(tokens=tokens, tokens_mask=new_mask, num_tokens=new_num_tokens)

    def _make_initial_hidden_state(self, encoded: Encoded) -> DecoderHiddenState:
        """Make an initial hidden state, if one is not provided."""
        batch_size, device = encoded.tokens.shape[1], encoded.tokens.device
        cum_alignment_padding = self.attn_rnn.attn.cum_alignment_padding
        window_length = self.attn_rnn.attn.window_length

        # [batch_size, attn_size * 2] →
        # [batch_size, num_frame_channels + 1 + attn_size] →
        # [batch_size, num_frame_channels], [batch_size, 1],
        # [batch_size, attn_size], [batch_size, attn_size]
        arange = torch.arange(0, batch_size, device=device)
        last_token = encoded.tokens[encoded.num_tokens - 1, arange]
        init_features = torch.cat([encoded.tokens[0], last_token], dim=1)
        state = self.init_state(init_features).split(self.init_state_segments, dim=-1)
        init_frame, init_cum_alignment, init_attn_context, beg_pad_token, end_pad_token = state

        padded_encoded = self._pad_encoded(encoded, beg_pad_token, end_pad_token)

        # NOTE: The `cum_alignment` or `cum_alignment` vector has a positive value for every
        # token that is has attended to. Assuming the model is attending to tokens from
        # left-to-right and the model starts reading at the first token, then any padding to the
        # left of the first token should be positive to be consistent.
        cum_alignment = torch.zeros(batch_size, encoded.tokens.shape[0], device=device)
        # [batch_size, 1] → [batch_size, cum_align_padding + window_length // 2]
        padding = window_length // 2 + cum_alignment_padding
        init_cum_alignment = init_cum_alignment.expand(-1, padding).abs()
        # [batch_size, num_tokens] →
        # [batch_size, num_tokens +  window_length // 2 + cum_alignment_padding]
        cum_alignment = torch.cat([init_cum_alignment, cum_alignment], -1)
        # [batch_size, num_tokens + cum_align_padding] →
        # [batch_size, num_tokens + 2 * cum_align_padding]
        cum_alignment = functional.pad(cum_alignment, [0, padding], mode="constant", value=0.0)
        attn_window_start = torch.zeros(batch_size, device=device, dtype=torch.long)
        attn_hidden_state = AttentionHiddenState(cum_alignment, attn_window_start)
        attn_rnn_hidden_state = AttentionRNNHiddenState(None, init_attn_context, attn_hidden_state)

        return DecoderHiddenState(
            last_frame=init_frame.unsqueeze(0),
            attn_rnn_hidden_state=attn_rnn_hidden_state,
            padded_encoded=padded_encoded,
            pre_net_hidden_state=None,
            lstm_hidden_state=None,
        )

    def __call__(
        self,
        encoded: Encoded,
        target_frames: typing.Optional[torch.Tensor] = None,
        hidden_state: typing.Optional[DecoderHiddenState] = None,
        **kwargs,
    ) -> Decoded:
        return super().__call__(encoded, target_frames, hidden_state=hidden_state, **kwargs)

    def forward(
        self,
        encoded: Encoded,
        target_frames: typing.Optional[torch.Tensor] = None,
        hidden_state: typing.Optional[DecoderHiddenState] = None,
        **kwargs,
    ) -> Decoded:
        """
        Args:
            ...
            target_frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Ground
                truth frames for "teacher forcing" and loss.
            hidden_state: Hidden state from previous time steps, used to predict the next time step.
        """
        message = "`target_frames` or `hidden_state` can be passed in, not both."
        assert target_frames is None or hidden_state is None, message

        state = self._make_initial_hidden_state(encoded) if hidden_state is None else hidden_state

        # NOTE: Shift target frames backwards one step to be the source frames.
        # TODO: For training in context, `last_frame` could be a real audio frame. The only
        # issue with that is that during inference we do not have a prior audio frame, so,
        # we would need to make sure that during training `last_frame` is sometimes an initial
        # frame.
        frames = state.last_frame
        if target_frames is not None:
            frames = torch.cat([state.last_frame, target_frames[0:-1]])

        # [num_frames, batch_size, num_frame_channels] →
        # [num_frames, batch_size, pre_net_hidden_size]
        pre_net_frames, pre_net_hidden_state = self.pre_net(frames, state.pre_net_hidden_state)

        attn_rnn_args = (pre_net_frames, state.padded_encoded, state.attn_rnn_hidden_state)
        attn_rnn_outs = self.attn_rnn(*attn_rnn_args, **kwargs)
        attn_rnn_out, attn_cntxts, alignments, window_starts, attn_rnn_hidden_state = attn_rnn_outs
        frames = (pre_net_frames + attn_rnn_out) / math.sqrt(2)

        # [num_frames, batch_size, hidden_size] (concat)
        # [num_frames, batch_size, attn_size] (concat)
        # [num_frames, batch_size, hidden_size + attn_size]
        block_input = torch.cat([frames, attn_cntxts], dim=2)

        # [num_frames, batch_size, hidden_size + attn_size] → [num_frames, batch_size]
        stop_token = self.linear_stop_token(block_input).squeeze(2)

        # [num_frames, batch_size, hidden_size + attn_size] → [num_frames, batch_size, hidden_size]
        lstm_out, lstm_hidden_state = self.lstm_out(block_input, state.lstm_hidden_state)
        frames = (pre_net_frames + lstm_out) / math.sqrt(2)

        # [num_frames, batch_size, hidden_size] → [num_frames, batch_size, num_frame_channels]
        frames = self.linear_out(frames)

        hidden_state = DecoderHiddenState(
            last_frame=frames[-1].unsqueeze(0),
            padded_encoded=state.padded_encoded,
            pre_net_hidden_state=pre_net_hidden_state,
            attn_rnn_hidden_state=attn_rnn_hidden_state,
            lstm_hidden_state=lstm_hidden_state,
        )

        # [num_frames, batch_size, num_tokens + window_length - 1] →
        # [num_frames, batch_size, num_tokens]
        pad_length = self.attn_rnn.attn.window_length // 2
        alignments = alignments.detach()[:, :, pad_length:-pad_length]
        alignments = alignments.masked_fill(~encoded.tokens_mask.unsqueeze(0), 0)

        assert (window_starts[-1] < encoded.num_tokens).all(), "Invariant failure"

        return Decoded(frames, stop_token, alignments, window_starts, hidden_state)
