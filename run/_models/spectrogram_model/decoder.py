import dataclasses
import math
import typing

import config as cf
import torch
import torch.nn
from torch.nn import functional

from lib.utils import LSTM, LSTMCell, lengths_to_mask, pad_tensor
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
        self.lstm = LSTMCell(input_size + attn_size, attn_size)
        self.attn = cf.partial(Attention)(attn_size)
        self.proj = torch.nn.Linear(attn_size, output_size)

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
            lstm_out = lstm_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attn_context [batch_size, attn_size]
            last_attn_context, alignment, attn_hidden_state = self.attn(
                encoded, lstm_out.unsqueeze(0), attn_hidden_state, **kwargs
            )

            lstm_out_list.append(lstm_out)
            attn_cntxts_list.append(last_attn_context)
            alignments_list.append(alignment)
            window_start_list.append(attn_hidden_state.window_start)

        lstm_out = torch.stack(lstm_out_list, dim=0)  # [seq_length, batch_size, attn_size]
        attn_cntxts = torch.stack(attn_cntxts_list, dim=0)  # [seq_length, batch_size, attn_size]
        alignments = torch.stack(alignments_list, dim=0)  # [seq_length, batch_size, num_tokens]
        window_starts = torch.stack(window_start_list, dim=0)  # [seq_length, batch_size]
        hidden_state = AttentionRNNHiddenState(
            lstm_hidden_state, last_attn_context, attn_hidden_state
        )

        # [seq_length, batch_size, attn_size] → [seq_length, batch_size, output_size]
        lstm_out = self.proj(lstm_out)

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
    """

    def __init__(self, num_frame_channels: int, hidden_size: int, attn_size: int):
        super().__init__()

        self.num_frame_channels = num_frame_channels
        self.hidden_size = hidden_size
        self.attn_size = attn_size
        self.init_state_parts = [self.num_frame_channels, 1] + [attn_size] * 5
        self.init_state = torch.nn.Sequential(
            torch.nn.Linear(attn_size * 4, attn_size),
            torch.nn.GELU(),
            torch.nn.Linear(attn_size, sum(self.init_state_parts)),
        )
        self.pre_net = cf.partial(PreNet)(num_frame_channels, hidden_size)
        self.attn_rnn = _AttentionRNN(hidden_size, hidden_size, attn_size)
        self.linear_stop_token = torch.nn.Sequential(
            torch.nn.Linear(hidden_size + attn_size, hidden_size),
            torch.nn.GELU(),
            cf.partial(torch.nn.LayerNorm)(hidden_size),
            torch.nn.Linear(hidden_size, 1),
        )
        self.lstm_out = LSTM(hidden_size + attn_size, hidden_size)
        self.linear_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, num_frame_channels),
        )

    def _pad_encoded(
        self,
        pad_len: int,
        encoded: Encoded,
        beg_pad_token: torch.Tensor,
        end_pad_token: torch.Tensor,
        beg_pad_key: torch.Tensor,
        end_pad_key: torch.Tensor,
    ) -> Encoded:
        """Add padding to `encoded` so that the attention module window has space at the beginning
        and end of the sequence.

        TODO: The encoder also requires padding for some components, like convolutions. It might
              be best to have the encoder own this padding.

        Args:
            ...
            beg_pad_token (torch.FloatTensor [batch_size, attn_size]): Pad token to
                add to the beginning of each sequence.
            end_pad_token (torch.FloatTensor [batch_size, attn_size]): Pad token to
                add to the end of each sequence.
            ...
        """
        batch_size, _, attn_size = encoded.tokens.shape

        # [batch_size, num_tokens, out_dim] → [batch_size, num_tokens + window_len - 1, out_dim]
        beg_pad_token = beg_pad_token.unsqueeze(1).expand(batch_size, pad_len, attn_size)
        tokens = pad_tensor(encoded.tokens, (pad_len, pad_len), dim=1)
        tokens[:, :pad_len] = beg_pad_token

        # [batch_size, out_dim, num_tokens] → [batch_size, out_dim, num_tokens + window_len - 1]
        beg_pad_key = beg_pad_key.unsqueeze(2).expand(batch_size, attn_size, pad_len)
        token_keys = functional.pad(encoded.token_keys, (pad_len, pad_len))
        token_keys[:, :, :pad_len] = beg_pad_key

        new_num_tokens = encoded.num_tokens + pad_len * 2
        new_mask = lengths_to_mask(new_num_tokens)

        # [batch_size, num_tokens] → [batch_size, num_tokens + window_length - 1]
        padded_mask = functional.pad(encoded.tokens_mask, (pad_len, 0), value=1.0)
        padded_mask = functional.pad(padded_mask, (0, pad_len))
        end_pad_idx = padded_mask.logical_xor(new_mask)

        # [batch_size, num_tokens]
        end_pad_idx_ = end_pad_idx.unsqueeze(1).expand(*token_keys.shape)
        tokens[end_pad_idx] = end_pad_token.unsqueeze(1).expand(*tokens.shape)[end_pad_idx]
        token_keys[end_pad_idx_] = end_pad_key.unsqueeze(2).expand(*token_keys.shape)[end_pad_idx_]

        return dataclasses.replace(
            encoded,
            tokens=tokens.masked_fill(~new_mask.unsqueeze(2), 0),
            token_keys=token_keys.masked_fill(~new_mask.unsqueeze(1), 0),
            tokens_mask=new_mask,
            num_tokens=new_num_tokens,
        )

    def _make_init_hidden_state(self, encoded: Encoded) -> DecoderHiddenState:
        """Make an initial hidden state, if one is not provided."""
        batch_size, device = encoded.tokens.shape[0], encoded.tokens.device

        idx = torch.arange(0, batch_size, device=device)
        last_token = encoded.tokens[idx, encoded.num_tokens - 1]
        last_token_key = encoded.token_keys[idx, :, encoded.num_tokens - 1]

        # [batch_size, attn_size * 2] → [batch_size, ...self.init_state_parts]
        feats = [encoded.tokens[:, 0], encoded.token_keys[:, :, 0], last_token, last_token_key]
        feats = torch.cat(feats, dim=1)
        state = self.init_state(feats).split(self.init_state_parts, dim=-1)
        init_frame, init_cum_alignment, init_attn_cntxt, *pad_tokens = state

        encoded_pad_len = self.attn_rnn.attn.window_len // 2
        encoded_padded = self._pad_encoded(encoded_pad_len, encoded, *pad_tokens)

        # NOTE: The `cum_alignment` or `cum_alignment` vector has a positive value for every
        # token that is has attended to. Assuming the model is attending to tokens from
        # left-to-right and the model starts reading at the first token, then any padding to the
        # left of the first token should be positive to be consistent.
        num_padded_tokens, padding = encoded_padded.tokens.shape[1], self.attn_rnn.attn.padding
        cum_alignment = init_cum_alignment.expand(-1, padding).abs()
        alignment = torch.zeros(batch_size, num_padded_tokens + padding * 2, device=device)
        alignment[:, encoded_pad_len - 1] = 1.0
        attn_hidden_state = AttentionHiddenState(
            alignment=alignment,
            cum_alignment=functional.pad(cum_alignment, (0, num_padded_tokens + padding)),
            window_start=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

        return DecoderHiddenState(
            last_frame=init_frame.unsqueeze(0),
            attn_rnn_hidden_state=AttentionRNNHiddenState(None, init_attn_cntxt, attn_hidden_state),
            encoded_padded=encoded_padded,
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

        state = self._make_init_hidden_state(encoded) if hidden_state is None else hidden_state

        # NOTE: Shift target frames backwards one step to be the source frames.
        # TODO: For training in context, `last_frame` could be a real audio frame. The only
        # issue with that is that during inference we do not have a prior audio frame, so,
        # we would need to make sure that during training `last_frame` is sometimes an initial
        # frame.
        # TODO: We've found huge performance improvements from conditioning the model on the
        # first spectrogram frame. It'd be helpful to have the last frame before the
        # `target_spectrogram`, so that it's available to condition the model. Additionally,
        # it might just be helpful to have a clip from the recording session as part of the
        # model conditional.
        frames = state.last_frame
        if target_frames is not None:
            frames = torch.cat([state.last_frame, target_frames[0:-1]])

        # [num_frames, batch_size, num_frame_channels] → [*, pre_net_hidden_size]
        pre_net_frames, pre_net_hidden_state = self.pre_net(frames, state.pre_net_hidden_state)

        attn_rnn_args = (pre_net_frames, state.encoded_padded, state.attn_rnn_hidden_state)
        attn_rnn_outs = self.attn_rnn(*attn_rnn_args, **kwargs)
        attn_rnn_out, attn_cntxts, alignments, window_starts, attn_rnn_hidden_state = attn_rnn_outs
        frames = (pre_net_frames + attn_rnn_out) / math.sqrt(2)

        # [num_frames, batch_size, hidden_size] (concat) [*, attn_size] →
        # [*, hidden_size + attn_size]
        block_input = torch.cat([frames, attn_cntxts], dim=2)

        # [num_frames, batch_size, hidden_size + attn_size] → [num_frames, batch_size]
        stop_token = self.linear_stop_token(block_input).squeeze(2)

        # [num_frames, batch_size, hidden_size + attn_size] → [*, hidden_size]
        lstm_out, lstm_hidden_state = self.lstm_out(block_input, state.lstm_hidden_state)
        frames = (pre_net_frames + lstm_out) / math.sqrt(2)

        # [num_frames, batch_size, hidden_size] → [*, num_frame_channels]
        frames = self.linear_out(frames)

        hidden_state = DecoderHiddenState(
            last_frame=frames[-1].unsqueeze(0),
            encoded_padded=state.encoded_padded,
            pre_net_hidden_state=pre_net_hidden_state,
            attn_rnn_hidden_state=attn_rnn_hidden_state,
            lstm_hidden_state=lstm_hidden_state,
        )
        return Decoded(frames, stop_token, alignments, window_starts, hidden_state)
