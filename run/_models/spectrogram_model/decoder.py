import typing

import torch
import torch.nn
from torch.nn import functional

from lib.utils import LSTM, LSTMCell
from run._models.spectrogram_model.attention import Attention
from run._models.spectrogram_model.containers import (
    AttentionHiddenState,
    Decoded,
    DecoderHiddenState,
    Encoded,
)
from run._models.spectrogram_model.pre_net import PreNet


class Decoder(torch.nn.Module):
    """Predicts the next spectrogram frame, given the previous spectrogram frame.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf

    Args:
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands")
        seq_meta_embed_size The size of the sequence metadata embedding.
        pre_net_size: The size of the pre-net hidden representation and output.
        lstm_hidden_size: The hidden size of the LSTM layers.
        encoder_output_size: The size of the attention context derived from the encoded sequence.
        stop_net_dropout: The dropout probability of the stop net.
    """

    def __init__(
        self,
        num_frame_channels: int,
        seq_meta_embed_size: int,
        pre_net_size: int,
        lstm_hidden_size: int,
        encoder_output_size: int,
        stop_net_dropout: float,
    ):
        super().__init__()

        self.num_frame_channels = num_frame_channels
        self.seq_meta_embed_size = seq_meta_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder_output_size = encoder_output_size
        input_size = seq_meta_embed_size + encoder_output_size
        initial_state_ouput_size = num_frame_channels + 1 + encoder_output_size
        self.initial_state = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, initial_state_ouput_size),
        )
        self.pre_net = PreNet(num_frame_channels, seq_meta_embed_size, pre_net_size)
        self.lstm_layer_one = LSTMCell(pre_net_size + input_size, lstm_hidden_size)
        self.lstm_layer_two = LSTM(lstm_hidden_size + input_size, lstm_hidden_size)
        self.attention = Attention(query_hidden_size=lstm_hidden_size)
        self.linear_out = torch.nn.Linear(lstm_hidden_size + input_size, num_frame_channels)
        self.linear_stop_token = torch.nn.Sequential(
            torch.nn.Dropout(stop_net_dropout),
            torch.nn.Linear(lstm_hidden_size, 1),
        )

    def _make_hidden_state(self, encoded: Encoded) -> DecoderHiddenState:
        """Make an initial hidden state, if one is not provided."""
        max_num_tokens, batch_size, _ = encoded.tokens.shape
        device = encoded.tokens.device
        cum_align_padding = self.attention.cumulative_alignment_padding

        segments = [self.num_frame_channels, 1, self.encoder_output_size]
        # [batch_size, seq_meta_embed_size + encoder_output_size] →
        # [batch_size, num_frame_channels + 1 + encoder_output_size] →
        # ([batch_size, num_frame_channels], [batch_size, 1], [batch_size, encoder_output_size])
        first_token = torch.cat([encoded.seq_metadata, encoded.tokens[0]], dim=1)
        state = self.initial_state(first_token).split(segments, dim=-1)
        initial_frame, initial_cum_align, initial_attention_context = state

        # NOTE: The `cum_align` or `cumulative_alignment` vector has a positive value for every
        # token that is has attended to. Assuming the model is attending to tokens from
        # left-to-right and the model starts reading at the first token, then any padding to the
        # left of the first token should be positive to be consistent.
        cum_align = torch.zeros(batch_size, max_num_tokens, device=device)
        # [batch_size, 1] → [batch_size, cum_align_padding]
        initial_cum_align = initial_cum_align.expand(-1, cum_align_padding).abs()
        # [batch_size, num_tokens] → [batch_size, num_tokens + cum_align_padding]
        cum_align = torch.cat([initial_cum_align, cum_align], -1)
        # [batch_size, num_tokens + cum_align_padding] →
        # [batch_size, num_tokens + 2 * cum_align_padding]
        cum_align = functional.pad(cum_align, [0, cum_align_padding], mode="constant", value=0.0)

        return DecoderHiddenState(
            last_attention_context=initial_attention_context,
            last_frame=initial_frame.unsqueeze(0),
            attention_hidden_state=AttentionHiddenState(
                cumulative_alignment=cum_align,
                window_start=torch.zeros(batch_size, device=device, dtype=torch.long),
            ),
            lstm_one_hidden_state=None,
            lstm_two_hidden_state=None,
        )

    def __call__(
        self,
        encoded: Encoded,
        target_frames: typing.Optional[torch.Tensor] = None,
        hidden_state: typing.Optional[DecoderHiddenState] = None,
        **kwargs,
    ) -> Decoded:
        return super().__call__(
            encoded=encoded, target_frames=target_frames, hidden_state=hidden_state, **kwargs
        )

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
        assert target_frames is None or hidden_state is None, (
            "Either the decoder is conditioned on `target_frames` or "
            "the `hidden_state` but not both."
        )

        hidden_state = self._make_hidden_state(encoded) if hidden_state is None else hidden_state
        (
            last_attention_context,
            last_frame,
            attention_hidden_state,
            lstm_one_hidden_state,
            lstm_two_hidden_state,
        ) = hidden_state

        # NOTE: Shift target frames backwards one step to be the source frames.
        frames = (
            last_frame if target_frames is None else torch.cat([last_frame, target_frames[0:-1]])
        )

        num_frames, _, _ = frames.shape

        del hidden_state

        # [num_frames, batch_size, num_frame_channels] →
        # [num_frames, batch_size, pre_net_hidden_size]
        pre_net_frames = self.pre_net(frames, encoded.seq_metadata)

        # Iterate over all frames for incase teacher-forcing; in sequential prediction, iterates
        # over a single frame.
        frames_list: typing.List[torch.Tensor] = []
        attention_contexts_list: typing.List[torch.Tensor] = []
        alignments_list: typing.List[torch.Tensor] = []
        window_start_list: typing.List[torch.Tensor] = []
        for frame in pre_net_frames.split(1, dim=0):
            frame = frame.squeeze(0)

            # [batch_size, pre_net_hidden_size] (concat)
            # [batch_size, seq_meta_embed_size] (concat)
            # [batch_size, encoder_output_size] →
            # [batch_size, pre_net_hidden_size + encoder_output_size + seq_meta_embed_size]
            frame = torch.cat([frame, last_attention_context, encoded.seq_metadata], dim=1)

            # frame [batch (batch_size),
            # input_size (pre_net_hidden_size + encoder_output_size + seq_meta_embed_size)]  →
            # [batch_size, lstm_hidden_size]
            lstm_one_hidden_state = self.lstm_layer_one(frame, lstm_one_hidden_state)
            assert lstm_one_hidden_state is not None
            frame = lstm_one_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attention_context [batch_size, encoder_output_size]
            query = frame.unsqueeze(0)
            last_attention_context, alignment, attention_hidden_state = self.attention(
                encoded=encoded, query=query, hidden_state=attention_hidden_state, **kwargs
            )

            frames_list.append(frame)
            attention_contexts_list.append(last_attention_context)
            alignments_list.append(alignment)
            window_start_list.append(attention_hidden_state.window_start)

            del alignment
            del frame

        # [num_frames, batch_size, num_tokens]
        alignments = torch.stack(alignments_list, dim=0)
        # [num_frames, batch_size, lstm_hidden_size]
        frames = torch.stack(frames_list, dim=0)
        # [num_frames, batch_size, encoder_output_size]
        attention_contexts = torch.stack(attention_contexts_list, dim=0)
        # [num_frames, batch_size]
        window_starts = torch.stack(window_start_list, dim=0)
        del alignments_list
        del frames_list
        del attention_contexts_list

        # [batch_size, seq_meta_embed_size] →
        # [1, batch_size, seq_meta_embed_size]
        seq_metadata = encoded.seq_metadata.unsqueeze(0)

        # [1, batch_size, seq_meta_embed_size] →
        # [num_frames, batch_size, seq_meta_embed_size]
        seq_metadata = seq_metadata.expand(num_frames, -1, -1)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, encoder_output_size] (concat)
        # [num_frames, batch_size, seq_meta_embed_size] →
        # [num_frames, batch_size, lstm_hidden_size + encoder_output_size + seq_meta_embed_size]
        frames = torch.cat([frames, attention_contexts, seq_metadata], dim=2)

        # frames [seq_len (num_frames), batch (batch_size),
        # input_size (lstm_hidden_size + encoder_output_size + seq_meta_embed_size)] →
        # [num_frames, batch_size, lstm_hidden_size]
        frames, lstm_two_hidden_state = self.lstm_layer_two(frames, lstm_two_hidden_state)

        # [num_frames, batch_size, pre_net_hidden_size + 2] →
        # [num_frames, batch_size]
        stop_token = self.linear_stop_token(frames).squeeze(2)

        # [num_frames, batch_size,
        #  lstm_hidden_size (concat) encoder_output_size (concat) seq_meta_embed_size] →
        # [num_frames, batch_size, num_frame_channels]
        frames = self.linear_out(torch.cat([frames, attention_contexts, seq_metadata], dim=2))

        hidden_state = DecoderHiddenState(
            last_attention_context=last_attention_context,
            last_frame=frames[-1].unsqueeze(0),
            attention_hidden_state=attention_hidden_state,
            lstm_one_hidden_state=lstm_one_hidden_state,
            lstm_two_hidden_state=lstm_two_hidden_state,
        )

        return Decoded(frames, stop_token, alignments.detach(), window_starts, hidden_state)
