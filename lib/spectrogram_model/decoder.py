import typing

import torch
import torch.nn
from hparams import HParam, configurable
from torch.nn import functional

from lib.spectrogram_model.attention import Attention, AttentionHiddenState
from lib.spectrogram_model.pre_net import PreNet
from lib.utils import LSTM, LSTMCell


class DecoderHiddenState(typing.NamedTuple):
    """Hidden state from previous time steps, used to predict the next time step.

    Args:
        last_attention_context (torch.FloatTensor [batch_size, encoder_output_size]):
            `Attention` last output.
        last_frame (torch.FloatTensor [1, batch_size, num_frame_channels], optional): The last
            predicted frame.
        attention_hidden_state: `Decoder.attention` hidden state.
        lstm_one_hidden_state: `Decoder.lstm_layer_one` hidden state.
        lstm_two_hidden_state: `Decoder.lstm_layer_two` hidden state.
    """

    last_attention_context: torch.Tensor
    last_frame: torch.Tensor
    attention_hidden_state: AttentionHiddenState
    lstm_one_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None
    lstm_two_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None


class DecoderOut(typing.NamedTuple):
    """
    Args:
        frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Spectrogram
            frame(s).
        stop_tokens (torch.FloatTensor [num_frames, batch_size]): Stopping probability for each
            frame in logits.
        alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention alignment
            between `frames` and `tokens`.
        window_starts (torch.LongTensor [num_frames, batch_size])
        hidden_state
    """

    frames: torch.Tensor
    stop_tokens: torch.Tensor
    alignments: torch.Tensor
    window_starts: torch.Tensor
    hidden_state: DecoderHiddenState


class Decoder(torch.nn.Module):
    """Predicts the next spectrogram frame, given the previous spectrogram frame.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf

    Args:
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands")
        speaker_embedding_size The size of the speaker embedding.
        pre_net_size: The size of the pre-net hidden representation and output.
        lstm_hidden_size: The hidden size of the LSTM layers.
        encoder_output_size: The size of the attention context derived from the encoded sequence.
        stop_net_dropout: The dropout probability of the stop net.
    """

    @configurable
    def __init__(
        self,
        num_frame_channels: int,
        speaker_embedding_size: int,
        pre_net_size: int = HParam(),
        lstm_hidden_size: int = HParam(),
        encoder_output_size: int = HParam(),
        stop_net_dropout: float = HParam(),
    ):
        super().__init__()

        self.num_frame_channels = num_frame_channels
        self.speaker_embedding_size = speaker_embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.encoder_output_size = encoder_output_size
        input_size = speaker_embedding_size + encoder_output_size
        initial_state_ouput_size = num_frame_channels + 1 + encoder_output_size
        self.initial_state = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size, initial_state_ouput_size),
        )
        self.pre_net = PreNet(num_frame_channels, speaker_embedding_size, pre_net_size)
        self.lstm_layer_one = LSTMCell(pre_net_size + input_size, lstm_hidden_size)
        self.lstm_layer_two = LSTM(lstm_hidden_size + input_size, lstm_hidden_size)
        self.attention = Attention(query_hidden_size=lstm_hidden_size)
        self.linear_out = torch.nn.Linear(lstm_hidden_size + input_size, num_frame_channels)
        self.linear_stop_token = torch.nn.Sequential(
            torch.nn.Dropout(stop_net_dropout),
            torch.nn.Linear(lstm_hidden_size, 1),
        )

    def _make_initial_hidden_state(
        self, tokens: torch.Tensor, speaker: torch.Tensor
    ) -> DecoderHiddenState:
        """Make an initial hidden state, if one is not provided.

        Args:
            tokens (torch.FloatTensor [num_tokens, batch_size, encoder_output_size])
            speaker (torch.FloatTensor [batch_size, speaker_embedding_dim])
        """
        max_num_tokens, batch_size, _ = tokens.shape
        device = tokens.device
        cumulative_alignment_padding = self.attention.cumulative_alignment_padding

        segments = [self.num_frame_channels, 1, self.encoder_output_size]
        # [batch_size, speaker_embedding_dim + encoder_output_size] →
        # [batch_size, num_frame_channels + 1 + encoder_output_size] →
        # ([batch_size, num_frame_channels], [batch_size, 1], [batch_size, encoder_output_size])
        state = self.initial_state(torch.cat([speaker, tokens[0]], dim=1)).split(segments, dim=-1)
        initial_frame, initial_cumulative_alignment, initial_attention_context = state

        # NOTE: The `cumulative_alignment` vector has a positive value for every token that is has
        # attended to. Assuming the model is attending to tokens from left-to-right and the model
        # starts reading at the first token, then any padding to the left of the first token should
        # be positive to be consistent.
        cumulative_alignment = torch.zeros(batch_size, max_num_tokens, device=device)
        # [batch_size, 1] → [batch_size, cumulative_alignment_padding]
        initial_cumulative_alignment = initial_cumulative_alignment.expand(
            -1, cumulative_alignment_padding
        ).abs()
        # [batch_size, num_tokens] → [batch_size, num_tokens + cumulative_alignment_padding]
        cumulative_alignment = torch.cat([initial_cumulative_alignment, cumulative_alignment], -1)
        # [batch_size, num_tokens + cumulative_alignment_padding] →
        # [batch_size, num_tokens + 2 * cumulative_alignment_padding]
        cumulative_alignment = functional.pad(
            cumulative_alignment,
            [0, cumulative_alignment_padding],
            mode="constant",
            value=0.0,
        )

        return DecoderHiddenState(
            last_attention_context=initial_attention_context,
            last_frame=initial_frame.unsqueeze(0),
            attention_hidden_state=AttentionHiddenState(
                cumulative_alignment=cumulative_alignment,
                window_start=torch.zeros(batch_size, device=device, dtype=torch.long),
            ),
            lstm_one_hidden_state=None,
            lstm_two_hidden_state=None,
        )

    def __call__(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        speaker: torch.Tensor,
        target_frames: typing.Optional[torch.Tensor] = None,
        hidden_state: typing.Optional[DecoderHiddenState] = None,
        **kwargs,
    ) -> DecoderOut:
        return super().__call__(
            tokens=tokens,
            tokens_mask=tokens_mask,
            num_tokens=num_tokens,
            speaker=speaker,
            target_frames=target_frames,
            hidden_state=hidden_state,
            **kwargs,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        speaker: torch.Tensor,
        target_frames: typing.Optional[torch.Tensor] = None,
        hidden_state: typing.Optional[DecoderHiddenState] = None,
        **kwargs,
    ) -> DecoderOut:
        """
        Args:
            tokens (torch.FloatTensor [num_tokens, batch_size, encoder_output_size])
            tokens_mask (torch.BoolTensor [batch_size, num_tokens])
            num_tokens (torch.LongTensor [batch_size])
            speaker (torch.FloatTensor [batch_size, speaker_embedding_dim])
            target_frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels],
                optional): Ground truth frames for "teacher forcing" and loss.
            hidden_state: Hidden state from previous time steps, used to predict the next time step.
        """
        assert target_frames is None or hidden_state is None, (
            "Either the decoder is conditioned on `target_frames` or "
            "the `hidden_state` but not both."
        )

        hidden_state = (
            self._make_initial_hidden_state(tokens, speaker)
            if hidden_state is None
            else hidden_state
        )

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
        pre_net_frames = self.pre_net(frames, speaker)

        # Iterate over all frames for incase teacher-forcing; in sequential prediction, iterates
        # over a single frame.
        frames_list: typing.List[torch.Tensor] = []
        attention_contexts_list: typing.List[torch.Tensor] = []
        alignments_list: typing.List[torch.Tensor] = []
        window_start_list: typing.List[torch.Tensor] = []
        for frame in pre_net_frames.split(1, dim=0):
            frame = frame.squeeze(0)

            # [batch_size, pre_net_hidden_size] (concat)
            # [batch_size, speaker_embedding_dim] (concat)
            # [batch_size, encoder_output_size] →
            # [batch_size, pre_net_hidden_size + encoder_output_size + speaker_embedding_dim]
            frame = torch.cat([frame, last_attention_context, speaker], dim=1)

            # frame [batch (batch_size),
            # input_size (pre_net_hidden_size + encoder_output_size + speaker_embedding_dim)]  →
            # [batch_size, lstm_hidden_size]
            lstm_one_hidden_state = self.lstm_layer_one(frame, lstm_one_hidden_state)
            assert lstm_one_hidden_state is not None
            frame = lstm_one_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attention_context [batch_size, encoder_output_size]
            last_attention_context, alignment, attention_hidden_state = self.attention(
                tokens=tokens,
                tokens_mask=tokens_mask,
                num_tokens=num_tokens,
                query=frame.unsqueeze(0),
                hidden_state=attention_hidden_state,
                **kwargs,
            )

            frames_list.append(frame)
            attention_contexts_list.append(last_attention_context)
            alignments_list.append(alignment)
            window_start_list.append(attention_hidden_state.window_start)

            del alignment
            del frame

        del tokens

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

        # [batch_size, speaker_embedding_dim] →
        # [1, batch_size, speaker_embedding_dim]
        speaker = speaker.unsqueeze(0)

        # [1, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, speaker_embedding_dim]
        speaker = speaker.expand(num_frames, -1, -1)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, encoder_output_size] (concat)
        # [num_frames, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, lstm_hidden_size + encoder_output_size + speaker_embedding_dim]
        frames = torch.cat([frames, attention_contexts, speaker], dim=2)

        # frames [seq_len (num_frames), batch (batch_size),
        # input_size (lstm_hidden_size + encoder_output_size + speaker_embedding_dim)] →
        # [num_frames, batch_size, lstm_hidden_size]
        frames, lstm_two_hidden_state = self.lstm_layer_two(frames, lstm_two_hidden_state)

        # [num_frames, batch_size, pre_net_hidden_size + 2] →
        # [num_frames, batch_size]
        stop_token = self.linear_stop_token(frames).squeeze(2)

        # [num_frames, batch_size,
        #  lstm_hidden_size (concat) encoder_output_size (concat) speaker_embedding_dim] →
        # [num_frames, batch_size, num_frame_channels]
        frames = self.linear_out(torch.cat([frames, attention_contexts, speaker], dim=2))

        hidden_state = DecoderHiddenState(
            last_attention_context=last_attention_context,
            last_frame=frames[-1].unsqueeze(0),
            attention_hidden_state=attention_hidden_state,
            lstm_one_hidden_state=lstm_one_hidden_state,
            lstm_two_hidden_state=lstm_two_hidden_state,
        )

        return DecoderOut(frames, stop_token, alignments.detach(), window_starts, hidden_state)
