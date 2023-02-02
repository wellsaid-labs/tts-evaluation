import dataclasses
import typing

import torch


@dataclasses.dataclass(frozen=True)
class Preds:
    """The model predictions and related metadata."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size, num_tokens]
    alignments: torch.Tensor

    # The number of frames in each sequence.
    # torch.LongTensor [batch_size]
    num_frames: torch.Tensor

    # Sequence mask(s) to deliminate `frames` padding with `False`.
    # torch.BoolTensor [batch_size, num_frames]
    frames_mask: torch.Tensor

    # The number of tokens in each sequence.
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor

    # Sequence mask(s) to deliminate token padding with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor

    # If `True` the sequence has reached `self.max_frames_per_token`.
    # torch.BoolTensor [batch_size]
    reached_max: torch.Tensor

    def __post_init__(self):
        self.check_invariants()

    def check_invariants(self):
        """Check various invariants for `Preds`."""
        # TODO: Let's consider writing invariants for the values each of these metrics have, for
        # example, `alignments` should be between 0 and 1.
        # TODO: Let's consider writing invariants to check everything is on the same device.
        batch_size = self.num_tokens.shape[0]
        num_frame_channels = self.frames.shape[2]
        num_frames = self.num_frames.max() if self.num_frames.numel() != 0 else 0
        num_tokens = self.num_tokens.max() if self.num_tokens.numel() != 0 else 0
        assert self.frames.shape == (num_frames, batch_size, num_frame_channels)
        assert self.frames.dtype == torch.float
        assert self.stop_tokens.shape == (num_frames, batch_size)
        assert self.stop_tokens.dtype == torch.float
        assert self.alignments.shape == (num_frames, batch_size, num_tokens)
        assert self.alignments.dtype == torch.float
        assert self.num_frames.shape == (batch_size,)
        assert self.num_frames.dtype == torch.long
        assert self.frames_mask.shape == (batch_size, num_frames)
        assert self.frames_mask.dtype == torch.bool
        assert self.num_tokens.shape == (batch_size,)
        assert self.num_frames.dtype == torch.long
        assert self.tokens_mask.shape == (batch_size, num_tokens)
        assert self.tokens_mask.dtype == torch.bool
        assert self.reached_max.shape == (batch_size,)
        assert self.reached_max.dtype == torch.bool

    def __len__(self):
        return self.num_tokens.shape[0]

    def __getitem__(self, key):
        num_frames = self.num_frames[key].max()
        num_tokens = self.num_tokens[key].max()
        return Preds(
            frames=self.frames[:num_frames, key],
            stop_tokens=self.stop_tokens[:num_frames, key],
            alignments=self.alignments[:num_frames, key, :num_tokens],
            num_frames=self.num_frames[key],
            frames_mask=self.frames_mask[key, :num_frames],
            num_tokens=self.num_tokens[key],
            tokens_mask=self.tokens_mask[key, :num_tokens],
            reached_max=self.reached_max[key],
        )


class Encoded(typing.NamedTuple):
    """The model inputs encoded."""

    # Batch of sequences
    # torch.FloatTensor [num_tokens, batch_size, out_dim]
    tokens: torch.Tensor

    # Sequence mask(s) to deliminate padding in `tokens` with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor

    # Number of tokens in each sequence.
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor

    # Sequence metadata encoded
    # torch.FloatTensor [batch_size, seq_meta_embed_size]
    seq_metadata: torch.Tensor


class AttentionHiddenState(typing.NamedTuple):
    """Attention hidden state from previous time steps, used to predict the next time step."""

    # torch.FloatTensor [batch_size, num_tokens + 2 * cum_alignment_padding]
    cum_alignment: torch.Tensor

    # torch.LongTensor [batch_size]
    window_start: torch.Tensor


class DecoderHiddenState(typing.NamedTuple):
    """Decoder hidden state from previous time steps, used to predict the next time step."""

    # `Attention` last output.
    # torch.FloatTensor [batch_size, encoder_out_size]
    last_attention_context: torch.Tensor

    # The last predicted frame.
    # torch.FloatTensor [1, batch_size, num_frame_channels]
    last_frame: torch.Tensor

    # `Decoder.attention` hidden state.
    attention_hidden_state: AttentionHiddenState

    # Padded encoding with space for the `attention` window.
    padded_encoded: Encoded

    # `Decoder.lstm_layer_one` hidden state.
    lstm_one_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None

    # `Decoder.lstm_layer_two` hidden state.
    lstm_two_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None


class Decoded(typing.NamedTuple):
    """The decoding of the encoding."""

    # Spectrogram frame(s).
    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame in logits.
    # torch.FloatTensor [num_frames, batch_size]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size, num_tokens]
    alignments: torch.Tensor

    # torch.LongTensor [num_frames, batch_size]
    window_starts: torch.Tensor

    # The last `Decoder` hidden state.
    hidden_state: DecoderHiddenState
