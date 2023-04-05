import dataclasses
import typing

import torch

from run._models.spectrogram_model.pre_net import PreNetHiddenState

PredsType = typing.TypeVar("PredsType", bound="Preds")


@dataclasses.dataclass(frozen=True)
class Preds:
    """The model predictions and related metadata."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size (optional)]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size (optional), num_tokens + attn.window_len - 1]
    alignments: torch.Tensor

    # The number of frames in each sequence.
    # torch.LongTensor [batch_size (optional)]
    num_frames: torch.Tensor

    # Sequence mask(s) to deliminate `frames` padding with `False`.
    # torch.BoolTensor [batch_size (optional), num_frames]
    frames_mask: torch.Tensor

    # The number of tokens in each sequence.
    # torch.LongTensor [batch_size (optional)]
    num_tokens: torch.Tensor

    # Sequence mask(s) to deliminate token padding with `False`.
    # torch.BoolTensor [batch_size (optional), num_tokens]
    tokens_mask: torch.Tensor

    # If `True` the sequence has reached `self.max_frames_per_token`.
    # torch.BoolTensor [batch_size (optional)]
    reached_max: torch.Tensor

    # The window length used to generate this prediction.
    attn_win_len: int

    def __post_init__(self):
        self.check_invariants()

    def check_invariants(self):
        """Check various invariants for `Preds`."""
        # TODO: Let's consider writing invariants for the values each of these metrics have, for
        # example, `alignments` should be between 0 and 1.
        # TODO: Let's consider writing invariants to check everything is on the same device.
        batch_size = len(self)
        num_frame_channels = self.frames.shape[-1]
        num_frames = self.num_frames.max() if self.num_frames.numel() != 0 else 0
        num_tokens = self.num_tokens.max() if self.num_tokens.numel() != 0 else 0
        assert self.frames.shape in (
            (num_frames, batch_size, num_frame_channels),
            (num_frames, num_frame_channels),
        )
        assert self.frames.dtype == torch.float
        assert self.stop_tokens.shape in ((num_frames, batch_size), (num_frames,))
        assert self.stop_tokens.dtype == torch.float
        assert self.alignments.shape in (
            (num_frames, batch_size, num_tokens + self.attn_win_len - 1),
            (num_frames, num_tokens + self.attn_win_len - 1),
        )
        assert self.alignments.dtype == torch.float
        assert self.num_frames.shape in ((batch_size,), ())
        assert self.num_frames.dtype == torch.long
        assert self.frames_mask.shape in ((batch_size, num_frames), (num_frames,))
        assert self.frames_mask.dtype == torch.bool
        assert self.num_tokens.shape in ((batch_size,), ())
        assert self.num_frames.dtype == torch.long
        assert self.tokens_mask.shape in ((batch_size, num_tokens), (num_tokens,))
        assert self.tokens_mask.dtype == torch.bool
        assert self.reached_max.shape in ((batch_size,), ())
        assert self.reached_max.dtype == torch.bool

    def __len__(self):
        return 1 if self.num_tokens.dim() == 0 else self.num_tokens.shape[0]

    def __getitem__(self, key):
        num_frames = self.num_frames[key].max()
        num_tokens = self.num_tokens[key].max()
        return Preds(
            frames=self.frames[:num_frames, key],
            stop_tokens=self.stop_tokens[:num_frames, key],
            alignments=self.alignments[:num_frames, key, : num_tokens + self.attn_win_len - 1],
            num_frames=self.num_frames[key],
            frames_mask=self.frames_mask[key, :num_frames],
            num_tokens=self.num_tokens[key],
            tokens_mask=self.tokens_mask[key, :num_tokens],
            reached_max=self.reached_max[key],
            attn_win_len=self.attn_win_len,
        )

    def apply(self: PredsType, call: typing.Callable[[torch.Tensor], torch.Tensor]) -> PredsType:
        applied = {f.name: call(getattr(self, f.name)) for f in dataclasses.fields(self)}
        return dataclasses.replace(self, **applied)


@dataclasses.dataclass(frozen=True)
class Encoded:
    """The model inputs encoded."""

    # Batch of sequences
    # torch.FloatTensor [batch_size, num_tokens, out_dim]
    tokens: torch.Tensor

    # Batch of sequences
    # torch.FloatTensor [batch_size, out_dim, num_tokens]
    token_keys: torch.Tensor

    # Sequence mask(s) to deliminate padding in `tokens` with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor

    # Number of tokens in each sequence.
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor

    def __getitem__(self, key):
        return Encoded(
            tokens=self.tokens[key],
            token_keys=self.token_keys[key],
            tokens_mask=self.tokens_mask[key],
            num_tokens=self.num_tokens[key],
        )


@dataclasses.dataclass(frozen=True)
class AttentionHiddenState:
    """Attention hidden state from previous time steps, used to predict the next time step."""

    # torch.FloatTensor [batch_size, num_tokens + 2 * cum_alignment_padding]
    alignment: torch.Tensor

    # torch.FloatTensor [batch_size, num_tokens + 2 * cum_alignment_padding]
    max_alignment: torch.Tensor

    # torch.FloatTensor [batch_size, num_tokens + 2 * cum_alignment_padding]
    cum_alignment: torch.Tensor

    # torch.LongTensor [batch_size]
    window_start: torch.Tensor

    def __getitem__(self, key):
        return AttentionHiddenState(
            alignment=self.alignment[key],
            max_alignment=self.max_alignment[key],
            cum_alignment=self.cum_alignment[key],
            window_start=self.window_start[key],
        )


class AttentionRNNHiddenState(typing.NamedTuple):
    """Attention RNN hidden state from previous time steps used to predict the next time step."""

    # `lstm` hidden state.
    lstm_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]]

    # torch.FloatTensor [batch_size, attn_size]
    last_attn_context: torch.Tensor

    # `attn` hidden state.
    attn_hidden_state: AttentionHiddenState


class DecoderHiddenState(typing.NamedTuple):
    """Decoder hidden state from previous time steps, used to predict the next time step."""

    # The last predicted frame.
    # torch.FloatTensor [1, batch_size, num_frame_channels]
    last_frame: torch.Tensor

    # `Decoder.attn_rnn` hidden state.
    attn_rnn_hidden_state: AttentionRNNHiddenState

    # Padded encoding with space for the `attention` window.
    encoded_padded: Encoded

    # `PreNet` hidden state.
    pre_net_hidden_state: PreNetHiddenState = None

    # `Decoder.lstm` hidden state.
    lstm_hidden_state: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None


class Decoded(typing.NamedTuple):
    """The decoding of the encoding."""

    # Spectrogram frame(s).
    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame in logits.
    # torch.FloatTensor [num_frames, batch_size]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size, num_tokens + attn.window_len - 1]
    alignments: torch.Tensor

    # torch.LongTensor [num_frames, batch_size]
    window_starts: torch.Tensor

    # The last `Decoder` hidden state.
    hidden_state: DecoderHiddenState
