import dataclasses
import typing

import torch

from lib.utils import lengths_to_mask


@dataclasses.dataclass(frozen=True)
class Inputs:
    """The model inputs.

    TODO: Use `tuple`s so these values cannot be reassigned.
    """

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each sequence
    seq_metadata: typing.List[typing.List[typing.Hashable]]

    # Metadata associated with each token in each sequence
    token_metadata: typing.List[typing.List[typing.List[typing.Hashable]]]

    # Embeddings associated with each token in each sequence
    # torch.FloatTensor [batch_size, num_tokens, *]
    token_embeddings: typing.Union[torch.Tensor, typing.List[torch.Tensor]]

    # Slice of tokens in each sequence to be voiced
    slices: typing.List[slice]

    device: torch.device = torch.device("cpu")

    # Number of tokens after `slices` is applied
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor = dataclasses.field(init=False)

    # Tokens mask after `slices` is applied
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor = dataclasses.field(init=False)

    def __post_init__(self):
        indices = [s.indices(len(t)) for s, t in zip(self.slices, self.tokens)]
        num_tokens = [b - a for a, b, _ in indices]
        num_tokens_ = torch.tensor(num_tokens, dtype=torch.long, device=self.device)
        object.__setattr__(self, "num_tokens", num_tokens_)
        object.__setattr__(self, "tokens_mask", lengths_to_mask(num_tokens, device=self.device))


class Preds(typing.NamedTuple):
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
    # torch.LongTensor [num_tokens]
    num_tokens: torch.Tensor

    # Sequence mask(s) to deliminate token padding with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor

    # If `True` the sequence has reached `self.max_frames_per_token`.
    # torch.BoolTensor [batch_size]
    reached_max: torch.Tensor


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
