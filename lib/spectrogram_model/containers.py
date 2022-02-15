import typing

import torch


class Inputs(typing.NamedTuple):
    """The model inputs."""

    # Batch of speakers
    speaker: typing.List[typing.Hashable]

    # Batch of recording sessions per speaker
    session: typing.List[typing.Hashable]

    # Batch of sequences of tokens
    tokens: typing.List[typing.List[typing.Hashable]]


class Infer(typing.NamedTuple):
    """The model inference returns."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size, num_tokens]
    alignments: torch.Tensor

    # The sequence length.
    # torch.LongTensor [1, batch_size]
    lengths: torch.Tensor

    # If `True` the sequence has reached `self.max_frames_per_token`.
    # torch.BoolTensor [1, batch_size]
    reached_max: torch.Tensor


class Forward(typing.NamedTuple):
    """The model forward pass returns."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size, num_tokens]
    alignments: torch.Tensor


class Encoded(typing.NamedTuple):
    """The model input encoded."""

    # Batch of sequences
    # torch.FloatTensor [num_tokens, batch_size, out_dim]
    tokens: torch.Tensor

    # Sequence mask(s) to deliminate padding in `tokens` with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor

    # Number of tokens in each sequence.
    # torch.LongTensor [num_tokens]
    num_tokens: torch.Tensor

    # Speaker encodings
    # torch.FloatTensor [batch_size, speaker_embed_size]
    speaker: torch.Tensor


class AttentionHiddenState(typing.NamedTuple):
    """Hidden state from previous time steps, used to predict the next time step."""

    # torch.FloatTensor [batch_size, num_tokens + 2 * cumulative_alignment_padding]
    cumulative_alignment: torch.Tensor

    # torch.LongTensor [batch_size]
    window_start: torch.Tensor


class DecoderHiddenState(typing.NamedTuple):
    """Hidden state from previous time steps, used to predict the next time step."""

    # `Attention` last output.
    # torch.FloatTensor [batch_size, encoder_output_size]
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

    hidden_state: DecoderHiddenState
