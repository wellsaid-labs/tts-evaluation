import contextlib
import enum
import logging
import math
import typing

import torch
import torch.nn
from hparams import HParam, configurable
from torchnlp.utils import lengths_to_mask
from tqdm import tqdm

from lib.spectrogram_model import decoder, encoder

logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    FORWARD: typing.Final = enum.auto()
    GENERATE: typing.Final = enum.auto()
    INFER: typing.Final = enum.auto()


class Forward(typing.NamedTuple):
    """The model forward pass return value."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size (optional)]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size (optional), num_tokens]
    alignments: torch.Tensor


class Infer(typing.NamedTuple):
    """The model inference return value."""

    # Spectrogram frames.
    # torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]
    frames: torch.Tensor

    # Stopping probability for each frame.
    # torch.FloatTensor [num_frames, batch_size (optional)]
    stop_tokens: torch.Tensor

    # Attention alignment between `frames` and `tokens`.
    # torch.FloatTensor [num_frames, batch_size (optional), num_tokens]
    alignments: torch.Tensor

    # The sequence length.
    # torch.LongTensor [1, batch_size (optional)]
    lengths: torch.Tensor

    # If `True` the sequence has reached `self.max_frames_per_token`.
    # torch.BoolTensor [1, batch_size (optional)]
    reached_max: torch.Tensor


Generator = typing.Iterator[Infer]


class Inputs(typing.NamedTuple):
    """Normalized tensors input tensors."""

    # Sequences
    # torch.LongTensor [batch_size, num_tokens]
    tokens: torch.Tensor

    # Speaker encodings
    # torch.FloatTensor [batch_size, speaker_embedding_size]
    speaker: torch.Tensor

    # Number of tokens in each sequence.
    # torch.LongTensor [batch_size]
    num_tokens: torch.Tensor

    # Sequence mask(s) to deliminate padding in `tokens` with `False`.
    # torch.BoolTensor [batch_size, num_tokens]
    tokens_mask: torch.Tensor


class Targets(typing.NamedTuple):
    """Normalized tensors target tensors."""

    # torch.FloatTensor [num_frames, batch_size, num_frame_channels]
    frames: torch.Tensor

    # torch.BoolTensor [num_frames, batch_size]
    mask: torch.Tensor


class Params(typing.NamedTuple):
    """Tensor inputs for running the model."""

    # tokens (torch.LongTensor [num_tokens, batch_size (optional)]): Sequences.
    tokens: torch.Tensor

    # speaker (torch.LongTensor [1, batch_size (optional)]): Speaker encodings.
    speaker: torch.Tensor

    # session (torch.LongTensor [1, batch_size (optional)]): Speaker recording session encodings.
    session: torch.Tensor

    # num_tokens (torch.LongTensor [1, batch_size (optional)] or None): Number of tokens in each
    #     sequence.
    num_tokens: typing.Optional[torch.Tensor] = None

    # tokens_mask (torch.BoolTensor [num_tokens, batch_size (optional)])
    tokens_mask: typing.Optional[torch.Tensor] = None


class SpectrogramModel(torch.nn.Module):
    """Sequence to sequence model from tokens to a spectrogram.

    TODO: Update our weight initialization to best practices like these:
    - https://github.com/pytorch/pytorch/issues/18182
    - Gated RNN init on last slide:
    https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture9.pdf
    - Kaiming init for RELu instead of Xavier:
    https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
    - Block orthogonal LSTM initilization:
    https://github.com/allenai/allennlp/pull/158
    - Kaiming and Xavier both assume the input has a mean of 0 and std of 1; therefore, the
    embeddings should be initialized with a normal distribution.
    - The PyTorch init has little backing:
    https://twitter.com/jeremyphoward/status/1107869607677681664

    TODO: Write a test ensuring that given a input with std 1.0 the output of the network also has
    an std of 1.0, following Kaiming's popular weight initialization approach:
    https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf

    Args:
        vocab_size: Maximum size of the vocabulary used to encode `tokens`.
        num_speakers
        num_sessions: The number of recording sessions.
        speaker_embedding_size: Size of the speaker embedding dimensions. The speaker embedding
            is composed of a speaker identifier and a recording session identifier.
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands").
        max_frames_per_token: The maximum sequential predictions to make before stopping.
        output_scalar: The output of this model is scaled up by this value.
        speaker_embed_dropout: The speaker embedding dropout probability.
        stop_threshold: If the stop probability exceeds this value, this model stops generating
            frames.
    """

    @configurable
    def __init__(
        self,
        vocab_size: int,
        num_speakers: int,
        num_sessions: int,
        speaker_embedding_size: int = HParam(),
        num_frame_channels: int = HParam(),
        max_frames_per_token: float = HParam(),
        output_scalar: float = HParam(),
        speaker_embed_dropout: float = HParam(),
        stop_threshold: float = HParam(),
    ):
        super().__init__()
        assert speaker_embedding_size % 2 == 0, "This must be even."
        self.max_frames_per_token = max_frames_per_token
        self.num_frame_channels = num_frame_channels
        self.stop_threshold = stop_threshold
        self.vocab_size = vocab_size
        self.num_speakers = num_speakers
        self.num_sessions = num_sessions
        self.embed_speaker = torch.nn.Embedding(num_speakers, speaker_embedding_size // 2)
        self.embed_session = torch.nn.Embedding(num_sessions, speaker_embedding_size // 2)
        self.speaker_embed_dropout = torch.nn.Dropout(speaker_embed_dropout)
        self.encoder = encoder.Encoder(vocab_size, speaker_embedding_size)
        self.decoder = decoder.Decoder(num_frame_channels, speaker_embedding_size)
        self.stop_sigmoid = torch.nn.Sigmoid()
        self.register_buffer("output_scalar", torch.tensor(output_scalar).float())
        self.grad_enabled = None

    def _is_stop(
        self,
        stop_token: torch.Tensor,
        num_tokens: torch.Tensor,
        window_start: torch.Tensor,
        reached_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        NOTE: This uses hard constraint to prevent stoppping unless all the characters were seen.
        TODO: Try training with the hard constraint for consistency with inference.

        Args:
            stop_token (torch.FloatTensor [*, batch_size, *])
            num_tokens (torch.LongTensor [batch_size])
            window_start (torch.LongTensor [batch_size])
            reached_max (torch.BoolTensor [batch_size])

        Returns:
            torch.BoolTensor [batch_size]
        """
        stop_token = stop_token.view(-1)
        larger_than_threshold = self.stop_sigmoid(stop_token) >= self.stop_threshold
        at_the_end = window_start >= num_tokens - self.decoder.attention.window_length
        return (larger_than_threshold & at_the_end) | reached_max

    def _infer_generator(
        self,
        inputs: Inputs,
        encoded_tokens: torch.Tensor,
        split_size: float,
        use_tqdm: bool,
        include_batch_dim: bool,
        **kwargs,
    ) -> Generator:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            ...
            tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            split_size
            use_tqdm: Add a progress bar for non-batch generation.
        """
        _, batch_size, _ = encoded_tokens.shape
        device = encoded_tokens.device

        assert (
            use_tqdm and batch_size == 1 or not use_tqdm
        ), "Progress bar not applicable for batch generation."

        hidden_state = None
        frames, stop_tokens, alignments = [], [], []
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_lengths = (inputs.num_tokens.float() * self.max_frames_per_token).long()
        max_lengths = torch.clamp(max_lengths, min=1)
        max_tokens = inputs.num_tokens.max().cpu().item() if use_tqdm else None
        progress_bar = tqdm(leave=True, unit="char(s)", total=max_tokens) if use_tqdm else None
        keep_going = lambda: (
            stopped.sum() < batch_size and lengths[~stopped].max() < max_lengths[~stopped].max()
        )
        maybe_squeeze = lambda t: t if include_batch_dim else t.squeeze(1)
        while keep_going():
            if self.grad_enabled is not None:
                assert torch.is_grad_enabled() == self.grad_enabled
            frame, stop_token, alignment, hidden_state = self.decoder(
                encoded_tokens,
                inputs.tokens_mask,
                inputs.num_tokens,
                inputs.speaker,
                hidden_state=hidden_state,
                **kwargs,
            )

            lengths[~stopped] += 1
            frame[:, stopped] *= 0
            reached_max = lengths == max_lengths
            window_start = hidden_state.attention_hidden_state.window_start
            stopped[self._is_stop(stop_token, inputs.num_tokens, window_start, reached_max)] = True

            frames.append(frame.squeeze(0) * self.output_scalar)
            stop_tokens.append(stop_token.squeeze(0))
            alignments.append(alignment.squeeze(0))

            if len(frames) > split_size or not keep_going():
                yield Infer(
                    maybe_squeeze(torch.stack(frames, dim=0)),
                    maybe_squeeze(torch.stack(stop_tokens, dim=0)),
                    maybe_squeeze(torch.stack(alignments, dim=0)),
                    maybe_squeeze(lengths.unsqueeze(0)),
                    maybe_squeeze(reached_max.unsqueeze(0)),
                )
                frames, stop_tokens, alignments = [], [], []

            if use_tqdm:
                half_window_length = self.decoder.attention.window_length // 2
                # NOTE: The `tqdm` will start at `half_window_length` and it'll end at negative
                # `half_window_length`; otherwise, it's an accurate representation of the
                # character progress.
                progress_bar.update(window_start.cpu().item() + half_window_length - progress_bar.n)

        if use_tqdm:
            progress_bar.close()

    def _make_inputs(self, params: Params) -> Inputs:
        assert params.tokens.dtype == torch.long
        assert params.tokens.shape[0] != 0, "`tokens` cannot be empty."
        # [num_tokens, batch_size] or [num_tokens] → [batch_size, num_tokens]
        tokens = params.tokens.view(params.tokens.shape[0], -1).transpose(0, 1)

        if params.num_tokens is None:
            assert tokens.shape[0] == 1, "Must provide `num_tokens` unless batch size is 1."
            num_tokens = torch.full((1,), tokens.shape[1], device=tokens.device, dtype=torch.long)
        else:
            assert params.num_tokens.dtype == torch.long
            # [1, batch_size] or [batch_size] or [] → [batch_size]
            num_tokens = params.num_tokens.view(-1)

        if params.tokens_mask is None:
            # NOTE: `lengths_to_mask` may cause a gpu-cpu synchronization, which may take sometime.
            # [batch_size, num_tokens]
            tokens_mask = lengths_to_mask(num_tokens, device=tokens.device)
        else:
            # [num_tokens] or [num_tokens, batch_size] → [batch_size, num_tokens]
            tokens_mask = params.tokens_mask.view(tokens.shape[1], -1).transpose(0, 1)

        assert params.speaker.dtype == torch.long
        speaker = params.speaker.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]
        assert params.session.dtype == torch.long
        session = params.session.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]

        # [batch_size] → [batch_size, speaker_embedding_size // 2]
        speaker = self.embed_speaker(speaker)
        # [batch_size] → [batch_size, speaker_embedding_size // 2]
        session = self.embed_session(session)
        speaker = self.speaker_embed_dropout(torch.cat([speaker, session], dim=1))

        return Inputs(tokens, speaker, num_tokens, tokens_mask)

    def _make_targets(
        self, frames: torch.Tensor, mask: typing.Optional[torch.Tensor] = None
    ) -> Targets:
        assert frames.dtype == torch.float
        assert frames.shape[0] != 0, "`target_frames` cannnot be empty."
        num_frames = frames.shape[0]
        device = frames.device
        # [num_frames, num_frame_channels] or [num_frames, batch_size, num_frame_channels] →
        # [num_frames, batch_size, num_frame_channels]
        target_frames = frames.view(num_frames, -1, self.num_frame_channels)

        if mask is None:
            # [num_frames, batch_size]
            target_mask = torch.ones(num_frames, target_frames.shape[1], device=device)
        else:
            # [num_frames] or [num_frames, batch_size] → [num_frames, batch_size]
            target_mask = mask.view(num_frames, -1)

        return Targets(target_frames / self.output_scalar, target_mask)

    def _forward(
        self,
        params: Params,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
    ) -> Forward:
        """Propagate the model forward for training.

        TODO: Explore speeding up training with `JIT`.

        Args:
            ...
            target_frames (torch.FloatTensor [num_frames, batch_size (optional),
                num_frame_channels]): Ground truth frames for "teacher forcing" and loss.
            target_mask (torch.BoolTensor [num_frames, batch_size (optional)])
        """
        include_batch_dim = len(params.tokens.shape) == 2
        inputs = self._make_inputs(params)
        targets = self._make_targets(target_frames, target_mask)
        out = self.decoder(
            self.encoder(inputs),
            inputs.tokens_mask,
            inputs.num_tokens,
            inputs.speaker,
            targets.frames,
        )
        frames = out.frames.masked_fill(~targets.mask.unsqueeze(2), 0) * self.output_scalar
        return Forward(
            frames if include_batch_dim else frames.squeeze(1),
            out.stop_tokens if include_batch_dim else out.stop_tokens.squeeze(1),
            out.alignments if include_batch_dim else out.alignments.squeeze(1),
        )

    def _generate(
        self,
        params: Params,
        split_size: float = 64,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
    ) -> Generator:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            ...
            split_size: The maximum length of a sequence returned by the generator.
            use_tqdm: If `True` then this adds a `tqdm` progress bar.
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a `logger.warning` will be logged.
        """
        with self._set_grad_enabled():
            inputs = self._make_inputs(params)
            yield from self._infer_generator(
                inputs=inputs,
                encoded_tokens=self.encoder(inputs),
                split_size=split_size,
                use_tqdm=use_tqdm,
                include_batch_dim=len(params.tokens.shape) == 2,
                token_skip_warning=token_skip_warning,
            )

    def _infer(self, *args, **kwargs) -> Infer:
        """Generates a spectrogram given `tokens`, `speaker`, etc."""
        kwargs.update({"split_size": float("inf")})
        items = list(self._generate(*args, **kwargs))
        assert len(items) == 1, "Invariant Violation: Double check `split_size` logic."
        item = items[0]
        if item.reached_max.sum() > 0:
            logger.warning("%d sequences reached max frames", item.reached_max.sum())
        return item

    def set_grad_enabled(self, enabled: typing.Optional[bool]):
        self.grad_enabled = enabled

    @contextlib.contextmanager
    def _set_grad_enabled(self):
        enable = self.grad_enabled
        with contextlib.nullcontext() if enable is None else torch.set_grad_enabled(enable):
            yield

    @typing.overload
    def __call__(
        self,
        params: Params,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
    ) -> Forward:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        params: Params,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.INFER] = Mode.INFER,
    ) -> Infer:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        params: Params,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.GENERATE] = Mode.GENERATE,
    ) -> Generator:
        ...  # pragma: no cover

    def __call__(self, *args, mode: Mode = Mode.FORWARD, **kwargs):
        return super().__call__(*args, mode=mode, **kwargs)

    def forward(self, *args, mode: Mode = Mode.FORWARD, **kwargs):
        """
        NOTE: The `forward` function is special, learn more:
        https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690

        NOTE: Since the `forward` function is required to be executed, we use the parameter `mode`
        to overload the function.
        """
        with self._set_grad_enabled():
            if mode == Mode.FORWARD:
                return self._forward(*args, **kwargs)
            elif mode == Mode.GENERATE:
                return self._generate(*args, **kwargs)
            else:
                return self._infer(*args, **kwargs)
