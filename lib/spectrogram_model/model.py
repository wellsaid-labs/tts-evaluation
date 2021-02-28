import enum
import logging
import math
import typing

import torch
import torch.nn
from hparams import HParam, configurable
from torchnlp.utils import lengths_to_mask
from tqdm import tqdm

from lib.spectrogram_model.decoder import AutoregressiveDecoder
from lib.spectrogram_model.encoder import Encoder

logger = logging.getLogger(__name__)

SpectrogramModelGenerator = typing.Iterator[
    typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]


class Mode(enum.Enum):
    INFER: typing.Final = enum.auto()
    GENERATE: typing.Final = enum.auto()
    FORWARD: typing.Final = enum.auto()


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
      speaker_embedding_size: Size of the speaker embedding dimensions.
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
        speaker_embedding_size: int = HParam(),
        num_frame_channels: int = HParam(),
        max_frames_per_token: float = HParam(),
        output_scalar: float = HParam(),
        speaker_embed_dropout: float = HParam(),
        stop_threshold: float = HParam(),
    ):
        super().__init__()

        self.max_frames_per_token = max_frames_per_token
        self.num_frame_channels = num_frame_channels
        self.stop_threshold = stop_threshold
        self.vocab_size = vocab_size
        self.num_speakers = num_speakers

        self.embed_speaker = torch.nn.Sequential(
            torch.nn.Embedding(num_speakers, speaker_embedding_size),
            torch.nn.Dropout(speaker_embed_dropout),
        )
        self.encoder = Encoder(vocab_size, speaker_embedding_size)
        self.decoder = AutoregressiveDecoder(num_frame_channels, speaker_embedding_size)
        self.stop_sigmoid = torch.nn.Sigmoid()
        self.register_buffer("output_scalar", torch.tensor(output_scalar).float())

    def _is_stop(
        self,
        stop_token: torch.Tensor,
        num_tokens: torch.Tensor,
        window_start: torch.Tensor,
        reached_max: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            stop_token (torch.FloatTensor [*, batch_size, *])
            num_tokens (torch.LongTensor [batch_size])
            window_start (torch.LongTensor [batch_size])
            reached_max (torch.BoolTensor [batch_size])

        Returns:
            (torch.BoolTensor [batch_size])
        """
        stop_token = stop_token.view(-1)
        larger_than_threshold = self.stop_sigmoid(stop_token) >= self.stop_threshold
        # NOTE: This is a hard constraint to prevent stoppping unless all the characters were
        # seen.
        # TODO: Try training with the hard constraint for consistency with inference.
        at_the_end = window_start >= num_tokens - self.decoder.attention.window_length
        return (larger_than_threshold & at_the_end) | reached_max

    def _infer_generator(
        self,
        tokens: torch.Tensor,
        split_size: float,
        num_tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        speaker: torch.Tensor,
        use_tqdm: bool,
        **kwargs,
    ) -> SpectrogramModelGenerator:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            split_size
            num_tokens (torch.LongTensor [batch_size])
            tokens_mask (torch.BoolTensor [batch_size, num_tokens])
            speaker (torch.LongTensor [batch_size, speaker_embedding_size])
            use_tqdm: Add a progress bar for non-batch generation.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [1, batch_size])
            reached_max (torch.BoolTensor [1, batch_size])
        """
        _, batch_size, _ = tokens.shape
        device = tokens.device

        assert (
            use_tqdm and batch_size == 1 or not use_tqdm
        ), "Progress bar not applicable for batch generation."

        hidden_state = None
        frames, stop_tokens, alignments = [], [], []
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_lengths = torch.clamp((num_tokens.float() * self.max_frames_per_token).long(), min=1)
        max_tokens = num_tokens.max().cpu().item() if use_tqdm else None
        progress_bar = tqdm(leave=True, unit="char(s)", total=max_tokens) if use_tqdm else None
        keep_going = lambda: (
            stopped.sum() < batch_size and lengths[~stopped].max() < max_lengths[~stopped].max()
        )
        while keep_going():
            frame, stop_token, alignment, hidden_state = self.decoder(
                tokens, tokens_mask, num_tokens, speaker, hidden_state=hidden_state, **kwargs
            )

            lengths[~stopped] += 1
            frame[:, stopped] *= 0
            reached_max = lengths == max_lengths
            window_start = hidden_state.attention_hidden_state.window_start
            stopped[self._is_stop(stop_token, num_tokens, window_start, reached_max)] = True

            frames.append(frame.squeeze(0) * self.output_scalar)
            stop_tokens.append(stop_token.squeeze(0))
            alignments.append(alignment.squeeze(0))

            if len(frames) > split_size or not keep_going():
                yield (
                    torch.stack(frames, dim=0),
                    torch.stack(stop_tokens, dim=0),
                    torch.stack(alignments, dim=0),
                    lengths.unsqueeze(0),
                    reached_max.unsqueeze(0),
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

    def _normalize_inputs(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Normalize the representation of these arguments. """
        assert tokens.dtype == torch.long
        assert tokens.shape[0] != 0, "`tokens` cannot be empty."
        # [num_tokens, batch_size] or [num_tokens] → [batch_size, num_tokens]
        tokens = tokens.view(tokens.shape[0], -1).transpose(0, 1)

        if num_tokens is None:
            assert tokens.shape[0] == 1, "Must provide `num_tokens` unless batch size is 1."
            num_tokens = torch.full((1,), tokens.shape[1], device=tokens.device, dtype=torch.long)
        else:
            assert num_tokens.dtype == torch.long
            num_tokens = num_tokens.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]

        if tokens_mask is None:
            # NOTE: `lengths_to_mask` may cause a gpu-cpu synchronization, which may take sometime.
            # [batch_size, num_tokens]
            tokens_mask = lengths_to_mask(num_tokens, device=tokens.device)
        else:
            # [num_tokens] or [num_tokens, batch_size] → [batch_size, num_tokens]
            tokens_mask = tokens_mask.view(tokens.shape[1], -1).transpose(0, 1)

        assert speaker.dtype == torch.long
        speaker = speaker.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]
        return tokens, num_tokens, speaker, tokens_mask

    def _normalize_targets(
        self, target_frames: torch.Tensor, target_mask: typing.Optional[torch.Tensor] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Normalize the representation of the target (ground truth) tensors. """
        assert target_frames.dtype == torch.float
        assert target_frames.shape[0] != 0, "`target_frames` cannnot be empty."
        num_frames = target_frames.shape[0]
        device = target_frames.device
        # [num_frames, num_frame_channels] or [num_frames, batch_size, num_frame_channels] →
        # [num_frames, batch_size, num_frame_channels]
        target_frames = target_frames.view(num_frames, -1, self.num_frame_channels)

        if target_mask is None:
            # [num_frames, batch_size]
            target_mask = torch.ones(num_frames, target_frames.shape[1], device=device)
        else:
            # [num_frames] or [num_frames, batch_size] → [num_frames, batch_size]
            target_mask = target_mask.view(num_frames, -1)

        return target_frames / self.output_scalar, target_mask

    def _forward(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        target_frames: torch.Tensor,
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
        target_mask: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate the model forward for training.

        TODO: Explore speeding up training with `JIT`.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size (optional)]): Sequences.
            speaker (torch.LongTensor [1, batch_size (optional)]): Speaker encodings.
            target_frames (torch.FloatTensor [num_frames, batch_size (optional),
                num_frame_channels]): Ground truth frames for "teacher forcing" and loss.
            num_tokens (torch.LongTensor [1, batch_size (optional)] or None): Number of tokens in
                each sequence.
            tokens_mask (torch.BoolTensor [num_tokens, batch_size (optional)])
            target_mask (torch.BoolTensor [num_frames, batch_size (optional)])

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]):
                Spectrogram frames.
            stop_token (torch.FloatTensor [num_frames, batch_size (optional)]): Stopping probability
                for each frame.
            alignments (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                Attention alignment between `frames` and `tokens`.
        """
        is_batch = len(tokens.shape) == 2

        inputs = (tokens, speaker, num_tokens, tokens_mask)
        tokens, num_tokens, speaker, tokens_mask = self._normalize_inputs(*inputs)

        target_frames, target_mask = self._normalize_targets(target_frames, target_mask)

        speaker = self.embed_speaker(speaker)  # [batch_size] → [batch_size, speaker_embedding_size]
        # [batch_size, num_tokens] → [num_tokens, batch_size, encoder_hidden_size]
        encoded_tokens = self.encoder(tokens, tokens_mask, num_tokens, speaker)
        inputs = (encoded_tokens, tokens_mask, num_tokens, speaker, target_frames)
        frames, stop_tokens, alignments, _ = self.decoder(*inputs)

        frames = frames.masked_fill(~target_mask.unsqueeze(2), 0) * self.output_scalar
        return_ = (frames, stop_tokens, alignments)
        return return_ if is_batch else tuple([t.squeeze(1) for t in return_])  # type: ignore

    def _generate(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
    ) -> SpectrogramModelGenerator:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size (optional)]): Sequences.
            speaker (torch.LongTensor [1, batch_size (optional)]): Speaker encodings.
            num_tokens (torch.LongTensor [1, batch_size (optional)] or None): Number of tokens in
                each sequence.
            tokens_mask (torch.BoolTensor [num_tokens, batch_size (optional)])
            split_size: The maximum length of a sequence returned by the generator.
            use_tqdm: If `True` then this adds a `tqdm` progress bar.
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a `logger.warning` will be logged.

        Generator Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]):
                Spectrogram frames.
            stop_token (torch.FloatTensor [num_frames, batch_size (optional)]): Stopping probability
                for each frame.
            alignments (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                Attention alignment between `frames` and `tokens`.
            lengths (torch.LongTensor [1, batch_size (optional)]): The sequence length, so far.
            reached_max (torch.BoolTensor [1, batch_size (optional)]): If `True` the sequence has
                reached `self.max_frames_per_token`.
        """
        is_batch = len(tokens.shape) == 2
        inputs = (tokens, speaker, num_tokens, tokens_mask)
        tokens, num_tokens, speaker, tokens_mask = self._normalize_inputs(*inputs)

        speaker = self.embed_speaker(speaker)  # [batch_size] → [batch_size, speaker_embedding_size]
        # [batch_size, num_tokens] → [num_tokens, batch_size, encoder_hidden_size]
        encoded_tokens = self.encoder(tokens, tokens_mask, num_tokens, speaker)
        args = (encoded_tokens, split_size, num_tokens, tokens_mask, speaker, use_tqdm)
        generator = self._infer_generator(*args, token_skip_warning=token_skip_warning)

        squeeze_ = lambda t: t.squeeze(1)
        yield from ((i if is_batch else map(squeeze_, i)) for i in generator)  # type: ignore

    def _infer(
        self, *args, **kwargs
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates a spectrogram given `tokens`, `speaker`, etc.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), num_frame_channels]):
                Spectrogram frames.
            stop_token (torch.FloatTensor [num_frames, batch_size (optional)]): Stopping probability
                for each frame.
            alignments (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                Attention alignment between `frames` and `tokens`.
            lengths (torch.LongTensor [1, batch_size (optional)]): The sequence length.
            reached_max (torch.BoolTensor [1, batch_size (optional)]): If `True` the sequence has
                reached `self.max_frames_per_token`.
        """
        kwargs.update({"split_size": float("inf")})
        items = list(self._generate(*args, **kwargs))
        assert len(items) == 1, "Invariant Violation: Double check `split_size` logic."
        frames, stop_tokens, alignments, lengths, reached_max = items[0]
        if reached_max.sum() > 0:
            logger.warning("%d sequences reached max frames", reached_max.sum())
        return frames, stop_tokens, alignments, lengths, reached_max

    @typing.overload
    def __call__(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        target_frames: torch.Tensor,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
        target_mask: typing.Optional[torch.Tensor] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        mode: typing.Literal[Mode.INFER],
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        mode: typing.Literal[Mode.GENERATE],
        num_tokens: typing.Optional[torch.Tensor] = None,
        tokens_mask: typing.Optional[torch.Tensor] = None,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
    ) -> SpectrogramModelGenerator:
        ...  # pragma: no cover

    def __call__(
        self,
        *args,
        mode: Mode = Mode.FORWARD,
        **kwargs,
    ):
        return super().__call__(*args, mode=mode, **kwargs)

    def forward(
        self,
        *args,
        mode: Mode = Mode.FORWARD,
        **kwargs,
    ):
        """
        NOTE:
            - The `forward` function is special, learn more:
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690
            - Since the `forward` function is required to be executed, we use the parameter
              `mode` to overload the function.
        """
        if mode == Mode.FORWARD:
            return self._forward(*args, **kwargs)
        elif mode == Mode.GENERATE:
            return self._generate(*args, **kwargs)
        else:
            return self._infer(*args, **kwargs)
