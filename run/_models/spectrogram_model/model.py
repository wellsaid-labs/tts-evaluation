import contextlib
import enum
import logging
import math
import typing

import config as cf
import torch
import torch.nn
from tqdm import tqdm

from lib.distributed import NumeralizePadEmbed
from lib.utils import lengths_to_mask
from run._models.spectrogram_model import decoder, encoder
from run._models.spectrogram_model.containers import Encoded, Preds
from run._models.spectrogram_model.inputs import Inputs

logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    FORWARD: typing.Final = enum.auto()
    GENERATE: typing.Final = enum.auto()
    INFER: typing.Final = enum.auto()


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
        max_tokens: The maximum number of tokens the modelsee.
        max_seq_meta_vals: The maximum number of sequence metadata values the model see.
        max_token_meta_vals: The maximum number of token metadata values the model see.
        max_word_vector_size: The maximum size of `inputs.anno_embed("token_embed")`.
        max_seq_vector_size: The maximum size of the sequence embedding.
        max_anno_vector_size: The maximum number of annotation features.
        annos: The annotations to use along with their corresponding mask.
        seq_embed_size: The size of the sequence metadata embedding.
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands").
        output_scalar: The output of this model is scaled up by this value.
        stop_threshold: If the stop probability exceeds this value, this model stops generating
            frames.
        stop_token_eps: The stop probability assigned to the initial frames.
    """

    def __init__(
        self,
        max_tokens: int,
        max_seq_meta_vals: typing.Tuple[int, ...],
        max_token_meta_vals: typing.Tuple[int, ...],
        max_word_vector_size: int,
        max_seq_vector_size: int,
        max_anno_vector_size: int,
        annos: typing.List[typing.Tuple[str, str]],
        seq_embed_size: int,
        num_frame_channels: int,
        output_scalar: float,
        stop_threshold: float,
        stop_token_eps: float = 1e-10,
    ):
        super().__init__()
        self.num_frame_channels = num_frame_channels
        self.stop_threshold = stop_threshold
        self.max_tokens = max_tokens
        self.encoder = encoder.Encoder(
            max_tokens=max_tokens,
            max_token_meta_vals=max_token_meta_vals,
            max_word_vector_size=max_word_vector_size,
            max_seq_vector_size=max_seq_vector_size,
            max_anno_vector_size=max_anno_vector_size,
            annos=annos,
            max_seq_meta_vals=max_seq_meta_vals,
            seq_embed_size=seq_embed_size,
            **cf.get(),
        )
        self.decoder = decoder.Decoder(
            num_frame_channels=num_frame_channels,
            seq_embed_size=seq_embed_size,
            **cf.get(),
        )
        self.output_scalar: torch.Tensor
        self.register_buffer("output_scalar", torch.tensor(output_scalar).float())
        self.stop_token_eps: torch.Tensor
        self.register_buffer("stop_token_eps", torch.logit(torch.tensor(stop_token_eps)))
        self.grad_enabled = None

    def allow_unk_on_eval(self, val: bool):
        """If `True` then the "unknown token" may be used during evaluation, otherwise this will
        error if a new token is encountered during evaluation."""
        for mod in self.modules():
            if isinstance(mod, NumeralizePadEmbed):
                mod.allow_unk_on_eval = val

    def _mask_stop_token(
        self, stop_token: torch.Tensor, num_tokens: torch.Tensor, window_start: torch.Tensor
    ) -> torch.Tensor:
        """Only consider the `stop_token` if the last token is within the attention window.

        Args:
            stop_token (torch.FloatTensor [num_frames (optional), batch_size])
            num_tokens (torch.LongTensor [1 (optional), batch_size])
            window_start (torch.LongTensor [num_frames (optional), batch_size])

        Returns:
            stop_token (torch.FloatTensor [num_frames (optional), batch_size])
        """
        at_the_end = window_start >= num_tokens - self.decoder.attention.window_length // 2 - 1
        return stop_token.masked_fill(~at_the_end, self.stop_token_eps)

    def _is_stop(
        self,
        stop_token: torch.Tensor,
        num_tokens: torch.Tensor,
        window_start: torch.Tensor,
        reached_max: torch.Tensor,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: This uses hard constraint to prevent stoppping unless all the characters were seen.

        Args:
            stop_token (torch.FloatTensor [*, batch_size, *])
            num_tokens (torch.LongTensor [batch_size])
            window_start (torch.LongTensor [batch_size])
            reached_max (torch.BoolTensor [batch_size])

        Returns:
            torch.BoolTensor [batch_size]
        """
        stop_token = stop_token.view(-1)
        stop_token = self._mask_stop_token(stop_token, num_tokens, window_start)
        is_stop = (torch.sigmoid(stop_token) >= self.stop_threshold) | reached_max
        return is_stop, stop_token

    def _infer_generator(
        self, inputs: Inputs, encoded: Encoded, split_size: float, use_tqdm: bool, **kwargs
    ) -> typing.Generator[Preds, None, None]:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        TODO: Should we consider masking `alignments`, `stop_token`, also?

        Args:
            ...
            tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            split_size
            use_tqdm: Add a progress bar for non-batch generation.
        """
        _, batch_size, _ = encoded.tokens.shape
        device = encoded.tokens.device
        num_tokens = encoded.num_tokens

        assert (
            use_tqdm and batch_size == 1 or not use_tqdm
        ), "Progress bar not applicable for batch generation."

        hidden_state = None
        frames, stop_tokens, alignments = [], [], []
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_tokens = num_tokens.max().cpu().item() if use_tqdm else None
        progress_bar = tqdm(leave=True, unit="char(s)", total=max_tokens) if use_tqdm else None
        keep_going = lambda: (
            stopped.sum() < batch_size
            and lengths[~stopped].max() < inputs.max_audio_len[~stopped].max()
        )
        while keep_going():
            if self.grad_enabled is not None:
                assert torch.is_grad_enabled() == self.grad_enabled
            frame, stop_token, alignment, _, hidden_state = self.decoder(
                encoded, hidden_state=hidden_state, **kwargs
            )

            lengths[~stopped] += 1
            frame = frame.masked_fill(stopped.view(1, -1, 1), 0)
            hidden_state = hidden_state._replace(last_frame=frame)  # type: ignore
            reached_max = lengths == inputs.max_audio_len
            window_start = hidden_state.attention_hidden_state.window_start
            is_stop, stop_token = self._is_stop(stop_token, num_tokens, window_start, reached_max)
            stopped[is_stop] = True

            frames.append(frame.squeeze(0) * self.output_scalar)
            stop_tokens.append(stop_token)
            alignments.append(alignment.squeeze(0))

            if len(frames) > split_size or not keep_going():
                yield Preds(
                    frames=torch.stack(frames, dim=0),
                    stop_tokens=torch.stack(stop_tokens, dim=0),
                    alignments=torch.stack(alignments, dim=0),
                    num_frames=lengths,
                    frames_mask=lengths_to_mask(lengths),
                    num_tokens=encoded.num_tokens,
                    tokens_mask=encoded.tokens_mask,
                    reached_max=reached_max,
                )
                frames, stop_tokens, alignments = [], [], []

            if use_tqdm:
                assert progress_bar is not None
                progress_bar.update(int(window_start.cpu().item()) - progress_bar.n)

        if use_tqdm:
            assert progress_bar is not None
            progress_bar.close()

    def _forward(
        self,
        inputs: Inputs,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
    ) -> Preds:
        """Propagate the model forward for training.

        TODO: Explore speeding up training with `JIT`.

        Args:
            ...
            target_frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Ground
                truth frames for "teacher forcing" and loss.
            target_mask (torch.BoolTensor [num_frames, batch_size])
        """
        if target_mask is None:
            target_mask = torch.ones(*target_frames.shape[:1], device=target_frames.device)
        assert target_mask.shape[:1] == target_frames.shape[:1], "Shapes must align."
        target_frames = target_frames / self.output_scalar
        encoded = self.encoder(inputs)
        out = self.decoder(encoded, target_frames)
        frames = out.frames.masked_fill(~target_mask.unsqueeze(2), 0) * self.output_scalar
        num_tokens = encoded.num_tokens.unsqueeze(0)
        stop_tokens = self._mask_stop_token(out.stop_tokens, num_tokens, out.window_starts)
        num_frames = target_mask.sum(dim=0)
        return Preds(
            frames=frames,
            stop_tokens=stop_tokens,
            alignments=out.alignments,
            num_frames=target_mask.sum(dim=0),
            frames_mask=target_mask.transpose(0, 1),
            num_tokens=encoded.num_tokens,
            tokens_mask=encoded.tokens_mask,
            reached_max=num_frames >= inputs.max_audio_len,
        )

    def _generate(
        self,
        inputs: Inputs,
        split_size: float = 64,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
    ) -> typing.Generator[Preds, None, None]:
        """Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            ...
            split_size: The maximum length of a sequence returned by the generator.
            use_tqdm: If `True` then this adds a `tqdm` progress bar.
            token_skip_warning: If the attention skips more than `token_skip_warning`, then
                a `logger.warning` will be logged.
        """
        with self._set_grad_enabled():
            encoded = self.encoder(inputs)
            yield from self._infer_generator(
                inputs=inputs,
                encoded=encoded,
                split_size=split_size,
                use_tqdm=use_tqdm,
                token_skip_warning=token_skip_warning,
            )

    def _infer(self, *args, **kwargs) -> Preds:
        """Generate the entire output at once."""
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
        inputs: Inputs,
        target_frames: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        mode: typing.Literal[Mode.FORWARD] = Mode.FORWARD,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: Inputs,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.INFER] = Mode.INFER,
    ) -> Preds:
        ...  # pragma: no cover

    @typing.overload
    def __call__(
        self,
        inputs: Inputs,
        split_size: float = 32,
        use_tqdm: bool = False,
        token_skip_warning: float = math.inf,
        mode: typing.Literal[Mode.GENERATE] = Mode.GENERATE,
    ) -> typing.Generator[Preds, None, None]:
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
