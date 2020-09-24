import logging
import typing
import typing_extensions

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.utils import lengths_to_mask
from tqdm import tqdm

import torch

from lib.spectrogram_model.decoder import AutoregressiveDecoder
from lib.spectrogram_model.encoder import Encoder

logger = logging.getLogger(__name__)

# TODO: Update our weight initialization to best practices like these:
# - https://github.com/pytorch/pytorch/issues/18182
# - Gated RNN init on last slide:
# https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/lectures/lecture9.pdf
# - Kaiming init for RELu instead of Xavier:
# https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
# - Block orthogonal LSTM initilization:
# https://github.com/allenai/allennlp/pull/158
# - Kaiming and Xavier both assume the input has a mean of 0 and std of 1; therefore, the embeddings
# should be initialized with a normal distribution.
# - The PyTorch init has little backing:
# https://twitter.com/jeremyphoward/status/1107869607677681664

# TODO: Write a test ensuring that given a input with std 1.0 the output of the network also has
# an std of 1.0, following Kaiming's popular weight initialization approach:
# https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

SpectrogramModelGenerator = typing.Generator[typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor], None, None]


class SpectrogramModel(nn.Module):
    """ Sequence to sequence model from tokens to a spectrogram.

      Reference:
          * Tacotron 2 Paper:
            https://arxiv.org/pdf/1712.05884.pdf

      Args:
        vocab_size: Maximum size of the vocabulary used to encode ``tokens``.
        num_speakers
        speaker_embedding_dim: Size of the speaker embedding dimensions.
        frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands")
        max_frames_per_token: The maximum sequential predictions to make before quitting; Used for
            testing and defensive design.
        output_scalar: The output of the model is scaled up by this value.
        speaker_embed_dropout: The speaker embedding dropout probability.
        stop_threshold: The threshold overwhich the model should stop generating.
      """

    @configurable
    def __init__(self,
                 vocab_size: int,
                 num_speakers: int,
                 speaker_embedding_dim: int = HParam(),
                 frame_channels: int = HParam(),
                 max_frames_per_token: float = HParam(),
                 output_scalar: float = HParam(),
                 speaker_embed_dropout: float = HParam(),
                 stop_threshold: float = HParam()):
        super().__init__()

        self.max_frames_per_token = max_frames_per_token
        self.frame_channels = frame_channels
        self.stop_threshold = stop_threshold

        self.encoder = Encoder(vocab_size, speaker_embedding_dim)
        self.decoder = AutoregressiveDecoder(
            frame_channels=frame_channels, speaker_embedding_dim=speaker_embedding_dim)
        self.stop_sigmoid = nn.Sigmoid()
        self.embed_speaker = nn.Sequential(
            nn.Embedding(num_speakers, speaker_embedding_dim), nn.Dropout(speaker_embed_dropout))

        self.register_buffer('output_scalar', torch.tensor(output_scalar).float())

        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def _infer_generator(self, encoded_tokens: torch.Tensor, split_size: float,
                         num_tokens: torch.Tensor, tokens_mask: torch.Tensor, speaker: torch.Tensor,
                         use_tqdm: bool) -> SpectrogramModelGenerator:
        """ Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            split_size
            num_tokens (torch.LongTensor [batch_size])
            tokens_mask (torch.BoolTensor [batch_size, num_tokens])
            speaker (torch.LongTensor [batch_size, speaker_embedding_dim])
            use_tqdm

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [1, batch_size])
            reached_max (torch.BoolTensor [1, batch_size])
        """
        _, batch_size, _ = encoded_tokens.shape
        device = encoded_tokens.device

        hidden_state = None
        frames, stop_tokens, alignments = [], [], []
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_lengths = torch.clamp((num_tokens.float() * self.max_frames_per_token).long(), min=1)
        max_tokens = num_tokens.max().cpu().item()
        progress_bar = tqdm(leave=True, unit='char(s)', total=max_tokens) if use_tqdm else None
        keep_going = lambda: (
            stopped.sum() < batch_size and lengths[~stopped].max() < max_lengths[~stopped].max())
        while keep_going():
            frame, stop_token, hidden_state, alignment = self.decoder(
                encoded_tokens, tokens_mask, speaker, num_tokens, hidden_state=hidden_state)

            lengths[~stopped] += 1
            frame[:, stopped] *= 0
            # NOTE: This is a hard constraint to prevent stoppping unless all the characters were
            # seen.
            # TODO: Train with this logic, and incorporate a similar signal into the stop token.
            has_stopped = (self.stop_sigmoid(stop_token).squeeze(0) >= self.stop_threshold) & (
                hidden_state.window_start >= num_tokens - self.decoder.attention.window_length)
            reached_max = lengths == max_lengths
            stopped[has_stopped | reached_max] = True

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
                progress_bar.update(hidden_state.window_start.cpu().item() + half_window_length -
                                    progress_bar.n)

        if use_tqdm:
            progress_bar.close()

    def _normalize_inputs(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        num_tokens: typing.Optional[torch.Tensor] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Normalize the representation of these inputs. """
        assert tokens.dtype == torch.long
        assert tokens.shape[0] != 0, '`tokens` cannot be empty.'
        # [num_tokens, batch_size] or [num_tokens] → [batch_size, num_tokens]
        tokens = tokens.view(tokens.shape[0], -1).transpose(0, 1)

        if num_tokens is None:
            assert tokens.shape[0] == 1, 'Must provide `num_tokens` unless batch size is 1.'
            num_tokens = torch.full((1,), tokens.shape[1], device=tokens.device, dtype=torch.long)
        else:
            assert num_tokens.dtype == torch.long
            num_tokens = num_tokens.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]

        assert speaker.dtype == torch.long
        speaker = speaker.view(-1)  # [1, batch_size] or [batch_size] or [] → [batch_size]
        return tokens, num_tokens, speaker

    def _normalize_targets(
        self,
        target_frames: torch.Tensor,
        target_lengths: typing.Optional[torch.Tensor] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """ Normalize the representation of the ground truth / target tensors. """
        assert target_frames.dtype == torch.float
        assert target_frames.shape[0] != 0, '`target_frames` cannnot be empty.'
        num_frames = target_frames.shape[0]
        device = target_frames.device
        # [num_frames, frame_channels] or [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, frame_channels]
        target_frames = target_frames.view(num_frames, -1, self.frame_channels)

        if target_lengths is None:
            assert num_frames == 1, ('Must provide `target_lengths` unless batch size is 1.')
            target_lengths = torch.full((1,), num_frames, device=device, dtype=torch.long)
        else:
            assert target_lengths.dtype == torch.long
            # [1, batch_size] or [batch_size] or [] → [batch_size]
            target_lengths = target_lengths.view(-1)

        return target_frames, target_lengths

    def _forward(
        self,
        tokens: torch.Tensor,
        speaker: torch.Tensor,
        target_frames: torch.Tensor,
        target_stop_token: torch.Tensor,
        num_tokens: typing.Optional[torch.Tensor] = None,
        target_lengths: typing.Optional[torch.Tensor] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Propagate the model forward for training.

        TODO: Explore speeding up training with `JIT`.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size (optional)]): Batched set of sequences.
            speaker (torch.LongTensor [1, batch_size (optional)]): Batched speaker encodings.
            target_frames (torch.FloatTensor [num_frames, batch_size (optional), frame_channels]):
                Ground truth frames for "teacher forcing" and loss.
            target_stop_token (torch.FloatTensor [num_frames, batch_size (optional)])
            num_tokens (torch.LongTensor [batch_size] or None): Number of tokens in each sequence.
            target_lengths (torch.LongTensor [batch_size] or None): Number of frames in each
                sequence.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), frame_channels]):
                Spectrogram frames.
            stop_token (torch.FloatTensor [num_frames, batch_size (optional)]): Stopping probability
                for each frame.
            alignments (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                Attention alignment between `frames` and `tokens`.
            spectrogram_loss (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                The difference between `target_frames` and predicted frames.
            stop_token_loss (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                The difference between `target_stop_token` and predicted stop token.
        """
        is_batch = len(tokens.shape) == 2
        tokens, num_tokens, speaker = self._normalize_inputs(tokens, speaker, num_tokens)
        target_frames, target_lengths = self._normalize_targets(target_frames, target_lengths)
        tokens_mask = lengths_to_mask(num_tokens, device=tokens.device)  # [batch_size, num_tokens]
        # [num_frames, batch_size]
        frames_mask = lengths_to_mask(target_lengths, device=target_frames.device).transpose(0, 1)

        speaker = self.embed_speaker(speaker)  # [batch_size] → [batch_size, speaker_embedding_dim]
        # [batch_size, num_tokens] → [num_tokens, batch_size, encoder_hidden_size]
        encoded_tokens = self.encoder(tokens, tokens_mask, speaker)
        frames, stop_tokens, hidden_state, alignments = self.decoder(
            encoded_tokens, tokens_mask, speaker, num_tokens, target_frames / self.output_scalar)

        frames = frames.masked_fill(~frames_mask.unsqueeze(2), 0) * self.output_scalar
        # [num_frames, batch_size, frame_channels] → [num_frames, batch_size, frame_channels]
        spectrogram_loss = self.mse_loss(frames, target_frames) * frames_mask.unsqueeze(2)
        # [num_frames, batch_size] → [num_frames, batch_size]
        stop_token_loss = self.bce_loss(stop_tokens, target_stop_token) * frames_mask

        return_ = (frames, stop_tokens, alignments, spectrogram_loss, stop_token_loss)
        return return_ if is_batch else tuple([t.squeeze(1) for t in return_])  # type: ignore

    def _generate(self,
                  tokens: torch.Tensor,
                  speaker: torch.Tensor,
                  num_tokens: typing.Optional[torch.Tensor] = None,
                  split_size: float = 32,
                  use_tqdm: bool = False) -> SpectrogramModelGenerator:
        """ Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size (optional)]): Batched set of sequences.
            speaker (torch.LongTensor [1, batch_size (optional)]): Batched speaker encodings.
            num_tokens (torch.LongTensor [batch_size] or None): Number of tokens in each sequence.
            split_size: The maximum length of a sequence returned by the generator.
            use_tqdm: If `True` then this adds a `tqdm` progress bar.

        Generator Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), frame_channels]):
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
        tokens, num_tokens, speaker = self._normalize_inputs(tokens, speaker, num_tokens)
        tokens_mask = lengths_to_mask(num_tokens, device=tokens.device)  # [batch_size, num_tokens]

        speaker = self.embed_speaker(speaker)  # [batch_size] → [batch_size, speaker_embedding_dim]
        # [batch_size, num_tokens] → [num_tokens, batch_size, encoder_hidden_size]
        encoded_tokens = self.encoder(tokens, tokens_mask, speaker)
        generator = self._infer_generator(encoded_tokens, split_size, num_tokens, tokens_mask,
                                          speaker, use_tqdm)

        squeeze_ = lambda t: t.squeeze(1)
        yield from ((i if is_batch else map(squeeze_, i)) for i in generator)  # type: ignore

    def _infer(
        self,
        *args,
        filter_reached_max: bool = False,
        **kwargs
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Generates a spectrogram given `tokens`, `speaker`, etc.

        Args:
            *args: See `self.generate` to learn more.
            filter_reached_max: If `True` filter out sequences that overflowed from the batch.
            **kwargs: See `self.generate` to learn more.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size (optional), frame_channels]):
                Spectrogram frames.
            stop_token (torch.FloatTensor [num_frames, batch_size (optional)]): Stopping probability
                for each frame.
            alignments (torch.FloatTensor [num_frames, batch_size (optional), num_tokens]):
                Attention alignment between `frames` and `tokens`.
            lengths (torch.LongTensor [1, batch_size (optional)]): The sequence length.
            reached_max (torch.BoolTensor [1, batch_size (optional)]): If `True` the sequence has
                reached `self.max_frames_per_token`.
        """
        # TODO: Update typing once this issue is resolved https://github.com/python/mypy/issues/2582
        items = list(self._generate(*args, split_size=float('inf'), **kwargs))  # type: ignore
        assert len(items) == 1, 'Invariant Violation: Double check `split_size` logic.'
        frames, stop_tokens, alignments, lengths, reached_max = items[0]
        if reached_max.sum() > 0:
            logger.warning('%d sequences reached max frames', reached_max.sum())
        if filter_reached_max:
            filter_ = ~reached_max.squeeze(0)
            lengths = lengths[:, filter_]
            frames, stop_tokens, alignments = tuple([
                t[:int(lengths.squeeze().max()), filter_] if lengths.numel() > 0 else t[:, filter_]
                for t in [frames, stop_tokens, alignments]
            ])
        return frames, stop_tokens, alignments, lengths, reached_max

    def forward(self,
                *args,
                mode: typing_extensions.Literal['infer', 'generate', 'forward'] = 'forward',
                **kwargs):
        """
        NOTE:
            - The `forward` function is special, learn more:
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690
            - Since the `forward` function is required to be executed, we use the parameter
              `mode` to overload the function.
        """
        if mode == 'forward':
            return self._forward(*args, **kwargs)
        elif mode == 'generate':
            return self._generate(*args, **kwargs)
        else:
            return self._infer(*args, **kwargs)
