import logging

from hparams import configurable
from hparams import HParam
from torch import nn
from torchnlp.utils import lengths_to_mask
from tqdm import tqdm

import torch

from src.spectrogram_model.decoder import AutoregressiveDecoder
from src.spectrogram_model.encoder import Encoder
from src.utils import pad_tensors

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


class SpectrogramModel(nn.Module):
    """ Character sequence consumed to predict a spectrogram.

    SOURCE (Tacotron 2):
        The network is composed of an encoder and a decoder with attention. The encoder converts a
        character sequence into a hidden feature representation which the decoder consumes to
        predict a spectrogram. Input characters are represented using a learned 512-dimensional
        character embedding, which are passed through a stack of 3 convolutional layers each
        containing 512 filters with shape 5 × 1, i.e., where each filter spans 5 characters,
        followed by batch normalization [18] and ReLU activations. As in Tacotron, these
        convolutional layers model longer-term context (e.g., N-grams) in the input character
        sequence. The output of the final convolutional layer is passed into a single
        bi-directional [19] LSTM [20] layer containing 512 units (256 in each direction) to
        generate the encoded features.

        The encoder output is consumed by an attention network which summarizes the full encoded
        sequence as a fixed-length context vector for each decoder output step. We use the
        location-sensitive attention from [21], which extends the additive attention mechanism
        [22] to use cumulative attention weights from previous decoder time steps as an additional
        feature. This encourages the model to move forward consistently through the input,
        mitigating potential failure modes where some subsequences are repeated or ignored by the
        decoder. Attention probabilities are computed after projecting inputs and location
        features to 128-dimensional hidden representations. Location features are computed using
        32 1-D convolution filters of length 31.

        The decoder is an autoregressive recurrent neural network which predicts a mel spectrogram
        from the encoded input sequence one frame at a time. The prediction from the previous time
        step is first passed through a small pre-net containing 2 fully connected layers of 256
        hidden ReLU units. We found that the pre-net acting as an information bottleneck was
        essential for learning attention. The prenet output and attention context vector are
        concatenated and passed through a stack of 2 uni-directional LSTM layers with 1024 units.
        The concatenation of the LSTM output and the attention context vector is projected through
        a linear transform to predict the target spectrogram frame. Finally, the predicted mel
        spectrogram is passed through a 5-layer convolutional post-net which predicts a residual to
        add to the prediction to improve the overall reconstruction. Each post-net layer is
        comprised of 512 filters with shape 5 × 1 with batch normalization, followed by tanh
        activations on all but the final layer.

        In parallel to spectrogram frame prediction, the concatenation of decoder LSTM output and
        the attention context is projected down to a scalar and passed through a sigmoid activation
        to predict the probability that the output sequence has completed. This “stop token”
        prediction is used during inference to allow the model to dynamically determine when to
        terminate generation instead of always generating for a fixed duration. Specifically,
        generation completes at the first frame for which this probability exceeds a threshold of
        0.5.

        The convolutional layers in the network are regularized using dropout [25] with probability
        0.5, and LSTM layers are regularized using zoneout [26] with probability 0.1. In order to
        introduce output variation at inference time, dropout with probability 0.5 is applied only
        to layers in the pre-net of the autoregressive decoder.

      Reference:
          * Tacotron 2 Paper:
            https://arxiv.org/pdf/1712.05884.pdf

      Args:
        vocab_size (int): Maximum size of the vocabulary used to encode ``tokens``.
        num_speakers (int)
        speaker_embedding_dim (int): Size of the speaker embedding dimensions.
        frame_channels (int): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")
        max_frames_per_token (float): The maximum sequential predictions to make before
            quitting; Used for testing and defensive design.
        output_scalar (float): The output of the model is scaled up by this value.
        speaker_embed_dropout (float): The speaker embedding dropout probability.
      """

    @configurable
    def __init__(self,
                 vocab_size,
                 num_speakers,
                 speaker_embedding_dim=HParam(),
                 frame_channels=HParam(),
                 max_frames_per_token=HParam(),
                 output_scalar=HParam(),
                 speaker_embed_dropout=HParam()):
        super().__init__()

        self.max_frames_per_token = max_frames_per_token
        self.frame_channels = frame_channels

        self.encoder = Encoder(vocab_size, speaker_embedding_dim)
        self.decoder = AutoregressiveDecoder(
            frame_channels=frame_channels, speaker_embedding_dim=speaker_embedding_dim)
        self.stop_sigmoid = nn.Sigmoid()
        self.embed_speaker = nn.Sequential(
            nn.Embedding(num_speakers, speaker_embedding_dim), nn.Dropout(speaker_embed_dropout))

        self.register_buffer('output_scalar', torch.tensor(output_scalar).float())

    def _aligned(self, encoded_tokens, tokens_mask, speaker, num_tokens, target_frames,
                 target_lengths):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            tokens_mask (torch.BoolTensor [batch_size, num_tokens])
            speaker (torch.LongTensor [batch_size, speaker_embedding_dim]): Batched speaker
                encoding.
            num_tokens (torch.LongTensor [batch_size]): The number of tokens in each sequence.
            target_frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            target_lengths (torch.LongTensor [batch_size]): The number of frames in each sequence.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
        """
        device = target_frames.device
        target_frames = target_frames / self.output_scalar
        mask = lengths_to_mask(target_lengths, device=device)

        frames, stop_tokens, hidden_state, alignments = self.decoder(
            encoded_tokens, tokens_mask, speaker, num_tokens, target_frames=target_frames)
        frames = frames.masked_fill(~mask.transpose(0, 1).unsqueeze(2), 0)
        frames_with_residual = frames

        frames_with_residual = frames_with_residual * self.output_scalar
        frames = frames * self.output_scalar

        return frames, frames_with_residual, stop_tokens, alignments

    @configurable
    def _infer_generator_helper(self,
                                encoded_tokens,
                                split_size,
                                max_lengths,
                                num_tokens,
                                tokens_mask,
                                speaker,
                                use_tqdm=False,
                                stop_threshold=HParam()):
        """ Generate frames from the decoder until a stop is predicted or `max_lengths` is reached.

        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            split_size (int): The maximum length of a sequence returned by the generator.
            max_lengths (torch.LongTensor [batch_size]): The maximum length to generate before
                stopping the generation.
            num_tokens (torch.LongTensor [batch_size]): The number of tokens in each sequence.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens])
            speaker (torch.LongTensor [batch_size, speaker_embedding_dim])
            use_tqdm (bool, optional): If `True` then this adds a `tqdm` progress bar.
            stop_threshold (float): The threshold overwhich the model should stop generating.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [batch_size]): The total number of frames in each sequence,
                so far.
        """
        _, batch_size, _ = encoded_tokens.shape
        device = encoded_tokens.device

        hidden_state = None
        max_lengths = max_lengths.clone()
        frames, stop_tokens, alignments = [], [], []
        lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)
        progress_bar = tqdm(
            leave=True, unit='char(s)', total=num_tokens.max().cpu().item()) if use_tqdm else None
        keep_going = lambda: stopped.sum() < batch_size and lengths.max() < max_lengths.max()
        while keep_going():
            frame, stop_token, hidden_state, alignment = self.decoder(
                encoded_tokens, tokens_mask, speaker, num_tokens, hidden_state=hidden_state)

            lengths[~stopped] += 1
            frame[:, stopped] *= 0
            stopped[(self.stop_sigmoid(stop_token).squeeze(0) >= stop_threshold) |
                    (lengths == max_lengths)] = True
            max_lengths[stopped] = lengths[stopped]

            frames.append(frame.squeeze(0))
            stop_tokens.append(stop_token.squeeze(0))
            alignments.append(alignment.squeeze(0))

            if len(frames) > split_size or not keep_going():
                yield (torch.stack(frames, dim=0), torch.stack(stop_tokens, dim=0),
                       torch.stack(alignments, dim=0), lengths)
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

    def _infer_generator(self, encoded_tokens, split_size, *args, **kwargs):
        """ Generate frames from the decoder that have been processed by the `post_net`.

        Args:
            encoded_tokens: See `_infer_generator_helper`.
            split_size (int): The maximum length of a sequence returned by the generator.
            *args: Arguments passed too `_infer_generator_helper`.
            **kwargs: Keyword arguments passed too `_infer_generator_helper`.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [1, batch_size])
        """
        device = encoded_tokens.device
        padding = 1
        last_item = None
        is_stop = False
        generator = self._infer_generator_helper(encoded_tokens, split_size, *args, **kwargs)
        while not is_stop:
            items = []
            while sum([i[0].shape[0]
                       for i in items]) < max(padding * 2, split_size) and not is_stop:
                try:
                    frames, stop_tokens, alignments, lengths = next(generator)
                    mask = torch.clamp(lengths - lengths.max() + frames.shape[0], min=0)
                    mask = lengths_to_mask(mask, device=device)
                    items.append((frames, mask, stop_tokens, alignments, lengths))
                except StopIteration:
                    is_stop = True

            padding_tuple = (0 if last_item else padding, padding if is_stop else 0)
            frames = ([last_item[0][-padding * 2:]] if last_item else []) + [i[0] for i in items]
            frames = pad_tensors(torch.cat(frames), pad=padding_tuple)
            mask = ([last_item[1][:, -padding * 2:]] if last_item else []) + [i[1] for i in items]
            mask = pad_tensors(torch.cat(mask, dim=1), pad=padding_tuple, dim=1)
            stop_tokens = ([last_item[2][-padding:]] if last_item else []) + [i[2] for i in items]
            stop_tokens = torch.cat(stop_tokens)
            alignments = ([last_item[3][-padding:]] if last_item else []) + [i[3] for i in items]
            alignments = torch.cat(alignments)

            yield (
                frames[padding:-padding] * self.output_scalar,
                (frames[padding:-padding]) * self.output_scalar,
                stop_tokens[:None if is_stop else -padding],
                alignments[:None if is_stop else -padding],
                (lengths if is_stop else torch.min(lengths.max() - padding, lengths)).unsqueeze(0),
            )

            last_item = (frames, mask, stop_tokens, alignments)

    def _infer(self,
               encoded_tokens,
               num_tokens,
               *args,
               is_generator=False,
               filter_reached_max=False,
               split_size=32,
               **kwargs):
        """
        NOTE: The intermediate outputs are not masked according to the `length`.

        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_hidden_size])
            num_tokens (torch.LongTensor [batch_size])
            *args: Arguments passed too `_infer_generator`.
            is_generator (bool): If `True` this returns a generator over the sequence.
            filter_reached_max (bool): If `True` this filters the batch, removing
                any sequences that reached the max frames.
            split_size: See `_infer_generator_helper`.
            **kwargs: Keyword arguments passed too `_infer_generator`.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels])
            stop_token (torch.FloatTensor [num_frames, batch_size])
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            lengths (torch.LongTensor [1, batch_size])

        Returns (`is_generator == False`):
            reached_max (torch.BoolTensor [1, batch_size]): The spectrogram sequences that
                reached the maximum number of frames as defined by `max_frames_per_token`.
        """
        _, batch_size, _ = encoded_tokens.shape
        split_size = split_size if is_generator else float('inf')
        max_lengths = torch.clamp((num_tokens.float() * self.max_frames_per_token).long(), min=1)

        generator = self._infer_generator(encoded_tokens, split_size, max_lengths, num_tokens,
                                          *args, **kwargs)
        if is_generator:
            return generator

        generator = list(generator)
        assert len(generator) == 1, 'Invariant Violation: Double check `split_size` logic.'
        frames, frames_with_residual, stop_tokens, alignments, lengths = generator[0]
        reached_max = (lengths == max_lengths).view(1, batch_size)
        if reached_max.sum() > 0:
            logger.warning('%d sequences reached max frames', reached_max.sum())

        if filter_reached_max:
            filter_ = ~reached_max.squeeze(0)
            lengths = lengths[:, filter_]
            frames, frames_with_residual, stop_tokens, alignments = tuple([
                t[:lengths.squeeze().max(), filter_] if lengths.numel() > 0 else t[:, filter_]
                for t in [frames, frames_with_residual, stop_tokens, alignments]
            ])

        return frames, frames_with_residual, stop_tokens, alignments, lengths, reached_max

    def _normalize_inputs(self, tokens, speaker, num_tokens, target_frames, target_lengths):
        """ Normalize the inputs and check some argument invariants.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size] or [num_tokens])
            speaker (torch.LongTensor [1, batch_size] or [batch_size] or [])
            num_tokens (torch.LongTensor [1, batch_size] or [batch_size] or [] or None)
            target_frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or
                [num_frames, frame_channels] or None)
            target_lengths (torch.LongTensor [1, batch_size] or [batch_size] or [])

        Returns:
            tokens (torch.LongTensor [batch_size, num_tokens])
            speaker (torch.LongTensor [batch_size])
            num_tokens (torch.LongTensor [batch_size])
            target_frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or None)
            target_lengths (torch.LongTensor [batch_size] or None)
        """
        if tokens.dtype != torch.long:
            raise ValueError('The `tokens` dtype must be a `torch.long`.')
        if tokens.shape[0] == 0:
            raise ValueError('`tokens` cannot be empty.')
        if speaker.dtype != torch.long:
            raise ValueError('The `speaker` dtype must be a `torch.long`.')

        batch_size = tokens.shape[1] if len(tokens.shape) == 2 else 1

        # [num_tokens, batch_size] or [num_tokens] → [batch_size, num_tokens]
        tokens = tokens.view(tokens.shape[0], batch_size).transpose(0, 1)

        if target_frames is not None:
            if target_frames.dtype != torch.float:
                raise ValueError('The `target_frames` dtype must be a `torch.float`.')
            if target_frames.shape[0] == 0:
                raise ValueError('`target_frames` cannnot be empty.')
            # [num_frames, batch_size, frame_channels]
            target_frames = target_frames.view(target_frames.shape[0], batch_size, -1)

        speaker = speaker.view(batch_size)  # [batch_size]

        if target_lengths is not None:
            if target_lengths.dtype != torch.long:
                raise ValueError('The `target_lengths` dtype must be a `torch.long`.')
            target_lengths = target_lengths.view(batch_size)  # [batch_size]

        if num_tokens is not None:
            if num_tokens.dtype != torch.long:
                raise ValueError('The `num_tokens` dtype must be a `torch.long`.')
            num_tokens = num_tokens.view(batch_size)  # [batch_size]
        elif num_tokens is None and batch_size == 1:
            num_tokens = torch.full((batch_size,),
                                    tokens.shape[1],
                                    device=tokens.device,
                                    dtype=torch.long)  # [batch_size]

        assert num_tokens is not None, 'Must provide `num_tokens` unless batch size is 1.'
        return tokens, speaker, num_tokens, target_frames, target_lengths

    def forward(self,
                tokens,
                speaker,
                num_tokens=None,
                target_frames=None,
                target_lengths=None,
                **kwargs):
        """
        TODO: Explore speeding up training with `JIT`.

        Args:
            tokens (torch.LongTensor [num_tokens, batch_size] or [num_tokens]): Batched set
                of sequences.
            speaker (torch.LongTensor [1, batch_size] or [batch_size] or []): Batched
                speaker encoding.
            num_tokens (torch.LongTensor [batch_size] or [] or None): The number of tokens in
                each sequence.
            target_frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or
                [num_frames, frame_channels]): Ground truth frames to do aligned prediction.
            target_lengths (torch.LongTensor [batch_size] or []): The number of frames in
                each sequence.
            **kwargs: Other key word arugments passed to ``_infer`` or ``_aligned``.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or [num_frames,
                frame_channels]) Predicted frames.
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels]
                or [num_frames, frame_channels]): Predicted frames with the post net residual added.
            stop_token (torch.FloatTensor [num_frames, batch_size] or [num_frames]): Probablity of
                stopping at each frame.
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens] or [num_frames,
                num_tokens]): Attention alignments.

        Returns (`target_frames is None and target_lengths is None`):
            lengths (torch.LongTensor [1, batch_size] or [1]): Number of frames predicted for each
                sequence in the batch.

        Returns (`is_generator == False and target_frames is None and target_lengths is None`):
            reached_max (torch.BoolTensor [1, batch_size] or [1]): The spectrogram sequences that
                reached the maximum number of frames as defined by `max_frames_per_token`.
        """
        is_unbatched = len(tokens.shape) == 1

        tokens, speaker, num_tokens, target_frames, target_lengths = self._normalize_inputs(
            tokens, speaker, num_tokens, target_frames, target_lengths)

        # [batch_size, num_tokens]
        tokens_mask = lengths_to_mask(num_tokens, device=tokens.device)

        # [batch_size] → [batch_size, speaker_embedding_dim]
        speaker = self.embed_speaker(speaker)

        # [batch_size, num_tokens] → [num_tokens, batch_size, encoder_hidden_size]
        encoded_tokens = self.encoder(tokens, tokens_mask, speaker)

        if target_frames is None:
            return_ = self._infer(encoded_tokens, num_tokens, tokens_mask, speaker, **kwargs)
        else:
            return_ = self._aligned(encoded_tokens, tokens_mask, speaker, num_tokens, target_frames,
                                    target_lengths, **kwargs)

        if isinstance(return_, tuple):
            return tuple([t.squeeze(1) for t in return_]) if is_unbatched else return_

        return (tuple([t.squeeze(1) for t in item]) if is_unbatched else item for item in return_)
