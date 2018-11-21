from torch import nn
from torchnlp.text_encoders import PADDING_INDEX
from tqdm import tqdm

import torch

from src.spectrogram_model.decoder import AutoregressiveDecoder
from src.spectrogram_model.encoder import Encoder
from src.spectrogram_model.post_net import PostNet

from src.hparams import configurable


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
        frame_channels (int, optional): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")
      """

    @configurable
    def __init__(self, vocab_size, num_speakers, frame_channels=80):

        super().__init__()

        self.encoder = Encoder(vocab_size, num_speakers)
        self.decoder = AutoregressiveDecoder(frame_channels=frame_channels)
        self.post_net = PostNet(frame_channels=frame_channels)

    def _get_stopped_indexes(self, predictions, stop_threshold):
        """ Get a list of indices that predicted stop.

        Args:
            stop_token (torch.FloatTensor [1, batch_size]): Probablity of stopping.
            stop_threshold (float): The threshold probability for deciding to stop.

        Returns:
            (list) Indices that predicted stop.
        """
        stopped = predictions.data.view(-1).ge(stop_threshold).nonzero()
        if stopped.dim() > 1:
            return stopped.squeeze(1).tolist()
        else:
            return []

    def _add_residual(self, frames):
        """ Add residual to frames.

        Args:
            fames (torch.FloatTensor [num_frames, batch_size, frame_channels]) Predicted frames.

        Returns:
            fames (torch.FloatTensor [num_frames, batch_size, frame_channels]) Predicted frames.
        """
        # ``frames`` is expected to have shape `[num_frames, batch_size, frame_channels]`.
        # The post net expect input of shape `[batch_size, frame_channels, num_frames]`. We thus
        # need to permute the tensor first.
        residual = frames.permute(1, 2, 0)
        residual = self.post_net(residual)

        # In order to add frames with the residual, we need to permute for their sizes to be
        # compatible.
        # [batch_size, frame_channels, num_frames] → [num_frames, batch_size, frame_channels]
        residual = residual.permute(2, 0, 1)

        # [num_frames, batch_size, frame_channels] +
        # [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, frame_channels]
        frames_with_residual = frames.add(residual)

        del residual

        return frames_with_residual

    def infer(self, tokens, speaker, max_recursion=2000, stop_threshold=0.5):
        """
        Args:
            tokens (torch.LongTensor [num_tokens, batch_size] or [num_tokens]): Batched set of
                sequences.
            speaker (torch.LongTensor [batch_size]): Batched speaker encoding.
            max_recursion (int, optional): The maximum sequential predictions to make before
                quitting; Used for testing and defensive design.
            stop_threshold (float, optional): The threshold probability for deciding to stop.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]) Predicted frames.
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with the post net residual added.
            stop_token (torch.FloatTensor [num_frames, batch_size]): Probablity of stopping.
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
                alignments.
            lengths (list of int or None): Lengths of predicted spectrograms. None is returned
                in a unbatched execution of the function.
        """
        is_unbatched = len(tokens.shape) == 1
        if is_unbatched:
            # [num_tokens] → [num_tokens, batch_size(1)]
            tokens = tokens.unsqueeze(1)

        # [num_tokens, batch_size]  → [batch_size, num_tokens]
        tokens = tokens.transpose(0, 1)

        # [batch_size, num_tokens]
        tokens_mask = tokens.detach().eq(PADDING_INDEX)
        encoded_tokens = self.encoder(tokens, speaker)

        # [num_tokens, batch_size, hidden_size]
        _, batch_size, _ = encoded_tokens.shape

        stopped = set()
        hidden_state = None
        alignments, frames, stop_tokens = [], [], []
        lengths = [max_recursion] * batch_size
        progress_bar = tqdm(leave=True, unit='frame(s)')
        while len(stopped) != batch_size and len(frames) < max_recursion:
            frame, stop_token, hidden_state, alignment = self.decoder(
                encoded_tokens, tokens_mask, hidden_state=hidden_state)
            to_stop = self._get_stopped_indexes(stop_token, stop_threshold=stop_threshold)
            stopped.update(to_stop)

            # Zero out stopped frames
            frame[:, list(stopped)] *= 0

            # Store results
            frames.append(frame.squeeze(0))
            stop_tokens.append(stop_token.squeeze(0))
            alignments.append(alignment.squeeze(0))

            progress_bar.update(1)
            progress_bar.total = len(frames)

            for stop_index in to_stop:
                lengths[stop_index] = len(frames)

        progress_bar.close()
        alignments = torch.stack(alignments, dim=0)
        frames = torch.stack(frames, dim=0)
        stop_tokens = torch.stack(stop_tokens, dim=0)
        frames_with_residual = self._add_residual(frames)

        if is_unbatched:
            return (frames.squeeze(1), frames_with_residual.squeeze(1), stop_tokens.squeeze(1),
                    alignments.squeeze(1), None)

        return frames, frames_with_residual, stop_tokens, alignments, lengths

    def forward(self, tokens, speaker, ground_truth_frames):
        """
        Args:
            tokens (torch.LongTensor [num_tokens, batch_size] or [num_tokens]): Batched set of
                sequences.
            speaker (torch.LongTensor [batch_size]): Batched speaker encoding.
            ground_truth_frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or
                [num_frames, frame_channels]): During training, ground truth frames for
                teacher-forcing.
            max_recursion (int, optional): The maximum sequential predictions to make before
                quitting; Used for testing and defensive design.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels] or [num_frames,
                frame_channels]) Predicted frames.
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels]
                or [num_frames, frame_channels]): Predicted frames with the post net residual added.
            stop_token (torch.FloatTensor [num_frames, batch_size] or [num_frames]): Probablity of
                stopping.
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens] or [num_frames,
                num_tokens]): Attention alignments.
        """
        is_unbatched = len(tokens.shape) == 1 and len(ground_truth_frames.shape) == 2
        if is_unbatched:
            # [num_tokens] → [num_tokens, batch_size(1)]
            tokens = tokens.unsqueeze(1)
            # [num_frames, frame_channels] → [num_frames, batch_size(1), frame_channels]
            ground_truth_frames = ground_truth_frames.unsqueeze(1)

        # [num_tokens, batch_size]  → [batch_size, num_tokens]
        tokens = tokens.transpose(0, 1)

        # [batch_size, num_tokens]
        tokens_mask = tokens.detach().eq(PADDING_INDEX)
        encoded_tokens = self.encoder(tokens, speaker)

        frames, stop_tokens, hidden_state, alignments = self.decoder(
            encoded_tokens, tokens_mask, ground_truth_frames=ground_truth_frames)

        frames_with_residual = self._add_residual(frames)

        if is_unbatched:
            return (frames.squeeze(1), frames_with_residual.squeeze(1), stop_tokens.squeeze(1),
                    alignments.squeeze(1))

        return frames, frames_with_residual, stop_tokens, alignments
