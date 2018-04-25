from torch import nn
from torch.autograd import Variable

import torch

from src.configurable import configurable
from src.feature_model.pre_net import PreNet
from src.feature_model.post_net import PostNet
from src.feature_model.location_sensative_attention import LocationSensitiveAttention


class AutoregressiveDecoderHiddenState(object):

    def __init__(self, last_attention_context, last_attention_alignment, last_frame,
                 lstm_one_hidden_state, lstm_two_hidden_state):
        """ For sequential prediction, decoder hidden state used to predict the next frame.

        Args:
            last_attention_context (torch.FloatTensor [batch_size, attention_context_size]): The
                last predicted attention context.
            last_attention_alignment (torch.FloatTensor [num_tokens, batch_size]): The last
                predicted attention alignment.
            last_frame (torch.FloatTensor [1, batch_size, frame_channels], optional): The last
                predicted frame.
            lstm_one_hidden_state (tuple): The last hidden state of the first LSTM in Tacotron.
            lstm_two_hidden_state (tuple): The last hidden state of the second LSTM in Tacotron.
        """
        self.last_attention_alignment = last_attention_alignment
        self.last_attention_context = last_attention_context
        self.last_frame = last_frame
        self.lstm_one_hidden_state = lstm_one_hidden_state
        self.lstm_two_hidden_state = lstm_two_hidden_state


class AutoregressiveDecoder(nn.Module):
    """ Decodes the sequence hidden feature representation into a mel-spectrogram.

    TODO: Add dropout

    SOURCE (Tacotron 2):
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

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf

    Args:
        frame_channels (int, optional): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")
        pre_net_hidden_size (int): Hidden size of the pre-net to use.
        encoder_hidden_size (int): Hidden size of the encoder used; for reference.
        lstm_hidden_size (int): Hidden size of both LSTM layers to use.
    """

    @configurable
    def __init__(self,
                 frame_channels=80,
                 pre_net_hidden_size=256,
                 encoder_hidden_size=512,
                 lstm_hidden_size=1024):

        super(AutoregressiveDecoder, self).__init__()

        self.frame_channels = frame_channels
        # Is this case, the encoder hidden feature representation size directly informs the size
        # of the attention context.
        self.attention_context_size = encoder_hidden_size
        self.pre_net = PreNet(hidden_size=pre_net_hidden_size)
        self.lstm_layer_one = nn.LSTM(
            input_size=pre_net_hidden_size + self.attention_context_size,
            hidden_size=lstm_hidden_size,
            num_layers=1)
        self.lstm_layer_two = nn.LSTM(
            input_size=pre_net_hidden_size + self.attention_context_size,
            hidden_size=lstm_hidden_size,
            num_layers=1)
        self.attention = LocationSensitiveAttention(encoder_hidden_size=self.attention_context_size)
        self.linear_out = nn.Linear(
            in_features=lstm_hidden_size + self.attention_context_size, out_features=frame_channels)
        self.post_net = PostNet(frame_channels=self.frame_channels)
        self.linear_stop_token = nn.Sequential(
            nn.Linear(in_features=lstm_hidden_size + self.attention_context_size, out_features=1),
            nn.Sigmoid())

    def _get_past_frames(self, batch_size, ground_truth_frames=None, hidden_state=None):
        """ Get the past frames to condition the decoder on.

        Args:
            batch_size (int): Size of the batch; used to shape initital tensor.
            ground_truth_frames (torch.FloatTensor [num_frames, batch_size, frame_channels],
                optional): Ground truth frames for teacher-forcing.
            hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Ground truth frames
                shifted back one timestep if ``ground_truth_frames is not None``; otherwise, the
                initial or last frame is returned.
        """
        assert ground_truth_frames is None or hidden_state is None, ("""Either the decoder is
conditioned on ``ground_truth_frames`` or the ``hidden_state`` but not both.""")

        if hidden_state is not None:
            return hidden_state.last_frame

        # Tacotron 2 authors confirmed that initially the decoder is conditioned on a fixed zero
        # frame.
        initial_frame = torch.LongTensor(1, batch_size, self.frame_channels).zero_()
        initial_frame = Variable(initial_frame, requires_grad=False)
        if self.is_cuda:
            initial_frame = initial_frame.cuda()

        if ground_truth_frames is None:
            return initial_frame

        # Tacotron 2 authors use teacher-forcing on the ground truth during training.
        # To use the ``initial_frame`` we concat it onto the beginning; furthermore, it does not
        # need to use the very last frame of the ``ground_truth_frames`` so we remove it.
        return torch.cat([initial_frame, ground_truth_frames[0:-1]])

    def _get_last_attention_context(self, batch_size, hidden_state=None):
        """ Get the last attention context to condition the decoder on.

        Args:
            batch_size (int): Size of the batch; used to shape initital tensor.
            hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.

        Returns:
            last_attention_context (torch.FloatTensor [batch_size, attention_context_size]):
                Last attention context to condition the decoder.
        """
        if hidden_state is not None:
            return hidden_state.last_attention_context

        # Tacotron 2 authors confirmed that initially the decoder is conditioned on a fixed zero
        # attention context.
        initial_attention_context = torch.LongTensor(batch_size,
                                                     self.attention_context_size).zero_()
        initial_attention_context = Variable(initial_attention_context, requires_grad=False)
        if self.is_cuda:
            initial_attention_context = initial_attention_context.cuda()
        return initial_attention_context

    def _get_last_attention_alignment(self, num_tokens, batch_size, hidden_state=None):
        """ Get the last attention alignment to condition the decoder on.

        Args:
            batch_size (int): Size of the batch; used to shape initital tensor.
            hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.

        Returns:
            last_attention_alignment (torch.FloatTensor [num_tokens, batch_size]): The last
                predicted attention alignment.
        """
        if hidden_state is not None:
            return hidden_state.last_attention_alignment

        # Tacotron 2 authors confirmed that initially the decoder is conditioned on a fixed zero
        # attention context.
        initial_attention_alignment = torch.LongTensor(num_tokens, batch_size).zero_()
        initial_attention_alignment = Variable(initial_attention_alignment, requires_grad=False)
        if self.is_cuda:
            initial_attention_alignment = initial_attention_alignment.cuda()
        return initial_attention_alignment

    def forward(self, encoded_tokens, ground_truth_frames=None, hidden_state=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, hidden_size]): Batched set of
                encoded sequences.
            ground_truth_frames (torch.FloatTensor [num_frames, batch_size, frame_channels],
                optional): During training, ground truth frames for teacher-forcing.
            hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]) Predicted frames.
            frames_with_residual (torch.FloatTensor [num_frames, batch_size, frame_channels]):
                Predicted frames with the post net residual added.
            stop_token (torch.FloatTensor [num_frames, batch_size]): Probablity of stopping.
            new_hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.
        """
        assert ground_truth_frames is None or hidden_state is None, ("""Either the decoder is
conditioned on ``ground_truth_frames`` or the ``hidden_state`` but not both.""")

        # TODO: Attention
        # TODO: Dropout + Zoneout
        num_tokens, batch_size, _ = encoded_tokens.shape

        # frames [num_frames, batch_size, frame_channels]
        frames = self._get_past_frames(
            batch_size=batch_size,
            ground_truth_frames=ground_truth_frames,
            hidden_state=hidden_state)

        # [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, pre_net_hidden_size]
        frames = self.pre_net(frames)

        # last_attention_context [batch_size, attention_context_size]
        last_attention_context = self._get_last_attention_context(
            batch_size=batch_size, hidden_state=hidden_state)
        # last_attention_context [num_tokens, batch_size]
        last_attention_alignment = self._get_last_attention_alignment(
            num_tokens=num_tokens, batch_size=batch_size, hidden_state=hidden_state)

        # Tacotron 2 authors confirmed over email this concat (``torch.cat``) strategy.

        attention_contexts = []
        # Iterate over all frames for incase teacher-forcing; in sequential prediction, iterates
        # over a single frame.
        for frame in frames:
            # [batch_size, pre_net_hidden_size] (concat)
            # [batch_size, self.attention_context_size] →
            # [batch_size, pre_net_hidden_size + self.attention_context_size]
            frame = torch.cat([frame, last_attention_context], dim=1)

            # Unsqueeze to match the expected LSTM intput with 3 dimensional matrix.
            # [batch_size, pre_net_hidden_size + self.attention_context_size] →
            # [1, batch_size, pre_net_hidden_size + self.attention_context_size]
            frame = frame.unsqueeze(0)

            # frame [seq_len (1), batch (batch_size),
            # input_size (pre_net_hidden_size + self.attention_context_size)]  →
            # [num_frames, batch_size, lstm_hidden_size]
            if hidden_state is None:
                frame, lstm_one_hidden_state = self.lstm_layer_one(frame)
            else:
                frame, lstm_one_hidden_state = self.lstm_layer_one(
                    frame, hidden_state.lstm_one_hidden_state)

            # Initial attention alignment, sometimes refered to as attention weights.
            # attention_context [batch_size, self.attention_context_size]
            attention_context, last_attention_alignment = self.attention(
                encoded_tokens=encoded_tokens,
                query=frame,
                previous_alignment=last_attention_alignment)
            attention_contexts.append(attention_context)

        # attention_context [num_frames (len(frames)), batch_size, self.attention_context_size]
        attention_contexts = torch.stack(attention_contexts)

        # [num_frames, batch_size, pre_net_hidden_size] (concat)
        # [num_frames, batch_size, self.attention_context_size] →
        # [num_frames, batch_size, pre_net_hidden_size + self.attention_context_size]
        frames = torch.cat([frames, attention_contexts], dim=2)

        # frames [seq_len (num_frames), batch (batch_size),
        # input_size (pre_net_hidden_size + self.attention_context_size)]  →
        # [num_frames, batch_size, lstm_hidden_size]
        if hidden_state is None:
            frames, lstm_two_hidden_state = self.lstm_layer_two(frames)
        else:
            frames, lstm_two_hidden_state = self.lstm_layer_two(frames,
                                                                hidden_state.lstm_two_hidden_state)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, self.attention_context_size] →
        # [num_frames, batch_size, lstm_hidden_size + self.attention_context_size]
        frames = torch.cat([frames, attention_contexts], dim=2)

        # [num_frames, batch_size, lstm_hidden_size + self.attention_context_size] →
        # [num_frames, batch_size, frame_channels]
        frames = self.linear_out(frames)  # First predicted Mel-Spectrogram before the post-net

        # Predict the stop token
        # [num_frames, batch_size, lstm_hidden_size + self.attention_context_size] →
        # [num_frames, batch_size, 1]
        stop_token = self.linear_stop_token(frames)

        # Our input is expected to have shape `[num_frames, batch_size, frame_channels]`.
        # The post net expect input of shape `[batch_size, frame_channels, num_frames]`. We thus
        # need to permute the tensor first.
        residual = frames.permute(1, 2, 0)
        residual = self.post_net(residual)

        # In order to add frames with the residual, we need to permute for their sizes to be
        # compatible.
        residual = frames.permute(2, 0, 1)

        # [num_frames, batch_size, frame_channels] +
        # [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, frame_channels]
        frames_with_residual = frames.add(residual)

        new_hidden_state = AutoregressiveDecoderHiddenState(
            last_attention_context=last_attention_context,
            last_attention_alignment=last_attention_alignment,
            last_frame=frames,  # Frames without the residual is used to condition in Tacotron
            lstm_one_hidden_state=lstm_one_hidden_state,
            lstm_two_hidden_state=lstm_two_hidden_state)

        # Loss is computed on both the ``frames`` and ``frames_with_residual`` to aid convergance.
        return frames, frames_with_residual, stop_token, new_hidden_state
