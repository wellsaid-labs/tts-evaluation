from collections import namedtuple

from hparams import configurable
from hparams import HParam
from torch import nn

import torch

from src.spectrogram_model.attention import LocationSensitiveAttention
from src.spectrogram_model.pre_net import PreNet

# For sequential prediction, decoder hidden state used to predict the next frame.
#
# Args:
#     last_attention_context (torch.FloatTensor [batch_size, encoder_output_size]): The
#         last predicted attention context.
#     initial_cumulative_alignment (torch.FloatTensor [batch_size, 1]): The cumulative alignment
#         padding value.
#     cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): The last
#         predicted attention alignment.
#     window_start (torch.LongTensor [batch_size]): During inference, the attention is
#         windowed to improve performance and this value determines the start of the window.
#     last_frame (torch.FloatTensor [1, batch_size, frame_channels], optional): The last
#         predicted frame.
#     lstm_one_hidden_state (tuple): The last hidden state of the first LSTM in Tacotron.
#     lstm_two_hidden_state (tuple): The last hidden state of the second LSTM in Tacotron.
#
AutoregressiveDecoderHiddenState = namedtuple('AutoregressiveDecoderHiddenState', [
    'last_attention_context', 'initial_cumulative_alignment', 'cumulative_alignment',
    'window_start', 'last_frame', 'lstm_one_hidden_state', 'lstm_two_hidden_state'
])


class AutoregressiveDecoder(nn.Module):
    """ Decodes the sequence hidden feature representation into a mel-spectrogram.

    SOURCE (Tacotron 2):
        The decoder is an autoregressive recurrent neural network which predicts a mel spectrogram
        from the encoded input sequence one frame at a time. The prediction from the previous time
        step is first passed through a small pre-net containing 2 fully connected layers of 256
        hidden ReLU units. We found that the pre-net acting as an information bottleneck was
        essential for learning attention. The prenet output and attention context vector are
        concatenated and passed through a stack of 2 uni-directional LSTM layers with 1024 units.
        The concatenation of the LSTM output and the attention context vector is projected through
        a linear transform to predict the target spectrogram frame.

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
        frame_channels (int): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")
        speaker_embedding_dim (int): Size of the speaker embedding dimensions.
        pre_net_hidden_size (int): Hidden size of the pre-net to use.
        lstm_hidden_size (int): Hidden size of both LSTM layers to use.
        encoder_output_size (int): The size of the attention context returned by the attention
            module.
        stop_net_dropout (float): The dropout probability of the stop net.
    """

    @configurable
    def __init__(self,
                 frame_channels,
                 speaker_embedding_dim,
                 pre_net_hidden_size=HParam(),
                 lstm_hidden_size=HParam(),
                 encoder_output_size=HParam(),
                 stop_net_dropout=HParam()):
        super().__init__()

        self.encoder_output_size = encoder_output_size
        self.frame_channels = frame_channels
        self.lstm_hidden_size = lstm_hidden_size
        hidden_size = lstm_hidden_size + self.encoder_output_size + speaker_embedding_dim

        self.initial_states = nn.Sequential(
            nn.Linear(speaker_embedding_dim + encoder_output_size,
                      speaker_embedding_dim + encoder_output_size), nn.ReLU(),
            nn.Linear(speaker_embedding_dim + encoder_output_size,
                      frame_channels + 1 + encoder_output_size))
        self.pre_net = PreNet(hidden_size=pre_net_hidden_size, frame_channels=frame_channels)
        self.lstm_layer_one = nn.LSTMCell(
            input_size=pre_net_hidden_size + self.encoder_output_size + speaker_embedding_dim,
            hidden_size=lstm_hidden_size)
        self.lstm_layer_two = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size)
        self.attention = LocationSensitiveAttention(query_hidden_size=lstm_hidden_size)
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=frame_channels)
        self.linear_stop_token = nn.Sequential(
            nn.Dropout(stop_net_dropout), nn.Linear(lstm_hidden_size, 1))

    def forward(self,
                encoded_tokens,
                tokens_mask,
                speaker,
                num_tokens,
                target_frames=None,
                hidden_state=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, encoder_output_size]):
                Batched set of encoded sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask where zero's
                represent padding in ``encoded_tokens``.
            speaker (torch.LongTensor [batch_size, speaker_embedding_dim]): Batched speaker
                encoding.
            num_tokens (torch.LongTensor [batch_size]): The number of tokens in each sequence.
            target_frames (torch.FloatTensor [num_frames, batch_size, frame_channels],
                optional): During training, ground truth frames for teacher-forcing.
            hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Predicted frames.
            stop_token (torch.FloatTensor [num_frames, batch_size]): Score for stopping.
            new_hidden_state (AutoregressiveDecoderHiddenState): For sequential prediction, decoder
                hidden state used to predict the next frame.
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
                alignment for every frame, stored for visualization and debugging.
        """
        assert target_frames is None or hidden_state is None, (
            "Either the decoder is"
            "conditioned on ``target_frames`` or the ``hidden_state`` but not both.")

        _, batch_size, _ = encoded_tokens.shape

        if hidden_state is None:
            (initial_frame, initial_cumulative_alignment,
             initial_attention_context) = self.initial_states(
                 torch.cat([speaker, encoded_tokens[0]],
                           dim=1)).split([self.frame_channels, 1, self.encoder_output_size], dim=-1)

        hidden_state = AutoregressiveDecoderHiddenState(
            last_attention_context=initial_attention_context,
            initial_cumulative_alignment=torch.abs(initial_cumulative_alignment),
            cumulative_alignment=None,
            window_start=None,
            last_frame=initial_frame.unsqueeze(0),
            lstm_one_hidden_state=None,
            lstm_two_hidden_state=None) if hidden_state is None else hidden_state

        # NOTE: Shift target frames backwards one step to be the source frames
        frames = hidden_state.last_frame if target_frames is None else torch.cat(
            [hidden_state.last_frame, target_frames[0:-1]])

        num_frames, _, _ = frames.shape

        (last_attention_context, initial_cumulative_alignment, cumulative_alignment, window_start,
         _, lstm_one_hidden_state, lstm_two_hidden_state) = hidden_state

        del hidden_state

        # [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, pre_net_hidden_size]
        pre_net_frames = self.pre_net(frames)

        # Iterate over all frames for incase teacher-forcing; in sequential prediction, iterates
        # over a single frame.
        updated_frames = []
        attention_contexts = []
        alignments = []
        cumulative_alignments = []
        for frame in pre_net_frames.split(1, dim=0):
            frame = frame.squeeze(0)

            # [batch_size, pre_net_hidden_size] (concat)
            # [batch_size, speaker_embedding_dim] (concat)
            # [batch_size, encoder_output_size] →
            # [batch_size, pre_net_hidden_size + encoder_output_size + speaker_embedding_dim]
            frame = torch.cat([frame, last_attention_context, speaker], dim=1)

            # frame [batch (batch_size),
            # input_size (pre_net_hidden_size + encoder_output_size + speaker_embedding_dim)]  →
            # [batch_size, lstm_hidden_size]
            lstm_one_hidden_state = self.lstm_layer_one(frame, lstm_one_hidden_state)
            frame = lstm_one_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attention_context [batch_size, encoder_output_size]
            last_attention_context, cumulative_alignment, alignment, window_start = self.attention(
                encoded_tokens=encoded_tokens,
                tokens_mask=tokens_mask,
                query=frame.unsqueeze(0),
                initial_cumulative_alignment=initial_cumulative_alignment,
                cumulative_alignment=cumulative_alignment,
                window_start=window_start)

            updated_frames.append(frame)
            attention_contexts.append(last_attention_context)
            alignments.append(alignment)
            cumulative_alignments.append(cumulative_alignment)

            del alignment  # Clear Memory
            del frame  # Clear Memory

        del encoded_tokens  # Clear Memory

        # [num_frames, batch_size, num_tokens]
        alignments = torch.stack(alignments, dim=0)
        # [num_frames, batch_size, num_tokens]
        cumulative_alignments = torch.stack(cumulative_alignments, dim=0)
        # [num_frames, batch_size, lstm_hidden_size]
        frames = torch.stack(updated_frames, dim=0)

        del updated_frames  # Clear Memory

        # [num_frames, batch_size, encoder_output_size]
        attention_contexts = torch.stack(attention_contexts, dim=0)

        # [batch_size, speaker_embedding_dim] →
        # [1, batch_size, speaker_embedding_dim]
        speaker = speaker.unsqueeze(0)

        # [1, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, speaker_embedding_dim]
        speaker = speaker.expand(num_frames, -1, -1)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, encoder_output_size] (concat)
        # [num_frames, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, lstm_hidden_size + encoder_output_size + speaker_embedding_dim]
        frames = torch.cat([frames, attention_contexts, speaker], dim=2)

        # frames [seq_len (num_frames), batch (batch_size),
        # input_size (lstm_hidden_size + encoder_output_size + speaker_embedding_dim)] →
        # [num_frames, batch_size, lstm_hidden_size]
        frames, lstm_two_hidden_state = self.lstm_layer_two(frames, lstm_two_hidden_state)

        # [num_frames, batch_size, pre_net_hidden_size + 2] →
        # [num_frames, batch_size]
        stop_token = self.linear_stop_token(frames).squeeze(2)

        # [num_frames, batch_size,
        #  lstm_hidden_size (concat) encoder_output_size (concat) speaker_embedding_dim] →
        # [num_frames, batch_size, frame_channels]
        frames = self.linear_out(torch.cat([frames, attention_contexts, speaker], dim=2))

        new_hidden_state = AutoregressiveDecoderHiddenState(
            last_attention_context=last_attention_context,
            initial_cumulative_alignment=initial_cumulative_alignment,
            cumulative_alignment=cumulative_alignment,
            window_start=window_start,
            last_frame=frames[-1].unsqueeze(0),
            lstm_one_hidden_state=lstm_one_hidden_state,
            lstm_two_hidden_state=lstm_two_hidden_state)

        return frames, stop_token, new_hidden_state, alignments.detach()
