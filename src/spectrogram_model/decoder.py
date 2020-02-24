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
#     last_attention_context (torch.FloatTensor [batch_size, attention_hidden_size]): The
#         last predicted attention context.
#     cumulative_alignment (torch.FloatTensor [batch_size, num_tokens]): The last
#         predicted attention alignment.
#     last_frame (torch.FloatTensor [1, batch_size, frame_channels], optional): The last
#         predicted frame.
#     lstm_one_hidden_state (tuple): The last hidden state of the first LSTM in Tacotron.
#     lstm_two_hidden_state (tuple): The last hidden state of the second LSTM in Tacotron.
#
AutoregressiveDecoderHiddenState = namedtuple('AutoregressiveDecoderHiddenState', [
    'last_attention_context', 'cumulative_alignment', 'last_frame', 'lstm_one_hidden_state',
    'lstm_two_hidden_state'
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
        attention_hidden_size (int): The size of the attention context returned by the attention
            module.
    """

    @configurable
    def __init__(self,
                 frame_channels,
                 speaker_embedding_dim,
                 pre_net_hidden_size=HParam(),
                 lstm_hidden_size=HParam(),
                 attention_hidden_size=HParam(),
                 min_spectrogram_magnitude=HParam()):
        super().__init__()

        self.attention_hidden_size = attention_hidden_size
        self.frame_channels = frame_channels
        self.min_spectrogram_magnitude = min_spectrogram_magnitude
        self.pre_net = PreNet(hidden_size=pre_net_hidden_size, frame_channels=frame_channels)
        self.lstm_layer_one = nn.LSTMCell(
            input_size=pre_net_hidden_size + self.attention_hidden_size + speaker_embedding_dim,
            hidden_size=lstm_hidden_size)
        hidden_size = lstm_hidden_size + self.attention_hidden_size + speaker_embedding_dim
        self.lstm_layer_two = nn.LSTM(input_size=hidden_size, hidden_size=lstm_hidden_size)
        self.attention = LocationSensitiveAttention(
            query_hidden_size=lstm_hidden_size, hidden_size=attention_hidden_size)
        self.linear_out = nn.Linear(in_features=hidden_size, out_features=frame_channels)
        self.linear_stop_token = nn.Linear(
            in_features=hidden_size - self.attention_hidden_size, out_features=1)

    def forward(self, encoded_tokens, tokens_mask, speaker, target_frames=None, hidden_state=None):
        """
        Args:
            encoded_tokens (torch.FloatTensor [num_tokens, batch_size, attention_hidden_size]):
                Batched set of encoded sequences.
            tokens_mask (torch.BoolTensor [batch_size, num_tokens]): Binary mask where zero's
                represent padding in ``encoded_tokens``.
            speaker (torch.LongTensor [batch_size, speaker_embedding_dim]): Batched speaker
                encoding.
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
        device = encoded_tokens.device

        hidden_state = AutoregressiveDecoderHiddenState(
            last_attention_context=torch.zeros(
                batch_size, self.attention_hidden_size, device=device),
            cumulative_alignment=None,
            last_frame=torch.full(
                (1, batch_size, self.frame_channels),
                fill_value=torch.log(torch.tensor(self.min_spectrogram_magnitude, device=device)),
                device=device),
            lstm_one_hidden_state=None,
            lstm_two_hidden_state=None) if hidden_state is None else hidden_state

        # NOTE: Shift target frames backwards one step to be the source frames
        frames = hidden_state.last_frame if target_frames is None else torch.cat(
            [hidden_state.last_frame, target_frames[0:-1]])

        num_frames, _, _ = frames.shape

        (last_attention_context, cumulative_alignment, _, lstm_one_hidden_state,
         lstm_two_hidden_state) = hidden_state

        del hidden_state

        # [num_frames, batch_size, frame_channels] →
        # [num_frames, batch_size, pre_net_hidden_size]
        frames = self.pre_net(frames)

        # Iterate over all frames for incase teacher-forcing; in sequential prediction, iterates
        # over a single frame.
        updated_frames = []
        attention_contexts = []
        alignments = []
        for frame in frames.split(1, dim=0):
            frame = frame.squeeze(0)

            # [batch_size, pre_net_hidden_size] (concat)
            # [batch_size, speaker_embedding_dim] (concat)
            # [batch_size, attention_hidden_size] →
            # [batch_size, pre_net_hidden_size + attention_hidden_size + speaker_embedding_dim]
            frame = torch.cat([frame, last_attention_context, speaker], dim=1)

            # frame [batch (batch_size),
            # input_size (pre_net_hidden_size + attention_hidden_size + speaker_embedding_dim)]  →
            # [batch_size, lstm_hidden_size]
            lstm_one_hidden_state = self.lstm_layer_one(frame, lstm_one_hidden_state)
            frame = lstm_one_hidden_state[0]

            # Initial attention alignment, sometimes refered to as attention weights.
            # attention_context [batch_size, attention_hidden_size]
            last_attention_context, cumulative_alignment, alignment = self.attention(
                encoded_tokens=encoded_tokens,
                tokens_mask=tokens_mask,
                query=frame.unsqueeze(0),
                cumulative_alignment=cumulative_alignment)

            updated_frames.append(frame)
            attention_contexts.append(last_attention_context)
            alignments.append(alignment.detach())

            del alignment  # Clear Memory
            del frame  # Clear Memory

        del encoded_tokens  # Clear Memory

        # [num_frames, batch_size, num_tokens]
        alignments = torch.stack(alignments, dim=0)
        # [num_frames, batch_size, lstm_hidden_size]
        frames = torch.stack(updated_frames, dim=0)

        del updated_frames  # Clear Memory

        # [num_frames, batch_size, attention_hidden_size]
        attention_contexts = torch.stack(attention_contexts, dim=0)

        # [batch_size, speaker_embedding_dim] →
        # [1, batch_size, speaker_embedding_dim]
        speaker = speaker.unsqueeze(0)

        # [1, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, speaker_embedding_dim]
        speaker = speaker.expand(num_frames, -1, -1)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, attention_hidden_size] (concat)
        # [num_frames, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, lstm_hidden_size + attention_hidden_size + speaker_embedding_dim]
        frames = torch.cat([frames, attention_contexts, speaker], dim=2)

        # frames [seq_len (num_frames), batch (batch_size),
        # input_size (lstm_hidden_size + attention_hidden_size + speaker_embedding_dim)] →
        # [num_frames, batch_size, lstm_hidden_size]
        frames, lstm_two_hidden_state = self.lstm_layer_two(frames, lstm_two_hidden_state)

        # [num_frames, batch_size, lstm_hidden_size] (concat)
        # [num_frames, batch_size, attention_hidden_size] (concat)
        # [num_frames, batch_size, speaker_embedding_dim] →
        # [num_frames, batch_size, lstm_hidden_size + attention_hidden_size + speaker_embedding_dim]

        # [num_frames, batch_size,
        #  lstm_hidden_size + attention_hidden_size + speaker_embedding_dim] →
        # [num_frames, batch_size, 1]
        stop_token = self.linear_stop_token(torch.cat([frames, speaker], dim=2)).squeeze(2)

        # [num_frames, batch_size,
        #  lstm_hidden_size + attention_hidden_size + speaker_embedding_dim] →
        # [num_frames, batch_size, frame_channels]
        frames = self.linear_out(torch.cat([frames, attention_contexts, speaker], dim=2))

        new_hidden_state = AutoregressiveDecoderHiddenState(
            last_attention_context=last_attention_context,
            cumulative_alignment=cumulative_alignment,
            last_frame=frames[-1].unsqueeze(0),
            lstm_one_hidden_state=lstm_one_hidden_state,
            lstm_two_hidden_state=lstm_two_hidden_state)

        return frames, stop_token, new_hidden_state, alignments
