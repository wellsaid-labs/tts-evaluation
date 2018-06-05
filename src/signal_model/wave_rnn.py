from torch import nn
from tqdm import tqdm

import torch

from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.utils.configurable import configurable
from src.audio import mu_law_decode


class WaveRNN(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".

    Args:
        signal_channels (int): Number of bins used to encode signal. [Parameter ``A`` in NV-WaveNet]
        rnn_size (int): Hidden size of the recurrent unit.
        conditional_size (int): Size of the conditional features inputed to the RNN.
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        local_features_size (int): Dimensionality of local features.
        upsample_chunks (int): Control the memory used by ``upsample_layers`` by breaking the
            operation up into chunks.
        receptive_field_size (int, optional): Set the context required to predict target sample.
            This is parameter is used to match WaveNet context. Note that WaveRNN is not
            bounded by receptive field size; therefore, we can explore using truncated back prop
            instead.
    """

    @configurable
    def __init__(self,
                 signal_channels=256,
                 rnn_size=896,
                 conditional_size=256,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_features_size=80,
                 receptive_field_size=509):
        super().__init__()
        self.signal_channels = signal_channels
        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            out_channels=conditional_size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            in_channels=local_features_size,
            num_layers=1,
            upsample_chunks=1)
        self.rnn = nn.GRU(input_size=conditional_size, hidden_size=rnn_size, num_layers=1)
        self.embed = torch.nn.Conv1d(in_channels=1, out_channels=conditional_size, kernel_size=1)
        torch.nn.init.xavier_uniform_(
            self.embed.weight, gain=torch.nn.init.calculate_gain('linear'))
        self.out = nn.Sequential(
            nn.Linear(in_features=rnn_size, out_features=rnn_size),
            nn.ReLU(),
            nn.Linear(in_features=rnn_size, out_features=self.signal_channels),
            nn.LogSoftmax(dim=2),
        )
        torch.nn.init.xavier_uniform_(self.out[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(
            self.out[2].weight, gain=torch.nn.init.calculate_gain('linear'))
        self.receptive_field_size = receptive_field_size

    def queue_kernel_update(self, *args, **kwargs):
        """ Stub method to match WaveNet API """
        pass

    def _step(self, conditional_features, gold_signal, rnn_hidden_state=None):
        """
        Args:
            conditional_features (torch.FloatTensor [batch_size, conditional_size, signal_length]):
                Feature to condition signal generation (e.g. spectrogram).
            gold_signal (torch.FloatTensor [batch_size, signal_length], optional): Signal used for
                teacher-forcing.
            rnn_hidden_state (any): Hidden state based back to RNN.

        Returns:
            predicted_signal (torch.FloatTensor [batch_size, signal_channels, signal_length]):
                Categorical distribution over ``signal_channels`` energy levels. The predicted
                signal is one time step ahead of the gold signal.
            rnn_hidden_state (any): Updated hidden state returned from RNN.
        """
        # [batch_size, signal_length] → [batch_size, 1, signal_length]
        gold_signal = gold_signal.float().unsqueeze(1)

        # [batch_size, 1, signal_length] → [batch_size, conditional_size, signal_length]
        gold_signal = self.embed(gold_signal)

        # features [batch_size, conditional_size, signal_length]
        features = gold_signal + conditional_features

        del gold_signal
        del conditional_features

        # RNN operater expects input of the form:
        # [batch_size (batch), conditional_size (input_size), signal_length (seq_len)] →
        # [seq_len, batch, input_size]
        features = features.permute(2, 0, 1)

        # [seq_len, batch, input_size] → [signal_length, batch_size, rnn_size]
        features, rnn_hidden_state = self.rnn(features, rnn_hidden_state)

        # [signal_length, batch_size, rnn_size] → [signal_length, batch_size, signal_channels]
        predicted = self.out(features)

        # [signal_length, batch_size, signal_channels] →
        # [batch_size, signal_channels, signal_length]
        predicted = predicted.permute(1, 2, 0)

        return predicted, rnn_hidden_state

    def forward(self, local_features, gold_signal=None):
        """
        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            gold_signal (torch.FloatTensor [batch_size, signal_length], optional): Signal used for
                teacher-forcing.

        Returns:
            predicted_signal: Returns former if ``gold_signal is not None`` else returns latter.
                * (torch.FloatTensor [batch_size, signal_channels, signal_length]): Categorical
                    distribution over ``signal_channels`` energy levels.
                * (torch.FloatTensor [batch_size, signal_length]): Categorical
                    distribution over ``signal_channels`` energy levels.
                The predicted signal is one time step ahead of the gold signal.
        """
        # [batch_size, local_length, local_features_size] →
        # [batch_size, 1, conditional_size, signal_length]
        conditional_features = self.conditional_features_upsample(local_features)

        if gold_signal is not None:
            assert conditional_features.shape[3] == gold_signal.shape[1], (
                "Upsampling parameters in tangent with signal shape and local features shape " +
                "must be the same length after upsampling.")

        # [batch_size, 1, conditional_size, signal_length] →
        # [batch_size, conditional_size, signal_length]
        conditional_features = conditional_features.squeeze(1)

        if gold_signal is None:
            # next_sample [batch_size, 1 (signal_length)]
            next_sample = conditional_features.new_zeros(conditional_features.shape[0], 1)
            rnn_hidden_state = None

            return_ = []  # Predicted signal

            for i in tqdm(range(conditional_features.shape[2])):
                # [batch_size, conditional_size, signal_length] →
                # [batch_size, conditional_size, 1]
                conditional_features_step = conditional_features[:, :, i].unsqueeze(2)

                # [batch_size, signal_channels, 1 (signal_length)]
                prediction, rnn_hidden_state = self._step(
                    conditional_features=conditional_features_step,
                    gold_signal=next_sample,
                    rnn_hidden_state=rnn_hidden_state)

                # [batch_size, 1 (signal_length)]
                next_sample = prediction.max(dim=1)[1]

                return_.append(next_sample)

                # [batch_size, 1 (signal_length)]
                next_sample = mu_law_decode(next_sample)

            # [batch_size, signal_length]
            return torch.cat(return_, dim=1)
        else:
            return self._step(conditional_features=conditional_features, gold_signal=gold_signal)[0]
