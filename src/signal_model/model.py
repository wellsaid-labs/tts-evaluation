import torch
import logging

from torch import nn

from src.signal_model.residual_block import ResidualBlock
from src.signal_model.upsample import ConditionalFeaturesUpsample
from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


def get_receptive_field_size(layers):
    """ Compute receptive field size.

    Args:
        layers (ResidualBlock): Layers of residual blocks used for Wavenet.

    Returns:
        receptive_field_size (int): Receptive field size in samples.
    """
    return sum([layer.dilation * (layer.kernel_size - 1) for layer in layers])


@configurable
class WaveNet(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".

    Args:
        mu (int): Mu used to encode signal with mu-law encoding.
        block_hidden_size (int): Hidden size of each residual block.
        num_layers (int): Number of residual layers to use with Wavenet.
        cycle_size (int): Cycles such that dilation is equal to:
            ``2**(i % cycle_size) {0 <= i < num_layers}``
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        local_features_size (int): Dimensionality of local features.
    """

    def __init__(self,
                 mu=255,
                 block_hidden_size=64,
                 skip_size=256,
                 num_layers=24,
                 cycle_size=6,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_features_size=80):
        super().__init__()

        self.mu = mu
        self.num_layers = num_layers
        self.embed = nn.Conv1d(
            in_channels=self.mu + 1, out_channels=block_hidden_size, kernel_size=1)
        self.layers = nn.ModuleList([
            ResidualBlock(
                hidden_size=block_hidden_size, dilation=2**(i % cycle_size), skip_size=skip_size)
            for i in range(num_layers)
        ])
        self.conditional_features_upsample = ConditionalFeaturesUpsample(
            block_hidden_size=block_hidden_size,
            upsample_repeat=upsample_repeat,
            upsample_convs=upsample_convs,
            local_features_size=local_features_size)
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(in_channels=skip_size, out_channels=self.mu + 1, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.mu + 1, out_channels=self.mu + 1, kernel_size=1),
            nn.Softmax(dim=1),
        )

        self.receptive_field_size = get_receptive_field_size(self.layers)
        logger.info('Receptive field size in samples: %d' % self.receptive_field_size)

    def forward(self, local_features, signal):
        """
        Args:
            local_features (torch.FloatTensor [local_length, batch_size, local_features_size]):
                Local feature to condition signal generation (e.g. spectrogram).
            signal (torch.FloatTensor [signal_length, batch_size]): Mu-law encoded signal used for
                teacher-forcing.

        Returns:
            predicted_signal (torch.FloatTensor [batch_size, mu + 1, signal_length]): Categorical
                distribution over ``mu + 1`` energy levels.
        """
        # [local_length, batch_size, local_features_size] →
        # [2 * block_hidden_size, batch_size, num_layers, signal_length]
        conditional_features = self.conditional_features_upsample(local_features)

        assert conditional_features.shape[3] == signal.shape[0], (
            "Upsampling parameters in tangent with signal shape and local features shape must " +
            "be partible")

        # [signal_length, batch_size] → [signal_length, batch_size, mu + 1]
        # Encode signal with one-hot encoding, reference:
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/25
        signal = (signal.unsqueeze(2) == torch.arange(self.mu + 1).reshape(1, 1,
                                                                           self.mu + 1)).float()

        # Convolution operater expects input_ of the form:
        # [batch_size, channels (mu + 1), signal_length]
        signal = signal.permute(1, 2, 0)

        # Using a convoluion, we compute an embedding from the one-hot encoding.
        # [batch_size, mu + 1, signal_length] → [batch_size, block_hidden_size, signal_length]
        signal_features = self.embed(signal)

        del signal

        # [batch_size, block_hidden_size, signal_length] →
        # [block_hidden_size, batch_size, signal_length]
        signal_features.transpose_(0, 1)

        cumulative_skip = None
        for i, layer in enumerate(self.layers):
            signal_features, skip = layer(
                signal_features=signal_features,
                conditional_features=conditional_features[:, :, i, :])

            if cumulative_skip is None:
                cumulative_skip = skip
            else:
                cumulative_skip += skip

        # [batch_size, skip_size, signal_length] → [batch_size, mu + 1, signal_length]
        return self.out(cumulative_skip)

    def export(self):
        pass
