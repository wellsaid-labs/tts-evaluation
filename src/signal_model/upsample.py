import torch

from torch import nn

from src.utils.configurable import configurable


class ConditionalFeaturesUpsample(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".

    Args:
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        local_features_size (int): Dimensionality of local features.
        block_hidden_size (int): Hidden size of each residual block.
        num_layers (int): Number of layers in Wavenet to condition.
    """

    @configurable
    def __init__(self,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 local_features_size=80,
                 block_hidden_size=64,
                 num_layers=24):
        super().__init__()
        self.block_hidden_size = block_hidden_size
        self.upsample_repeat = upsample_repeat
        self.num_layers = num_layers
        self.upsample_signal_length = None
        if upsample_convs is not None:
            self.upsample_signal_length = nn.Sequential(*tuple([
                nn.ConvTranspose1d(
                    in_channels=local_features_size,
                    out_channels=local_features_size,
                    kernel_size=size,
                    stride=size) for size in upsample_convs
            ]))
        self.upsample_layers = nn.Conv1d(
            in_channels=local_features_size,
            out_channels=block_hidden_size * 2 * num_layers,
            kernel_size=1)
        torch.nn.init.xavier_uniform_(
            self.upsample_layers.weight, gain=torch.nn.init.calculate_gain('tanh'))

    def forward(self, local_features):
        """
        TODO:
            * Support global conditioning

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, local_features_size]):
                Local features to condition signal generation (e.g. spectrogram).

        Returns:
            conditional_features (torch.FloatTensor [2 * block_hidden_size, batch_size, num_layers,
                signal_length]): Upsampled local conditional features.
        """
        # Convolution operater expects input_ of the form:
        # [batch_size, local_features_size, signal_length (local_length)]
        local_features = local_features.permute(0, 2, 1)

        # [batch_size, local_features_size, local_length] →
        # [batch_size, local_features_size, signal_length]
        if self.upsample_signal_length is not None:
            local_features = self.upsample_signal_length(local_features)
        local_features = local_features.repeat(1, 1, self.upsample_repeat)

        # [batch_size, local_features_size, signal_length] →
        # [batch_size, block_hidden_size * 2 * num_layers, signal_length]
        local_features = self.upsample_layers(local_features)

        batch_size, _, signal_length = local_features.shape

        # [batch_size, local_features_size, signal_length] →
        # [batch_size, num_layers, block_hidden_size * 2, signal_length]
        local_features = local_features.view(batch_size, self.num_layers,
                                             self.block_hidden_size * 2, signal_length)

        # [batch_size, num_layers, block_hidden_size * 2, signal_length] →
        # [block_hidden_size * 2, batch_size, num_layers, signal_length]
        return local_features.permute(2, 0, 1, 3)
