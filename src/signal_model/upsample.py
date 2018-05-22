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
        self.upsample_repeat = upsample_repeat
        self.num_layers = num_layers
        self.upsample_convs = nn.Sequential(*tuple([
            nn.ConvTranspose1d(
                in_channels=local_features_size,
                out_channels=local_features_size,
                kernel_size=size,
                stride=size,
                bias=True) for size in upsample_convs
        ]))
        self.project_local_features = nn.Conv1d(
            in_channels=local_features_size, out_channels=block_hidden_size, kernel_size=1)

    def forward(self, local_features):
        """
        TODO:
            * Support global conditioning

        Args:
            local_features (torch.FloatTensor [local_length, batch_size, local_features_size]):
                Local features to condition signal generation (e.g. spectrogram).

        Returns:
            conditional_features (torch.FloatTensor [2 * block_hidden_size, batch_size, num_layers,
                signal_length]): Upsampled global and local conditional features.
        """
        # Convolution operater expects input_ of the form:
        # [batch_size, local_features_size, signal_length (local_length)]
        local_features = local_features.permute(1, 2, 0)

        # [batch_size, local_features_size, local_length] →
        # [batch_size, local_features_size, signal_length]
        local_features = self.upsample_convs(local_features)
        _, _, length = local_features.shape
        local_features = local_features.repeat(1, 1, self.upsample_repeat)

        # [batch_size, local_features_size, signal_length] →
        # [batch_size, block_hidden_size, signal_length]
        local_features = self.project_local_features(local_features)

        # [batch_size, block_hidden_size, signal_length] →
        # [block_hidden_size, batch_size, signal_length]
        local_features = local_features.transpose_(0, 1)

        # Stub for global features
        # global_features [block_hidden_size, batch_size, signal_length]
        global_features = local_features.new_zeros(local_features.shape)

        # Interweave features; the first ``block_hidden_size`` chunk conditions the nonlinearity
        # while the last ``block_hidden_size`` chunk conditions the ``sigmoid`` gate.
        global_features_left, global_features_right = tuple(torch.chunk(global_features, 2, dim=0))
        local_features_left, local_features_right = tuple(torch.chunk(local_features, 2, dim=0))

        # conditional_features [2 * block_hidden_size, batch_size, signal_length]
        conditional_features = torch.cat(
            (global_features_left, local_features_left, global_features_right,
             local_features_right),
            dim=0)

        # [2 * block_hidden_size, batch_size, signal_length] →
        # [2 * block_hidden_size, batch_size, 1, signal_length]
        conditional_features = conditional_features.unsqueeze(2)

        # [2 * block_hidden_size, batch_size, 1, signal_length] →
        # [2 * block_hidden_size, batch_size, num_layers, signal_length]
        conditional_features = conditional_features.repeat(1, 1, self.num_layers, 1)

        return conditional_features
