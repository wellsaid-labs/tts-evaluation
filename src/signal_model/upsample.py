from torch import nn


class ConditionalFeaturesUpsample(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".
        * Inspired by SOTA super resolution model:
          https://github.com/pytorch/examples/tree/master/super_resolution

    Args:
        in_channels (int): Dimensionality of input features.
        out_channels (int): Dimensionality of outputed features.
        upsample_learned (int): Number of times to repeat frames with a learned upsampling.
        upsample_repeat (int): Number of times to repeat frames.
    """

    def __init__(self, in_channels=80, out_channels=64, upsample_learned=4, upsample_repeat=75):
        super().__init__()
        self.out_channels = out_channels
        self.upsample_repeat = upsample_repeat
        self.upsample_learned = upsample_learned

        self.upsample_length = nn.ConvTranspose2d(
            1, 1, kernel_size=(1, upsample_learned), stride=(1, upsample_learned))

        self.upsample_channels = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def _repeat(self, local_features):
        """ Repeat similar to this 3x repeat [1, 2, 3] → [1, 1, 1, 2, 2, 2, 3, 3, 3].

        Notes:
            * Learn more:
              https://stackoverflow.com/questions/35227224/torch-repeat-tensor-like-numpy-repeat

        Args:
            local_features (torch.FloatTensor [batch_size, in_channels, local_length]):
                Local features to repeat.

        Returns:
            local_features (torch.FloatTensor [batch_size, in_channels, local_length * repeat]):
                Local features repeated.
        """
        # [batch_size, in_channels, local_length] →
        # [batch_size, in_channels, local_length, 1]
        local_features = local_features.unsqueeze(3)

        # [batch_size, in_channels, local_length] →
        # [batch_size, in_channels, local_length, num_repeat]
        local_features = local_features.repeat(1, 1, 1, self.upsample_repeat)

        # [batch_size, in_channels, local_length, num_repeat] →
        # [batch_size, in_channels, local_length * num_repeat]
        return local_features.view(local_features.shape[0], local_features.shape[1], -1)

    def forward(self, local_features):
        """
        TODO: Support global conditioning

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, in_channels]):
                Local features to condition signal generation (e.g. spectrogram).

        Returns:
            conditional_features (torch.FloatTensor [batch_size, out_channels, signal_length]):
                Upsampled local conditional features.
        """
        batch_size, local_length, in_channels = local_features.shape

        # [batch_size, local_length, in_channels] →
        # [batch_size, in_channels, local_length]
        local_features = local_features.transpose(1, 2)

        # [batch_size, in_channels, local_length] →
        # [batch_size, 1, in_channels, local_length]
        local_features = local_features.unsqueeze(1)

        # [batch_size, in_channels, local_length] →
        # [batch_size, 1, in_channels, local_length * upsample_learned]
        local_features = self.upsample_length(local_features)

        # [batch_size, in_channels, local_length] →
        # [batch_size, in_channels, local_length * upsample_learned]
        local_features = local_features.squeeze(1)

        # [batch_size, in_channels, local_length * upsample_learned] →
        # [batch_size, out_channels, local_length * upsample_learned]
        local_features = self.upsample_channels(local_features)

        # [batch_size, out_channels, signal_length / upsample_repeat] →
        # [batch_size, out_channels, signal_length]
        return self._repeat(local_features)
