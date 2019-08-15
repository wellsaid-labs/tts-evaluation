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
        num_filters (list of int): Filters to be used with each kernel. The last kernel is used
            for upsampling the length.
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        kernels (list of tuples): Sizes of kernels used for upsampling, every kernel has an
            associated number of filters.
    """

    def __init__(self, in_channels, out_channels, num_filters, upsample_repeat, kernels):
        super().__init__()
        self.out_channels = out_channels
        self.upsample_repeat = upsample_repeat
        self.padding = sum([(kernel[0] - 1) for kernel in kernels])

        assert all(all(s % 2 == 1 and s < in_channels for s in kernel) for kernel in kernels), (
            'Kernel size must be odd and must be less than ``in_channels``')
        assert self.padding % 2 == 0, 'Padding invariant violated.'

        self.initial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters[0],
            kernel_size=kernels[0],
            padding=tuple([int((s - 1) / 2) for s in kernels[0]]))

        self.pre_net = nn.Sequential(*[
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=(num_filters[0] if i == 0 else num_filters[i - 1]),
                    out_channels=num_filters[i],
                    kernel_size=kernel,
                    padding=(0, int((kernel[1] - 1) / 2)))) for i, kernel in enumerate(kernels)
        ])

        # Multiplier for sequential size ``local_length``
        self.scale_factor = self.upsample_repeat * num_filters[-1]
        self.post_net = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, local_features, pad=False):
        """
        Args:
            local_features (torch.FloatTensor [batch_size, local_length + padding, in_channels]):
                Local features to condition signal generation (e.g. spectrogram). Upsample does
                pad the convolution operations to process the spectrograms; therefore, we require
                that a user pads ``local_features`` time domain instead with
                ``sum([(kernel[0] - 1) for kernel in kernels])`` padding.
            pad (bool, optional): Pad the spectrogram with zeros on the ends, assuming that the
                spectrogram has no context on the ends.

        Returns:
            conditional_features (torch.FloatTensor [batch_size, out_channels, signal_length]):
                Upsampled local conditional features.
        """
        batch_size, local_length, in_channels = local_features.shape

        if pad:
            local_features = nn.functional.pad(local_features,
                                               (0, 0, int(self.padding / 2), int(self.padding / 2)))
        assert local_features.shape[1] > self.padding, (
            'Remember to pad local_features as described in the above docs.')

        # [batch_size, local_length + padding, in_channels] →
        # [batch_size, 1, local_length + padding, in_channels]
        local_features = local_features.unsqueeze(1)

        # [batch_size, 1, local_length + padding, in_channels] →
        # [batch_size, num_filters[0], local_length + padding - kernels[0] + 1, in_channels]
        local_features = self.initial_conv(local_features)

        # [batch_size, num_filters[0], local_length + padding - kernels[0] + 1, in_channels] →
        # [batch_size, num_filters[-1], local_length, in_channels]
        local_features = self.pre_net(local_features)

        # [batch_size, num_filters[-1], local_length, in_channels] →
        # [batch_size, local_length, num_filters[-1], in_channels]
        local_features = local_features.transpose(1, 2).contiguous()

        # [batch_size, local_length, num_filters[-1], in_channels] →
        # [batch_size, local_length * num_filters[-1], in_channels]
        local_features = local_features.view(batch_size, -1, in_channels)

        # [batch_size, local_length * num_filters[-1], in_channels] →
        # [batch_size, in_channels, local_length * num_filters[-1]]
        local_features = local_features.transpose(1, 2)

        # [batch_size, in_channels, local_length * num_filters[-1]] →
        # [batch_size, out_channels, local_length * num_filters[-1]]
        local_features = self.post_net(local_features)

        # [batch_size, out_channels, local_length * num_filters[-1]] →
        # [batch_size, out_channels,
        #  signal_length (local_length * num_filters[-1] * upsample_repeat)]
        # Repeat similar to [1, 2, 3] → [1, 1, 1, 2, 2, 2, 3, 3, 3]
        return nn.functional.interpolate(
            local_features, scale_factor=self.upsample_repeat, mode='nearest')
