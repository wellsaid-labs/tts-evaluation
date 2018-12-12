from torch import nn

from src.hparams import configurable


class Identity(nn.Module):
    """ Identity block returns the input. """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        if len(args) == 1:
            return args[0]

        return args


class ResidualBlock(nn.Module):
    """ Residual block applied during upsampling.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        padding (int or tuple, optional): Zero-padding added to both sides of the input.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding))

        self.shortcut = (
            Identity() if in_channels == out_channels else nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, tensor):
        """
        Args:
            tensor (torch.FloatTensor [batch_size, in_channels, width, height])

        Returns:
            tensor (torch.FloatTensor [batch_size, out_channels, width, height]):
        """
        residual = self.net(tensor)

        # The Conv2D may reduce the size of the tensor; in this case, we just reduce size similarly.
        less_width = int((tensor.shape[2] - residual.shape[2]) / 2)
        less_height = int((tensor.shape[3] - residual.shape[3]) / 2)
        tensor = tensor[:, :, less_width:tensor.shape[2] - less_width, less_height:tensor.shape[3] -
                        less_height]

        return self.shortcut(tensor) + residual


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
        kernels (list of tuples): Sizes of kernels used for upsampling, every kernel has an
            associated number of filters.
        num_filters (list of int): Filters to be used with each kernel. The last kernel is used
            for upsampling the length.
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
    """

    @configurable
    def __init__(self,
                 in_channels=80,
                 out_channels=64,
                 kernels=[(5, 5), (3, 3), (3, 3), (3, 3)],
                 num_filters=[64, 64, 32, 10],
                 upsample_repeat=30):
        super().__init__()
        self.out_channels = out_channels
        self.upsample_repeat = upsample_repeat
        self.min_padding = sum([(kernel[0] - 1) for kernel in kernels])

        assert all(all(s % 2 == 1 and s < in_channels for s in kernel) for kernel in kernels), (
            'Kernel size must be odd and must be less than ``in_channels``')
        assert self.min_padding % 2 == 0, 'Padding invariant violated.'

        self.initial_conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters[0],
            kernel_size=kernels[0],
            padding=tuple([int((s - 1) / 2) for s in kernels[0]]))

        self.pre_net = nn.Sequential(*[
            ResidualBlock(
                in_channels=(num_filters[0] if i == 0 else num_filters[i - 1]),
                out_channels=num_filters[i],
                kernel_size=kernel,
                padding=(0, int((kernel[1] - 1) / 2))) for i, kernel in enumerate(kernels)
        ])

        self.post_net = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

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

    def forward(self, local_features, pad=False):
        """
        TODO: Support global conditioning

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
            local_features = nn.functional.pad(
                local_features, (0, 0, int(self.min_padding / 2), int(self.min_padding / 2)))
        assert local_features.shape[1] > self.min_padding, (
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
        return self._repeat(local_features)
