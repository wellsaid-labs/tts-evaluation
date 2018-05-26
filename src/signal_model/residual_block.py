from torch import nn
from torch.nn import functional

import torch

from src.utils.configurable import configurable


class ResidualBlock(nn.Module):

    @configurable
    def __init__(self,
                 hidden_size=64,
                 skip_size=256,
                 kernel_size=2,
                 dilation=1,
                 is_last_layer=False):
        """
        Args:
            hidden_size (int, optional): The number of channels for the ``signal_features`` and
                output of the residual block. [Parameter ``R`` in NV-WaveNet]
            skip_size (int, optional): The number of channels for the skip output connection
                of the residual block. [Parameter ``S`` in NV-WaveNet]
            kernel_size (int, optional): Size of the convolving kernel applied onto
                ``signal_features``. [NV-WaveNet assumes kernel size 2]
            dilation (int, optional): Spacing between kernel elements applied onto
                ``signal_features``.
            is_last_layer (bool, optional): If True, residual block does not return additional
                signal features.
        """

        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.is_last_layer = is_last_layer

        self.dilated_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,  # [NV-WaveNet assumes biases for the dilated convolution]
            padding=0)

        # SOURCE WaveNet:
        # By using causal convolutions ... For 1-D data such as audio one can more easily
        # implement this by shifting the output of a normal convolution by a few timesteps.
        self.padding = (dilation * (kernel_size - 1), 0)

        if not self.is_last_layer:
            self.out_conv = nn.Conv1d(
                in_channels=hidden_size,
                out_channels=hidden_size,
                kernel_size=1,
                bias=True  # [NV-WaveNet assumes biases for the residual connection]
            )

        self.skip_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=skip_size,
            kernel_size=1,
            bias=True  # [NV-WaveNet assumes biases for the residual connection]
        )

    def forward(self, signal_features, conditional_features):
        """
        Args:
            signal_features (torch.FloatTensor [hidden_size, batch_size, signal_length]): Signal
                features.
            conditional_features (torch.FloatTensor [2 * hidden_size, batch_size, signal_length]):
                Local and global conditional features (e.g. speaker or spectrogram features). The
                conditional first ``hidden_size`` chunk is used to condition the nonlinearly while
                the last ``hidden_size`` chunk is used to condition the signmoid.
        Returns:
            out (torch.FloatTensor [hidden_size, batch_size, signal_length]): Signal features post
                residual.
            skip (torch.FloatTensor [batch_size, skip_size, signal_length]): Skip features post
                residual.
        """
        # Convolution operater expects signal_features of the form:
        # [batch_size, channels (hidden_size), signal_length]
        # [hidden_size, batch_size, signal_length] → [batch_size, hidden_size, signal_length]
        signal_features = signal_features.transpose(0, 1)

        # original [batch_size, hidden_size, signal_length]
        original = signal_features

        # [batch_size, hidden_size, signal_length] →
        # [batch_size, hidden_size, signal_length + self.padding[0]]
        signal_features = functional.pad(signal_features, self.padding)

        # [batch_size, hidden_size, signal_length + self.padding[0]] →
        # [batch_size, 2 * hidden_size, signal_length]
        signal_features = self.dilated_conv(signal_features)

        # [batch_size, 2 * hidden_size, signal_length] →
        # [2 * hidden_size, batch_size, signal_length]
        signal_features = signal_features.transpose_(0, 1)

        # [2 * hidden_size, batch_size, signal_length]
        signal_features = conditional_features + signal_features

        # [2 * hidden_size, batch_size, signal_length] →
        # [batch_size, 2 * hidden_size, signal_length]
        signal_features = signal_features.transpose_(1, 0)

        # left, right [batch_size, hidden_size, signal_length]
        left, right = tuple(torch.chunk(signal_features, 2, dim=1))

        # signal_features [batch_size, hidden_size, signal_length]
        signal_features = functional.tanh(left) * functional.sigmoid(right)  # TODO: tanh init

        del left
        del right

        # [batch_size, hidden_size, signal_length] → [batch_size, skip_size, signal_length]
        skip = self.skip_conv(signal_features)  # TODO: relu init

        if self.is_last_layer:
            return None, skip

        # [batch_size, hidden_size, signal_length] → [batch_size, hidden_size, signal_length]
        residual = self.out_conv(signal_features)

        # [batch_size, hidden_size, signal_length] + [batch_size, hidden_size, signal_length] →
        # [batch_size, hidden_size, signal_length]
        signal_features = residual + original

        # Tranpose to the same shape as ``signal_features`` originally
        # [batch_size, hidden_size, signal_length] → [hidden_size, batch_size, signal_length]
        signal_features = signal_features.transpose_(0, 1)

        return signal_features, skip
