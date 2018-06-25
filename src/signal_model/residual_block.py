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
        self.hidden_size = hidden_size

        self.dilated_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=2 * hidden_size,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,  # [NV-WaveNet assumes biases for the dilated convolution]
            padding=0)
        torch.nn.init.xavier_uniform_(
            self.dilated_conv.weight, gain=torch.nn.init.calculate_gain('tanh'))

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
            torch.nn.init.xavier_uniform_(
                self.out_conv.weight, gain=torch.nn.init.calculate_gain('linear'))

        self.skip_conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=skip_size,
            kernel_size=1,
            bias=True  # [NV-WaveNet assumes biases for the residual connection]
        )
        torch.nn.init.xavier_uniform_(
            self.skip_conv.weight, gain=torch.nn.init.calculate_gain('relu'))

    def forward(self, signal_features, conditional_features):
        """
        Args:
            signal_features (torch.FloatTensor [batch_size, hidden_size, signal_length]): Signal
                features.
            conditional_features (torch.FloatTensor [batch_size, 2 * hidden_size, signal_length]):
                Local and global conditional features (e.g. speaker or spectrogram features). The
                conditional first ``hidden_size`` chunk is used to condition the nonlinearly while
                the last ``hidden_size`` chunk is used to condition the signmoid.
        Returns:
            out (torch.FloatTensor [batch_size, hidden_size, signal_length]): Signal features post
                residual.
            skip (torch.FloatTensor [batch_size, skip_size, signal_length]): Skip features post
                residual.
        """
        # residual [batch_size, hidden_size, signal_length]
        residual = signal_features

        # NOTE: This has the same speed as the padding parameter in Conv1D, practically.
        # [batch_size, hidden_size, signal_length] →
        # [batch_size, hidden_size, signal_length + self.padding[0]]
        signal_features = functional.pad(signal_features, self.padding)

        # Convolution operater expects signal_features of the form:
        # signal_features [batch_size, channels (hidden_size), signal_length]
        # [batch_size, hidden_size, signal_length + self.padding[0]] →
        # [batch_size, 2 * hidden_size, signal_length]
        signal_features = self.dilated_conv(signal_features)

        # [batch_size, 2 * hidden_size, signal_length]
        signal_features = conditional_features + signal_features

        # [batch_size, hidden_size, signal_length] → [batch_size, 2 * hidden_size, signal_length]
        signal_features = (
            functional.tanh(signal_features[:, :self.hidden_size]) * functional.sigmoid(
                signal_features[:, self.hidden_size:]))

        # [batch_size, hidden_size, signal_length] → [batch_size, skip_size, signal_length]
        skip = self.skip_conv(signal_features)

        if self.is_last_layer:
            return None, skip

        # [batch_size, hidden_size, signal_length] → [batch_size, hidden_size, signal_length]
        signal_features = self.out_conv(signal_features)

        # [batch_size, hidden_size, signal_length]
        signal_features += residual

        return signal_features, skip