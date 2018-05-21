from torch import nn
from torch.nn import functional

import torch


class ResidualBlock(nn.Module):

    def __init__(self, hidden_size=64, skip_size=256, kernel_size=2, dilation=1):
        """
        Args:
            hidden_size (int, optional): The number of channels for the ``input_`` and output of
                the residual block. [Parameter ``R`` in NV-WaveNet]
            skip_size (int, optional): The number of channels for the skip output connection
                of the residual block. [Parameter ``S`` in NV-WaveNet]
            kernel_size (int, optional): Size of the convolving kernel applied onto ``input_``.
                [NV-WaveNet assumes kernel size 2]
            dilation (int, optional): Spacing between kernel elements applied onto ``input_``.
        """

        super().__init__()

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

    def forward(self, input_, conditional):
        """
        Args:
            input_ (torch.FloatTensor [hidden_size, batch_size, num_samples]): Audio features.
            conditional (torch.FloatTensor [2 * hidden_size, batch_size, num_samples]): Local and
                global conditional features (e.g. speaker or spectrogram features).
        """
        # Convolution operater expects input_ of the form:
        # [batch_size, channels (hidden_size), signal_length (num_samples)]
        # [hidden_size, batch_size, num_samples] → [batch_size, hidden_size, num_samples]
        input_ = input_.transpose_(0, 1)

        # original [batch_size, hidden_size, num_samples]
        original = input_

        # [batch_size, hidden_size, num_samples] →
        # [batch_size, hidden_size, num_samples + self.padding[0]]
        input_ = functional.pad(input_, self.padding)

        # [batch_size, hidden_size, num_samples + self.padding[0]] →
        # [batch_size, 2 * hidden_size, num_samples]
        input_ = self.dilated_conv(input_)

        # [batch_size, 2 * hidden_size, num_samples] → [2 * hidden_size, batch_size, num_samples]
        input_ = input_.transpose_(0, 1)

        # [2 * hidden_size, batch_size, num_samples]
        input_ = conditional + input_

        # [2 * hidden_size, batch_size, num_samples] → [batch_size, 2 * hidden_size, num_samples]
        input_ = input_.transpose_(1, 0)

        # left, right [batch_size, hidden_size, num_samples]
        left, right = tuple(torch.chunk(input_, 2, dim=1))

        # input_ [batch_size, hidden_size, num_samples]
        input_ = functional.tanh(left) * functional.sigmoid(right)

        del left
        del right

        # [batch_size, hidden_size, num_samples] → [batch_size, skip_size, num_samples]
        skip = self.skip_conv(input_)

        # [batch_size, hidden_size, num_samples] → [batch_size, hidden_size, num_samples]
        residual = self.out_conv(input_)

        del input_

        # TODO: Last residual output is not used, it can be removed.

        # [batch_size, hidden_size, num_samples] + [batch_size, hidden_size, num_samples] →
        # [batch_size, hidden_size, num_samples]
        out = residual + original

        return out, skip
