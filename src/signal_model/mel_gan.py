import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


class PixelShuffle1d(nn.Module):

    def __init__(self, upscale_factor):
        super().__init__()

        self.upscale_factor = upscale_factor

    def forward(self, tensor):
        """
        Inspired by: https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b

        Example:
            >>> t = torch.arange(0, 12).view(1, 3, 4).transpose(1, 2)
            >>> t
            tensor([[[ 0,  4,  8],
                    [ 1,  5,  9],
                    [ 2,  6, 10],
                    [ 3,  7, 11]]])
            >>> t[0, :, 0] # First frame
            tensor([0, 1, 2, 3])
            >>> pixel_shuffle_1d(t, 4)
            tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]]])

        Args:
            tensor (torch.Tensor [batch_size, channels, sequence_length])

        Returns:
            tensor (torch.Tensor [batch_size, channels // self.upscale_factor,
                sequence_length * self.upscale_factor])
        """
        batch_size, channels, steps = tensor.size()
        channels //= self.upscale_factor
        input_view = tensor.contiguous().view(batch_size, channels, self.upscale_factor, steps)
        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
        return shuffle_out.view(batch_size, channels, steps * self.upscale_factor)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.block = nn.Sequential(
            nn.GELU(),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.GELU(),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x)[:, :, self.dilation:-self.dilation] + self.block(x)


class Generator(nn.Module):

    def __init__(self, input_size=128, ngf=32, n_residual_layers=3, padding=7):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        self.padding = padding
        self.pad = nn.ConstantPad1d(padding, 0.0)
        mult = int(2**len(ratios))

        model = [
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.GELU(),
                WNConv1d(
                    mult * ngf,
                    (mult * ngf // 2) * r,
                    kernel_size=3,
                    padding=0,
                ),
                PixelShuffle1d(r)
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3**j)]

            mult //= 2

        model += [
            nn.GELU(),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, spectrogram, pad_input=True):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])

        Args:
            signal (torch.FloatTensor [batch_size, signal_length])
        """
        has_batch_dim = len(spectrogram.shape) == 3

        # [batch_size, num_frames, frame_channels]
        spectrogram = spectrogram.view(-1, spectrogram.shape[-2], spectrogram.shape[-1])
        batch_size, num_frames, frame_channels = spectrogram.shape

        # [batch_size, num_frames, frame_channels] → [batch_size, frame_channels, num_frames]
        spectrogram = spectrogram.transpose(1, 2)

        spectrogram = self.pad(spectrogram) if pad_input else spectrogram
        num_frames = num_frames if pad_input else num_frames - self.padding * 2

        # [batch_size, frame_channels, num_frames] → [batch_size, signal_length + excess_padding]
        signal = self.model(spectrogram).squeeze(1)

        excess_padding = signal.shape[1] - num_frames * self.hop_length
        assert excess_padding < self.hop_length * 2  # Not too much padding
        assert excess_padding > 0  # Not too little padding
        assert excess_padding % 2 == 0  # Invariant
        # signal [batch_size, num_frames * self.hop_length]
        signal = signal[:, excess_padding // 2:-excess_padding // 2]
        assert signal.shape == (batch_size, self.hop_length * num_frames)

        return signal if has_batch_dim else signal.squeeze(0)
