import numpy as np

from torch.nn.utils import weight_norm

import torch
import torch.nn as nn


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
        torch.nn.init.orthogonal_(m.weight)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def trim_padding(x, y):
    """ Trim the `x` and `y` so that their shapes match in the third dimension. """
    # [batch_size, frame_channels, num_frames]
    excess_padding = abs(x.shape[2] - y.shape[2])
    assert excess_padding > 0  # Not too little padding
    assert excess_padding % 2 == 0  # Invariant
    if x.shape[2] > y.shape[2]:
        return x[:, :, excess_padding // 2:-excess_padding // 2], y
    return x, y[:, :, excess_padding // 2:-excess_padding // 2]


class GBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.shortcut = nn.Sequential(
            WNConv1d(
                in_channels,
                out_channels * scale_factor,
                kernel_size=1,
            ), PixelShuffle1d(scale_factor))
        self.block = nn.Sequential(
            nn.GELU(),
            WNConv1d(
                in_channels,
                out_channels * scale_factor,
                kernel_size=3,
            ),
            PixelShuffle1d(scale_factor),
            nn.GELU(),
            WNConv1d(out_channels, out_channels, kernel_size=3, dilation=2),
        )
        self.other_block = nn.Sequential(
            nn.GELU(),
            WNConv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                dilation=4,
            ),
            nn.GELU(),
            WNConv1d(out_channels, out_channels, kernel_size=3, dilation=8),
        )

    def forward(self, x):
        x = torch.add(*trim_padding(self.shortcut(x), self.block(x)))
        return torch.add(*trim_padding(x, self.other_block(x)))


class Generator(nn.Module):

    def __init__(self, input_size=128, ngf=32, padding=35):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.hop_length = np.prod(ratios)
        self.padding = padding
        self.pad = nn.ConstantPad1d(padding, 0.0)
        mult = int(2**len(ratios))

        model = [
            WNConv1d(input_size, mult * ngf, kernel_size=3, padding=0),
            GBlock(mult * ngf, mult * ngf, scale_factor=1),
            GBlock(mult * ngf, mult * ngf, scale_factor=1),
        ]

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [GBlock(mult * ngf, mult * ngf // 2, scale_factor=r)]

            mult //= 2

        model += [
            WNConv1d(ngf, 1, kernel_size=3, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, spectrogram, pad_input=True, mu=255):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            padding (bool, optional)
            mu (int, optional): Mu for the u-law scaling. Learn more:
                https://en.wikipedia.org/wiki/%CE%9C-law_algorithm.

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

        # Mu-law expantion, learn more here:
        # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#mu_expand
        signal = torch.sign(signal) / mu * (torch.pow(1 + mu, torch.abs(signal)) - 1)

        return signal if has_batch_dim else signal.squeeze(0)
