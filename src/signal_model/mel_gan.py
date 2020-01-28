import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResnetBlock(nn.Module):

    def __init__(self, dim, dilation=1):
        super().__init__()
        self.dilation = dilation
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x)[:, :, self.dilation:-self.dilation] + self.block(x)


class Generator(nn.Module):

    def __init__(self, input_size=128, ngf=32, n_residual_layers=3, padding=5):
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
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3**j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
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
        assert excess_padding > 0 and excess_padding % 2 == 0
        # signal [batch_size, num_frames * self.hop_length]
        signal = signal[:, excess_padding // 2:-excess_padding // 2]

        return signal if has_batch_dim else signal.squeeze(0)
