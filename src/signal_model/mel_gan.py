import itertools
import logging

from torchnlp.utils import get_total_parameters

import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


def trim_padding(x, y):
    """ Trim the `x` and `y` so that their shapes match in the third dimension. """
    # [batch_size, frame_channels, num_frames]
    excess_padding = abs(x.shape[2] - y.shape[2])
    assert excess_padding % 2 == 0, 'Uneven padding, %d' % excess_padding
    if x.shape[2] > y.shape[2]:
        return x[:, :, excess_padding // 2:-excess_padding // 2], y
    return x, y[:, :, excess_padding // 2:-excess_padding // 2]


class Block(torch.nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()

        self.shortcut = torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor))

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, in_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=scale_factor * 2, stride=scale_factor),
            torch.nn.GELU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )

        self.other_block = torch.nn.Sequential(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
            torch.nn.GELU(),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, x):
        # shape = x.shape  # [batch_size, frame_channels, num_frames]
        x = torch.add(*trim_padding(self.shortcut(x), self.block(x)))
        x = torch.add(*trim_padding(x, self.other_block(x)))
        # TODO: Fix this...
        # assert (shape[2] - x.shape[2]) % 2 == 0
        return x


class Generator(torch.nn.Module):

    def __init__(self,
                 input_size=128,
                 hidden_size=32,
                 padding=6,
                 oversample=4,
                 sample_rate=24000,
                 ratios=[8, 8, 4, 4]):
        super().__init__()

        self.scale_factor = np.prod(ratios) // oversample
        self.padding = padding
        self.pad = torch.nn.ConstantPad1d(padding, 0.0)
        self.register_buffer('sample_rate', torch.tensor(sample_rate))
        self.register_buffer('oversample', torch.tensor(oversample))
        self.ratios = ratios
        self.hidden_size = hidden_size

        self.model = torch.nn.Sequential(*itertools.chain([
            torch.nn.Conv1d(input_size, self.get_channel_size(0), kernel_size=3, padding=0),
            Block(self.get_channel_size(0), self.get_channel_size(0), scale_factor=1),
            Block(self.get_channel_size(0), self.get_channel_size(0), scale_factor=1),
        ], [
            Block(self.get_channel_size(i), self.get_channel_size(i + 1), scale_factor=r)
            for i, r in enumerate(ratios)
        ], [
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_size, 1, kernel_size=3, padding=0),
            torch.nn.Tanh(),
        ]))

        self.reset_parameters()

        # NOTE: We initialize the convolution parameters weight norm factorizes them.
        for module in self.get_weight_norm_modules():
            torch.nn.utils.weight_norm(module)

        logger.info('Number of parameters is: %d', get_total_parameters(self))
        logger.info('Initialized model: %s', self)

    def get_weight_norm_modules(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                yield module

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d) or isinstance(module, torch.nn.ConvTranspose1d):
                assert isinstance(module.weight, torch.nn.Parameter)
                torch.nn.init.orthogonal_(module.weight)
                assert isinstance(module.bias, torch.nn.Parameter)
                torch.nn.init.zeros_(module.bias)
            elif get_total_parameters(module) > 0:
                # TODO: Use a recursive approach to filter out modules where all children parameters
                # were set, and only pass up legitmate messages.
                logger.warning('%s module parameters may have not been set.', type(module))

    def get_channel_size(self, i):
        """ Get the hidden size of layer `i` based on the final hidden size `self.hidden_size`.

        Args:
            i (int): The index of the layer.

        Returns:
            (int): The number of units.
        """
        assert i <= len(self.ratios)

        return (int(2**len(self.ratios)) * self.hidden_size) // 2**i

    def forward(self, spectrogram, pad_input=True, mu=255, input_scalar=5.0):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            padding (bool, optional)
            mu (int, optional): Mu for the u-law scaling. Learn more:
                https://en.wikipedia.org/wiki/%CE%9C-law_algorithm.
            input_scalar (int, optional): This scales the input so that it's within a suitable
                range.

        Args:
            signal (torch.FloatTensor [batch_size, signal_length])
        """
        has_batch_dim = len(spectrogram.shape) == 3

        # TODO: Remove `input_scalar` and instead ensure the spectrogram initially
        # is corrected.
        # [batch_size, num_frames, frame_channels]
        spectrogram = spectrogram.view(-1, spectrogram.shape[-2],
                                       spectrogram.shape[-1]) * input_scalar
        batch_size, num_frames, frame_channels = spectrogram.shape

        # [batch_size, num_frames, frame_channels] → [batch_size, frame_channels, num_frames]
        spectrogram = spectrogram.transpose(1, 2)

        spectrogram = self.pad(spectrogram) if pad_input else spectrogram
        num_frames = num_frames if pad_input else num_frames - self.padding * 2

        # [batch_size, frame_channels, num_frames] → [batch_size, signal_length + excess_padding]
        signal = self.model(spectrogram).squeeze(1)

        # Mu-law expantion, learn more here:
        # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#mu_expand
        signal = torch.sign(signal) / mu * (torch.pow(1 + mu, torch.abs(signal)) - 1)

        # TODO: Add a parameter to `resample_waveform` to not pad the signal.
        # TODO: Look into what happens with signal is not an exact multiple.
        # NOTE: Ensure there is enough padding for `resample_waveform`.
        assert signal.shape[1] - num_frames * self.scale_factor > 24, signal.shape[
            1] - num_frames * self.scale_factor

        # NOTE: Ensure that there even padding after the downsample.
        signal = signal[:, 2:-2]

        assert signal.shape[1] % self.oversample.item() == 0, (
            signal.shape[1] % self.oversample.item())
        signal = torchaudio.compliance.kaldi.resample_waveform(signal,
                                                               self.sample_rate * self.oversample,
                                                               self.sample_rate)

        excess_padding = signal.shape[1] - num_frames * self.scale_factor
        assert excess_padding < self.scale_factor * 2, 'Too much padding, %d' % excess_padding
        assert excess_padding >= 0, 'Too little padding, %d' % excess_padding
        assert excess_padding % 2 == 0, 'Uneven padding, %d' % excess_padding
        if excess_padding > 0:  # [batch_size, num_frames * self.scale_factor]
            signal = signal[:, excess_padding // 2:-excess_padding // 2]
        assert signal.shape == (batch_size, self.scale_factor * num_frames), signal.shape

        num_clipped_samples = ((signal > 1.0) | (signal < -1.0)).sum().item()
        if num_clipped_samples > 0:
            logger.warning('%d samples clipped.', num_clipped_samples)

        signal = torch.clamp(signal, -1.0, 1.0)

        return signal if has_batch_dim else signal.squeeze(0)
