import logging

from torchnlp.utils import get_total_parameters

import numpy as np
import torch

logger = logging.getLogger(__name__)


def trim(*args, dim=2):
    """ Trim such that all tensors sizes match on `dim`.

    Args:
        *args (torch.Tensor)
        dim (int): The dimension to trim.

    Returns:
        *args (torch.Tensor)
    """
    minimum = min(a.shape[dim] for a in args)
    assert all((a.shape[dim] - minimum) % 2 == 0 for a in args), 'Uneven padding'
    return (a.narrow(dim, (a.shape[dim] - minimum) // 2, minimum) for a in args)


class PixelShuffle1d(torch.nn.Module):

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
            >>> module = PixelShuffle1d(4)
            >>> module(t)
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


class Block(torch.nn.Module):
    """ Building block for the model.

    Args:
        in_channels (int)
        out_channels (int)
        scale_factor (int): The upsample to scale the input.
    """

    def __init__(self, in_channels, out_channels, scale_factor=1):
        super().__init__()

        self.shortcut = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels * scale_factor,
                kernel_size=1,
            ), PixelShuffle1d(scale_factor))

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, in_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(
                in_channels,
                out_channels * scale_factor,
                kernel_size=3,
            ),
            PixelShuffle1d(scale_factor),
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

        self.scale_factor = scale_factor

    def forward(self, input_):
        """
        Args:
            input_ (torch.FloatTensor [batch_size, frame_channels, ~num_frames])

        Returns:
            torch.FloatTensor [batch_size, frame_channels, ~num_frames * scale_factor]
        """
        shape = input_.shape  # [batch_size, frame_channels, num_frames]
        input_ = torch.add(*trim(self.shortcut(input_), self.block(input_)))
        input_ = torch.add(*trim(input_, self.other_block(input_)))
        assert (shape[2] * self.scale_factor - input_.shape[2]) % 2 == 0
        return input_


class SignalModel(torch.nn.Module):
    """
    Args:
        input_size (int): The channel size of the input.
        hidden_size (int): The input size of the final convolution. The rest of the convolution
            sizes are a multiple of `hidden_size`.
        padding (int): The input padding required.
        ratios (list of int): A list of scale factors for upsampling.
    """

    def __init__(self, input_size=128, hidden_size=32, padding=14, ratios=[2] * 8):
        super().__init__()

        self.padding = padding
        self.ratios = ratios
        self.hidden_size = hidden_size
        self.scale_factor = np.prod(ratios)
        self.pad = torch.nn.ConstantPad1d(padding, 0.0)

        self.network = torch.nn.Sequential(*tuple([
            torch.nn.Conv1d(input_size, self.get_layer_size(0), kernel_size=3, padding=0),
            Block(self.get_layer_size(0), self.get_layer_size(0)),
            Block(self.get_layer_size(0), self.get_layer_size(0))
        ] + [
            Block(self.get_layer_size(i), self.get_layer_size(i + 1), r)
            for i, r in enumerate(ratios)
        ] + [
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
            if isinstance(module, torch.nn.Conv1d):
                yield module

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                assert isinstance(module.weight, torch.nn.Parameter)
                torch.nn.init.orthogonal_(module.weight)
                assert isinstance(module.bias, torch.nn.Parameter)
                torch.nn.init.zeros_(module.bias)
            elif get_total_parameters(module) > 0:
                # TODO: Use a recursive approach to filter out modules where all children parameters
                # were set, and only pass up legitmate messages.
                logger.warning('%s module parameters may have not been set.', type(module))

    def get_layer_size(self, i, max_size=512):
        """ Get the hidden size of layer `i` based on the final hidden size `self.hidden_size`.

        Args:
            i (int): The index of the layer.
            max_size (int, optional): Max channel size.

        Returns:
            (int): The number of units.
        """
        assert i <= len(self.ratios)

        return min((int(2**(len(self.ratios) // 2)) * self.hidden_size) // 2**(i // 2), max_size)

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
        signal = self.network(spectrogram).squeeze(1)

        # Mu-law expantion, learn more here:
        # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#mu_expand
        signal = torch.sign(signal) / mu * (torch.pow(1 + mu, torch.abs(signal)) - 1)

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
