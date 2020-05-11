import logging
import math

from hparams import configurable
from hparams import HParam
from torch.nn.utils.weight_norm import WeightNorm

import numpy as np
import torch

from src.utils import pad_tensors
from src.utils import trim_tensors
import src.distributed

logger = logging.getLogger(__name__)


class L1L2Loss(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.l1_loss = torch.nn.L1Loss(*args, **kwargs)
        self.l2_loss = torch.nn.MSELoss(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.l1_loss(*args, **kwargs) + self.l2_loss(*args, **kwargs)


class ConditionalConcat(torch.nn.Module):
    """ Concat's conditional features injected by a parent module.

    Args:
        size (int): The hidden size of the conditional.
        scale_factor (int): Scale factor used to interpolate the features before applying them.
    """

    def __init__(self, size, scale_factor):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def forward(self, in_, conditioning):
        """
        Args:
            in_ (torch.FloatTensor [batch_size, frame_channels, num_frames])
            conditioning (torch.FloatTensor [batch_size, frame_channels, num_frames])

        Returns:
            torch.FloatTensor [batch_size, frame_channels + self.size, num_frames]
        """
        conditioning = torch.nn.functional.interpolate(conditioning, scale_factor=self.scale_factor)
        assert conditioning.shape[1] == self.size
        assert conditioning.shape[2] >= in_.shape[2], 'Scale factor is too small.'
        return torch.cat(list(trim_tensors(in_, conditioning)), dim=1)


class Mask(torch.nn.Module):
    """ Features are masked with mask injected by a parent module.

    Args:
        scale_factor (int): Scale factor used to interpolate the mask before applying it.
    """

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, in_, mask):
        """
        Args:
            in_ (torch.FloatTensor [batch_size, frame_channels, num_frames])
            mask (torch.BoolTensor [batch_size, num_frames])

        Returns:
            torch.FloatTensor [batch_size, frame_channels, num_frames]
        """
        with torch.no_grad():
            mask = torch.nn.functional.interpolate(
                mask.float(), scale_factor=self.scale_factor).bool()
        assert mask.shape[2] >= in_.shape[2], 'Scale factor is too small.'
        in_, mask = trim_tensors(in_, mask)
        return in_.masked_fill(~mask, 0.0)


class PixelShuffle1d(torch.nn.Module):
    """ The 1d version to PyTorch's `torch.nn.PixelShuffle`. """

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


class Sequential(torch.nn.Sequential):

    def forward(self, input_, mask=None, conditioning=None):
        for module in self:
            if isinstance(module, ConditionalConcat):
                input_ = module(input_, conditioning[:, :module.size])
            elif isinstance(module, Mask):
                input_ = module(input_, mask)
            elif isinstance(module, Block):
                input_ = module(input_, mask, conditioning)
            else:
                input_ = module(input_)
        return input_


class Block(torch.nn.Module):
    """ Building block for `SignalModel`.

    Args:
        in_channels (int): The input channel size.
        out_channels (int): The output channel size.
        upscale_factor (int): The upsample to scale the input.
    """

    def __init__(self, in_channels, out_channels, upscale_factor=1, input_scale=1):
        super().__init__()

        self.shortcut = Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels * upscale_factor,
                kernel_size=1,
            ), PixelShuffle1d(upscale_factor))

        self.block = Sequential(
            ConditionalConcat(in_channels, input_scale),
            torch.nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            torch.nn.GELU(),
            Mask(input_scale),
            torch.nn.Conv1d(
                in_channels,
                out_channels * upscale_factor,
                kernel_size=3,
            ),
            PixelShuffle1d(upscale_factor),
            torch.nn.GELU(),
            Mask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )

        self.other_block = Sequential(
            ConditionalConcat(out_channels, input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            torch.nn.GELU(),
            Mask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
            torch.nn.GELU(),
            Mask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )

        self.upscale_factor = upscale_factor

        output_scale = input_scale * upscale_factor
        self.padding_required = (self.block[4].kernel_size[0] // 2) / input_scale
        self.padding_required += (self.block[-1].kernel_size[0] // 2) / output_scale
        self.padding_required += (self.other_block[4].kernel_size[0] // 2) / output_scale
        self.padding_required += (self.other_block[-1].kernel_size[0] // 2) / output_scale

    def forward(self, input_, mask, conditioning):
        """
        Args:
            input_ (torch.FloatTensor [batch_size, frame_channels, ~num_frames])

        Returns:
            torch.FloatTensor [batch_size, frame_channels, ~num_frames * upscale_factor]
        """
        shape = input_.shape  # [batch_size, frame_channels, num_frames]
        input_ = torch.add(
            *trim_tensors(self.shortcut(input_), self.block(input_, mask, conditioning)))
        input_ = torch.add(*trim_tensors(input_, self.other_block(input_, mask, conditioning)))
        assert (shape[2] * self.upscale_factor - input_.shape[2]) % 2 == 0
        return input_


class LayerNorm(torch.nn.LayerNorm):

    def forward(self, tensor):
        return super().forward(tensor.transpose(1, 2)).transpose(1, 2)


def generate_waveform(model, spectrogram, spectrogram_mask=None):
    """
    TODO: Similar to WaveNet, we could incorperate a "Fast WaveNet" approach. This basically means
    we don't need to recompute the padding for each split.
    TODO: The dataset is based in 16-bits while the signal model outputs in 32-bits. This should
    be resolved.
    TODO: Consider adding another "forward" function like `forward_generator` to `SignalModel` and
    incorperating this functionality.

    Args:
        model (SignalModel): The model to synthesize the waveform with.
        spectrogram (generator)
        spectrogram_mask (generator, optional)

    Returns:
        signal (torch.FloatTensor [batch_size, signal_length] or [signal_length])
    """
    padding = model.padding
    last_item = None
    spectrogram_mask = spectrogram_mask if spectrogram_mask is None else iter(spectrogram_mask)
    spectrogram = iter(spectrogram)
    is_stop = False
    while not is_stop:
        items = []
        while sum([i[0].shape[1] for i in items]) < padding * 2 and not is_stop:
            try:
                frames = next(spectrogram)  # [batch_size (optional), num_frames, frame_channels]
                has_batch_dim = len(frames.shape) == 3
                mask = None if spectrogram_mask is None else next(spectrogram_mask)
                items.append(model._normalize_input(frames, mask, False))
            except StopIteration:
                is_stop = True

        padding_tuple = (0 if last_item else padding, padding if is_stop else 0)
        frames = ([last_item[0][:, -padding * 2:]] if last_item else []) + [i[0] for i in items]
        frames = pad_tensors(torch.cat(frames, dim=1), pad=padding_tuple, dim=1)
        mask = ([last_item[1][:, -padding * 2:]] if last_item else []) + [i[1] for i in items]
        mask = pad_tensors(torch.cat(mask, dim=1), pad=padding_tuple, dim=1)

        waveform = model(frames, mask, pad_input=False)
        yield waveform if has_batch_dim else waveform.squeeze(0)

        last_item = (frames, mask)


def has_weight_norm(module, name='weight'):
    """ Check if module has `WeightNorm` decorator. """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            return True
    return False


class SignalModel(torch.nn.Module):
    """ Predicts a signal given a spectrogram.

    Args:
        input_size (int): The channel size of the input.
        hidden_size (int): The input size of the final convolution. The rest of the convolution
            sizes are a multiple of `hidden_size`.
        ratios (list of int): A list of scale factors for upsampling.
        max_channel_size (int): The maximum convolution channel size.
        mu (int): Mu for the u-law scaling. Learn more:
            https://en.wikipedia.org/wiki/%CE%9C-law_algorithm.
    """

    @configurable
    def __init__(self,
                 input_size=HParam(),
                 hidden_size=HParam(),
                 ratios=HParam(),
                 max_channel_size=HParam(),
                 mu=HParam()):
        super().__init__()

        self.ratios = ratios
        self.hidden_size = hidden_size
        self.max_channel_size = max_channel_size
        self.mu = mu
        self.upscale_factor = np.prod(ratios)

        self.pre_net = Sequential(
            Mask(1),
            torch.nn.Conv1d(input_size, self.get_layer_size(0), kernel_size=3, padding=0),
            LayerNorm(self.get_layer_size(0)),
        )
        self.network = Sequential(*tuple([
            Block(self.get_layer_size(0), self.get_layer_size(0)),
            Block(self.get_layer_size(0), self.get_layer_size(0))
        ] + [
            Block(self.get_layer_size(i), self.get_layer_size(i + 1), r, np.prod(ratios[:i]))
            for i, r in enumerate(ratios)
        ] + [
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            torch.nn.GELU(),
            Mask(self.upscale_factor),
            torch.nn.Conv1d(hidden_size, 2, kernel_size=3, padding=0),
            Mask(self.upscale_factor)
        ]))

        max_size = max([m.size for m in self.modules() if isinstance(m, ConditionalConcat)])
        self.condition = torch.nn.Conv1d(self.get_layer_size(0), max_size, kernel_size=1)

        self.padding = self.pre_net[1].kernel_size[0] // 2
        self.padding += (1 / (self.upscale_factor) * (self.network[-2].kernel_size[0] // 2))
        self.padding += sum([m.padding_required for m in self.modules() if isinstance(m, Block)])
        self.excess_padding = math.ceil(self.padding) - self.padding
        self.padding = math.ceil(self.padding)
        self.pad = torch.nn.ConstantPad1d(self.padding, 0.0)

        self.reset_parameters()

        # NOTE: We initialize the convolution parameters before weight norm factorizes them.
        # NOTE: The `torch.nn.Module` by default is initialized to `self.training = True`.
        self.train(mode=True)

    def train(self, *args, **kwargs):
        """ Sets the module in training or evaluation mode.

        Learn more more: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train
        """
        return_ = super().train(*args, **kwargs)
        if not src.distributed.is_initialized() or self.training:
            for module in self._get_weight_norm_modules():
                if self.training and not has_weight_norm(module):
                    torch.nn.utils.weight_norm(module)
                if not self.training and has_weight_norm(module):
                    torch.nn.utils.remove_weight_norm(module)
        return return_

    def _get_weight_norm_modules(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                yield module

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.orthogonal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def get_layer_size(self, i):
        """ Get the hidden size of layer `i` based on the final hidden size `self.hidden_size`.

        Args:
            i (int): The index of the layer.

        Returns:
            (int): The number of units.
        """
        assert i <= len(self.ratios)

        return min((int(2**(len(self.ratios) // 2)) * self.hidden_size) // 2**(i // 2),
                   self.max_channel_size)

    def _normalize_input(self, spectrogram, spectrogram_mask, pad_input):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels] or
                [num_frames, frame_channels])
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames] or [num_frames] or None)
            pad_input (bool): If `True` padding is applied to the input.

        Returns:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames])
        """
        # [batch_size, num_frames, frame_channels]
        spectrogram = spectrogram.view(-1, spectrogram.shape[-2], spectrogram.shape[-1])

        device = spectrogram.device
        if spectrogram_mask is None:
            spectrogram_mask = torch.ones(*spectrogram.shape[:2], device=device, dtype=torch.bool)
        spectrogram_mask = spectrogram_mask.view(*spectrogram.shape[:2])

        if pad_input:
            spectrogram = pad_tensors(spectrogram, (self.padding, self.padding), dim=1)
            spectrogram_mask = pad_tensors(spectrogram_mask, (self.padding, self.padding), dim=1)

        return spectrogram, spectrogram_mask

    def forward(self, spectrogram, spectrogram_mask=None, pad_input=True):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels] or
                [num_frames, frame_channels])
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames] or [num_frames], optional):
                The mask elements on either boundary of the spectrogram so that the corresponding
                output is not affected.
            pad_input (bool, optional): If `True` padding is applied to the input.

        Returns:
            signal (torch.FloatTensor [batch_size, signal_length] or [signal_length])
        """
        has_batch_dim = len(spectrogram.shape) == 3

        spectrogram, spectrogram_mask = self._normalize_input(spectrogram, spectrogram_mask,
                                                              pad_input)

        batch_size, num_frames, frame_channels = spectrogram.shape
        num_frames = num_frames - self.padding * 2

        # [batch_size, num_frames, frame_channels] → [batch_size, frame_channels, num_frames]
        spectrogram = spectrogram.transpose(1, 2)

        # [batch_size, num_frames] → [batch_size, 1, num_frames]
        spectrogram_mask = spectrogram_mask.unsqueeze(1)

        # [batch_size, frame_channels, num_frames] →
        # [batch_size, self.get_layer_size(0), num_frames]
        spectrogram = self.pre_net(spectrogram, spectrogram_mask)

        conditioning = self.condition(spectrogram)  # [batch_size, *, num_frames]

        # [batch_size, frame_channels, num_frames] → [batch_size, 2, signal_length + excess_padding]
        signal = self.network(spectrogram, spectrogram_mask, conditioning)

        # [batch_size, 2, signal_length + excess_padding] →
        # [batch_size, signal_length + excess_padding]
        signal = torch.sigmoid(signal[:, 0]) * torch.tanh(signal[:, 1])

        # Mu-law expantion, learn more here:
        # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#mu_expand
        signal = torch.sign(signal) / self.mu * (torch.pow(1 + self.mu, torch.abs(signal)) - 1)

        excess_padding = int(self.excess_padding * self.upscale_factor)
        if excess_padding > 0:  # [batch_size, num_frames * self.upscale_factor]
            signal = signal[:, excess_padding:-excess_padding]
        assert signal.shape == (batch_size, self.upscale_factor * num_frames), signal.shape

        # Remove clipped samples
        num_clipped_samples = ((signal > 1.0) | (signal < -1.0)).sum().item()
        if num_clipped_samples > 0:
            logger.warning('%d samples clipped.', num_clipped_samples)
        signal = torch.clamp(signal, -1.0, 1.0)

        return signal if has_batch_dim else signal.squeeze(0)


class SpectrogramDiscriminator(torch.nn.Module):
    """ Discriminates between predicted and real spectrograms.

    Args:
        fft_length (int)
        num_mel_bins (int)
        hidden_size (int): The size of the hidden layers.
    """

    @configurable
    def __init__(self, fft_length, num_mel_bins, hidden_size=HParam()):
        super().__init__()

        input_size = fft_length + num_mel_bins + 2

        self.layers = Sequential(
            torch.nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            LayerNorm(hidden_size),
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_size, 1, kernel_size=3, padding=1),
        )

        # NOTE: We initialize the convolution parameters before weight norm factorizes them.
        self.reset_parameters()

        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.utils.weight_norm(module)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.orthogonal_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def forward(self, spectrogram, db_spectrogram, db_mel_spectrogram):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])

        Returns:
            (torch.FloatTensor [batch_size]): A score that discriminates between predicted and
                real spectrogram.
        """
        features = torch.cat([spectrogram, db_spectrogram, db_mel_spectrogram], dim=2)
        features = features.transpose(-1, -2)
        return self.layers(features).squeeze(1).mean(dim=-1)
