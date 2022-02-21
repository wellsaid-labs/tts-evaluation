# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import logging
import math
import typing
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn
from hparams import HParam, configurable
from torch.nn import functional
from torch.nn.utils.weight_norm import WeightNorm, remove_weight_norm, weight_norm

from lib.utils import PaddingAndLazyEmbedding, log_runtime, pad_tensor, trim_tensors

logger = logging.getLogger(__name__)


class _InterpolateAndConcat(torch.nn.Module):
    """Interpolates `concat` tensor, and concatenates it to `tensor`.

    Args:
        size: The hidden size of `concat`.
        scale_factor: Scale factor used to interpolate the `concat`.
    """

    def __init__(self, size: int, scale_factor: int):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor

    def __call__(self, tensor: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor, concat)

    def forward(self, tensor: torch.Tensor, concat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.FloatTensor [batch_size, frame_channels, num_frames])
            concat (torch.FloatTensor [batch_size, self.size, >= num_frames / self.scale_factor])

        Returns:
            torch.FloatTensor [batch_size, frame_channels + self.size, num_frames]
        """
        concat = functional.interpolate(concat, scale_factor=self.scale_factor)
        assert concat.shape[1] == self.size
        assert concat.shape[2] >= tensor.shape[2], "Scale factor is too small."
        return torch.cat(list(trim_tensors(tensor, concat)), dim=1)


class _InterpolateAndMask(torch.nn.Module):
    """Interpolates `mask` tensor, and masks `tensor`.

    Args:
        scale_factor: Scale factor used to interpolate the `mask`.
    """

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def __call__(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor, mask)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (torch.FloatTensor [batch_size, frame_channels, num_frames])
            mask (torch.BoolTensor [batch_size, >= num_frames / self.scale_factor])

        Returns:
            torch.FloatTensor [batch_size, frame_channels, num_frames]
        """
        mask = functional.interpolate(mask.float(), scale_factor=self.scale_factor).bool()
        assert mask.shape[2] >= tensor.shape[2], "Scale factor is too small."
        tensor, mask = trim_tensors(tensor, mask)
        return tensor.masked_fill(~mask, 0.0)


class _PixelShuffle1d(torch.nn.Module):
    """The 1d version to PyTorch's `torch.nn.PixelShuffle`."""

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.upscale_factor = upscale_factor

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Inspired by: https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b

        Example:
            >>> t = torch.arange(0, 12).view(1, 3, 4).transpose(1, 2)
            >>> t
            tensor([[[ 0,  4,  8],
                     [ 1,  5,  9],
                     [ 2,  6, 10],
                     [ 3,  7, 11]]])
            >>> t[0, :, 0]
            tensor([0, 1, 2, 3])
            >>> module = PixelShuffle1d(4)
            >>> module(t)
            tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]]])

        Args:
            tensor (torch.Tensor [batch_size, channels, sequence_length])

        Returns:
            tensor (torch.Tensor [batch_size, channels / upscale_factor,
                sequence_length * upscale_factor])
        """
        batch_size, channels, steps = tensor.size()
        channels //= self.upscale_factor
        input_view = tensor.contiguous().view(batch_size, channels, self.upscale_factor, steps)
        shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
        return shuffle_out.view(batch_size, channels, steps * self.upscale_factor)


_SequentialSelfType = typing.TypeVar("_SequentialSelfType", bound="_Sequential")


class _Sequential(torch.nn.Sequential):
    def __call__(
        self,
        input_: torch.Tensor,
        mask: typing.Optional[torch.Tensor] = None,
        conditioning: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return super().__call__(input_, mask, conditioning)

    def forward(
        self,
        input_: torch.Tensor,
        mask: typing.Optional[torch.Tensor] = None,
        conditioning: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for module in self:
            if isinstance(module, _InterpolateAndConcat):
                assert conditioning is not None
                # NOTE: Drop elements from `conditioning` similar too:
                # https://arxiv.org/abs/1809.11096
                input_ = module(input_, conditioning[:, : module.size])
            elif isinstance(module, _InterpolateAndMask):
                assert mask is not None
                input_ = module(input_, mask)
            elif isinstance(module, _Block):
                assert mask is not None
                assert conditioning is not None
                input_ = module(input_, mask, conditioning)
            else:
                input_ = module(input_)
        return input_

    def __getitem__(
        self: _SequentialSelfType, idx: typing.Union[int, slice]
    ) -> typing.Union[_SequentialSelfType, torch.nn.Module]:
        return super().__getitem__(idx)


class _Block(torch.nn.Module):
    """Building block for `SignalModel`.

    Args:
        in_channels: The input channel dimension size.
        out_channels: The output channel dimension size. Note `out_channels` must be smaller or
            equal to `in_channels`.
        upscale_factor: The `input_` is upsampled by this factor.
        input_scale: The `input_` is longer than `conditioning` and `mask` by this factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upscale_factor: int = 1,
        input_scale: int = 1,
    ):
        super().__init__()
        assert out_channels <= in_channels
        self.shortcut = _Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels * upscale_factor,
                kernel_size=1,
            ),
            _PixelShuffle1d(upscale_factor),
        )
        self.block = _Sequential(
            _InterpolateAndConcat(in_channels, input_scale),
            torch.nn.Conv1d(in_channels * 2, in_channels, kernel_size=1),
            torch.nn.GELU(),
            _InterpolateAndMask(input_scale),
            torch.nn.Conv1d(
                in_channels,
                out_channels * upscale_factor,
                kernel_size=3,
            ),
            _PixelShuffle1d(upscale_factor),
            torch.nn.GELU(),
            _InterpolateAndMask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )
        self.other_block = _Sequential(
            _InterpolateAndConcat(out_channels, input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels * 2, out_channels, kernel_size=1),
            torch.nn.GELU(),
            _InterpolateAndMask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
            torch.nn.GELU(),
            _InterpolateAndMask(input_scale * upscale_factor),
            torch.nn.Conv1d(out_channels, out_channels, kernel_size=3),
        )
        self.upscale_factor = upscale_factor
        output_scale = input_scale * upscale_factor
        self.padding_required = 0.0
        for module, scale_factor in [
            (self.block[4], input_scale),
            (self.block[-1], output_scale),
            (self.other_block[4], output_scale),
            (self.other_block[-1], output_scale),
        ]:
            assert isinstance(module, torch.nn.Conv1d)
            self.padding_required += (module.kernel_size[0] // 2) / scale_factor

    def __call__(
        self, input_: torch.Tensor, mask: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return super().__call__(input_, mask, conditioning)

    def forward(
        self, input_: torch.Tensor, mask: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        """
        NOTE: `other_num_frames` is defined as `other_num_frames >= num_frames / input_scale`
        and `other_num_frames % 2 == 0`.

        Args:
            input_ (torch.FloatTensor [batch_size, in_channels,
                num_frames + ceil(module.padding_required) * input_scale * 2])
            mask (torch.BoolTensor [batch_size, 1, other_num_frames])
            conditioning (torch.FloatTensor [batch_size, in_channels, other_num_frames])

        Returns:
            torch.FloatTensor [batch_size, out_channels, (num_frames +
                (ceil(padding_required) - padding_required) * input_scale * 2) * upscale_factor]
        """
        shape = input_.shape
        input_ = torch.add(
            *trim_tensors(self.shortcut(input_), self.block(input_, mask, conditioning))
        )
        input_ = torch.add(*trim_tensors(input_, self.other_block(input_, mask, conditioning)))
        assert (shape[2] * self.upscale_factor - input_.shape[2]) % 2 == 0
        return input_


class _LayerNorm(torch.nn.LayerNorm):
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().__call__(tensor)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return super().forward(tensor.transpose(1, 2)).transpose(1, 2)


def _has_weight_norm(module: torch.nn.Module, name: str = "weight") -> bool:
    """Check if module has `WeightNorm` decorator."""
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            return True
    return False


class _Encoder(torch.nn.Module):
    """Encode signal model inputs.

    Args:
        max_seq_meta_values: The maximum number of metadata values the model will be trained on.
        frame_size: The size of each spectrogram frame.
        seq_meta_embed_size: The size of the sequence metadata embedding.
    """

    def __init__(
        self, max_seq_meta_values: typing.Tuple[int, ...], seq_meta_embed_size: int, frame_size: int
    ):
        super().__init__()

        message = "`seq_meta_embed_size` must be divisable by the number of metadata attributes."
        assert seq_meta_embed_size % len(max_seq_meta_values) == 0, message
        self.embed_metadata = torch.nn.ModuleList(
            PaddingAndLazyEmbedding(n, seq_meta_embed_size // len(max_seq_meta_values))
            for n in max_seq_meta_values
        )
        self.out_size = seq_meta_embed_size + frame_size

    def __call__(
        self,
        spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
    ) -> torch.Tensor:
        return super().__call__(spectrogram, seq_metadata)

    def forward(
        self,
        spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
    ):
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_size])
            seq_metadata: Metadata associated with each sequence

        Returns: torch.FloatTensor [batch_size, num_frames, out_size]
        """
        # [batch_size] → [batch_size, seq_meta_embed_size]
        seq_metadata_ = [
            embed([metadata[i] for metadata in seq_metadata], batch_first=True)[0]
            for i, embed in enumerate(self.embed_metadata)
        ]
        seq_metadata_ = torch.cat(seq_metadata_, dim=1)
        # [batch_size, seq_meta_embed_size] → [batch_size, num_frames, seq_meta_embed_size]
        seq_metadata_ = seq_metadata_.unsqueeze(1).expand(-1, spectrogram.shape[1], -1)
        # [batch_size, num_frames, seq_meta_embed_size] (cat)
        # [batch_size, num_frames, frame_size] →
        # [batch_size, num_frames, frame_size + seq_meta_embed_size]
        return torch.cat([spectrogram, seq_metadata_], dim=2)


class SignalModel(torch.nn.Module):
    """Predicts a signal given a spectrogram.

    Args:
        max_seq_meta_values: The maximum number of metadata values the model will be trained on.
        seq_meta_embed_size
        frame_size
        hidden_size: The channal dimension size of the final convolution(s). The rest of the modules
            are a multiple of `hidden_size` as determined by `get_layer_size`.
        ratios: List of up scale factors for each `SignalModel` layer. The output is
            `np.prod(ratios)` longer than the input.
        max_channel_size: The maximum convolution channel dimension size.
        mu: Mu-law scaling parameter. Learn more: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """

    @configurable
    def __init__(
        self,
        max_seq_meta_values: typing.Tuple[int, ...],
        seq_meta_embed_size: int = HParam(),
        frame_size: int = HParam(),
        hidden_size: int = HParam(),
        ratios: typing.List[int] = HParam(),
        max_channel_size: int = HParam(),
        mu: int = HParam(),
    ):
        super().__init__()
        self.ratios = ratios
        self.hidden_size = hidden_size
        self.max_channel_size = max_channel_size
        self.mu = mu
        self.upscale_factor = int(np.prod(ratios))
        self.grad_enabled = None

        self.encoder = _Encoder(max_seq_meta_values, seq_meta_embed_size, frame_size)
        self.pre_net = _Sequential(
            _InterpolateAndMask(1),
            torch.nn.Conv1d(
                self.encoder.out_size,
                self.get_layer_size(0),
                kernel_size=3,
                padding=0,
            ),
            _LayerNorm(self.get_layer_size(0)),
        )
        _network: typing.List[torch.nn.Module] = [
            _Block(self.get_layer_size(0), self.get_layer_size(0)),
            _Block(self.get_layer_size(0), self.get_layer_size(0)),
        ]
        _network += [
            _Block(
                self.get_layer_size(i),
                self.get_layer_size(i + 1),
                r,
                int(np.prod(ratios[:i])),
            )
            for i, r in enumerate(ratios)
        ]
        _network += [
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            torch.nn.GELU(),
            _InterpolateAndMask(self.upscale_factor),
            torch.nn.Conv1d(hidden_size, 2, kernel_size=3, padding=0),
            _InterpolateAndMask(self.upscale_factor),
        ]
        self.network = _Sequential(*tuple(_network))

        max_size = max([m.size for m in self.modules() if isinstance(m, _InterpolateAndConcat)])
        self.condition = torch.nn.Conv1d(self.get_layer_size(0), max_size, kernel_size=1)

        padding: float = typing.cast(torch.nn.Conv1d, self.pre_net[1]).kernel_size[0] // 2
        post_net_padding = typing.cast(torch.nn.Conv1d, self.network[-2]).kernel_size[0] // 2
        padding += post_net_padding / self.upscale_factor
        padding += sum([m.padding_required for m in self.modules() if isinstance(m, _Block)])
        self.excess_padding: int = round((math.ceil(padding) - padding) * self.upscale_factor)
        self.padding: int = math.ceil(padding)

        # NOTE: We initialize the convolution parameters before weight norm factorizes them.
        self.reset_parameters()

        # NOTE: Learn more about `weight_norm` compatibility with DDP:
        # https://github.com/pytorch/pytorch/issues/35191
        [weight_norm(module) for module in self._get_weight_norm_modules()]

    def del_weight_norm_temp_tensor_(self):
        """Delete the temporary "weight" tensor created every forward pass by `weight_norm`.

        NOTE: It can cause issues like:
        https://github.com/pytorch/pytorch/issues/28594
        """
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm) and hasattr(module, hook.name):
                    delattr(module, hook.name)

    def set_weight_norm_temp_tensor_(self):
        """Re-create the temporary "weight" tensor created every forward pass by `weight_norm`."""
        for module in self.modules():
            for _, hook in module._forward_pre_hooks.items():
                if isinstance(hook, WeightNorm) and not hasattr(module, hook.name):
                    hook(module, None)

    def remove_weight_norm_(self):
        """Remove `weight_norm` from `self`.

        WARNING: `remove_weight_norm` creates and deletes model parameters. For example, if an
        optimizer depends on the current model parameters, this will break that connection.
        """
        # NOTE: `remove_weight_norm` requires that the temporary tensor exists.
        self.set_weight_norm_temp_tensor_()
        [remove_weight_norm(m) for m in self._get_weight_norm_modules() if _has_weight_norm(m)]

    def _get_weight_norm_modules(self) -> typing.Iterator[torch.nn.Module]:
        """Get all modules that should have their weight(s) normalized."""
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                yield module

    def allow_unk_on_eval(self, val: bool):
        """If `True` then the "unknown token" may be used during evaluation, otherwise this will
        error if a new token is encountered during evaluation."""
        for mod in self.modules():
            if isinstance(mod, PaddingAndLazyEmbedding):
                mod.allow_unk_on_eval = val

    @log_runtime
    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def get_layer_size(self, i: int) -> int:
        """Get the hidden size of layer `i`."""
        assert i <= len(self.ratios)
        initial_size = int(2 ** (len(self.ratios) // 2)) * self.hidden_size
        layer_size = initial_size // 2 ** (i // 2)
        return min(layer_size, self.max_channel_size)

    def _normalize_input(
        self,
        spectrogram: torch.Tensor,
        spectrogram_mask: typing.Optional[torch.Tensor],
        pad_input: bool,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames] or None)
            pad_input: If `True` padding is applied to the input.

        Returns:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames])
        """
        device = spectrogram.device
        if spectrogram_mask is None:
            spectrogram_mask = torch.ones(*spectrogram.shape[:2], device=device, dtype=torch.bool)

        if pad_input:
            padding = (self.padding, self.padding)
            spectrogram = pad_tensor(spectrogram, padding, dim=1)
            spectrogram_mask = pad_tensor(spectrogram_mask, padding, dim=1)

        return spectrogram, spectrogram_mask

    def set_grad_enabled(self, enabled: typing.Optional[bool]):
        self.grad_enabled = enabled

    def __call__(
        self,
        spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
        spectrogram_mask: typing.Optional[torch.Tensor] = None,
        pad_input: bool = True,
    ) -> torch.Tensor:
        return super().__call__(spectrogram, seq_metadata, spectrogram_mask, pad_input)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        grad_enabled = self.grad_enabled
        with nullcontext() if grad_enabled is None else torch.set_grad_enabled(grad_enabled):
            return self._forward(*args, **kwargs)

    def _forward(
        self,
        spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
        spectrogram_mask: typing.Optional[torch.Tensor] = None,
        pad_input: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, frame_channels])
            seq_metadata: Metadata associated with each sequence
            spectrogram_mask (torch.BoolTensor [batch_size, num_frames] or None):
                The mask elements on either boundary of the spectrogram so that the corresponding
                output is not affected.
            pad_input: If `True` padding is applied to the input.

        Returns:
            signal (torch.FloatTensor [batch_size, signal_length] or [signal_length])
        """
        has_batch_dim = len(spectrogram.shape) == 3

        spectrogram, spectrogram_mask = self._normalize_input(
            spectrogram, spectrogram_mask, pad_input
        )

        batch_size, num_frames, _ = spectrogram.shape
        num_frames = num_frames - self.padding * 2

        # [batch_size, num_frames, frame_channels] → [batch_size, num_frames, encoder.out_size]
        encoded = self.encoder(spectrogram, seq_metadata)

        # [batch_size, num_frames, encoder.out_size] → [batch_size, encoder.out_size, num_frames]
        encoded = encoded.transpose(1, 2)

        # [batch_size, num_frames] → [batch_size, 1, num_frames]
        spectrogram_mask = spectrogram_mask.unsqueeze(1)

        # [batch_size, encoder.out_size, num_frames] →
        # [batch_size, self.get_layer_size(0), num_frames]
        encoded = self.pre_net(encoded, spectrogram_mask)

        conditioning = self.condition(encoded)  # [batch_size, *, num_frames]

        # [batch_size, self.get_layer_size(0), num_frames] →
        # [batch_size, 2, signal_length + excess_padding]
        signal = self.network(encoded, spectrogram_mask, conditioning)

        # [batch_size, 2, signal_length + excess_padding] →
        # [batch_size, signal_length + excess_padding]
        signal = torch.sigmoid(signal[:, 0]) * torch.tanh(signal[:, 1])

        # Mu-law expantion, learn more here:
        # https://librosa.github.io/librosa/_modules/librosa/core/audio.html#mu_expand
        signal = torch.sign(signal) / self.mu * (torch.pow(1 + self.mu, torch.abs(signal)) - 1)

        if self.excess_padding > 0:  # [batch_size, num_frames * self.upscale_factor]
            signal = signal[:, self.excess_padding : -self.excess_padding]
        assert signal.shape == (batch_size, self.upscale_factor * num_frames), signal.shape

        # Remove clipped samples
        num_clipped_samples = ((signal > 1.0) | (signal < -1.0)).sum().item()
        if num_clipped_samples > 0:
            logger.warning("%d samples clipped.", num_clipped_samples)
        signal = torch.clamp(signal, -1.0, 1.0)

        return signal if has_batch_dim else signal.squeeze(0)


class SpectrogramDiscriminator(torch.nn.Module):
    """Discriminates between predicted and real spectrograms.

    Args:
        fft_length
        num_mel_bins
        ...
        hidden_size: The size of the hidden layers.
    """

    @configurable
    def __init__(
        self,
        fft_length: int,
        num_mel_bins: int,
        max_seq_meta_values: typing.Tuple[int, ...],
        seq_meta_embed_size: int = HParam(),
        hidden_size: int = HParam(),
    ):
        super().__init__()
        self.fft_length = fft_length
        frame_size = fft_length + num_mel_bins + 2
        self.encoder = _Encoder(max_seq_meta_values, seq_meta_embed_size, frame_size)
        self.layers = _Sequential(
            torch.nn.Conv1d(self.encoder.out_size, hidden_size, kernel_size=3, padding=1),
            _LayerNorm(hidden_size),
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden_size, 1, kernel_size=3, padding=1),
        )

        # NOTE: We initialize the convolution parameters before weight norm factorizes them.
        self.reset_parameters()

        [weight_norm(m) for m in self.modules() if isinstance(m, torch.nn.Conv1d)]

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def __call__(
        self,
        spectrogram: torch.Tensor,
        db_spectrogram: torch.Tensor,
        db_mel_spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
    ) -> torch.Tensor:
        return super().__call__(spectrogram, db_spectrogram, db_mel_spectrogram, seq_metadata)

    def forward(
        self,
        spectrogram: torch.Tensor,
        db_spectrogram: torch.Tensor,
        db_mel_spectrogram: torch.Tensor,
        seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
    ) -> torch.Tensor:
        """
        Args:
            spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
            db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
            ...

        Returns:
            (torch.FloatTensor [batch_size]): A score that discriminates between predicted and
                real spectrogram.
        """
        spectrogram = torch.cat([spectrogram, db_spectrogram, db_mel_spectrogram], dim=2)
        encoded = self.encoder(spectrogram, seq_metadata)
        encoded = encoded.transpose(-1, -2)
        return self.layers(encoded).squeeze(1).mean(dim=-1)


def generate_waveform(
    model: SignalModel,
    spectrogram: typing.Iterator[torch.Tensor],
    seq_metadata: typing.List[typing.Tuple[typing.Hashable, ...]],
    spectrogram_mask: typing.Optional[typing.Iterator[torch.Tensor]] = None,
) -> typing.Iterator[torch.Tensor]:
    """
    TODO: Similar to WaveNet, we could incorperate a "Fast WaveNet" approach. This basically means
    we don't need to recompute the padding for each split.
    TODO: Consider adding another "forward" function like `forward_generator` to `SignalModel` and
    incorperating this functionality.

    Args:
        model: The model to synthesize the waveform with.
        ...

    Returns:
        signal (torch.FloatTensor [batch_size, signal_length])
    """
    padding = model.padding
    last_item: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None
    spectrogram_mask = spectrogram_mask if spectrogram_mask is None else iter(spectrogram_mask)
    spectrogram = iter(spectrogram)
    is_stop = False
    has_batch_dim = None
    while not is_stop:
        items: typing.List[typing.Tuple[torch.Tensor, torch.Tensor]] = []
        while sum([i[0].shape[1] for i in items]) < padding * 2 and not is_stop:
            try:
                # [batch_size, num_frames, frame_channels]
                item_frames = next(spectrogram)
                has_batch_dim = len(item_frames.shape) == 3
                item_mask = None if spectrogram_mask is None else next(spectrogram_mask)
                items.append(model._normalize_input(item_frames, item_mask, False))
            except StopIteration:
                is_stop = True
        assert has_batch_dim is not None, "Spectrogram iterator must not be empty."

        padding_tuple = (0 if last_item else padding, padding if is_stop else 0)
        frames = ([last_item[0][:, -padding * 2 :]] if last_item else []) + [i[0] for i in items]
        frames_ = pad_tensor(torch.cat(frames, dim=1), pad=padding_tuple, dim=1)
        mask = ([last_item[1][:, -padding * 2 :]] if last_item else []) + [i[1] for i in items]
        mask_ = pad_tensor(torch.cat(mask, dim=1), pad=padding_tuple, dim=1)

        waveform = model(frames_, seq_metadata, mask_, pad_input=False)
        yield waveform if has_batch_dim else waveform.squeeze(0)

        last_item = (frames_, mask_)
