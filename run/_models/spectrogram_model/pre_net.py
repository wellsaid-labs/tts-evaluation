import typing

import config as cf
import torch
import torch.nn
from torch.nn.init import _no_grad_trunc_normal_

from lib.utils import LSTM


class _GaussianNoise(torch.nn.Module):
    """Gaussian noise adds additive noise to the input `x`.

    Args:
        p: The additive noise will have standard deviation `sqrt(p / (1 - p))`. Learn more:
            https://proceedings.mlr.press/v28/wang13a.html
        sigma: The additive noise will have a maximum range of `std * sigma`.
    """

    def __init__(self, p, sigma: float = 4.0):
        super(_GaussianNoise, self).__init__()
        assert p >= 0 and p < 1
        self.p = p
        self.stddev = (self.p / (1.0 - self.p)) ** 0.5
        self.bound = self.stddev * sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.FloatTensor [*, hidden_size])

        Returns:
            x (torch.FloatTensor [*, hidden_size])
        """
        if self.p != 0:
            noise = torch.empty_like(x)
            noise = _no_grad_trunc_normal_(noise, 0, self.stddev, -self.bound, self.bound)
            return x.std(dim=-1, keepdim=True) * noise + x
        else:
            return x


class PreNet(torch.nn.Module):
    """Pre-net processes encodes spectrogram frames.

    SOURCE (Tacotron 2):
        ...small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that
        the pre-net acting as an information bottleneck was essential for learning attention.

        ...

        In order to introduce output variation at inference time, dropout with probability 0.5 is
        applied only to layers in the pre-net of the autoregressive decoder.

    Args:
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands").
        seq_embed_size
        size: The size of the hidden representation and output.
        num_layers: Number of fully connected layers of ReLU units.
        dropout: Probability of an element to be zeroed.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    def __init__(
        self,
        num_frame_channels: int,
        seq_embed_size: int,
        size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encode = LSTM(num_frame_channels, size, num_layers)
        self.out = torch.nn.Sequential(
            _GaussianNoise(p=dropout),
            cf.partial(torch.nn.LayerNorm)(size),
        )

    def __call__(
        self,
        frames: torch.Tensor,
        seq_embed: torch.Tensor,
        lstm_hidden_state: typing.Optional[typing.Tuple] = None,
    ) -> typing.Tuple[torch.Tensor, typing.Tuple]:
        return super().__call__(frames, seq_embed, lstm_hidden_state)

    def forward(
        self,
        frames: torch.Tensor,
        seq_embed: torch.Tensor,
        lstm_hidden_state: typing.Optional[typing.Tuple] = None,
    ) -> typing.Tuple[torch.Tensor, typing.Optional[typing.Tuple]]:
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Spectrogram
                frames.
            seq_embed (torch.FloatTensor [batch_size, seq_embed_size])
            lstm_hidden_state

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size])
            lstm_hidden_state
        """
        frames, lstm_hidden_state = self.encode(frames, lstm_hidden_state)
        return self.out(frames), lstm_hidden_state
