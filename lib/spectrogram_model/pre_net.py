from hparams import configurable
from hparams import HParam

import torch


class _AlwaysDropout(torch.nn.Dropout):
    """ Adaptation of `nn.Dropout` to apply dropout during both evaluation and training. """

    def forward(self, input):
        return torch.nn.functional.dropout(
            input=input, p=self.p, training=True, inplace=self.inplace)


class PreNet(torch.nn.Module):
    """ Pre-net processes encodes spectrogram frames.

    SOURCE (Tacotron 2):
        ...small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that
        the pre-net acting as an information bottleneck was essential for learning attention.

        ...

        In order to introduce output variation at inference time, dropout with probability 0.5 is
        applied only to layers in the pre-net of the autoregressive decoder.

    Args:
        num_frame_channels: Number of channels in each frame (sometimes refered to as
            "Mel-frequency bins" or "FFT bins" or "FFT bands").
        size: The size of the hidden representation and output.
        num_layers: Number of fully connected layers of ReLU units.
        dropout: Probability of an element to be zeroed.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(self,
                 num_frame_channels: int,
                 size: int,
                 num_layers: int = HParam(),
                 dropout: float = HParam()):
        super().__init__()
        self.layers = torch.nn.Sequential(*tuple([
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=num_frame_channels if i == 0 else size, out_features=size),
                torch.nn.ReLU(inplace=True),
                torch.nn.LayerNorm(size),
                _AlwaysDropout(p=dropout),
            ) for i in range(num_layers)
        ]))

        for module in self.layers.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(
                    module.weight, gain=torch.nn.init.calculate_gain('relu'))

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        return super().__call__(frames)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Spectrogram
                frames.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size])
        """
        return self.layers(frames)
