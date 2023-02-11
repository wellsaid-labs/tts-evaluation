import config as cf
import torch
import torch.nn


class GaussianDropout(torch.nn.Module):
    def __init__(self, p):
        super(GaussianDropout, self).__init__()
        if p < 0 or p >= 1:
            raise Exception("p value should accomplish 0 <= p < 1")
        self.p = p

    def forward(self, x):
        if self.training and self.p != 0:
            stddev = (self.p / (1.0 - self.p)) ** 0.5
            epsilon = torch.randn_like(x) * stddev
            return x * epsilon
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
        _layers = [
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=num_frame_channels if i == 0 else size, out_features=size
                ),
                torch.nn.GELU(),
                GaussianDropout(p=dropout),
                torch.nn.LayerNorm(size, **cf.get()),
            )
            for i in range(num_layers)
        ]
        self.layers = torch.nn.ModuleList(_layers)

        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                gain = torch.nn.init.calculate_gain("relu")
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)

    def __call__(self, frames: torch.Tensor, seq_embed: torch.Tensor) -> torch.Tensor:
        return super().__call__(frames, seq_embed)

    def forward(self, frames: torch.Tensor, seq_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Spectrogram
                frames.
            seq_embed (torch.FloatTensor [batch_size, seq_embed_size])

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size])
        """
        for layer in self.layers:
            frames = layer(frames)
        return frames
