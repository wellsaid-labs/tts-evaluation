import torch
import torch.nn
from hparams import HParam, configurable
from torch.nn import functional


class _AlwaysDropout(torch.nn.Dropout):
    """Adaptation of `nn.Dropout` to apply dropout during both evaluation and training."""

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return functional.dropout(input=input_, p=self.p, training=True, inplace=self.inplace)


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
        seq_meta_embed_size
        size: The size of the hidden representation and output.
        num_layers: Number of fully connected layers of ReLU units.
        dropout: Probability of an element to be zeroed.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(
        self,
        num_frame_channels: int,
        seq_meta_embed_size: int,
        size: int,
        num_layers: int = HParam(),
        dropout: float = HParam(),
    ):
        super().__init__()
        _layers = [
            torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=(num_frame_channels if i == 0 else size) + seq_meta_embed_size,
                    out_features=size,
                ),
                torch.nn.ReLU(inplace=True),
                torch.nn.LayerNorm(size),
                _AlwaysDropout(p=dropout),
            )
            for i in range(num_layers)
        ]
        self.layers = torch.nn.ModuleList(_layers)

        for module in self.layers.modules():
            if isinstance(module, torch.nn.Linear):
                gain = torch.nn.init.calculate_gain("relu")
                torch.nn.init.xavier_uniform_(module.weight, gain=gain)

    def __call__(self, frames: torch.Tensor, seq_metadata: torch.Tensor) -> torch.Tensor:
        return super().__call__(frames, seq_metadata)

    def forward(self, frames: torch.Tensor, seq_metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, num_frame_channels]): Spectrogram
                frames.
            seq_metadata (torch.FloatTensor [batch_size, seq_meta_embed_size])

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size])
        """
        # [batch_size, seq_meta_embed_size] → [num_frames, batch_size, seq_meta_embed_size]
        seq_metadata = seq_metadata.unsqueeze(0).expand(frames.shape[0], -1, -1)
        for layer in self.layers:
            frames = layer(torch.cat([seq_metadata, frames], dim=2))
        return frames
