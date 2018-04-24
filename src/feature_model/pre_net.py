from functools import partial

from torch import nn

from src.configurable import configurable

# NOTE: `momentum=0.01` to match Tensorflow defaults
nn.BatchNorm1d = partial(nn.BatchNorm1d, momentum=0.01)


class PreNet(nn.Module):
    """ Pre-net processes the last frame of the spectrogram.

    SOURCE (Tacotron 2):
        ...small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that
        the pre-net acting as an information bottleneck was essential for learning attention.

    Args:
        frame_channels (int, optional): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands").
        num_layers (int): Number of fully connected layers of ReLU units.
        hidden_size (int): Number of hidden units in each layer.
        nonlinearity (torch.nn.Module): A non-linear differentiable function to use.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(self, frame_channels=80, num_layers=2, hidden_size=256, nonlinearity=nn.ReLU):
        super(PreNet, self).__init__()
        self.layers = nn.Sequential(*tuple([
            nn.Sequential(
                nn.Linear(
                    in_features=hidden_size
                    if i != 0 else frame_channels, out_features=hidden_size), nonlinearity())
            for i in range(num_layers)
        ]))

    def forward(self, frames):
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Batched set of
                spectrogram frames.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size]): Batched set of
                spectrogram frames processed by the Pre-net.
        """
        return self.layers(frames)
