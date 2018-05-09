from torch import nn

from src.configurable import configurable


class PreNet(nn.Module):
    """ Pre-net processes the last frame of the spectrogram.

    SOURCE (Tacotron 2):
        ...small pre-net containing 2 fully connected layers of 256 hidden ReLU units. We found that
        the pre-net acting as an information bottleneck was essential for learning attention.

        ...

        In order to introduce output variation at inference time, dropout with probability 0.5 is
        applied only to layers in the pre-net of the autoregressive decoder.

    Args:
        frame_channels (int, optional): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands").
        num_layers (int, optional): Number of fully connected layers of ReLU units.
        hidden_size (int, optional): Number of hidden units in each layer.
        dropout (float, optional): Probability of an element to be zeroed.
        nonlinearity (torch.nn.Module, optional): A non-linear differentiable function to use.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(self, frame_channels=80, num_layers=2, hidden_size=256, dropout=0.5):
        super(PreNet, self).__init__()
        self.layers = nn.Sequential(*tuple([
            nn.Sequential(
                nn.Linear(
                    in_features=frame_channels
                    if i == 0 else hidden_size, out_features=hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout)) for i in range(num_layers)
        ]))

        # Initialize weights
        for module in self.layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, frames):
        """
        Args:
            frames (torch.FloatTensor [num_frames, batch_size, frame_channels]): Batched set of
                spectrogram frames.

        Returns:
            frames (torch.FloatTensor [num_frames, batch_size, hidden_size]): Batched set of
                spectrogram frames processed by the Pre-net.
        """
        # NOTE: ``.train()`` to ensure dropout does not turn off during inference as mentioned in
        # the Tacotron 2 paper.
        for layer in self.layers:
            layer[2].train()
        return self.layers(frames)
