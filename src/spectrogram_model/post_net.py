from torch import nn

from src.hparams import configurable
from src.hparams import ConfiguredArg


class PostNet(nn.Module):
    """ Post-net processes a frame of the spectrogram.

    SOURCE (Tacotron 2):
        Finally, the predicted mel spectrogram is passed through a 5-layer convolutional post-net
        which predicts a residual to add to the prediction to improve the overall reconstruction.
        Each post-net layer is comprised of 512 filters with shape 5 × 1 with batch normalization,
        followed by tanh activations on all but the final layer.

        ...

        Since it is not possible to use the information of predicted future frames before they have
        been decoded, we use a convolutional postprocessing network to incorporate past and future
        frames after decoding to improve the feature predictions.

    NOTE: Google Tacotron 2 authors mentioned they did not add dropout to the PostNet over GChat.

    Args:
        frame_channels (int): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")
        num_convolution_layers (int): Number of convolution layers to apply.
        num_convolution_filters (odd :clas:`int`): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int): Size of the convolving kernel.
        convolution_dropout (float): Probability of an element to be zeroed.

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(self,
                 frame_channels,
                 num_convolution_layers=ConfiguredArg(),
                 num_convolution_filters=ConfiguredArg(),
                 convolution_filter_size=ConfiguredArg(),
                 convolution_dropout=ConfiguredArg()):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('``convolution_filter_size`` must be odd')

        self.layers = [
            nn.Sequential(
                # SOURCE (Tacotron 2): Each post-net layer is comprised of 512 filters with shape
                # 5 × 1 with batch normalization, followed by tanh activations on all but the
                # final layer.
                nn.Conv1d(
                    in_channels=num_convolution_filters if i != 0 else frame_channels,
                    out_channels=num_convolution_filters,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)),
                nn.BatchNorm1d(num_features=num_convolution_filters),
                nn.Tanh(),
                nn.Dropout(p=convolution_dropout)) for i in range(num_convolution_layers - 1)
        ]

        # Initialize weights
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))

        self.layers.append(
            # SOURCE: (Tacotron 2):
            # followed by tanh activations on all but the final layer.
            # Last Layer without ``nn.Tanh`` and different number of ``out_channels``
            nn.Sequential(
                nn.Conv1d(
                    in_channels=num_convolution_filters,
                    out_channels=frame_channels,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)),
                nn.BatchNorm1d(num_features=frame_channels), nn.Dropout(p=convolution_dropout)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, frames, mask):
        """
        Args:
            frames (torch.FloatTensor [batch_size, frame_channels, num_frames]): Batched set of
                spectrogram frames.
            mask (torch.ByteTensor [batch_size, num_frames]): Mask such that the padding tokens
                are zeros.

        Returns:
            residual (torch.FloatTensor [batch_size, frame_channels, num_frames]): Residual to add
                to the frames to improve the overall reconstruction.
        """
        for layer in self.layers:
            frames = frames.masked_fill(~mask.unsqueeze(1), 0)
            frames = layer(frames)
        return frames.masked_fill(~mask.unsqueeze(1), 0)