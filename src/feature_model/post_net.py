from functools import partial

from torch import nn

from src.configurable import configurable

# NOTE: `momentum=0.01` to match Tensorflow defaults
nn.BatchNorm1d = partial(nn.BatchNorm1d, momentum=0.01)


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

    Args:
        num_convolution_layers (int, optional): Number of convolution layers to apply.
        num_convolution_filters (odd :clas:`int`, optional): Number of dimensions (channels)
            produced by the convolution.
        convolution_filter_size (int, optional): Size of the convolving kernel.
        frame_channels (int, optional): Number of channels in each frame (sometimes refered to
            as "Mel-frequency bins" or "FFT bins" or "FFT bands")

    Reference:
        * Tacotron 2 Paper:
          https://arxiv.org/pdf/1712.05884.pdf
    """

    @configurable
    def __init__(self,
                 num_convolution_layers=5,
                 num_convolution_filters=512,
                 convolution_filter_size=5,
                 frame_channels=80):
        super(PostNet, self).__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('`convolution_filter_size` must be odd')

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
                nn.Tanh()) for i in range(num_convolution_layers - 1)
        ]
        self.layers.append(
            # SOURCE: (Tacotron 2): followed by tanh activations on all but the final layer.
            # Last Layer without ``nn.Tanh`` and different number of ``out_channels``
            nn.Sequential(
                nn.Conv1d(
                    in_channels=num_convolution_filters,
                    out_channels=frame_channels,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)),
                nn.BatchNorm1d(num_features=frame_channels)))
        self.layers = nn.Sequential(*tuple(self.layers))

    def forward(self, frames):
        """
        Args:
            frames (torch.FloatTensor [batch_size, frame_channels, num_frames]): Batched set of
                spectrogram frames.

        Returns:
            residual (torch.FloatTensor [batch_size, frame_channels, num_frames]): Residual to add
                to the frames to improve the overall reconstruction.
        """
        return self.layers(frames)
