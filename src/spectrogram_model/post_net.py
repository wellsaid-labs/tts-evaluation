from hparams import configurable
from hparams import HParam
from torch import nn


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
                 num_convolution_layers=HParam(),
                 num_convolution_filters=HParam(),
                 convolution_filter_size=HParam(),
                 convolution_dropout=HParam()):
        super().__init__()

        # LEARN MORE:
        # https://datascience.stackexchange.com/questions/23183/why-convolutions-always-use-odd-numbers-as-filter-size
        assert convolution_filter_size % 2 == 1, ('``convolution_filter_size`` must be odd')

        self.layers = nn.ModuleList([
            nn.Sequential(
                # SOURCE (Tacotron 2): Each post-net layer is comprised of 512 filters with shape
                # 5 × 1 with batch normalization, followed by tanh activations on all but the
                # final layer.
                # NOTE: We learned in Comet experiments in December 2019 that RELu + LayerNorm
                # combination was more effective.
                nn.Conv1d(
                    in_channels=num_convolution_filters if i != 0 else frame_channels,
                    out_channels=num_convolution_filters,
                    kernel_size=convolution_filter_size,
                    padding=int((convolution_filter_size - 1) / 2)),
                nn.ReLU(),
                nn.Dropout(p=convolution_dropout)) for i in range(num_convolution_layers - 1)
        ])

        self.normalization_layers = nn.ModuleList(
            [nn.LayerNorm(num_convolution_filters) for i in range(num_convolution_layers - 1)])

        # Initialize weights
        for layer in self.layers:
            for module in layer.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))

        # SOURCE: (Tacotron 2):
        # followed by tanh activations on all but the final layer.
        # NOTE: Last Layer without ``nn.Tanh`` and different number of ``out_channels``
        self.last_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=num_convolution_filters,
                out_channels=frame_channels,
                kernel_size=convolution_filter_size,
                padding=int((convolution_filter_size - 1) / 2)), nn.Dropout(p=convolution_dropout))

    def forward(self, frames, mask):
        """
        Args:
            frames (torch.FloatTensor [batch_size, frame_channels, num_frames]): Batched set of
                spectrogram frames.
            mask (torch.BoolTensor [batch_size, num_frames]): Mask such that the padding tokens
                are zeros.

        Returns:
            residual (torch.FloatTensor [batch_size, frame_channels, num_frames]): Residual to add
                to the frames to improve the overall reconstruction.
        """
        frames = frames.masked_fill(~mask.unsqueeze(1), 0)

        for i, (layer, normalize) in enumerate(zip(self.layers, self.normalization_layers)):
            # NOTE: Ignore the first residual because the shapes dont match.
            frames = layer(frames) if i == 0 else layer(frames) + frames

            # frames [batch_size, frame_channels, num_frames]
            frames = frames.transpose(1, 2)
            frames = normalize(frames)
            frames = frames.transpose(1, 2)

            frames = frames.masked_fill(~mask.unsqueeze(1), 0)

        return self.last_layer(frames).masked_fill(~mask.unsqueeze(1), 0)
