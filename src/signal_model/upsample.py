import torch

from torch import nn


class ConditionalFeaturesUpsample(nn.Module):
    """
    Notes:
        * Tacotron 2 authors mention on Google Chat:  "We upsample 4x with the layers and then
          repeat each value 75x".

    Args:
        upsample_convs (list of int): Size of convolution layers used to upsample local features
            (e.g. 256 frames x 4 x ...).
        upsample_repeat (int): Number of times to repeat frames, another upsampling technique.
        in_channels (int): Dimensionality of input features.
        output_size (int): Dimensionality of outputed features.
        num_layers (int): Number of layers in Wavenet to condition.
        upsample_chunks (int): Control the memory used by ``upsample_layers`` by breaking the
            operation up into chunks.
        local_feature_processing_layers (int): Number of Conv1D for processing the spectrogram.
    """

    def __init__(self,
                 upsample_convs=[4],
                 upsample_repeat=75,
                 in_channels=80,
                 out_channels=64,
                 num_layers=24,
                 upsample_chunks=3,
                 local_feature_processing_layers=4):
        super().__init__()
        self.out_channels = out_channels
        self.upsample_repeat = upsample_repeat
        self.num_layers = num_layers

        assert self.num_layers % upsample_chunks == 0, (
            "For simplicity, we only support whole chunking")

        # TODO: Document
        self.preprocess = None
        if local_feature_processing_layers is not None:
            self.preprocess = nn.Sequential(*[
                nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
                for i in range(local_feature_processing_layers)
            ])

        self.upsample_signal_length = None
        if upsample_convs is not None:
            # Similar to:
            # https://github.com/kan-bayashi/PytorchWaveNetVocoder/blob/fe99175470977d993cfae6df5e8610b6aab8ce90/src/nets/wavenet.py#L131
            self.upsample_signal_length = nn.Sequential(*tuple([
                nn.ConvTranspose2d(1, 1, kernel_size=(1, size), stride=(1, size))
                for size in upsample_convs
            ]))

        self.upsample_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels * int(num_layers / upsample_chunks),
                kernel_size=1) for _ in range(upsample_chunks)
        ])
        for conv in self.upsample_layers:
            torch.nn.init.xavier_uniform_(conv.weight, gain=torch.nn.init.calculate_gain('tanh'))

    def _repeat(self, local_features):
        """ Repeat similar to this 3x repeat [1, 2, 3] → [1, 1, 1, 2, 2, 2, 3, 3, 3].

        Notes:
            * Learn more:
              https://stackoverflow.com/questions/35227224/torch-repeat-tensor-like-numpy-repeat

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, in_channels]):
                Local features to repeat.

        Returns:
            local_features (torch.FloatTensor [batch_size, local_length,
                in_channels * repeat]): Local features to repeated.
        """
        # TODO: Even without a speaker, we can try a speaker embedding
        # [batch_size, in_channels, upsample_length] →
        # [batch_size, in_channels, upsample_length, 1]
        local_features = local_features.unsqueeze(3)

        # [batch_size, in_channels, upsample_length] →
        # [batch_size, in_channels, upsample_length, num_repeat]
        local_features = local_features.repeat(1, 1, 1, self.upsample_repeat)

        # [batch_size, in_channels, upsample_length, num_repeat] →
        # [batch_size, in_channels, upsample_length * num_repeat]
        return local_features.view(local_features.shape[0], local_features.shape[1], -1)

    def forward(self, local_features):
        """
        TODO: Support global conditioning

        Args:
            local_features (torch.FloatTensor [batch_size, local_length, in_channels]):
                Local features to condition signal generation (e.g. spectrogram).

        Returns:
            conditional_features (torch.FloatTensor [batch_size, num_layers, out_channels,
                signal_length]): Upsampled local conditional features.
        """
        # Convolution operater expects input_ of the form:
        # [batch_size, in_channels, signal_length (local_length)]
        local_features = local_features.transpose(1, 2)

        # [batch_size, in_channels, local_length]
        # [batch_size, in_channels, local_length] →
        if self.preprocess is not None:
            local_features = self.preprocess(local_features)

        # [batch_size, in_channels, local_length] →
        # [batch_size, out_channels, signal_length]
        local_features = local_features.unsqueeze(1)
        if self.upsample_signal_length is not None:
            local_features = self.upsample_signal_length(local_features)
        local_features = local_features.squeeze(1)
        local_features = self._repeat(local_features)

        # [batch_size, out_channels, signal_length] →
        # [batch_size, out_channels * num_layers, signal_length]
        local_features = [conv(local_features) for conv in self.upsample_layers]
        local_features = torch.cat(local_features, dim=1)

        batch_size, _, signal_length = local_features.shape

        # [batch_size, out_channels, signal_length] →
        # [batch_size, num_layers, out_channels, signal_length]
        local_features = local_features.view(batch_size, self.num_layers, self.out_channels,
                                             signal_length)

        return local_features
