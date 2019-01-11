import torch

from src.signal_model.upsample import ConditionalFeaturesUpsample


def test_conditional_features_upsample():
    batch_size = 2
    length = 16
    in_channels = 12
    out_channels = 24
    upsample_learned = 6
    upsample_repeat = 2
    signal_length = length * upsample_learned * upsample_repeat

    net = ConditionalFeaturesUpsample(
        in_channels=in_channels,
        out_channels=out_channels,
        upsample_repeat=upsample_repeat,
        num_filters=[64, 64, 32, upsample_learned],
        kernels=[(5, 5), (3, 3), (3, 3), (3, 3)])  # Padding of 2 + 1 + 1 + 1 = 5 on either side

    local_features = torch.FloatTensor(batch_size, length + 10, in_channels)
    upsampled = net(local_features)

    assert upsampled.shape == (batch_size, out_channels, signal_length)

    # Smoke test back prop
    upsampled.sum().backward()
