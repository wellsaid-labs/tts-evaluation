import torch
import numpy as np

from src.signal_model.upsample import ConditionalFeaturesUpsample


def test_conditional_features_upsample():
    batch_size = 2
    length = 16
    in_channels = 80
    upsample_learned = 4
    upsample_repeat = 2
    out_channels = 3
    signal_length = length * upsample_learned * upsample_repeat

    upsample = ConditionalFeaturesUpsample(
        upsample_learned=upsample_learned,
        upsample_repeat=upsample_repeat,
        out_channels=out_channels,
        in_channels=in_channels)

    local_features = torch.FloatTensor(batch_size, length, in_channels)
    upsampled = upsample(local_features)

    assert upsampled.shape == (batch_size, out_channels, signal_length)

    # Smoke test back prop
    upsampled.sum().backward()


def test_conditional_features_upsample_repeat():
    upsample = ConditionalFeaturesUpsample(upsample_repeat=2)
    local_features = torch.tensor([1, 2, 3]).view(1, 1, 3)
    local_features = upsample._repeat(local_features)
    local_features = local_features.view(-1)
    np.testing.assert_allclose(local_features, torch.tensor([1, 1, 2, 2, 3, 3]))
