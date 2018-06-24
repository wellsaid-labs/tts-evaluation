import torch
import numpy as np

from src.signal_model.upsample import ConditionalFeaturesUpsample


def test_conditional_features_upsample():
    batch_size = 2
    length = 16
    channels = 80
    upsample_convs = [2, 3]
    upsample_repeat = 2
    num_layers = 5
    block_hidden_size = 3
    signal_length = length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    upsample = ConditionalFeaturesUpsample(
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        out_channels=block_hidden_size * 2,
        in_channels=channels,
        num_layers=num_layers,
        upsample_chunks=5)

    local_features = torch.FloatTensor(batch_size, length, channels)
    upsampled = upsample(local_features)

    assert upsampled.shape == (batch_size, num_layers, 2 * block_hidden_size, signal_length)

    # Smoke test back prop
    upsampled.sum().backward()


def test_conditional_features_upsample_repeat():
    upsample = ConditionalFeaturesUpsample(upsample_repeat=2)
    local_features = torch.tensor([1, 2, 3]).view(1, 1, 3)
    local_features = upsample._repeat(local_features)
    local_features = local_features.view(-1)
    np.testing.assert_allclose(local_features, torch.tensor([1, 1, 2, 2, 3, 3]))
