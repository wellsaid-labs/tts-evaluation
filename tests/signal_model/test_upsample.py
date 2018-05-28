import torch

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
        local_features_size=channels,
        block_hidden_size=block_hidden_size,
        num_layers=num_layers,
        upsample_chunks=5)

    local_features = torch.FloatTensor(batch_size, length, channels)
    upsampled = upsample(local_features)

    assert upsampled.shape == (2 * block_hidden_size, batch_size, num_layers, signal_length)

    # Smoke test back prop
    upsampled.sum().backward()
