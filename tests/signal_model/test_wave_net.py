import torch

from src.signal_model import WaveNet


def test_wave_net():
    batch_size = 2
    skip_size = 32
    local_length = 16
    local_features_size = 80
    signal_channels = 256
    upsample_convs = [2, 3]
    upsample_repeat = 2
    signal_length = local_length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    local_features = torch.FloatTensor(batch_size, local_length, local_features_size)
    signal = torch.randint(0, signal_channels, (batch_size, signal_length), dtype=torch.long)

    net = WaveNet(
        num_layers=3,
        signal_channels=signal_channels,
        skip_size=skip_size,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size,
        upsample_chunks=3)
    predicted = net(local_features, signal)

    assert predicted.shape == (batch_size, signal_channels, signal_length)

    # Smoke test back prop
    predicted.sum().backward()


def test_wave_net_infer():
    if not torch.cuda.is_available():
        return

    batch_size = 2
    local_length = 16
    local_features_size = 80
    upsample_convs = [2, 3]
    upsample_repeat = 2
    signal_channels = 256
    signal_length = local_length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    local_features = torch.FloatTensor(batch_size, local_length, local_features_size)

    net = WaveNet(
        signal_channels=signal_channels,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size).eval()
    predicted = net(local_features)

    assert torch.min(predicted) >= 0
    assert torch.max(predicted) <= signal_channels
    assert predicted.shape == (batch_size, signal_length)
    # Everything is below ``signal_channels``
    assert (predicted < signal_channels).sum() == batch_size * signal_length
