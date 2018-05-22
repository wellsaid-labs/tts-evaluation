import pytest
import torch

from src.signal_model import SignalModel


def test_signal_model():
    batch_size = 2
    skip_size = 32
    local_length = 16
    local_features_size = 80
    mu = 255
    upsample_convs = [2, 3]
    upsample_repeat = 2
    signal_length = local_length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    local_features = torch.FloatTensor(local_length, batch_size, local_features_size)
    signal = torch.randint(0, mu + 1, (signal_length, batch_size))

    net = SignalModel(
        num_layers=3,
        mu=mu,
        skip_size=skip_size,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size)
    predicted = net(local_features, signal)

    assert predicted.shape == (batch_size, mu + 1, signal_length)
    for i in range(batch_size):
        for j in range(signal_length):
            assert predicted[i, :, j].sum().item() == pytest.approx(1, 0.0001)

    # Smoke test back prop
    predicted.sum().backward()
