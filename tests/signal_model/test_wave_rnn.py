import torch

from src.signal_model import WaveRNN


def test_wave_rnn():
    bits = 16
    batch_size = 2
    local_length = 16
    local_features_size = 80
    upsample_convs = [2, 3]
    upsample_repeat = 2
    signal_length = local_length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    local_features = torch.FloatTensor(batch_size, local_length, local_features_size)
    target_coarse = torch.rand(batch_size, signal_length, 1)
    input_signal = torch.rand(batch_size, signal_length, 2)

    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size)
    predicted_coarse, predicted_fine = net(local_features, input_signal, target_coarse)

    assert predicted_coarse.shape == (batch_size, signal_length, net.bins)
    assert predicted_fine.shape == (batch_size, signal_length, net.bins)

    # Smoke test back prop
    (predicted_coarse + predicted_fine).sum().backward()


def test_wave_rnn_inference():
    bits = 16
    batch_size = 2
    local_length = 16
    local_features_size = 80
    upsample_convs = [2, 3]
    upsample_repeat = 2
    signal_length = local_length * upsample_convs[0] * upsample_convs[1] * upsample_repeat

    local_features = torch.FloatTensor(batch_size, local_length, local_features_size)

    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size).eval()
    predicted_coarse, predicted_fine = net(local_features)

    assert predicted_coarse.shape == (batch_size, signal_length, net.bins)
    assert predicted_fine.shape == (batch_size, signal_length, net.bins)

    # Softmax
    assert torch.min(predicted_coarse) >= 0
    assert torch.max(predicted_coarse) <= 1
    assert torch.min(predicted_fine) >= 0
    assert torch.max(predicted_fine) <= 1
