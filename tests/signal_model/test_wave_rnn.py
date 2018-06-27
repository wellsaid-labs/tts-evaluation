import torch
import numpy as np

from src.signal_model import WaveRNN
from src.utils import split_signal


def test_wave_rnn_inference_train_equivilance():
    bits = 16
    batch_size = 1
    local_length = 2
    local_features_size = 80
    upsample_convs = [1, 1]
    upsample_repeat = 1
    local_features = torch.FloatTensor(batch_size, local_length, local_features_size)

    # Run inference
    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_convs=upsample_convs,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size).eval()
    predicted_coarse, predicted_fine, hidden = net(local_features)

    # [batch_size, signal_length] → [batch_size, signal_length - 1, 2]
    input_signal = torch.stack((predicted_coarse[:, :-1], predicted_fine[:, :-1]), dim=2)
    coarse, fine = split_signal(torch.zeros(batch_size))
    # [batch_size] → [batch_size, 1, 2]
    go_signal = torch.stack((coarse, fine), dim=1).unsqueeze(1).long()
    # [batch_size, signal_length - 1, 2] → [batch_size, signal_length, 2]
    input_signal = torch.cat((go_signal, input_signal), dim=1)

    _, _, other_hidden = net(
        local_features, input_signal=input_signal, target_coarse=predicted_coarse.unsqueeze(2))

    np.testing.assert_allclose(hidden.detach().numpy(), other_hidden.detach().numpy(), atol=1e-04)


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
    predicted_coarse, predicted_fine, _ = net(local_features, input_signal, target_coarse)

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
    predicted_coarse, predicted_fine, _ = net(local_features)

    assert predicted_coarse.shape == (batch_size, signal_length)
    assert predicted_fine.shape == (batch_size, signal_length)

    # Softmax
    assert torch.min(predicted_coarse) >= 0
    assert torch.max(predicted_coarse) < 256
    assert torch.min(predicted_fine) >= 0
    assert torch.max(predicted_fine) < 256
