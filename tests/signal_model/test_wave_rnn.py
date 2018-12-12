import torch
import numpy as np

from src.signal_model import WaveRNN
from src.signal_model.wave_rnn import _scale
from src.audio import split_signal


def test_wave_rnn_infer_equals_forward():
    bits = 16
    local_length = 4
    batch_size = 1
    local_features_size = 80
    upsample_learned = 2
    upsample_repeat = 2
    hidden_size = 32
    upsample_kernels = [(5, 5), (3, 3), (3, 3), (3, 3)]
    length_padding = sum([(kernel[0] - 1) for kernel in upsample_kernels])

    net = WaveRNN(
        hidden_size=hidden_size,
        bits=bits,
        upsample_num_filters=[64, 64, 32, upsample_learned],
        upsample_kernels=upsample_kernels,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size).eval()

    # Ensure that each parameter a reasonable value to affect the output
    for parameter in net.parameters():
        if parameter.requires_grad:
            torch.nn.init.normal_(parameter, std=0.1)

    # Inputs
    # NOTE: # Multiply by 0.1 make the gru more unstable
    local_features = torch.randn(batch_size, local_length + length_padding,
                                 local_features_size) * 0.1
    hidden_state = torch.randn(hidden_size, batch_size)
    go_coarse, go_fine = split_signal(torch.zeros(batch_size))
    infer_hidden_state = (go_coarse.long(), go_fine.long(), hidden_state[:int(hidden_size / 2)],
                          hidden_state[int(hidden_size / 2):])
    forward_hidden_state = hidden_state.transpose(0, 1)

    # Run inference
    # NOTE: argmax to ensure forward and infer sample the deterministically
    infer_predicted_coarse, infer_predicted_fine, infer_hidden_state = net.infer(
        local_features, hidden_state=infer_hidden_state, argmax=True, pad=False)

    # [batch_size, signal_length] → [batch_size, signal_length - 1, 2]
    forward_input_signal = torch.stack((infer_predicted_coarse, infer_predicted_fine),
                                       dim=2)[:, :-1]
    # [batch_size] → [batch_size, 1, 2]
    go_signal = torch.stack((go_coarse, go_fine), dim=1).unsqueeze(1).long()
    # [batch_size, signal_length - 1, 2] → [batch_size, signal_length, 2]
    forward_input_signal = torch.cat((go_signal, forward_input_signal), dim=1)
    # [batch_size, signal_length] → [batch_size, signal_length, 1]
    forward_target_coarse = infer_predicted_coarse.unsqueeze(2)

    forward_predicted_coarse, forward_predicted_fine, forward_hidden_state = net.forward(
        local_features,
        input_signal=forward_input_signal,
        target_coarse=forward_target_coarse,
        hidden_state=forward_hidden_state)

    # Ensure infer hidden state is equal to forward hidden state
    _, _, infer_coarse_hidden, infer_fine_hidden = infer_hidden_state
    infer_hidden_state = torch.cat((infer_coarse_hidden, infer_fine_hidden), dim=0).transpose(0, 1)
    np.testing.assert_allclose(
        forward_hidden_state.detach().numpy(), infer_hidden_state.detach().numpy(), atol=1e-04)

    forward_predicted_coarse = forward_predicted_coarse.max(dim=2)[1]
    np.testing.assert_allclose(
        infer_predicted_coarse.detach().numpy(),
        forward_predicted_coarse.detach().numpy(),
        atol=1e-04)

    forward_predicted_fine = forward_predicted_fine.max(dim=2)[1]
    np.testing.assert_allclose(
        infer_predicted_fine.detach().numpy(), forward_predicted_fine.detach().numpy(), atol=1e-04)


def test_wave_rnn_scale():
    original = torch.linspace(0, 255, steps=256)
    scaled = _scale(original, 256)
    assert torch.min(scaled) == -1.0
    assert torch.max(scaled) == 1.0
    reconstructed = (scaled + 1.0) * 127.5
    np.testing.assert_allclose(original.numpy(), reconstructed.numpy(), atol=1e-04)


def test_wave_rnn_forward():
    bits = 16
    batch_size = 2
    local_length = 16
    local_features_size = 80
    upsample_learned = 6
    upsample_repeat = 2
    upsample_kernels = [(5, 5), (3, 3), (3, 3), (3, 3)]
    length_padding = sum([(kernel[0] - 1) for kernel in upsample_kernels])
    signal_length = local_length * upsample_learned * upsample_repeat

    local_features = torch.rand(batch_size, local_length + length_padding, local_features_size)
    target_coarse = torch.rand(batch_size, signal_length, 1)
    input_signal = torch.rand(batch_size, signal_length, 2)

    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_num_filters=[64, 64, 32, upsample_learned],
        upsample_kernels=upsample_kernels,
        upsample_repeat=upsample_repeat,
        local_features_size=local_features_size)
    predicted_coarse, predicted_fine, _ = net.forward(local_features, input_signal, target_coarse)

    assert predicted_coarse.shape == (batch_size, signal_length, net.bins)
    assert predicted_fine.shape == (batch_size, signal_length, net.bins)

    # Smoke test back prop
    (predicted_coarse + predicted_fine).sum().backward()


def test_wave_rnn_infer():
    bits = 16
    local_length = 16
    batch_size = 2
    local_features_size = 80
    upsample_learned = 6
    upsample_repeat = 2
    upsample_kernels = [(5, 5), (3, 3), (3, 3), (3, 3)]
    signal_length = local_length * upsample_learned * upsample_repeat

    # TODO: With out ``fill_(1)`` there are strange runtime errors, investigate
    local_features = torch.FloatTensor(batch_size, local_length, local_features_size).fill_(1)

    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_num_filters=[64, 64, 32, upsample_learned],
        upsample_repeat=upsample_repeat,
        upsample_kernels=upsample_kernels,  # Local
        local_features_size=local_features_size).eval()
    predicted_coarse, predicted_fine, _ = net.infer(local_features, pad=True)

    assert predicted_coarse.shape == (batch_size, signal_length)
    assert predicted_fine.shape == (batch_size, signal_length)

    # Softmax
    assert torch.min(predicted_coarse) >= 0
    assert torch.max(predicted_coarse) < 256
    assert torch.min(predicted_fine) >= 0
    assert torch.max(predicted_fine) < 256


def test_wave_rnn_infer_unbatched():
    bits = 16
    local_length = 16
    local_features_size = 80
    upsample_learned = 6
    upsample_repeat = 2
    upsample_kernels = [(5, 5), (3, 3), (3, 3), (3, 3)]
    signal_length = local_length * upsample_learned * upsample_repeat

    # TODO: With out ``fill_(1)`` there are strange runtime errors, investigate
    local_features = torch.FloatTensor(local_length, local_features_size).fill_(1)

    net = WaveRNN(
        hidden_size=32,
        bits=bits,
        upsample_num_filters=[64, 64, 32, upsample_learned],
        upsample_repeat=upsample_repeat,
        upsample_kernels=upsample_kernels,
        local_features_size=local_features_size).eval()
    predicted_coarse, predicted_fine, _ = net.infer(local_features, pad=True)

    assert predicted_coarse.shape == (signal_length,)
    assert predicted_fine.shape == (signal_length,)
