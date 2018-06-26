from torch import nn

import torch
import numpy as np

from src.signal_model.stripped_gru import StrippedGRU


def equivalent_gru(size, hidden, input_, weight_hh_l0, bias_ih_l0):
    # ... [size]
    bias_reset_gate, bias_update_gate, bias_memory = torch.split(bias_ih_l0, size)

    project_hidden = nn.Linear(size, 3 * size, bias=False)
    project_hidden.weight = weight_hh_l0

    # [1, batch_size, size] → [1, batch_size, 3 * size]
    projected_hidden = project_hidden(hidden)
    # ... [1, batch_size, size]
    hidden_reset_gate, hidden_update_gate, hidden_memory = torch.split(
        projected_hidden, size, dim=2)
    input_reset_gate, input_update_gate, input_memory = torch.split(input_, size, dim=2)

    reset_gate = torch.nn.functional.sigmoid(input_reset_gate + hidden_reset_gate + bias_reset_gate)
    update_gate = torch.nn.functional.sigmoid(
        input_update_gate + hidden_update_gate + bias_update_gate)
    next_hidden = torch.nn.functional.tanh(input_memory + bias_memory + reset_gate * hidden_memory)
    return (1.0 - update_gate) * next_hidden + update_gate * hidden


def test_stripped_gru():
    size = 10
    batch_size = 3
    seq_len = 1
    input_ = torch.randn(seq_len, batch_size, size * 3)
    hidden = torch.randn(1, batch_size, size)

    stripped_gru = StrippedGRU(size)
    _, output = stripped_gru(input_, hidden)

    expected_output = equivalent_gru(size, hidden, input_, stripped_gru.gru.weight_hh_l0,
                                     stripped_gru.gru.bias_ih_l0)
    np.testing.assert_allclose(
        output.detach().numpy(), expected_output.detach().numpy(), rtol=1e-04)


def test_stripped_gru_cuda():
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.deterministic = True

    size = 10
    batch_size = 3
    seq_len = 1
    input_ = torch.randn(seq_len, batch_size, size * 3).cuda()
    hidden = torch.randn(1, batch_size, size).cuda()

    stripped_gru = StrippedGRU(size).cuda()
    _, output = stripped_gru(input_, hidden)

    expected_output = equivalent_gru(size, hidden, input_, stripped_gru.gru.weight_hh_l0,
                                     stripped_gru.gru.bias_ih_l0)
    np.testing.assert_allclose(
        output.detach().cpu().numpy(), expected_output.detach().cpu().numpy(), rtol=1e-04)


def test_stripped_gru_cuda_sequence():
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.deterministic = True

    size = 10
    batch_size = 3
    seq_len = 2
    input_ = torch.randn(seq_len, batch_size, size * 3).cuda()
    hidden = torch.randn(1, batch_size, size).cuda()

    stripped_gru = StrippedGRU(size).cuda()
    # output [seq_len, batch_size, size]
    output, _ = stripped_gru(input_, hidden)

    expected_output_one = equivalent_gru(size, hidden, input_[0:1], stripped_gru.gru.weight_hh_l0,
                                         stripped_gru.gru.bias_ih_l0)
    expected_output_two = equivalent_gru(size, expected_output_one, input_[1:2],
                                         stripped_gru.gru.weight_hh_l0, stripped_gru.gru.bias_ih_l0)
    np.testing.assert_allclose(
        output[0:1].detach().cpu().numpy(), expected_output_one.detach().cpu().numpy(), rtol=1e-04)
    np.testing.assert_allclose(
        output[1:2].detach().cpu().numpy(), expected_output_two.detach().cpu().numpy(), rtol=1e-04)
