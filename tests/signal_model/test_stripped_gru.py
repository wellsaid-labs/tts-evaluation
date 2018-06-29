from torch import nn

import torch
import numpy as np

from src.signal_model.stripped_gru import StrippedGRU


def equivalent_gru(size, hidden, input_, weight_hh_l0, bias_ih_l0, bias_hh_l0):
    """ GRU implemented manually to check the output of Stripped GRU """
    # ... [size]
    bias_r, bias_u, bias_e = torch.split(bias_ih_l0, size)

    project_hidden = nn.Linear(size, 3 * size)
    project_hidden.weight = weight_hh_l0
    project_hidden.bias = bias_hh_l0

    # [1, batch_size, size] → [1, batch_size, 3 * size]
    projected_hidden = project_hidden(hidden)
    # ... [1, batch_size, size]
    hidden_r, hidden_u, hidden_e = projected_hidden.split(size, dim=2)
    input_r, input_u, input_e = input_.split(size, dim=2)

    r = torch.nn.functional.sigmoid(input_r + hidden_r + bias_r)
    u = torch.nn.functional.sigmoid(input_u + hidden_u + bias_u)
    next_hidden = torch.nn.functional.tanh(input_e + bias_e + r * hidden_e)
    return (1.0 - u) * next_hidden + u * hidden


def test_stripped_gru():
    """ Test a CPU GRU implementation on seq length of 1 """
    size = 10
    batch_size = 3
    seq_len = 1
    input_ = torch.randn(seq_len, batch_size, size * 3)
    hidden = torch.randn(1, batch_size, size)

    stripped_gru = StrippedGRU(size)
    for parameter in stripped_gru.parameters():
        if parameter.requires_grad:
            # Ensure that each parameter a reasonable value to affect the output
            torch.nn.init.normal_(parameter)

    _, output = stripped_gru(input_, hidden)

    expected_output = equivalent_gru(size, hidden, input_, stripped_gru.gru.weight_hh_l0,
                                     stripped_gru.gru.bias_ih_l0, stripped_gru.gru.bias_hh_l0)
    np.testing.assert_allclose(
        output.detach().numpy(), expected_output.detach().numpy(), rtol=1e-04)


def test_stripped_gru_cuda():
    """ Test a CUDA GRU implementation on seq length of 1"""
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.deterministic = True

    size = 10
    batch_size = 3
    seq_len = 1
    input_ = torch.randn(seq_len, batch_size, size * 3).cuda()
    hidden = torch.randn(1, batch_size, size).cuda()

    stripped_gru = StrippedGRU(size).cuda()
    for parameter in stripped_gru.parameters():
        if parameter.requires_grad:
            # Ensure that each parameter a reasonable value to affect the output
            torch.nn.init.normal_(parameter)

    _, output = stripped_gru(input_, hidden)

    expected_output = equivalent_gru(size, hidden, input_, stripped_gru.gru.weight_hh_l0,
                                     stripped_gru.gru.bias_ih_l0, stripped_gru.gru.bias_hh_l0)
    np.testing.assert_allclose(
        output.detach().cpu().numpy(), expected_output.detach().cpu().numpy(), rtol=1e-04)


def test_stripped_gru_cuda_sequence():
    """ Test a CUDA GRU implementation on a longer sequence"""
    if not torch.cuda.is_available():
        return

    torch.backends.cudnn.deterministic = True

    size = 10
    batch_size = 3
    seq_len = 2
    input_ = torch.randn(seq_len, batch_size, size * 3).cuda()
    hidden = torch.randn(1, batch_size, size).cuda()

    stripped_gru = StrippedGRU(size).cuda()
    for parameter in stripped_gru.parameters():
        if parameter.requires_grad:
            # Ensure that each parameter a reasonable value to affect the output
            torch.nn.init.normal_(parameter)

    # output [seq_len, batch_size, size]
    output, _ = stripped_gru(input_, hidden)

    expected_output_one = equivalent_gru(size, hidden, input_[0:1], stripped_gru.gru.weight_hh_l0,
                                         stripped_gru.gru.bias_ih_l0, stripped_gru.gru.bias_hh_l0)
    expected_output_two = equivalent_gru(size, expected_output_one, input_[1:2],
                                         stripped_gru.gru.weight_hh_l0, stripped_gru.gru.bias_ih_l0,
                                         stripped_gru.gru.bias_hh_l0)
    np.testing.assert_allclose(
        output[0:1].detach().cpu().numpy(), expected_output_one.detach().cpu().numpy(), rtol=1e-04)
    np.testing.assert_allclose(
        output[1:2].detach().cpu().numpy(), expected_output_two.detach().cpu().numpy(), rtol=1e-04)
