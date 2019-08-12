from copy import deepcopy
from unittest import mock

import math
import unittest

import torch
import numpy

from src.optimizers import AutoOptimizer
from src.optimizers import Lamb
from src.optimizers import Optimizer
from tests._utils import MockCometML


def test_lamb_optimizer__adam():
    net_lamb = torch.nn.LSTM(10, 20, 2)
    net_adam = torch.nn.LSTM(10, 20, 2)

    net_adam.load_state_dict(deepcopy(net_lamb.state_dict()))  # Same weights as `net_lamb`

    # `trust_ratio=1.0` ensures equality with AdamW (similar to Adam with a different weight decay).
    lamb = Lamb(
        params=net_lamb.parameters(),
        amsgrad=False,
        lr=10**-3,
        min_trust_ratio=1,
        max_trust_ratio=1,
        l2_regularization=0.01)
    adam = torch.optim.Adam(
        params=net_adam.parameters(), amsgrad=False, lr=10**-3, weight_decay=0.01)

    input_ = torch.randn(5, 3, 10, requires_grad=False)

    output_lamb, _ = net_lamb(input_)
    lamb.zero_grad()
    output_lamb.sum().backward()
    lamb.step()

    output_adam, _ = net_adam(input_)
    adam.zero_grad()
    output_adam.sum().backward()
    adam.step()

    # The first step for LAMB should have an Adam update
    for p1, p2 in zip(net_lamb.parameters(), net_adam.parameters()):
        numpy.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy(), rtol=1e-4)


def test_lamb_optimizer__amsgrad():
    net_lamb = torch.nn.LSTM(10, 20, 2)
    net_adam = torch.nn.LSTM(10, 20, 2)

    net_adam.load_state_dict(deepcopy(net_lamb.state_dict()))  # Same weights as `net_lamb`

    # `trust_ratio=1.0` ensures equality with AdamW (similar to Adam with a different weight decay).
    lamb = Lamb(
        params=net_lamb.parameters(), amsgrad=True, lr=10**-3, min_trust_ratio=1, max_trust_ratio=1)
    adam = torch.optim.Adam(params=net_adam.parameters(), lr=10**-3, amsgrad=True)

    input_ = torch.randn(5, 3, 10, requires_grad=False)

    output_lamb, _ = net_lamb(input_)
    lamb.zero_grad()
    output_lamb.sum().backward()
    lamb.step()

    output_adam, _ = net_adam(input_)
    adam.zero_grad()
    output_adam.sum().backward()
    adam.step()

    # The first step for LAMB should have an Adam update
    for p1, p2 in zip(net_lamb.parameters(), net_adam.parameters()):
        numpy.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy(), rtol=1e-4)


def test_lamb_optimizer__adam_w():  # Smoke test
    net_lamb = torch.nn.LSTM(10, 20, 2)
    net_adam = torch.nn.LSTM(10, 20, 2)

    net_adam.load_state_dict(deepcopy(net_lamb.state_dict()))  # Same weights as `net_lamb`

    # `trust_ratio=1.0` ensures equality with AdamW (similar to Adam with a different weight decay).
    # NOTE: Our implementation includes bias correction in `AdamW` while `torch.optim.AdamW`
    # doesn't.
    lamb = Lamb(
        params=net_lamb.parameters(),
        amsgrad=False,
        lr=10**-3,
        min_trust_ratio=1,
        max_trust_ratio=1,
        weight_decay=0.01,
        betas=(0, 0))
    adam = torch.optim.AdamW(
        params=net_adam.parameters(), amsgrad=False, lr=10**-3, weight_decay=0.01, betas=(0, 0))

    input_ = torch.randn(5, 3, 10, requires_grad=False)

    output_lamb, _ = net_lamb(input_)
    lamb.zero_grad()
    output_lamb.sum().backward()
    lamb.step()

    output_adam, _ = net_adam(input_)
    adam.zero_grad()
    output_adam.sum().backward()
    adam.step()

    # The first step for LAMB should have an Adam update
    for p1, p2 in zip(net_lamb.parameters(), net_adam.parameters()):
        numpy.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy(), rtol=1e-4)


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            Optimizer(torch.optim.Adam(params))
        except:
            self.fail('__init__ failed.')

    def test_auto_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            AutoOptimizer(torch.optim.Adam(params), window_size=10)
        except:
            self.fail('__init__ failed.')

    def test_to(self):
        net = torch.nn.GRU(10, 20, 2)
        adam = torch.optim.Adam(net.parameters())

        # Ensure there is some state.
        optim = Optimizer(adam)
        input_ = torch.randn(5, 3, 10)
        output, _ = net(input_)
        output.sum().backward()
        optim.step(max_grad_norm=None, comet_ml=MockCometML())

        optim.to(torch.device('cpu'))

    @mock.patch('torch.nn.utils.clip_grad_norm_')
    def test_step_max_grad_norm(self, mock_clip_grad_norm):
        net = torch.nn.GRU(10, 20, 2)
        adam = torch.optim.Adam(net.parameters())
        optim = Optimizer(adam)
        input_ = torch.randn(5, 3, 10)
        output, _ = net(input_)
        output.sum().backward()
        mock_clip_grad_norm.return_value = 1.0
        optim.step(max_grad_norm=5, comet_ml=MockCometML())
        mock_clip_grad_norm.assert_called_once()

    @mock.patch('torch.nn.utils.clip_grad_norm_')
    def test_auto_step_max_grad_norm(self, mock_clip_grad_norm):
        mock_clip_grad_norm.return_value = 1.0
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        params[0].grad = torch.randn(2, 3, 4)
        optim = AutoOptimizer(torch.optim.Adam(params), window_size=2)
        assert optim.max_grad_norm is None
        optim.step()
        assert optim.max_grad_norm is not None
        old_max_grad_norm = optim.max_grad_norm
        params[0].grad = torch.randn(2, 3, 4)
        optim.step()
        assert old_max_grad_norm != optim.max_grad_norm  # Max grad norm updates
        mock_clip_grad_norm.assert_called_once()  # Max grad norm is only called on the second step

        # Test sliding window stabilizes
        optim.step()
        old_max_grad_norm = optim.max_grad_norm
        optim.step()
        assert old_max_grad_norm == optim.max_grad_norm

    def test_ignore_step(self):
        did_step = False

        def _step(*args, **kwargs):
            nonlocal did_step
            did_step = True

        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        params[0].grad = torch.randn(2, 3, 4)
        adam = torch.optim.Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML())
        assert did_step

        # Test ``inf``
        did_step = False
        params = [torch.nn.Parameter(torch.randn(1))]
        params[0].grad = torch.tensor([math.inf])
        adam = torch.optim.Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML())
        assert not did_step

        # Test ``nan``
        did_step = False
        params[0].grad = torch.tensor([float('nan')])
        adam = torch.optim.Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML())
        assert not did_step
