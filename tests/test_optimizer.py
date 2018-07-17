import unittest

import torch
import mock

from src.optimizer import Optimizer
from src.optimizer import AutoOptimizer


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            Optimizer(torch.optim.Adam(params))
        except:
            self.fail("__init__ failed.")

    def test_auto_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            AutoOptimizer(torch.optim.Adam(params))
        except:
            self.fail("__init__ failed.")

    def test_to(self):
        net = torch.nn.GRU(10, 20, 2)
        adam = torch.optim.Adam(net.parameters())
        optim = Optimizer(adam)
        input_ = torch.randn(5, 3, 10)
        output, _ = net(input_)
        output.sum().backward()
        optim.step(max_grad_norm=None)  # Set state

        optim.to(torch.device('cpu'))

    @mock.patch("torch.nn.utils.clip_grad_norm_")
    def test_step_max_grad_norm(self, mock_clip_grad_norm):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        optim = Optimizer(torch.optim.Adam(params))
        optim.step(max_grad_norm=5)
        mock_clip_grad_norm.assert_called_once()

    @mock.patch("torch.nn.utils.clip_grad_norm_")
    def test_auto_step_max_grad_norm(self, mock_clip_grad_norm):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        params[0].grad = torch.randn(2, 3, 4)
        optim = AutoOptimizer(torch.optim.Adam(params))
        assert optim.max_grad_norm is None
        optim.step()
        assert optim.max_grad_norm is not None
        old_max_grad_norm = optim.max_grad_norm
        optim.step()
        assert old_max_grad_norm != optim.max_grad_norm  # Max grad norm updates
        mock_clip_grad_norm.assert_called_once()  # Max grad norm is only called on the second step

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
        optim.step()
        assert did_step

        # Test ``inf``
        did_step = False
        params = [torch.nn.Parameter(torch.randn(1))]
        params[0].grad = torch.tensor([float('inf')])
        adam = torch.optim.Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        assert not did_step

        # Test ``nan``
        did_step = False
        params[0].grad = torch.tensor([float('nan')])
        adam = torch.optim.Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        assert not did_step
