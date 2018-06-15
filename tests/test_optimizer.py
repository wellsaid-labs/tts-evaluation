import unittest

import torch
import mock

from src.optimizer import Optimizer


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            Optimizer(torch.optim.Adam(params), max_grad_norm=0.0)
        except:
            self.fail("__init__ failed.")

    @mock.patch("torch.nn.utils.clip_grad_norm_")
    def test_step(self, mock_clip_grad_norm):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        optim = Optimizer(torch.optim.Adam(params), max_grad_norm=5)
        optim.step()
        mock_clip_grad_norm.assert_called_once()
