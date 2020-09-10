from unittest import mock

import math
import unittest

from torch.optim import Adam

import pytest
import torch

from src.environment import TEMP_PATH
from src.optimizers import AutoOptimizer
from src.optimizers import ExponentialMovingParameterAverage
from src.optimizers import Optimizer
from src.utils import Checkpoint
from tests._utils import MockCometML


class _TestExponentialMovingParameterAverageCheckpointModule(torch.nn.Module):

    def __init__(self, value):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.full((1,), value))


def test_exponential_moving_parameter_average__checkpoint():
    """ Test to ensure that the state is saved correctly during checkpointing. """

    values = [1.0, 2.0]

    module = _TestExponentialMovingParameterAverageCheckpointModule(values[0])
    exponential_moving_parameter_average = ExponentialMovingParameterAverage(
        module.parameters(), beta=0.98)

    checkpoint = Checkpoint(
        directory=TEMP_PATH,
        step=0,
        model=module,
        exponential_moving_parameter_average=exponential_moving_parameter_average)
    checkpoint_path = checkpoint.save()

    del checkpoint

    checkpoint = Checkpoint.from_path(checkpoint_path)

    del exponential_moving_parameter_average
    del module

    exponential_moving_parameter_average = checkpoint.exponential_moving_parameter_average
    module = checkpoint.model

    # Ensure that `apply_shadow` is set to `values[0]`
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == values[0]
    exponential_moving_parameter_average.restore()

    # Ensure that `update` / `apply_shadow` responds to the parameter update from the model loaded
    # from disk
    module.parameter.data = torch.tensor([values[1]])
    exponential_moving_parameter_average.update()
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == (0.0196 * values[0] + 0.02 * values[1]) / 0.0396

    # Ensure that `restore` is able to restore the model loaded from disk
    exponential_moving_parameter_average.restore()
    assert module.parameter.data[0] == values[1]


def test_exponential_moving_parameter_average__identity():

    class Module(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))

    module = Module()
    exponential_moving_parameter_average = ExponentialMovingParameterAverage(
        module.parameters(), beta=0)

    module.parameter.data[0] = 1.0
    exponential_moving_parameter_average.update()
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == 1.0
    exponential_moving_parameter_average.restore()

    module.parameter.data[0] = 2.0
    exponential_moving_parameter_average.update()
    exponential_moving_parameter_average.apply_shadow()
    assert module.parameter.data[0] == 2.0
    exponential_moving_parameter_average.restore()


def test_exponential_moving_parameter_average():
    """ Test bias correction implementation via this video:
    https://pt.coursera.org/lecture/deep-neural-network/www.deeplearning.ai-XjuhD
    """
    values = [1.0, 2.0]

    class Module(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.full((2,), values[0]))

    module = Module()
    exponential_moving_parameter_average = ExponentialMovingParameterAverage(
        module.parameters(), beta=0.98)

    assert module.parameter.data[0] == values[0]
    assert module.parameter.data[1] == values[0]

    module.parameter.data = torch.tensor([values[1], values[1]])

    exponential_moving_parameter_average.update()

    assert module.parameter.data[0] == values[1]
    assert module.parameter.data[1] == values[1]

    exponential_moving_parameter_average.apply_shadow()

    assert module.parameter.data[0] == (0.0196 * values[0] + 0.02 * values[1]) / 0.0396
    assert module.parameter.data[1] == (0.0196 * values[0] + 0.02 * values[1]) / 0.0396

    exponential_moving_parameter_average.restore()

    assert module.parameter.data[0] == values[1]
    assert module.parameter.data[1] == values[1]


class TestOptimizer(unittest.TestCase):

    def test_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            Optimizer(Adam(params))
        except:
            self.fail('__init__ failed.')

    def test_auto_init(self):
        params = [torch.nn.Parameter(torch.randn(2, 3, 4))]
        try:
            AutoOptimizer(Adam(params), window_size=10)
        except:
            self.fail('__init__ failed.')

    def test_to(self):
        net = torch.nn.GRU(10, 20, 2)
        adam = Adam(net.parameters())

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
        adam = Adam(net.parameters())
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
        optim = AutoOptimizer(Adam(params), window_size=2)
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
        adam = Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML())
        assert did_step

        # Test ``inf``
        did_step = False
        params = [torch.nn.Parameter(torch.randn(1))]
        params[0].grad = torch.tensor([math.inf])
        adam = Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML(), skip_batch=True)
        assert not did_step

        # Test ``nan``
        did_step = False
        params[0].grad = torch.tensor([float('nan')])
        adam = Adam(params)
        adam.step = _step
        optim = Optimizer(adam)
        optim.step(comet_ml=MockCometML(), skip_batch=True)
        assert not did_step

        with pytest.raises(ValueError):
            # Test ``nan``
            did_step = False
            params[0].grad = torch.tensor([float('nan')])
            adam = Adam(params)
            adam.step = _step
            optim = Optimizer(adam)
            optim.step(comet_ml=MockCometML(), skip_batch=False)
            assert not did_step

        with pytest.raises(ValueError):
            # Test ``inf``
            did_step = False
            params = [torch.nn.Parameter(torch.randn(1))]
            params[0].grad = torch.tensor([math.inf])
            adam = Adam(params)
            adam.step = _step
            optim = Optimizer(adam)
            optim.step(comet_ml=MockCometML(), skip_batch=False)
