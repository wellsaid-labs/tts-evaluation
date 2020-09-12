import math

import numpy as np
import pytest
import torch

from tests._utils import assert_almost_equal

import lib


def test_adaptive_gradient_norm_clipping():
    """ Test `AdaptiveGradientNormClipping` clips the gradient norm correctly. """
    clippers = lib.optimizers.AdaptiveGradientNormClipping(10, float('inf'))

    parameters = torch.tensor([1.0, 2.0, 3.0])
    parameters.grad = torch.tensor([1.0, 2.0, 3.0])
    clippers.clip_(parameters)
    assert_almost_equal(parameters.grad, torch.tensor([1.0, 2.0, 3.0]))

    parameters = torch.tensor([5.0, 5.0, 5.0])
    parameters.grad = torch.tensor([5.0, 5.0, 5.0])
    clippers.clip_(parameters)
    assert_almost_equal(parameters.grad, torch.tensor([3.0, 3.0, 3.0]), decimal=4)


def test_adaptive_gradient_norm_clipping__window():
    """ Test `AdaptiveGradientNormClipping` manages the window correctly. """
    clippers = lib.optimizers.AdaptiveGradientNormClipping(3, float('inf'))

    parameters = torch.tensor([4.0])
    parameters.grad = torch.tensor([4.0])
    clippers.clip_(parameters)
    np.testing.assert_almost_equal(clippers.window, [4.0])
    np.testing.assert_almost_equal(clippers.sorted_window, [4.0])
    assert clippers._get_median() == 4.0

    parameters = torch.tensor([3.0])
    parameters.grad = torch.tensor([3.0])
    clippers.clip_(parameters)
    np.testing.assert_almost_equal(clippers.window, [4.0, 3.0])
    np.testing.assert_almost_equal(clippers.sorted_window, [3.0, 4.0])
    assert clippers._get_median() == 3.5

    parameters = torch.tensor([2.0])
    parameters.grad = torch.tensor([2.0])
    clippers.clip_(parameters)
    np.testing.assert_almost_equal(clippers.window, [4.0, 3.0, 2.0])
    np.testing.assert_almost_equal(clippers.sorted_window, [2.0, 3.0, 4.0])
    assert clippers._get_median() == 3.0

    parameters = torch.tensor([1.0])
    parameters.grad = torch.tensor([1.0])
    clippers.clip_(parameters)
    np.testing.assert_almost_equal(clippers.window, [3.0, 2.0, 1.0])
    np.testing.assert_almost_equal(clippers.sorted_window, [1.0, 2.0, 3.0])
    assert clippers._get_median() == 2.0


def test_adaptive_gradient_norm_clipping__large_gradient():
    """ Test `AdaptiveGradientNormClipping` errors given a large gradient. """
    clippers = lib.optimizers.AdaptiveGradientNormClipping(3, float('inf'))

    parameters = torch.tensor([math.nan])
    parameters.grad = torch.tensor([math.nan])
    with pytest.raises(ValueError):
        clippers.clip_(parameters)

    parameters = torch.tensor([math.inf])
    parameters.grad = torch.tensor([math.inf])
    with pytest.raises(ValueError):
        clippers.clip_(parameters)


def test_exponential_moving_parameter_average():
    """ Test `ExponentialMovingParameterAverage`'s bias correction implementation via this video:
    https://pt.coursera.org/lecture/deep-neural-network/www.deeplearning.ai-XjuhD
    """
    values = [1.0, 2.0]
    beta = 0.98
    parameters = [torch.full((2,), values[0])]
    ema = lib.optimizers.ExponentialMovingParameterAverage(parameters, beta=beta)
    assert parameters[0].data[0] == values[0]
    assert parameters[0].data[1] == values[0]

    parameters[0].data = torch.tensor([values[1], values[1]])
    ema.update()
    assert parameters[0].data[0] == values[1]
    assert parameters[0].data[1] == values[1]

    ema.apply()
    expected = ((1 - beta) * beta * values[0] + (1 - beta) * values[1]) / ((1 - beta) +
                                                                           (1 - beta) * beta)
    assert parameters[0].data[0] == expected
    assert parameters[0].data[1] == expected

    ema.restore()
    assert parameters[0].data[0] == values[1]
    assert parameters[0].data[1] == values[1]


def test_exponential_moving_parameter_average__identity():
    """ Test `ExponentialMovingParameterAverage` is an identity function when `beta==0.0`. """
    beta = 0.0
    parameters = [torch.zeros(1)]
    ema = lib.optimizers.ExponentialMovingParameterAverage(parameters, beta=beta)

    parameters[0].data[0] = 1.0
    ema.update()
    with ema:
        assert parameters[0].data[0] == 1.0

    parameters[0].data[0] = 2.0
    ema.update()
    with ema:
        assert parameters[0].data[0] == 2.0


def test_exponential_moving_parameter_average__to():
    """ Test `ExponentialMovingParameterAverage.to` executes. """
    device = torch.device('cpu')
    parameters = [torch.zeros(1)]
    parameters[0].to(device)

    ema = lib.optimizers.ExponentialMovingParameterAverage(parameters, beta=0.98)
    ema = ema.to(device)
    ema.update()
    ema = ema.to(device)


def test_warmup_lr_multiplier_schedule():
    assert lib.optimizers.warmup_lr_multiplier_schedule(1, 10) == 1 / 10
    assert lib.optimizers.warmup_lr_multiplier_schedule(9, 10) == 9 / 10
    assert lib.optimizers.warmup_lr_multiplier_schedule(10, 10) == 1.0
    assert lib.optimizers.warmup_lr_multiplier_schedule(11, 10) == 1.0
    assert lib.optimizers.warmup_lr_multiplier_schedule(20, 10) == 1.0
