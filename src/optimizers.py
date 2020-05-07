from bisect import insort
from bisect import bisect_left
from math import floor

import itertools
import math

from hparams import configurable
from hparams import HParam
from third_party import get_parameter_norm

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class Optimizer(object):
    """ Encapsulates ``torch.optim`` package adding additional functionality.

    Args:
        optim (torch.optim.Optimizer): Optimizer object. Note the parameters to be optimized
          should be given when instantiating ``optim`` (e.g. ``torch.optim.SGD(params)``)
    """

    def __init__(self, optim):
        self.optimizer = optim

        # Common functions
        self.zero_grad = self.optimizer.zero_grad
        self.state_dict = self.optimizer.state_dict
        self.load_state_dict = self.optimizer.load_state_dict

    def step(self, comet_ml=None, max_grad_norm=None, skip_batch=False):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            comet_ml (comet_ml.Experiment, optional): Visualization library.
            max_grad_norm (float, optional): Clip gradient norm to this maximum.
            skip_batch (bool, optional): If `True`, this skips none finite batches; otherwise, this
                raises an error.

        Returns:
            parameter_norm (float): Total norm of the parameters.
        """
        params = list(
            itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups]))
        parameter_norm = get_parameter_norm(params)
        parameter_norm_inf = get_parameter_norm(params, norm_type=math.inf)

        if max_grad_norm is not None:
            if comet_ml is not None:
                comet_ml.log_metric('step/grad_norm/clip_max', max_grad_norm)
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

        # Take a step if norm is finite (e.g. no ``inf`` or ``nan`` values in the gradient)
        if np.isfinite(parameter_norm):
            if comet_ml is not None:
                comet_ml.log_metric('step/grad_norm/two', parameter_norm)
                comet_ml.log_metric('step/grad_norm/infinity', parameter_norm_inf)
                for i, param_group in enumerate(self.optimizer.param_groups):
                    comet_ml.log_metric('step/parameters_%d/lr' % i, param_group['lr'])
            self.optimizer.step()
        elif comet_ml is not None:
            if skip_batch:
                logger.warning('Gradient was too large "%s", skipping batch.', str(parameter_norm))
            else:
                raise ValueError('Gradient was too large "%s".' % str(parameter_norm))

        return parameter_norm

    def to(self, device):
        """ Move the optimizer state to ``device``.

        Side effects:
            - Any parameter specific state in the optimizer will be located on ``device``.

        Args:
            device (torch.device)
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param_state = self.optimizer.state[param]
                for k in param_state.keys():
                    if torch.is_tensor(param_state[k]):
                        param_state[k] = param_state[k].to(device)

        return self


# TODO: Consider adding a complementary function to ``torch.nn.utils.clip_grad_norm_`` that auto
# sets the grad norm cutoff.


class AutoOptimizer(Optimizer):
    """ Encapsulates ``torch.optim`` package adding additional functionality.

    TODO: The `state_dict` function does not save `window` and `sorted_window`.

    Args:
        optim (torch.optim.Optimizer): Optimizer object. Note the parameters to be optimized
          should be given when instantiating ``optim`` (e.g. ``torch.optim.SGD(params)``)
        window_size (int): Size of the sliding window used to compute max gradient norm.
    """

    @configurable
    def __init__(self, optim, window_size=HParam()):
        super().__init__(optim)
        self.window_size = window_size
        self.window = []
        self.sorted_window = []
        self.max_grad_norm = None

    def step(self, *args, **kwargs):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            *args (list, optional): Arguments to pass on to ``Optimizer.step``
            **kwargs (dict, optional): Keyword arguments to pass on to ``Optimizer.step``

        Returns:
            parameter_norm (float): Total norm of the parameters,
        """
        parameter_norm = super().step(*args, max_grad_norm=self.max_grad_norm, **kwargs)

        if np.isfinite(parameter_norm):
            if len(self.window) == self.window_size:
                old_value = self.window.pop(0)
                del self.sorted_window[bisect_left(self.sorted_window, old_value)]

            self.window.append(parameter_norm)
            insort(self.sorted_window, parameter_norm)

            half = len(self.sorted_window) / 2
            if len(self.sorted_window) % 2 == 1:
                self.max_grad_norm = self.sorted_window[int(floor(half))]
            else:
                half = int(half)
                self.max_grad_norm = (self.sorted_window[half] + self.sorted_window[half - 1]) / 2

        return parameter_norm


class ExponentialMovingParameterAverage():
    """ Average the model parameters over time.

    Inspired by: http://www.programmersought.com/article/28492072406/

    Learn more about EMA, here: https://arxiv.org/abs/1806.04498

    Args:
        model (torch.nn.Module): The model w/ parameters to average.
        beta (float): Beta used to weight the exponential mean.
    """

    @configurable
    def __init__(self, parameters, beta=HParam()):
        self.parameters = list(parameters)
        self.beta = beta
        self.shadow = [param.clone().detach() * (1.0 - self.beta) for param in self.parameters]
        self.backup = []
        self.step = 1

    def update(self):
        """ Update the parameter average.
        """
        for i, param in enumerate(self.parameters):
            self.shadow[i] = (1.0 - self.beta) * param.clone().detach() + self.beta * self.shadow[i]
        self.step += 1

    def apply_shadow(self):
        """ Replace the model with it's averaged parameters.

        TODO: Investigate implementing this as a context manager.
        """
        self.backup = [param.clone().detach() for param in self.parameters]
        for param, shadow in zip(self.parameters, self.shadow):
            # The initial 0.0 average values introduce bias that is corrected, learn more:
            # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
            with torch.no_grad():
                param.copy_(shadow / (1 - self.beta**(self.step)))

    def restore(self):
        """ Restore the model's old parameters.
        """
        for param, backup in zip(self.parameters, self.backup):
            with torch.no_grad():
                param.copy_(backup)
        self.backup = []

    def to(self, device):
        """ Move the state to ``device``.

        Args:
            device (torch.device)
        """
        for list_ in [self.parameters, self.shadow, self.backup]:
            for i, param in enumerate(list_):
                list_[i] = param.to(device)
        return self
