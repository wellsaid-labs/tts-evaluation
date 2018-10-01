from bisect import insort
from bisect import bisect_left
from math import floor

import itertools

import torch
import numpy as np
import logging

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


def get_parameter_norm(parameters, norm_type=2):
    """Compute the total norm of the parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Inspired by:
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_

    Args:
        parameters (Iterable[Tensor]): An iterable of Tensors.
        norm_type (float or int, optional): Type of the used p-norm. Can be ``'inf'`` for infinity
            norm.

    Return:
        Total norm of the parameters (viewed as a single vector).
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type).item()
            total_norm += param_norm**norm_type
        total_norm = total_norm**(1. / norm_type)
    return total_norm


class Optimizer(object):
    """ Encapsulates ``torch.optim`` package adding gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): Optimizer object, the parameters to be optimized
            should be given when instantiating the object (e.g. ``torch.optim.SGD(params)``)
    """

    def __init__(self, optim):
        self.optimizer = optim

        # Common functions
        self.zero_grad = self.optimizer.zero_grad
        self.state_dict = self.optimizer.state_dict
        self.load_state_dict = self.optimizer.load_state_dict

    def step(self, tensorboard=None, max_grad_norm=None, eps=10**-3):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            tensorboard (tensorboardX.SummaryWriter, optional): Tensorboard for logging infinite
                gradient.
            max_grad_norm (float, optional): Clip gradient norm to this maximum.
            eps (float, optional): Parameter used to sanity check ``parameter_norm`` equality.

        Returns:
            parameter_norm (float): Total norm of the parameters.
        """
        params = list(
            itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups]))
        parameter_norm = get_parameter_norm(params)
        parameter_norm_inf = get_parameter_norm(params, norm_type=float('inf'))

        if max_grad_norm is not None:
            if tensorboard is not None:
                tensorboard.add_scalar('max_grad_norm/step', max_grad_norm)
            other_parameter_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

            # Both callables should compute the same value
            # TODO: Make this a relative check
            assert abs(parameter_norm - other_parameter_norm) < eps

        # Take a step if norm is finite (e.g. no ``inf`` or ``nan`` values in the gradient)
        if np.isfinite(parameter_norm):
            if tensorboard is not None:
                tensorboard.add_scalar('parameter_norm/step', parameter_norm)
                tensorboard.add_scalar('parameter_inf_norm/step', parameter_norm_inf)
            self.optimizer.step()
        elif tensorboard is not None:
            tensorboard.add_text('event/anomaly', 'Gradient was too large "%s", skipping batch.',
                                 str(parameter_norm))

        return parameter_norm

    def to(self, device):
        """ Move the optimizer state to ``device``. After calling, any parameter specific state in
        the optimizer will be located on ``device``.

        Args:
            device (torch.device)
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param_state = self.optimizer.state[param]
                for k in param_state.keys():
                    if torch.is_tensor(param_state[k]):
                        param_state[k] = param_state[k].to(device)


class AutoOptimizer(Optimizer):
    """ Encapsulates ``torch.optim`` package adding automatic gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): Optimizer object, the parameters to be optimized
            should be given when instantiating the object (e.g. ``torch.optim.SGD(params)``)
        window_size (int): Size of the sliding window used to compute max gradient norm.
    """

    @configurable
    def __init__(self, optim, window_size):
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
            parameter_norm (float): Total norm of the parameters if ``max_grad_norm > 0``;
                otherwise, returns None.
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
