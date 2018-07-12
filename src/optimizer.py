import itertools

import torch
import numpy as np
import logging

from src.utils.configurable import configurable

logger = logging.getLogger(__name__)


def get_parameter_norm(parameters, norm_type=2):
    """Compute the total 2-norm of the parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Inspired by:
    https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_

    Args:
        parameters (Iterable[Tensor]): An iterable of Tensors.
        norm_type (float or int): Type of the used p-norm. Can be ``'inf'`` for infinity norm.

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
        beta (float, optional): Smoothing parameter for estimating the average gradient norm.
    """

    @configurable
    def __init__(self, optim, beta=0.98):
        self.optimizer = optim

        # Common functions
        self.zero_grad = self.optimizer.zero_grad
        self.state_dict = self.optimizer.state_dict
        self.load_state_dict = self.optimizer.load_state_dict
        self.average_norm = 0
        self.beta = beta
        self.steps = 0

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Returns:
            parameter_norm (float): Total norm of the parameters if ``max_grad_norm > 0``;
                otherwise, returns None.
            max_grad_norm (float): Predicted max grad norm.
        """
        params = itertools.chain.from_iterable(
            [group['params'] for group in self.optimizer.param_groups])
        parameter_norm = get_parameter_norm(params)
        self.average_norm = self.beta * self.average_norm + (1 - self.beta) * parameter_norm
        smoothed_norm = self.average_norm / (1 - self.beta**(self.steps + 1))
        self.steps += 1

        torch.nn.utils.clip_grad_norm_(params, max_norm=smoothed_norm)

        # Take a step if norm is finite (e.g. no ``inf`` or ``nan`` values in the gradient)
        if np.isfinite(parameter_norm):
            self.optimizer.step()
        else:
            logger.warn('Gradient was not finite, skipping batch.')

        return parameter_norm, smoothed_norm

    def to(self, device):
        """ Move the optimizer state to ``device``. After calling, any parameter specific state in
        the optimizer will be located on ``device``.
        """
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                param_state = self.optimizer.state[param]
                for k in param_state.keys():
                    if torch.is_tensor(param_state[k]):
                        param_state[k] = param_state[k].to(device)
