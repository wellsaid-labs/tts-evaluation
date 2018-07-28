import itertools

import torch
import numpy as np
import logging

from src.utils.configurable import configurable
from src.utils import ExponentiallyWeightedMovingAverage

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
    """

    def __init__(self, optim):
        self.optimizer = optim
        self.max_grad_norm = max_grad_norm

        # Common functions
        self.zero_grad = self.optimizer.zero_grad
        self.state_dict = self.optimizer.state_dict
        self.load_state_dict = self.optimizer.load_state_dict

    def step(self, tensorboard=None, max_grad_norm=None):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            tensorboard (tensorboardX.SummaryWriter): Tensorboard for logging infinite gradient.
            max_grad_norm (float, optional): Clip gradient norm to this maximum.

        Returns:
            parameter_norm (float): Total norm of the parameters.
        """
        params = itertools.chain.from_iterable(
            [group['params'] for group in self.optimizer.param_groups])
        parameter_norm = get_parameter_norm(params)

        if max_grad_norm is not None:
            if tensorboard is not None:
                tensorboard.add_scalar('max_grad_norm/step', max_grad_norm)
            torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)

        # Take a step if norm is finite (e.g. no ``inf`` or ``nan`` values in the gradient)
        if np.isfinite(parameter_norm):
            if tensorboard is not None:
                tensorboard.add_scalar('parameter_norm/step', parameter_norm)
            self.optimizer.step()
        elif tensorboard is not None:
            tensorboard.add_text('event/anomaly', 'Gradient was too large "%s", skipping batch.',
                                 str(parameter_norm))

        return parameter_norm

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


class AutoOptimizer(Optimizer):
    """ Encapsulates ``torch.optim`` package adding automatic gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): Optimizer object, the parameters to be optimized
            should be given when instantiating the object (e.g. ``torch.optim.SGD(params)``)
        beta (float, optional): Smoothing parameter for estimating the average gradient norm.
    """

    @configurable
    def __init__(self, optim, beta=0.99):
        super().__init__(optim)
        self.max_grad_norm = None
        self.stats = ExponentiallyWeightedMovingAverage(beta=beta)

    def step(self, tensorboard=None):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            tensorboard (tensorboardX.SummaryWriter): Tensorboard for logging infinite gradient.

        Returns:
            parameter_norm (float): Total norm of the parameters if ``max_grad_norm > 0``;
                otherwise, returns None.
            max_grad_norm (float): Predicted max grad norm.
        """
        parameter_norm = super().step(tensorboard=tensorboard, max_grad_norm=self.max_grad_norm)

        if np.isfinite(parameter_norm):
            # Update max gradient norm to the average parameter norm
            self.max_grad_norm, _ = self.stats.step(parameter_norm)

        return parameter_norm, self.max_grad_norm
