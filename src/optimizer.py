import itertools

import torch

from src.utils.configurable import configurable


class Optimizer(object):
    """ Encapsulates ``torch.optim`` package adding gradient norm clipping.

    Args:
        optim (torch.optim.Optimizer): Optimizer object, the parameters to be optimized
            should be given when instantiating the object (e.g. ``torch.optim.SGD(params)``)
        max_grad_norm (float, optional): Value used for gradient norm clipping, set None to disable.
    """

    @configurable
    def __init__(self, optim, max_grad_norm=None):
        self.optimizer = optim
        self.max_grad_norm = max_grad_norm

        # Common functions
        self.zero_grad = self.optimizer.zero_grad
        self.state_dict = self.optimizer.state_dict
        self.load_state_dict = self.optimizer.load_state_dict

    def step(self):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Returns:
            parameter_norm (float): Total norm of the parameters if ``max_grad_norm > 0``;
                otherwise, returns None.
        """
        parameter_norm = None
        if self.max_grad_norm is not None:
            params = itertools.chain.from_iterable(
                [group['params'] for group in self.optimizer.param_groups])
            parameter_norm = torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
        self.optimizer.step()
        return parameter_norm
