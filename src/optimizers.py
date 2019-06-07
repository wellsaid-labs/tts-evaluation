from bisect import insort
from bisect import bisect_left
from math import floor

import itertools
import math

from third_party import get_parameter_norm

import logging
import numpy as np
import torch

from src.hparams import configurable
from src.hparams import ConfiguredArg

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

    def step(self, comet_ml=None, max_grad_norm=None):
        """ Performs a single optimization step, including gradient norm clipping if necessary.

        Args:
            comet_ml (comet_ml.Experiment, optional): Visualization library.
            max_grad_norm (float, optional): Clip gradient norm to this maximum.

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
            logger.warning('Gradient was too large "%s", skipping batch.', str(parameter_norm))

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


# TODO: Consider adding a complementary function to ``torch.nn.utils.clip_grad_norm_`` that auto
# sets the grad norm cutoff.


class AutoOptimizer(Optimizer):
    """ Encapsulates ``torch.optim`` package adding additional functionality.

    Args:
        optim (torch.optim.Optimizer): Optimizer object. Note the parameters to be optimized
          should be given when instantiating ``optim`` (e.g. ``torch.optim.SGD(params)``)
        window_size (int): Size of the sliding window used to compute max gradient norm.
    """

    @configurable
    def __init__(self, optim, window_size=ConfiguredArg()):
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


class Lamb(torch.optim.Optimizer):
    r"""Implements Lamb algorithm.

    It was proposed in `Reducing BERT Pre-Training Time from 3 Days to 76 Minutes`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay as proposed in
            `Decoupled Weight Decay Regularization` (default: 0)
        max_trust_ratio (float, optional): the maximum trust ratio per layer (default: 10)
        min_trust_ratio (float, optional): the minimum trust ratio per layer (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)


    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    .. _Reducing BERT Pre-Training Time from 3 Days to 76 Minutes:
        https://arxiv.org/abs/1904.00962
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    """

    @configurable
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 max_trust_ratio=10,
                 min_trust_ratio=0,
                 amsgrad=False):
        if not 0.0 <= max_trust_ratio:
            raise ValueError("Invalid maximum trust ratio: {}".format(max_trust_ratio))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            max_trust_ratio=max_trust_ratio,
            min_trust_ratio=min_trust_ratio)
        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1**state['step']
                bias_correction2 = 1 - beta2**state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # References:
                # https://github.com/pytorch/pytorch/issues/18414
                # https://github.com/NVIDIA/apex/blob/d74fda260c403f775817470d87f810f816f3d615/apex/parallel/LARC.py
                # https://github.com/noahgolmant/pytorch-lars/blob/master/lars.py
                # https://github.com/cybertronai/pytorch-lamb
                # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/opt/python/training/lars_optimizer.py
                adam_update = (
                    exp_avg / denom) + group['weight_decay'] * p.data  # AdamW implementation
                r_1 = p.data.norm(2)
                r_2 = adam_update.norm(2)
                trust_ratio = 1.0 if r_1 == 0 or r_2 == 0 else r_1 / r_2
                trust_ratio = max(
                    min(trust_ratio, group['max_trust_ratio']), group['min_trust_ratio'])
                p.data.add_(-step_size * trust_ratio, adam_update)

        return loss
