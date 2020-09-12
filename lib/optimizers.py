# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from bisect import bisect_left
from bisect import insort
from math import floor
from types import TracebackType

import typing

from hparams import configurable
from hparams import HParam
from third_party import get_parameter_norm

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AdaptiveGradientNormClipping():
    """ Clip gradient norm based on the median gradient norm.

    Source (On the difficulty of training Recurrent Neural Networks):
        The proposed clipping is simple to implement and computationally efficient, but it does
        however introduce an additional hyper-parameter, namely the threshold. One good heuristic
        for setting this threshold is to look at statistics on the average norm over a sufficiently
        large number of updates. In our experiments we have noticed that for a given task and model
        size, training is not very sensitive to this hyperparameter and the algorithm behaves well
        even for rather small thresholds.

    Source (The Best of Both Worlds: Combining Recent Advances in Neural Machine Translation):
        To further stabilize training, we also use adaptive gradient clipping. We discard a training
        step completely if an anomaly in the gradient norm value is detected, which is usually an
        indication of an imminent gradient explosion. More specifically, we keep track of a moving
        average and a moving standard deviation of the log of the gradient norm values, and we abort
        a step if the norm of the gradient exceeds four standard deviations of the moving average.

    Args:
        window_size: Number of data points used to calculate the median gradient norm.
        norm_type: The "p" in p-norm. This includes `inf` for infinity norm.
    """

    @configurable
    def __init__(self, window_size: int = HParam(), norm_type: float = HParam()):
        super().__init__()
        self.max_norm: typing.Optional[float] = None
        self.norm_type: float = norm_type
        self.sorted_window: typing.List[float] = []
        self.window_size: int = window_size
        self.window: typing.List[float] = []

    def _insert(self, norm: float):
        """ Insert `norm` into `window` and `sorted_window`, and remove the oldest value. """
        if len(self.window) == self.window_size:
            old_value = self.window.pop(0)
            del self.sorted_window[bisect_left(self.sorted_window, old_value)]
        self.window.append(norm)
        insort(self.sorted_window, norm)

    def _get_median(self) -> float:
        """ Get the middle value in `sorted_window`. """
        half = len(self.sorted_window) / 2
        if len(self.sorted_window) % 2 == 1:
            return self.sorted_window[int(floor(half))]
        half = int(half)
        return (self.sorted_window[half] + self.sorted_window[half - 1]) / 2

    def clip_(self, parameters: typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]]):
        """ Clips gradient norm of an iterable of parameters, and update gradient norm history.

        Args:
            parameters: Tensor(s) that will have their gradients normalized.
        """
        if self.max_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters, max_norm=self.max_norm, norm_type=self.norm_type)
        norm = get_parameter_norm(parameters, self.norm_type)
        if not np.isfinite(norm):
            raise ValueError(f"Gradient is not finite: {norm}")
        self._insert(norm)
        self.max_norm = self._get_median()


class ExponentialMovingParameterAverage():
    """ Average the model parameters over time.

    Learn more:
    - On EMA implementation: http://www.programmersought.com/article/28492072406/
    - On EMA effectiveness: https://arxiv.org/abs/1806.04498

    Args:
        parameters: Tensor(s) that'll be tracked over time.
        beta: Beta used to weight the exponential mean.
    """

    @configurable
    def __init__(self, parameters: typing.Iterable[torch.Tensor], beta: float = HParam()):
        self.parameters = list(parameters)
        self.beta = beta
        self.shadow = [param.clone().detach() * (1.0 - self.beta) for param in self.parameters]
        self.backup: typing.List[torch.Tensor] = []
        self.step = 1

    def update(self):
        """ Update the parameter average. """
        for i, param in enumerate(self.parameters):
            self.shadow[i] = (1.0 - self.beta) * param.clone().detach() + self.beta * self.shadow[i]
        self.step += 1

    def apply(self):
        """ Replace the parameters with their averaged counterpart. """
        self.backup = [param.clone().detach() for param in self.parameters]
        for param, shadow in zip(self.parameters, self.shadow):
            # The initial 0.0 average values introduce bias that is corrected, learn more:
            # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
            with torch.no_grad():
                param.copy_(shadow / (1 - self.beta**(self.step)))

    def restore(self):
        """ Restore the parameter values after `self.apply`. """
        for param, backup in zip(self.parameters, self.backup):
            with torch.no_grad():
                param.copy_(backup)
        self.backup = []

    def __enter__(self) -> ExponentialMovingParameterAverage:
        self.apply()
        return self

    def __exit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_value: typing.Optional[BaseException],
        exc_traceback: typing.Optional[TracebackType],
    ):
        self.restore()

    def to(self, device: torch.device) -> ExponentialMovingParameterAverage:
        """ Move the parameters to `device`. """
        # NOTE: `self.parameters` are external and should be moved outside of this function.
        for list_ in [self.shadow, self.backup]:
            for i, param in enumerate(list_):
                list_[i] = param.to(device)
        return self


def warmup_lr_multiplier_schedule(step: int, warmup: int) -> float:
    """ Basic learning rate multiplier schedule.

    Args:
        step: The current step.
        warmup: The number of warmup steps.

    Returns:
        Base learning rate multiplier.
    """
    if step < warmup:
        return step / warmup
    return 1.0
