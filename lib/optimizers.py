# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import logging
import math
import typing
from bisect import bisect_left, insort
from math import floor
from types import TracebackType

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

logger = logging.getLogger(__name__)


class AdaptiveGradientNormClipper:
    """Clip gradient norm based on the median gradient norm.

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
        parameters
        window_size: Number of data points used to calculate the median gradient norm.
        norm_type: The "p" in p-norm. This includes `inf` for infinity norm.
    """

    def __init__(
        self,
        parameters: typing.Iterable[torch.Tensor],
        window_size: int,
        norm_type: float,
    ):
        super().__init__()
        self.max_norm = math.inf
        self.norm_type: float = norm_type
        self.sorted_window: typing.List[float] = []
        self.window_size: int = window_size
        self.window: typing.List[float] = []
        # NOTE: `self.parameters` is a reference, and it shouldn't be broken.
        self.parameters = parameters

    def _insert(self, norm: float):
        """Insert `norm` into `window` and `sorted_window`, and remove the oldest value."""
        if len(self.window) == self.window_size:
            old_value = self.window.pop(0)
            del self.sorted_window[bisect_left(self.sorted_window, old_value)]
        self.window.append(norm)
        insort(self.sorted_window, norm)

    def _get_median(self) -> float:
        """Get the middle value in `sorted_window`."""
        half = len(self.sorted_window) / 2
        if len(self.sorted_window) % 2 == 1:
            return self.sorted_window[int(floor(half))]
        half = int(half)
        return (self.sorted_window[half] + self.sorted_window[half - 1]) / 2

    def clip(self) -> float:
        """Clips gradient norm of an iterable of `self.parameters`, and update gradient norm
        history."""
        assert all([p.grad is not None for p in self.parameters]), "`None` gradients found."
        norm = clip_grad_norm_(self.parameters, self.max_norm, self.norm_type)
        if not torch.isfinite(norm):  # type: ignore
            raise ValueError(f"Gradient is not finite: {norm}")
        item = typing.cast(float, norm.item())
        self._insert(item)
        self.max_norm = self._get_median()
        return item


class ExponentialMovingParameterAverage:
    """Average the model parameters over time.

    Learn more:
    - On EMA implementation: http://www.programmersought.com/article/28492072406/
    - On EMA effectiveness: https://arxiv.org/abs/1806.04498

    NOTE: A `to` function is not built in for consistency with `torch.optim`. The expectation
    is that `ExponentialMovingParameterAverage` is instantiated on or `lib.environment.load` onto
    the device it'll be used on. Learn more about how PyTorch handles this:
    https://github.com/pytorch/pytorch/pull/3658
    https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py#L132

    Args:
        parameters: Tensor(s) that'll be tracked over time.
        beta: Beta used to weight the exponential mean.
    """

    def __init__(self, parameters: typing.Iterable[torch.Tensor], beta: float):
        # NOTE: `self.parameters` is a reference, and it shouldn't be broken.
        self.parameters = list(parameters)
        self.beta = beta
        self.shadow = [param.clone().detach() * (1.0 - self.beta) for param in self.parameters]
        self.backup: typing.List[torch.Tensor] = []
        self.step = 1

    def update(self):
        """Update the parameter average."""
        for i, param in enumerate(self.parameters):
            self.shadow[i] = (1.0 - self.beta) * param.clone().detach() + self.beta * self.shadow[i]
        self.step += 1

    def apply(self):
        """Replace the parameters with their averaged counterpart."""
        self.backup = [param.clone().detach() for param in self.parameters]
        for param, shadow in zip(self.parameters, self.shadow):
            # The initial 0.0 average values introduce bias that is corrected, learn more:
            # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
            with torch.no_grad():
                param.copy_(shadow / (1 - self.beta ** (self.step)))

    def restore(self):
        """Restore the parameter values after `self.apply`."""
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


def warmup_lr_multiplier_schedule(step: int, warmup: int) -> float:
    """Basic learning rate multiplier schedule.

    Args:
        step: The current step.
        warmup: The number of warmup steps.

    Returns:
        Base learning rate multiplier.
    """
    if step < warmup:
        return step / warmup
    return 1.0
