# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from math import isclose

import itertools
import logging
import math
import multiprocessing.pool
import random
import statistics
import time
import typing

import torch
import torch.utils.data

import lib

logger = logging.getLogger(__name__)

_RandomSampleReturnType = typing.TypeVar('_RandomSampleReturnType')


def random_sample(list_: typing.List[_RandomSampleReturnType],
                  sample_size: int) -> typing.List[_RandomSampleReturnType]:
    """ Random sample function that doesn't error if `list_` is smaller than `sample_size`. """
    return random.sample(list_, min(len(list_), sample_size))


def nested_to_flat_dict(dict_: typing.Dict[str, typing.Any],
                        delimitator: str = '.') -> typing.Dict[str, typing.Any]:
    """ Convert nested dictionary to a flat dictionary by concatenating keys with a `delimitator`.

    Args:
        dict_
        delimitator: Delimitator used to join keys.
    """
    return _nested_to_flat_dict(dict_=dict_, delimitator=delimitator, keys=[])


def _nested_to_flat_dict(dict_: typing.Dict[str, typing.Any], delimitator: str,
                         keys: typing.List[str]) -> typing.Dict[str, typing.Any]:
    ret_ = {}
    for key in dict_:
        if isinstance(dict_[key], dict):
            ret_.update(_nested_to_flat_dict(dict_[key], delimitator, keys + [key]))
        else:
            ret_[delimitator.join(keys + [key])] = dict_[key]
    return ret_


def mean(list_: typing.Iterable[float]) -> float:
    """ Mean function that does not return an error if `list_` is empty. """
    list_ = list(list_)
    if len(list_) == 0:
        return math.nan
    return statistics.mean(list_)  # NOTE: `statistics.mean` returns an error for an empty list


_ChunksReturnType = typing.TypeVar('_ChunksReturnType')


def get_chunks(list_: typing.List[_ChunksReturnType],
               n: int) -> typing.Generator[typing.List[_ChunksReturnType], None, None]:
    """ Yield successive `n`-sized chunks from `list_`. """
    for i in range(0, len(list_), n):
        yield list_[i:i + n]


def get_weighted_stdev(tensor: torch.Tensor,
                       dim: int = 0,
                       mask: typing.Optional[torch.Tensor] = None) -> float:
    """ Computed the average weighted standard deviation accross a dimension in `tensor`.

    NOTE:
    - `tensor` must sum up to 1.0 on `dim`.
    - The value of an element in a `tensor` corresponds to it's weight.
    - The index of an element in a `tensor` corresponds to it's position.

    Learn more:
    - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    - https://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean

    Args:
        tensor (torch.FloatTensor [*, dim, *])
        dim: Compute standard deviation along `dim` in `tensor`.
        mask (torch.BoolTensor [*, *])
    """
    # Expects normalized weightes total of 0, 1 to ensure correct variance decisions
    assert all([isclose(value, 1, abs_tol=1e-3) for value in tensor.sum(dim=dim).view(-1).tolist()])

    # Create position matrix where the index is the position and the value is the weight
    indicies = torch.arange(0, tensor.shape[dim], dtype=tensor.dtype, device=tensor.device)
    shape = [1] * len(tensor.shape)
    shape[dim] = tensor.shape[dim]
    indicies = indicies.view(*shape).expand_as(tensor).float()

    weighted_mean = (indicies * tensor).sum(dim=dim) / tensor.sum(dim=dim)
    weighted_variance = ((indicies - weighted_mean.unsqueeze(dim=dim))**2 * tensor).sum(dim=dim)
    weighted_standard_deviation = weighted_variance**0.5

    assert not torch.isnan(weighted_standard_deviation.min()), 'NaN detected.'

    if mask is not None:
        weighted_standard_deviation = weighted_standard_deviation.masked_select(mask)

    return weighted_standard_deviation.mean().item()


""" Flatten a list of lists into a list. """
flatten = lambda l: [item for sublist in l for item in sublist]


def flatten_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """ Apply `flatten_parameters` to `model`. """
    lambda_ = lambda m: m.flatten_parameters() if hasattr(m, 'flatten_parameters') else None
    return model.apply(lambda_)


_IdentityReturnType = typing.TypeVar('_IdentityReturnType')


def identity(x: _IdentityReturnType) -> _IdentityReturnType:
    return x


_AccumulateAndSplitReturnType = typing.TypeVar('_AccumulateAndSplitReturnType')


def accumulate_and_split(
        list_: typing.List[_AccumulateAndSplitReturnType],
        thresholds: typing.List[float],
        get_value=identity
) -> typing.Generator[typing.List[_AccumulateAndSplitReturnType], None, None]:
    """ Split `list_` when the accumulated sum passes a threshold.

    Args:
        list
        thresholds
        get_value: Given a list item, determine the value of the list item.

    Returns:
        Slice(s) of the list.
    """
    totals = list(itertools.accumulate([get_value(i) for i in list_]))
    index = 0
    for threshold in thresholds:
        lambda_ = lambda x: x < threshold + (totals[index - 1] if index > 0 else 0)
        count = len(list(itertools.takewhile(lambda_, totals[index:])))
        yield list_[index:index + count]
        index = index + count
    if index < len(list_):
        yield list_[index:]


def seconds_to_string(seconds: float) -> str:
    """ Rewrite seconds as a string.

    Example:
        >>> seconds_to_string(123)
        '2m 3s 0ms'
        >>> seconds_to_string(123.100)
        '2m 3s 100ms'
        >>> seconds_to_string(86400)
        '1d 0h 0m 0s 0ms'
        >>> seconds_to_string(3600)
        '1h 0m 0s 0ms'
    """
    assert seconds > 0, 'Seconds must be positive.'
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return '%dd %dh %dm %ds %dms' % (days, hours, minutes, seconds, milliseconds)
    elif hours > 0:
        return '%dh %dm %ds %dms' % (hours, minutes, seconds, milliseconds)
    elif minutes > 0:
        return '%dm %ds %dms' % (minutes, seconds, milliseconds)
    elif seconds > 0:
        return '%ds %dms' % (seconds, milliseconds)
    else:
        return '%dms' % (milliseconds)


_LogRuntimeFunction = typing.TypeVar('_LogRuntimeFunction', bound=typing.Callable[..., typing.Any])


def log_runtime(function: _LogRuntimeFunction) -> _LogRuntimeFunction:
    """ Decorator for measuring the execution time of a function. """

    @wraps(function)
    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        elapsed = seconds_to_string(time.time() - start)
        logger.info('`%s` ran for %s', function.__qualname__, elapsed)
        return result

    return typing.cast(_LogRuntimeFunction, decorator)


_SortTogetherItem = typing.TypeVar('_SortTogetherItem')


def sort_together(list_: typing.Iterable[_SortTogetherItem], key: typing.Iterable[typing.Any],
                  **kwargs) -> typing.Iterable[_SortTogetherItem]:
    """ Sort `list_` with `key` iterable.

    Args:
        list_
        key
        **kwargs: Additional keyword arguments passed to `sorted`.

    Returns:
        Sorted `list_`.
    """
    return [item for _, item in sorted(zip(key, list_), key=lambda p: p[0], **kwargs)]


@contextmanager
def Pool(*args, **kwargs) -> typing.Iterator[multiprocessing.pool.Pool]:
    """ Alternative implementation of a `Pool` context manager. The original `multiprocessing.Pool`
    context manager calls `terminate` rather than `close` followed by `join`.

    Furthermore, it defaults to a 'forkserver' pool, a safer alternative to 'fork'.
    """
    # Learn more: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    # Learn more about `forkserver` / `spawn` / `fork`:
    # https://github.com/pytorch/pytorch/issues/2245
    # https://codewithoutrules.com/2018/09/04/python-multiprocessing/
    # https://pythontic.com/multiprocessing/multiprocessing/introduction
    pool = torch.multiprocessing.get_context('forkserver').Pool(*args, **kwargs)
    yield pool
    pool.close()  # Marks the pool as closed.
    pool.join()  # Waits for workers to exit.


def pad_tensor(input_: torch.Tensor,
               pad: typing.Tuple[int, int],
               dim: int = 0,
               **kwargs) -> torch.Tensor:
    """ Pad a tensor dimension.

    Args:
        input_
        pad: The padding to apply.
        dim: The dimension to apply padding.
        **kwargs: Keyword arguments passed onto `torch.nn.functional.pad`.
    """
    padding = [(0, 0) for _ in range(input_.dim())]
    padding[dim] = pad
    # NOTE: `torch.nn.functional.pad` accepts the last dimension first.
    flat: typing.List[int] = flatten(reversed(padding))
    return torch.nn.functional.pad(input_, flat, **kwargs)


def trim_tensors(*args: torch.Tensor, dim: int = 2) -> typing.Iterable[torch.Tensor]:
    """ Trim the edges of each tensor in `args` such that each tensor's dimension `dim` has the same
    size.

    Args:
        *args
        dim: The dimension to trim.
    """
    minimum = min(a.shape[dim] for a in args)
    assert all((a.shape[dim] - minimum) % 2 == 0 for a in args), 'Uneven padding'
    return tuple([a.narrow(dim, (a.shape[dim] - minimum) // 2, minimum) for a in args])


class LSTM(torch.nn.LSTM):
    """ LSTM with a trainable initial hidden state.

    TODO: Add support for `PackedSequence`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_directions = 2 if self.bidirectional else 1
        self.initial_hidden_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size))
        self.initial_cell_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        if hx is None:
            batch_size = input.shape[0] if self.batch_first else input.shape[1]
            hx = (self.initial_hidden_state.expand(-1, batch_size, -1).contiguous(),
                  self.initial_cell_state.expand(-1, batch_size, -1).contiguous())
        return super().forward(input, hx=hx)


class LSTMCell(torch.nn.LSTMCell):
    """ LSTMCell with a trainable initial hidden state.

    TODO: Add support for `PackedSequence`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_hidden_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))
        self.initial_cell_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = (self.initial_hidden_state.expand(input.shape[0], -1).contiguous(),
                  self.initial_cell_state.expand(input.shape[0], -1).contiguous())
        return super().forward(input, hx=hx)


class Average():
    """ Track the average. """

    def __init__(self):
        self.reset()

    def reset(self) -> typing.Optional[float]:
        """ Reset the metric statistics and return the mean. """
        average = self.total_value / self.total_count if (hasattr(self, 'total_value') and hasattr(
            self, 'total_count') and self.total_count > 0) else None
        self.last_update_value: typing.Optional[float] = None
        self.total_value: float = 0.0
        self.total_count: float = 0.0
        return average

    def update(self,
               value: typing.Union[torch.Tensor, float],
               count: typing.Union[torch.Tensor, int] = 1) -> Average:
        """ Update the mean.

        Args:
            value
            count: Number of times to add value / frequency of value.
        """
        value = typing.cast(float, value.item()) if isinstance(value, torch.Tensor) else value
        count = typing.cast(int, count.item()) if isinstance(count, torch.Tensor) else count
        assert count > 0, f"Count ({count}) must be positive."
        self.total_value += value * count
        self.total_count += count
        self.last_update_value = value
        return self


class DistributedAverage(Average):
    """ Track the average in a distributed environment. """

    def reset(self) -> typing.Optional[float]:
        super().reset()
        average = self.post_sync_total_value / self.post_sync_total_count if (
            hasattr(self, 'post_sync_total_value') and hasattr(self, 'post_sync_total_value') and
            self.post_sync_total_count > 0) else None
        self.post_sync_total_value: float = 0.0
        self.post_sync_total_count: float = 0.0
        return average

    def sync(self) -> DistributedAverage:
        """ Synchronize measurements accross multiple processes. """
        last_post_sync_total_value = self.post_sync_total_value
        last_post_sync_total_count = self.post_sync_total_count
        torch_ = torch.cuda if torch.cuda.is_available() else torch
        packed = torch_.FloatTensor([self.total_value, self.total_count])  # type: ignore
        torch.distributed.reduce(packed, dst=lib.distributed.get_master_rank())
        self.post_sync_total_value, self.post_sync_total_count = tuple(packed.tolist())
        self.last_update_value = (self.post_sync_total_value - last_post_sync_total_value) / (
            self.post_sync_total_count - last_post_sync_total_count)
        return self
