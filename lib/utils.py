# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import dataclasses
import functools
import itertools
import logging
import math
import multiprocessing.pool
import random
import statistics
import time
import typing
from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
import torch
import torch.multiprocessing
import torch.nn
import torch.nn.functional
import torch.utils.data

logger = logging.getLogger(__name__)


def round_(x: float, bucket_size: float) -> float:
    """Bin `x` into buckets."""
    return bucket_size * round(x / bucket_size)


_RandomSampleReturnType = typing.TypeVar("_RandomSampleReturnType")


def random_sample(
    list_: typing.List[_RandomSampleReturnType], sample_size: int
) -> typing.List[_RandomSampleReturnType]:
    """ Random sample function that doesn't error if `list_` is smaller than `sample_size`. """
    return random.sample(list_, min(len(list_), sample_size))


def mean(list_: typing.Iterable[float]) -> float:
    """ Mean function that does not return an error if `list_` is empty. """
    list_ = list(list_)
    if len(list_) == 0:
        return math.nan
    return statistics.mean(list_)  # NOTE: `statistics.mean` returns an error for an empty list


_ChunksReturnType = typing.TypeVar("_ChunksReturnType")


def get_chunks(
    list_: typing.List[_ChunksReturnType], n: int
) -> typing.Iterator[typing.List[_ChunksReturnType]]:
    """ Yield successive `n`-sized chunks from `list_`. """
    for i in range(0, len(list_), n):
        yield list_[i : i + n]


def get_weighted_std(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Computed the weighted standard deviation accross a dimension in `tensor`.

    NOTE:
    - `tensor` must sum up to 1.0 on `dim`.
    - The value of an element in a `tensor` corresponds to it's weight.
    - The index of an element in a `tensor` corresponds to it's position.

    Learn more:
    - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    - https://mathoverflow.net/questions/11803/unbiased-estimate-of-the-variance-of-a-weighted-mean
    - https://www.rapidtables.com/calc/math/standard-deviation-calculator.html

    Args:
        tensor (torch.FloatTensor [*, dim, *])
        dim: Compute standard deviation along `dim` in `tensor`.
    """
    # Expects normalized weightes total of 0, 1 to ensure correct variance decisions
    assert all(
        math.isclose(value, 1, abs_tol=1e-3) for value in tensor.sum(dim=dim).view(-1).tolist()
    )

    # Create position matrix where the index is the position and the value is the weight
    indicies = torch.arange(0, tensor.shape[dim], dtype=tensor.dtype, device=tensor.device)
    shape = [1] * len(tensor.shape)
    shape[dim] = tensor.shape[dim]
    indicies = indicies.view(*shape).expand_as(tensor).float()

    weighted_mean = (indicies * tensor).sum(dim=dim) / tensor.sum(dim=dim)
    weighted_variance = ((indicies - weighted_mean.unsqueeze(dim=dim)) ** 2 * tensor).sum(dim=dim)
    weighted_standard_deviation = weighted_variance ** 0.5

    numel = weighted_standard_deviation.numel()
    assert numel == 0 or not torch.isnan(weighted_standard_deviation.min()), "NaN detected."

    return weighted_standard_deviation


# NOTE: Due the issues around recursive typing, we've included a `flatten_2d` and
# `flatten_3d` which don't have recursive typing.
# NOTE: Learn more about this typing:
# https://github.com/microsoft/pyright/issues/1147
_FlattenReturnType = typing.TypeVar("_FlattenReturnType")
_FlattenInputType = typing.Union[
    _FlattenReturnType, typing.Sequence[typing.Union[_FlattenReturnType, "_FlattenInputType"]]
]


def flatten_2d(
    l: typing.Iterable[typing.Iterable[_FlattenReturnType]],
) -> typing.List[_FlattenReturnType]:
    """Flatten a 2d list into a 1d list.

    Learn more: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    """
    return [item for sublist in l for item in sublist]


def flatten_3d(
    l: typing.Iterable[typing.Iterable[typing.Iterable[_FlattenReturnType]]],
) -> typing.List[_FlattenReturnType]:
    """Flatten a 3d list into a 1d list."""
    return [item for sublist in l for subsublist in sublist for item in subsublist]


def flatten(l: _FlattenInputType) -> typing.List[_FlattenReturnType]:
    """Flatten a list of lists into a list."""
    if isinstance(l, list):
        return sum(map(flatten, l), [])
    return [l]


_ListToTupleVariable = typing.TypeVar("_ListToTupleVariable")
_ListToTupleInputType = typing.Union[
    _ListToTupleVariable,
    typing.Sequence[typing.Union[_ListToTupleVariable, "_ListToTupleInputType"]],
]


def list_to_tuple(l: _ListToTupleInputType) -> _ListToTupleInputType:
    """Turn a list of lists into a tuple of tuples."""
    if isinstance(l, list):
        return tuple(map(list_to_tuple, l))
    return l


_TupleToListVariable = typing.TypeVar("_TupleToListVariable")
_TupleToListInputType = typing.Union[
    _TupleToListVariable,
    typing.Sequence[typing.Union[_TupleToListVariable, "_TupleToListInputType"]],
]


def tuple_to_list(t: _TupleToListInputType) -> _TupleToListInputType:
    """Turn a tuple of tuples into a list of lists."""
    if isinstance(t, tuple):
        return list(map(tuple_to_list, t))
    return t


def flatten_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """ Apply `flatten_parameters` to `model`. """
    lambda_ = lambda m: m.flatten_parameters() if hasattr(m, "flatten_parameters") else None
    return model.apply(lambda_)


_IdentityReturnType = typing.TypeVar("_IdentityReturnType")


def identity(x: _IdentityReturnType) -> _IdentityReturnType:
    return x


_SplitReturnType = typing.TypeVar("_SplitReturnType")


def split(
    list_: typing.List[_SplitReturnType], splits: typing.List[float], value=identity
) -> typing.Iterator[typing.List[_SplitReturnType]]:
    """Split `list_` when the accumulated sum passes a threshold.

    Args:
        ...
        value: Given a list item, determine the value of the list item.

    Returns: Slice(s) of the list.
    """
    totals = list(itertools.accumulate([value(i) for i in list_]))
    index = 0
    for split in splits:
        lambda_ = lambda x: x <= split + (totals[index - 1] if index > 0 else 0)
        count = len(list(itertools.takewhile(lambda_, totals[index:])))
        yield list_[index : index + count]
        index = index + count
    if index < len(list_):
        yield list_[index:]


def seconds_to_str(seconds: float) -> str:
    """Rewrite seconds as a string.

    Example:
        >>> seconds_to_str(123)
        '2m 3s 0ms'
        >>> seconds_to_str(123.100)
        '2m 3s 100ms'
        >>> seconds_to_str(86400)
        '1d 0h 0m 0s 0ms'
        >>> seconds_to_str(3600)
        '1h 0m 0s 0ms'
    """
    assert seconds >= 0, "Seconds must be positive."
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return "%dd %dh %dm %ds %dms" % (days, hours, minutes, seconds, milliseconds)
    elif hours > 0:
        return "%dh %dm %ds %dms" % (hours, minutes, seconds, milliseconds)
    elif minutes > 0:
        return "%dm %ds %dms" % (minutes, seconds, milliseconds)
    elif seconds > 0:
        return "%ds %dms" % (seconds, milliseconds)
    else:
        return "%dms" % (milliseconds)


_LogRuntimeFunction = typing.TypeVar("_LogRuntimeFunction", bound=typing.Callable[..., typing.Any])


def log_runtime(function: _LogRuntimeFunction) -> _LogRuntimeFunction:
    """ Decorator for measuring the execution time of a function. """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        elapsed = seconds_to_str(time.time() - start)
        logger.info("`%s` ran for %s", function.__qualname__, elapsed)
        return result

    return typing.cast(_LogRuntimeFunction, decorator)


_SortTogetherItem = typing.TypeVar("_SortTogetherItem")


def sort_together(
    list_: typing.Iterable[_SortTogetherItem],
    key: typing.Iterable[typing.Any],
    **kwargs,
) -> typing.Iterable[_SortTogetherItem]:
    """Sort `list_` with `key` iterable.

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
    """Alternative implementation of a `Pool` context manager. The original `multiprocessing.Pool`
    context manager calls `terminate` rather than `close` followed by `join`.

    Furthermore, it defaults to a 'forkserver' pool, a safer alternative to 'fork'.
    """
    # Learn more: https://pytest-cov.readthedocs.io/en/latest/subprocess-support.html
    # Learn more about `forkserver` / `spawn` / `fork`:
    # https://github.com/pytorch/pytorch/issues/2245
    # https://codewithoutrules.com/2018/09/04/python-multiprocessing/
    # https://pythontic.com/multiprocessing/multiprocessing/introduction
    pool = torch.multiprocessing.get_context("forkserver").Pool(*args, **kwargs)
    yield pool
    pool.close()  # Marks the pool as closed.
    pool.join()  # Waits for workers to exit.


def pad_tensor(
    input_: torch.Tensor, pad: typing.Tuple[int, int], dim: int = 0, **kwargs
) -> torch.Tensor:
    """Pad a tensor dimension.

    Args:
        input_
        pad: The padding to apply.
        dim: The dimension to apply padding.
        **kwargs: Keyword arguments passed onto `torch.nn.functional.pad`.
    """
    padding = [[0, 0] for _ in range(input_.dim())]
    padding[dim] = list(pad)
    # NOTE: `torch.nn.functional.pad` accepts the last dimension first.
    flat: typing.List[int] = flatten_2d(list(reversed(padding)))
    return torch.nn.functional.pad(input_, flat, **kwargs)


def trim_tensors(*args: torch.Tensor, dim: int = 2) -> typing.Iterable[torch.Tensor]:
    """Trim the edges of each tensor in `args` such that each tensor's dimension `dim` has the same
    size.

    Args:
        *args
        dim: The dimension to trim.
    """
    minimum = min(a.shape[dim] for a in args)
    assert all((a.shape[dim] - minimum) % 2 == 0 for a in args), "Uneven padding"
    return tuple([a.narrow(dim, (a.shape[dim] - minimum) // 2, minimum) for a in args])


class LSTM(torch.nn.LSTM):
    """LSTM with a trainable initial hidden state.

    TODO: Add support for `PackedSequence`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_directions = 2 if self.bidirectional else 1
        self.initial_hidden_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size)
        )
        self.initial_cell_state = torch.nn.Parameter(
            torch.randn(self.num_layers * num_directions, 1, self.hidden_size)
        )

    def forward(  # type: ignore
        self,
        input: torch.Tensor,
        hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
        if hx is None:
            batch_size = input.shape[0] if self.batch_first else input.shape[1]
            hx = (
                self.initial_hidden_state.expand(-1, batch_size, -1).contiguous(),
                self.initial_cell_state.expand(-1, batch_size, -1).contiguous(),
            )
        return super().forward(input, hx=hx)


class LSTMCell(torch.nn.LSTMCell):
    """LSTMCell with a trainable initial hidden state.

    TODO: Add support for `PackedSequence`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_hidden_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))
        self.initial_cell_state = torch.nn.Parameter(torch.randn(1, self.hidden_size))

    def forward(
        self,
        input: torch.Tensor,
        hx: typing.Optional[typing.Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        if hx is None:
            hx = (
                self.initial_hidden_state.expand(input.shape[0], -1).contiguous(),
                self.initial_cell_state.expand(input.shape[0], -1).contiguous(),
            )
        return super().forward(input, hx=hx)


_ClampReturnType = typing.TypeVar("_ClampReturnType")


def clamp(a: _ClampReturnType, min_: float = -math.inf, max_: float = math.inf) -> _ClampReturnType:
    return max(min(a, max_), min_)


_CallOnceReturnType = typing.TypeVar("_CallOnceReturnType")


@functools.lru_cache(maxsize=None)
def call_once(
    callable_: typing.Callable[..., _CallOnceReturnType], *args, **kwargs
) -> _CallOnceReturnType:
    """Call `callable_` only once with `args` and `kwargs` within the same process."""
    return callable_(*args, **kwargs)


_MappedIteratorItem = typing.TypeVar("_MappedIteratorItem")


class MappedIterator(typing.Mapping[int, _MappedIteratorItem], typing.Generic[_MappedIteratorItem]):
    """ Wrap an iterator with a mapping. """

    def __init__(self, iterator: typing.Iterable[_MappedIteratorItem]):
        self.iterator = iterator
        self.iter = None
        self.offset = 0

    def __getitem__(self, index) -> _MappedIteratorItem:
        assert index >= self.offset, "Items may only be accessed once."
        self.iter = iter(self.iterator) if self.iter is None else self.iter
        for _ in range(index - self.offset):
            next(self.iter)
        self.offset = index + 1
        return next(self.iter)

    def __iter__(self):
        return [self[i] for i in range(len(self))]

    def __len__(_):
        raise NotImplementedError()


_TupleVar = typing.TypeVar("_TupleVar")


class Tuple(typing.Hashable, typing.Sequence[_TupleVar]):
    """Abstract class for a `tuple`."""

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other):
        raise NotImplementedError


def stow(
    items: typing.Sequence[_TupleVar], dtype: typing.Optional[np.dtype] = None
) -> Tuple[_TupleVar]:
    """Make a data structure for storing `items` efficiently with a `Tuple` inferface."""
    return typing.cast(Tuple[_TupleVar], tuple(items)) if len(items) < 2 else _Tuple(items, dtype)


__TupleType = typing.TypeVar("__TupleType", bound="_Tuple")


class _Tuple(Tuple[_TupleVar]):
    """Datastructure for efficiently storing, and retrieving data.

    NOTE: Learn more about reducing memory usage: https://habr.com/en/post/458518/
    NOTE: This will not error if the data type doesn't accurately represent the data. For example:
    ```
    >>> import numpy as np
    >>> np.array([10000000], np.int16)
    array([-27008], dtype=int16)
    ```
    NOTE: This doesn't support items with `numpy` objects.
    """

    __slots__ = "storage", "type"

    def __init__(
        self,
        items: typing.Union[typing.Sequence[_TupleVar], np.ndarray],
        dtype: typing.Optional[np.dtype] = None,
        type_: typing.Optional[typing.Type[_TupleVar]] = None,
    ):
        self.storage = np.array(items, dtype=dtype)
        self.type: typing.Optional[typing.Type[_TupleVar]]
        if isinstance(items, np.ndarray):
            self.type = type_
        elif isinstance(items, list) and len(items) > 0:
            self.type = items[0].__class__
        elif isinstance(items, list):
            self.type = None
        else:
            raise TypeError("Unsupported arguments.")

    def _convert(self, item: typing.Tuple) -> _TupleVar:
        """ Convert numpy `item()` to `self.type`. """
        assert self.type is not None
        if self.type is tuple:
            return typing.cast(_TupleVar, item)
        return self.type(*item)

    @typing.overload
    def __getitem__(self, index: int) -> _TupleVar:
        ...

    @typing.overload
    def __getitem__(self: __TupleType, index: slice) -> __TupleType:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(self.storage[index], dtype=self.storage.dtype, type_=self.type)
        elif isinstance(index, int):
            return self._convert(self.storage[index].item())
        else:
            raise TypeError("Invalid argument type: {}".format(type(index)))

    def __len__(self) -> int:
        return len(self.storage)

    def __iter__(self) -> typing.Iterator[_TupleVar]:
        return (self._convert(i) for i in self.storage.tolist())

    def __contains__(self, item: _TupleVar) -> bool:
        if len(self.storage) == 0:
            return False
        return np.array(item, dtype=self.storage.dtype) in self.storage

    def _to_tuple(self):
        return tuple(iter(self))

    def __eq__(self, other):
        if type(other) is type(self):
            return other._to_tuple() == self._to_tuple()
        return False

    def __hash__(self):
        return hash(self._to_tuple())

    def __str__(self):
        return str(self._to_tuple())

    def __repr__(self):
        return repr(self._to_tuple())


def mazel_tov() -> str:
    return random.choice(["🎉", "✨", "🤗", "🍾", "🥂", "🥳"])


_CorrectedRandomChoiceVar = typing.TypeVar("_CorrectedRandomChoiceVar")


def corrected_random_choice(
    distribution: typing.Dict[_CorrectedRandomChoiceVar, float], eps=1
) -> _CorrectedRandomChoiceVar:
    """Choose a key in `distribution` that would help even out the distribution.
    NOTE: In order to make the `distribution` uniform, we'd need to sample each key
    `max(values) - value` times; therefore, we use that expectation as a `weight`.
    NOTE: `eps` is added to ensure each key has some chance of getting sampled.
    """
    assert all(v >= 0 for v in distribution.values()), "Must be a valid distribution."
    keys = list(distribution.keys())
    max_ = max(distribution.values())
    return random.choices(keys, [(max_ - v + eps) for v in distribution.values()])[0]


def dataclass_as_dict(data):
    """Shallow copy `dataclass` to `dict`."""
    return {f.name: getattr(data, f.name) for f in dataclasses.fields(data) if f.init}
