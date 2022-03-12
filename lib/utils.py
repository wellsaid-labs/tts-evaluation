# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import dataclasses
import enum
import functools
import itertools
import logging
import math
import multiprocessing.pool
import pathlib
import pickle
import random
import statistics
import time
import typing
from abc import abstractmethod
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.nn.functional
import torch.utils.data
from tqdm import tqdm

import lib.distributed

logger = logging.getLogger(__name__)


def round_(x: float, bucket_size: float) -> float:
    """Bin `x` into buckets."""
    return bucket_size * round(x / bucket_size)


_RandomSampleReturnType = typing.TypeVar("_RandomSampleReturnType")


def random_sample(
    list_: typing.List[_RandomSampleReturnType], sample_size: int
) -> typing.List[_RandomSampleReturnType]:
    """Random sample function that doesn't error if `list_` is smaller than `sample_size`."""
    return random.sample(list_, min(len(list_), sample_size))


def mean(list_: typing.Iterable[float]) -> float:
    """Mean function that does not return an error if `list_` is empty."""
    list_ = list(list_)
    if len(list_) == 0:
        return math.nan
    return statistics.mean(list_)  # NOTE: `statistics.mean` returns an error for an empty list


_ChunksReturnType = typing.TypeVar("_ChunksReturnType")


def get_chunks(
    list_: typing.List[_ChunksReturnType], n: int
) -> typing.Iterator[typing.List[_ChunksReturnType]]:
    """Yield successive `n`-sized chunks from `list_`."""
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

    Learn more:
    https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
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


def flatten_parameters(model: torch.nn.Module) -> torch.nn.Module:
    """Apply `flatten_parameters` to `model`."""
    lambda_ = lambda m: m.flatten_parameters() if hasattr(m, "flatten_parameters") else None
    return model.apply(lambda_)


_IdentityReturnType = typing.TypeVar("_IdentityReturnType")


def identity(x: _IdentityReturnType) -> _IdentityReturnType:
    return x


_SplitReturnType = typing.TypeVar("_SplitReturnType")


def split(
    list_: typing.List[_SplitReturnType],
    splits: typing.List[float],
    value: typing.Callable[[_SplitReturnType], float] = identity,
) -> typing.Iterator[typing.List[_SplitReturnType]]:
    """Split `list_` when the accumulated sum passes a threshold.

    Args:
        ...
        value: Given a list item, determine the value of the list item.

    Returns: Slice(s) of the list.
    """
    index = 0
    for split in splits:
        if math.isinf(split):
            yield list_[index:]
            index = len(list_)
        elif split == 0:
            yield []
        else:
            totals = itertools.accumulate(value(i) for i in list_[index:])
            count = sum(1 for _ in itertools.takewhile(lambda x: x <= split, totals))
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


AnyCallable = typing.Callable[..., typing.Any]

_LogRuntimeFunction = typing.TypeVar("_LogRuntimeFunction", bound=AnyCallable)


def log_runtime(function: _LogRuntimeFunction) -> _LogRuntimeFunction:
    """Decorator for measuring the execution time of a function."""

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        elapsed = seconds_to_str(time.time() - start)
        logger.info("`%s` ran for %s", function.__qualname__, elapsed)
        return result

    return typing.cast(_LogRuntimeFunction, decorator)


_CacheReturnDecoratorFunction = typing.TypeVar("_CacheReturnDecoratorFunction", bound=AnyCallable)


def disk_cache(path: pathlib.Path):
    """Decorator for caching the return value in a file regardless of the arguments."""

    def decorator(function: _CacheReturnDecoratorFunction) -> _CacheReturnDecoratorFunction:
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if path.exists():
                logger.warn(
                    f"Loading cache for `{function.__qualname__}` from `{path}`. "
                    f"Please delete `{path}` and rerun if you'd like to not use the "
                    "cache."
                )
                with path.open("rb") as f:
                    return pickle.load(f)

            result = function(*args, **kwargs)
            with path.open("wb") as f:
                logger.info(f"Caching return value for `{function.__qualname__}` to `{path}`.")
                pickle.dump(result, f)

            return result

        wrapper.clear_cache = lambda: path.unlink()

        return typing.cast(_CacheReturnDecoratorFunction, wrapper)

    return decorator


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


Hashable1d2dList = typing.Union[
    typing.List[typing.Hashable], typing.List[typing.List[typing.Hashable]]
]


class NumeralizePadEmbed(torch.nn.Module):
    """An `Embedding` layer with `max_embeddings` that progressively maps new tokens to embeddings.

    NOTE: This layer is intended to simplify the boilerplate code required to numeralize, pad,
          and embed a simple sequence. In order to support this end-to-end, this embedding layer
          only works with inputs in the form of [batch_size] or [batch_size, num_tokens].

    NOTE: In a former version of this, we tried to track the `unk_token` embedding to see if it
          had changed, in order to trigger a vocab update. This turned out to be difficult due
          to the fancy optimizers with various second order and ema optimization techniques.

          With that in mind, it's theoritically possible, that we could have a fancy update
          detection mechanism based on changes in `unk_token`.

    Args:
        max_embeddings: The maximum number of embeddings needed, in addition to the standard unknown
            and padding embedding.
        allow_unk_on_eval: Iff then the "unknown token" may be used during evaluation, otherwise
            this will error if a new token is encountered during evaluation.
        *args: Arguments passed to `torch.nn.Embedding`.
        **kwargs: Keyword arguments passed to `torch.nn.Embedding`.
    """

    class _Tokens(enum.Enum):
        """
        Unique hashtable objects for `pad_token` and `unk_token`, that's unique from any object
        that could be inputted to `forward`.

        NOTE: It's standard to use 0 for padding.
        """

        PAD_TOKEN: typing.Final = 0
        UNK_TOKEN: typing.Final = 1

    def __init__(
        self,
        max_embeddings: int,
        *args,
        allow_unk_on_eval: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.allow_unk_on_eval = allow_unk_on_eval
        self._training_forward_pass_counter = 0
        self.reset()

        self.pad_idx = self._Tokens.PAD_TOKEN.value
        self.unk_idx = self._Tokens.UNK_TOKEN.value
        self.pad_token = self._Tokens.PAD_TOKEN
        self.unk_token = self._Tokens.UNK_TOKEN

        self.vocab: typing.Dict[typing.Hashable, int]
        self.vocab = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}

        max_embeddings = len(self.vocab) + max_embeddings
        self.embed = torch.nn.Embedding(max_embeddings, *args, padding_idx=self.pad_idx, **kwargs)
        self.weight = self.embed.weight
        self.num_embeddings = self.embed.num_embeddings
        self.embedding_dim = self.embed.embedding_dim
        self.padding_idx = self.embed.padding_idx

        self._new_tokens = set()
        self._unk_tokens = set()  # NOTE: Track unknown tokens seen during evaluation

    def _queue_new_tokens(self, sequences: typing.List[typing.List[typing.Hashable]]):
        """Queue up tokens for a vocab update."""
        self._new_tokens.update([t for s in sequences for t in s if t not in self.vocab])
        if len(self._unk_tokens) > 0:
            self._unk_tokens = set()
        if len(self._new_tokens) + len(self.vocab) > self.num_embeddings:
            raise ValueError("The number of tokens exceeds the allocated number of embeddings.")

    def reset(self):
        """Reset the step counter, so that updates happen again."""
        self.update_every = 1

    def update_tokens(
        self,
        tokens: typing.List[typing.Hashable],
        embeddings: typing.Optional[torch.Tensor] = None,
    ):
        """Add or update tokens in `self.vocab`.

        NOTE: This doesn't support a distributed context, yet.

        Args:
            tokens: The tokens to add or update.
            embeddings: The corresponding embeddings for each token.
        """
        if lib.distributed.is_initialized():
            raise ValueError("This doesn't support distributed context.")

        self._queue_new_tokens([tokens])
        if len(self._new_tokens) > 0:
            self._update_vocab()

        if embeddings is not None:
            if embeddings.shape != (len(tokens), self.embed.embedding_dim):
                raise ValueError("The updated `embeddings` are the wrong shape.")

            with torch.no_grad():
                self.weight[[self.vocab[t] for t in tokens]] = embeddings

    def _update_vocab(self):
        """Update `self.vocab` with `self._new_tokens`."""
        new_tokens = list(self._new_tokens)

        if lib.distributed.is_initialized():
            outputs = [None for _ in range(lib.distributed.get_world_size())]
            torch.distributed.all_gather_object(outputs, new_tokens)
            outputs = typing.cast(typing.List[typing.List[typing.Hashable]], outputs)
            new_tokens = [t for l in outputs for t in l]
            if len(new_tokens) > 0:
                try:
                    # NOTE: Ensure that the order `new_tokens` are added in is consistent for
                    # external observers.
                    # NOTE: `set` may have a different order between processes, so it must be
                    # sorted.
                    new_tokens = sorted(set(new_tokens))  # type: ignore
                except TypeError:
                    pass

        for token in new_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        if len(new_tokens) == 0:
            self.update_every *= 2

        self._new_tokens = set()

    def _check_invariants(self):
        """Ensure the data structure invariants hold."""
        for token in self._new_tokens:
            assert token not in self.vocab, "Invariant failure."

    def _tok_to_idx(self, token: typing.Hashable) -> int:
        """Get the index of `token` and return `unk_token` if `token` is not found.

        Raises:
            KeyError: Iff the module is in evaluation mode and "unknown token" is disabled this
                will error iff `token` isn't in `self.vocab`.
        """
        unk_idx = self.unk_idx
        idx = self.vocab.get(token, unk_idx)
        is_unknown = idx is unk_idx

        if not self.training and is_unknown:
            if not self.allow_unk_on_eval:
                raise KeyError(f"Token not found: {token}")
            if token not in self._unk_tokens:
                logger.info(f"[Evaluation] Marking '{token}' token as unknown token")
                # NOTE: Track unknown tokens so that they are not logged over and over.
                self._unk_tokens.add(token)

        if self.training and is_unknown:
            logger.info(f"[Training] Using unknown token in-place of '{token}'.")

        return idx

    def _should_update(self):
        return (
            self._training_forward_pass_counter % self.update_every == 0
            or not lib.distributed.is_initialized()
        )

    def __call__(
        self, tokens: Hashable1d2dList, **kwargs
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(tokens, **kwargs)

    def forward(
        self, tokens: Hashable1d2dList, batch_first: bool = False, **kwargs
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input [batch_size, num_tokens (optional)]: A 1 or 2-dimensional list of tokens.
            batch_first: In the return tensor, iff `True` the batch dimension is first, otherwise
                it's second.

        Returns:
            embedded (torch.FloatTensor [batch_size, num_tokens, embedding_dim] or
                [num_tokens, batch_size, embedding_dim] or [batch_size, embedding_dim])
            mask (torch.BoolTensor [batch_size, num_tokens] or [num_tokens, batch_size] or
                [batch_size])
        """
        is_one_dim = not isinstance(tokens[0], list)
        get, pad_idx = self._tok_to_idx, self.pad_idx
        sequences = typing.cast(
            typing.List[typing.List[typing.Hashable]], [[s] if is_one_dim else s for s in tokens]
        )

        if self.training:
            self._training_forward_pass_counter += 1
            self._queue_new_tokens(sequences)
            if self._should_update():
                self._update_vocab()

        max_len = max(len(s) for s in sequences)
        indices = [[get(t) for t in s] + [pad_idx] * (max_len - len(s)) for s in sequences]
        indices_ = torch.tensor(indices, device=self.weight.device, dtype=torch.long)
        indices_ = indices_.squeeze(1) if is_one_dim else indices_
        indices_ = indices_ if batch_first or is_one_dim else indices_.transpose(0, 1)
        mask = indices_ != pad_idx
        return self.embed(indices_), mask


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
    """Wrap an iterator with a mapping."""

    def __init__(self, iterator: typing.Iterable[_MappedIteratorItem]):
        self.iterator = iterator
        self.iter = None
        self.offset = 0

    def __getitem__(self, index: int) -> _MappedIteratorItem:
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
_TupleType = typing.TypeVar("_TupleType", bound="Tuple")


class Tuple(typing.Hashable, typing.Sequence[_TupleVar]):
    """Abstract class for a `tuple`."""

    @typing.overload
    @abstractmethod
    def __getitem__(self, index: int) -> _TupleVar:
        ...

    @typing.overload
    @abstractmethod
    def __getitem__(self: _TupleType, index: slice) -> _TupleType:
        ...

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: typing.Any) -> bool:
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
        items = items if isinstance(items, np.ndarray) else list(items)
        self.storage = np.asarray(items, dtype=dtype)
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
        """Convert numpy `item()` to `self.type`.

        TODO: `_convert` is slow. Instead of creating an object, we should consider using
        `recarray`. It behaves similarly to a `NamedTuple`.
        """
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
    distr: typing.Dict[_CorrectedRandomChoiceVar, float],
    expect: typing.Optional[typing.Dict[_CorrectedRandomChoiceVar, float]] = None,
    eps: float = 10e-8,
) -> _CorrectedRandomChoiceVar:
    """Choose a key in `distribution` that would help even out the distribution.

    NOTE: In order to make the `distribution` uniform, we'd need to sample each key
    `max(values) - value` times; therefore, we use that expectation as a `weight`.
    NOTE: `eps` is added to ensure each key has some chance of getting sampled.
    """
    vals = list(distr.values())
    keys = list(distr.keys())
    assert all(v >= 0 for v in vals), "Must be a valid distribution."
    expect_ = [1 for _ in keys] if expect is None else [expect[k] for k in keys]
    ratio = max([b / a for a, b in zip(expect_, vals)])
    expect_ = [a * ratio for a in expect_]
    diff = [a - b + eps for a, b in zip(expect_, vals)]
    return random.choices(keys, diff)[0]


def dataclass_as_dict(data):
    """Shallow copy `dataclass` to `dict`."""
    assert dataclasses.is_dataclass(data), "Argument should be a data class."
    return {f.name: getattr(data, f.name) for f in dataclasses.fields(data) if f.init}


def to_str(object, *attributes: str) -> str:
    """Create a string representation of `object` given it's `attributes`."""
    values = ", ".join(f"{a}={getattr(object, a)}" for a in attributes)
    return f"{object.__class__.__name__}({values})"


FloatFloat = typing.Tuple[float, float]


class Timeline:
    """Store and query a sequence of sorted `intervals`.

    NOTE: This is optimized for the case where the intervals can be sorted. There are other more
    general interval data structures like:
    - https://pandas.pydata.org/docs/reference/api/pandas.Interval.html
    - https://github.com/chaimleib/intervaltree
    - https://github.com/brentp/quicksect
    - Our previous implementation which bucketed intervals into a `dict`.
    """

    __slots__ = "_intervals", "dtype"

    def __init__(self, intervals: typing.List[FloatFloat], dtype=np.float64):
        intervals = sorted(intervals, key=lambda k: k[0])
        message = "Timeline only accepts ordered intervals."
        assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(intervals, intervals[1:])), message
        self.dtype = dtype
        self._intervals = np.array(intervals, dtype=self.dtype).T.reshape(2, len(intervals))
        self._intervals = np.ascontiguousarray(self._intervals)

    def start(self, index: int) -> np.ndarray:
        """Get the start of the interval at `index`."""
        return self._intervals[0, index]

    def stop(self, index: int) -> np.ndarray:
        """Get the stop of the interval at `index`."""
        return self._intervals[1, index]

    def make_slice(self, interval: typing.Union[int, float, slice]) -> slice:
        """Get a `slice` of intervals overlapping `interval`.

        TODO: For short slices, instead of calling `searchsorted` twice, we can call it once. If
        we know how many elements `np.where` can traverse in the same time it takes
        `np.searchsorted` to run, then we can proactively check if the `length` is within that
        number of elements. If so, we'd call `np.where` instead of `np.searchsorted`.

        NOTE: Learn more about the performance of `searchsorted`:
        https://github.com/numpy/numpy/issues/13566
        http://sociograph.blogspot.com/2011/12/gotcha-with-numpys-searchsorted.html
        https://github.com/numpy/numpy/issues/5370
        https://stackoverflow.com/questions/15139299/performance-of-numpy-searchsorted-is-poor-on-structured-arrays

        NOTE: Learn more about the performance of `transpose`:
        https://stackoverflow.com/questions/48509674/numpy-transpose-functions-speed-and-use-cases
        """
        if isinstance(interval, slice):
            start, stop = interval.start, interval.stop
        elif isinstance(interval, (int, float)):
            start, stop = interval, interval
        else:
            raise TypeError("Invalid argument type: {}".format(type(interval)))
        assert start <= stop, "Start must be smaller than stop."
        start_ = self._intervals[1].searchsorted(self.dtype(start), side="left")
        length = self._intervals[0][start_:].searchsorted(self.dtype(stop), side="right")
        return slice(start_, start_ + length)

    def indicies(self, interval: typing.Union[int, float, slice]) -> typing.Iterable[int]:
        """Similar to `make_slice` except this returns an `Iterable` of indicies."""
        return range(*self.make_slice(interval).indices(self._intervals.shape[1]))

    def __getitem__(self, interval: typing.Union[int, float, slice]) -> np.ndarray:
        """Get the intervals overlapping `interval`."""
        return self._intervals[:, self.make_slice(interval)].T

    def intervals(
        self, interval: typing.Optional[typing.Union[int, float, slice]] = None
    ) -> typing.List[FloatFloat]:
        array = self._intervals.T if interval is None else self[interval]
        return [(float(i[0]), float(i[1])) for i in array]

    def num_intervals(self) -> int:
        return self._intervals.shape[1]


_TimelineVar = typing.TypeVar("_TimelineVar")


class TimelineMap(Timeline, typing.Generic[_TimelineVar]):
    """A mapping from `intervals` to values."""

    __slots__ = "intervals", "dtype", "vals"

    def __init__(self, intervals: typing.List[typing.Tuple[FloatFloat, _TimelineVar]]):
        intervals = sorted(intervals, key=lambda k: k[0])
        super().__init__([i[0] for i in intervals])
        self.vals = tuple(i[1] for i in intervals)

    def __getitem__(self, key: typing.Union[int, float, slice]) -> typing.Tuple[_TimelineVar, ...]:
        return self.vals[self.make_slice(key)]


_TripletsVar = typing.TypeVar("_TripletsVar")
_TripletsItem = typing.Tuple[
    typing.Optional[_TripletsVar], typing.Optional[_TripletsVar], typing.Optional[_TripletsVar]
]


def triplets(items: typing.List[_TripletsVar]) -> typing.Iterator[_TripletsItem[_TripletsVar]]:
    """Get triples of `items` bounded by `None`."""
    items_ = typing.cast(typing.List[typing.Optional[_TripletsVar]], items)
    none = typing.cast(typing.List[typing.Optional[_TripletsVar]], [None])
    return zip(none + items_[:-1], items, items_[1:] + none)


_TqdmVar = typing.TypeVar("_TqdmVar")


def tqdm_(iterator: typing.Iterable[_TqdmVar], **kwargs) -> typing.Iterable[_TqdmVar]:
    """`tqdm` with typing."""
    return tqdm(iterator, **kwargs)
