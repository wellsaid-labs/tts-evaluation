# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import enum
import gc
import logging
import typing

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing
import torch.nn
import torch.nn.functional
import torch.utils.data

from lib.environment import IS_TESTING_ENVIRONMENT

logger = logging.getLogger(__name__)


# TODO: Rename `master` to `main`, learn more:
# https://www.wired.com/story/tech-confronts-use-labels-master-slave/


def is_initialized() -> bool:
    """Return `True` if distributed mode is initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_master_rank() -> typing.Literal[0]:
    """Returns the rank of the master processs."""
    return 0


def get_rank():
    if IS_TESTING_ENVIRONMENT and not is_initialized():
        return get_master_rank()
    return torch.distributed.get_rank()


def is_master() -> bool:
    """Returns `True` if distributed isn't initialized or if this process is the master process."""
    if IS_TESTING_ENVIRONMENT and not is_initialized():
        return True
    return torch.distributed.get_rank() == get_master_rank()


def get_device_count() -> int:
    if IS_TESTING_ENVIRONMENT and not torch.cuda.is_available():
        return 1
    return torch.cuda.device_count()


def get_world_size() -> int:
    if IS_TESTING_ENVIRONMENT and not is_initialized():
        return 1
    return torch.distributed.get_world_size()


def spawn(*args, nprocs=None, **kwargs):
    """`torch.multiprocessing.spawn` wrapper.

    NOTE (michael): Without an assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    crashes in a nondescript way.
    """
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0, "Unable to find CUDA devices."
        nprocs = torch.cuda.device_count() if nprocs is None else nprocs
    return torch.multiprocessing.spawn(*args, nprocs=nprocs, **kwargs)  # type: ignore


ListedDictKey = typing.TypeVar("ListedDictKey")
ListedDictValue = typing.TypeVar("ListedDictValue")


class ListedDict(typing.Generic[ListedDictKey, ListedDictValue]):
    """Store lists of dictionaries efficiently."""

    def __init__(self):
        super().__init__()
        self._data: typing.List[typing.List[typing.Dict[ListedDictKey, ListedDictValue]]] = []
        self._keys = set()

    def append(self, data: typing.List[typing.Dict[ListedDictKey, ListedDictValue]]):
        """Append an additional list of dictionaries."""
        self._data.append(data)
        for dict_ in data:
            self._keys.update(dict_.keys())

    @typing.overload
    def __getitem__(self, key: slice) -> ListedDict[ListedDictKey, ListedDictValue]:
        """Get a `slice` of dictionaries."""
        ...

    @typing.overload
    def __getitem__(self, key: ListedDictKey) -> typing.List[typing.List[ListedDictValue]]:
        """Get all the values for `key` in `ListedDict` in each dictionary."""
        ...

    def __getitem__(self, key):
        if isinstance(key, slice):
            data = ListedDict()
            data._data = self._data[key]
            data._keys = self._keys
            return data

        if key not in self._keys:
            raise KeyError(key)

        return [[d[key] for d in r if key in d] for r in self._data]

    def keys(self) -> typing.Iterator[ListedDictKey]:
        return iter(self._keys)

    def __iter__(self) -> typing.Iterator[ListedDictKey]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __contains__(self, key) -> bool:
        return key in self._keys


GatherVar = typing.TypeVar("GatherVar")


class DictStore:
    """DictStore gathers `dict`s from workers on master.

    TODO: Look into other compression algorithms like Zstandard:
    https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

    NOTE: `torch.distributed.gather_object` and `torch.distributed.TCPStore` benchmarked similarly,
    so we used `gather_object` because it's more widely used, more reliable, and it's easier to
    use.

    NOTE: Speed up `pickle.loads` with this approach:
    https://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram

    TODO: Add a method for queuing up `update` and `log` requests, and doing them all at once.

    Args:
        data: On the master process, this is a merged collection of data from the worker processes.
        cache_keys: Instead of pickling, sending, and receiving keys, this maps each key to a
            unique identifier which is used in it's stead. This setting is particularly
            helpful to use if there are a limited set of keys that are slow to `pickle`. Python
            objects like `NamedTuple` or `dataclasses` are slow to `pickle`, for example.
    """

    sync_every = 1
    new_keys: typing.List[typing.Any] = []
    key_cache: typing.Dict[typing.Any, str] = {}
    reverse_key_cache: typing.Dict[str, typing.Any] = {}
    # NOTE: It's 5x faster to serialize and deserialize a unique string with pickle than a
    # unique object (i.e. subclassing `int`), so we are using `key_cache_prefix` as a prefix for
    # generating a unique string.
    key_cache_prefix = "__"

    def __init__(self, cache_keys: bool = False):
        self.data = ListedDict()
        self.cache_keys = cache_keys
        self._operation = -1
        self._world_size = get_world_size()
        self._is_master = is_master()

    def _all_gather(self, data: GatherVar) -> typing.List[GatherVar]:
        outputs = [None for _ in range(self._world_size)]
        torch.distributed.all_gather_object(outputs, data)
        return typing.cast(typing.List[GatherVar], outputs)

    def _gather(self, data: GatherVar) -> typing.Optional[typing.List[GatherVar]]:
        outputs = [None for _ in range(self._world_size)] if self._is_master else None
        torch.distributed.gather_object(data, outputs)
        return typing.cast(typing.Optional[typing.List[GatherVar]], outputs)

    def _encode(self, data: typing.Dict) -> typing.Dict:
        """Encode `data` with `key_cache`."""
        if not self.cache_keys:
            return data

        # NOTE: `try` / `except` is faster than `if` / `else`, learn more here:
        # https://www.geeksforgeeks.org/try-except-vs-if-in-python/
        encoded = {}
        for key, value in data.items():
            try:
                encoded[DictStore.key_cache[key]] = value
            except KeyError:
                encoded[key] = value
        return encoded

    def _decode(self, encoded: typing.Dict) -> typing.Dict:
        """Decode `data` with `key_cache`."""
        if not self.cache_keys:
            return encoded

        decoded = {}
        for key, value in encoded.items():
            try:
                decoded[DictStore.reverse_key_cache[key]] = value
            except KeyError:
                DictStore.new_keys.append(key)
                decoded[key] = value
        return decoded

    def _update_vocab(self, new_keys_batch: typing.List[typing.List]):
        """Update the `key_cache` with new keys."""
        if not self.cache_keys:
            return

        len_ = len(DictStore.key_cache)
        for new_keys in new_keys_batch:
            for key in new_keys:
                if key not in DictStore.key_cache:
                    id = DictStore.key_cache_prefix + str(len(DictStore.key_cache))
                    DictStore.key_cache[key] = id
                    DictStore.reverse_key_cache[id] = key
                    assert len(DictStore.key_cache) == len(DictStore.reverse_key_cache)

        if len_ == len(DictStore.key_cache):
            DictStore.sync_every *= 2

    def update(self, data: typing.Dict):
        """Shallow update the master process `self.data` with `data`."""
        gc.disable()
        self._operation += 1
        gathered = self._gather(self._encode(data))
        if self._operation > 0 and self._operation % DictStore.sync_every == 0:
            self._update_vocab(self._all_gather(DictStore.new_keys))
        if self._is_master and gathered is not None:
            self.data.append([self._decode(d) for d in gathered])
        gc.enable()


_NumeralizePadEmbedVar = typing.TypeVar("_NumeralizePadEmbedVar", bound=typing.Hashable)


class NumeralizePadEmbed(torch.nn.Module, typing.Generic[_NumeralizePadEmbedVar]):
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

        VocabKey = typing.Union[_NumeralizePadEmbedVar, NumeralizePadEmbed._Tokens]
        self.vocab: typing.Dict[VocabKey, int]
        self.vocab = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}

        max_embeddings = len(self.vocab) + max_embeddings
        self.embed = torch.nn.Embedding(max_embeddings, *args, padding_idx=self.pad_idx, **kwargs)
        self.weight = self.embed.weight
        self.num_embeddings = self.embed.num_embeddings
        self.embedding_dim = self.embed.embedding_dim
        self.padding_idx = self.embed.padding_idx

        self._new_tokens = set()
        self._unk_tokens = set()  # NOTE: Track unknown tokens seen during evaluation

    def _queue_new_tokens(self, sequences: typing.List[typing.List[_NumeralizePadEmbedVar]]):
        """Queue up tokens for a vocab update."""
        self._new_tokens.update([t for s in sequences for t in s if t not in self.vocab])
        if len(self._unk_tokens) > 0:
            self._unk_tokens = set()
        if len(self._new_tokens) + len(self.vocab) > self.num_embeddings:
            raise ValueError(
                f"The number of tokens exceeds the allocated "
                f"number of embeddings, {self.num_embeddings}."
            )

    def reset(self):
        """Reset the step counter, so that updates happen again."""
        self.update_every = 1

    def update_tokens(
        self,
        tokens: typing.List[_NumeralizePadEmbedVar],
        embeddings: typing.Optional[torch.Tensor] = None,
    ):
        """Add or update tokens in `self.vocab`.

        NOTE: This doesn't support a distributed context, yet.

        Args:
            tokens: The tokens to add or update.
            embeddings: The corresponding embeddings for each token.
        """
        if is_initialized():
            raise ValueError("This doesn't support distributed context.")

        self._queue_new_tokens([tokens])
        if len(self._new_tokens) > 0:
            self._update_vocab()

        if embeddings is not None:
            if embeddings.shape != (len(tokens), self.embed.embedding_dim):
                raise ValueError("The updated `embeddings` are the wrong shape.")

            with torch.no_grad():
                self.weight[[self.vocab[t] for t in tokens]] = embeddings

    def tokens(self) -> typing.List[_NumeralizePadEmbedVar]:
        """Get all the tokens in the vocabulary excluding the default tokens."""
        return [t for t in self.vocab.keys() if not isinstance(t, NumeralizePadEmbed._Tokens)]

    def _update_vocab(self):
        """Update `self.vocab` with `self._new_tokens`."""
        new_tokens = list(self._new_tokens)

        if is_initialized():
            outputs = [None for _ in range(get_world_size())]
            torch.distributed.all_gather_object(outputs, new_tokens)
            outputs = typing.cast(typing.List[typing.List[_NumeralizePadEmbedVar]], outputs)
            new_tokens: typing.List[_NumeralizePadEmbedVar] = [t for l in outputs for t in l]
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

        if len(self.vocab) > self.num_embeddings:
            raise ValueError(
                f"The number of tokens exceeds the allocated "
                f"number of embeddings, {self.num_embeddings}."
            )

        if len(new_tokens) == 0:
            self.update_every *= 2

        self._new_tokens = set()

    def _check_invariants(self):
        """Ensure the data structure invariants hold."""
        for token in self._new_tokens:
            assert token not in self.vocab, "Invariant failure."

    def _tok_to_idx(self, token: _NumeralizePadEmbedVar) -> int:
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
            logger.debug(f"[Training] Using unknown token in-place of '{token}'.")

        return idx

    def _should_update(self):
        return self._training_forward_pass_counter % self.update_every == 0 or not is_initialized()

    def __call__(
        self,
        tokens: typing.Union[
            typing.List[_NumeralizePadEmbedVar], typing.List[typing.List[_NumeralizePadEmbedVar]]
        ],
        **kwargs,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return super().__call__(tokens, **kwargs)

    def forward(
        self,
        tokens: typing.Union[
            typing.List[_NumeralizePadEmbedVar], typing.List[typing.List[_NumeralizePadEmbedVar]]
        ],
        batch_first: bool = False,
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
            typing.List[typing.List[_NumeralizePadEmbedVar]],
            [[s] if is_one_dim else s for s in tokens],
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
