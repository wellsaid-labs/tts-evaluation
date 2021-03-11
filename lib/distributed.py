# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import asyncio
import collections
import gzip
import itertools
import json
import logging
import typing

import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.nn.functional

from lib.environment import IS_TESTING_ENVIRONMENT

logger = logging.getLogger(__name__)


# TODO: Rename `master` to `main`, learn more:
# https://www.wired.com/story/tech-confronts-use-labels-master-slave/


def is_initialized() -> bool:
    """ Return `True` if distributed mode is initialized. """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_master_rank() -> typing.Literal[0]:
    """ Returns the rank of the master processs. """
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


DictStoreValue = typing.Union[
    str, int, float, bool, None, typing.List["DictStoreValue"], typing.Dict[str, "DictStoreValue"]
]
DictStoreData = typing.Dict[str, DictStoreValue]
DictStoreDataCollection = typing.Dict[str, typing.List[typing.Tuple[DictStoreValue]]]


class DictStore:
    """DictStore gathers `dict`s from workers on master.

    TODO: Look into other compression algorithms like Zstandard:
    https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

    TODO: Support multiple simultaneous `update` operations on `master`.

    Args:
        data: On the master process, this is a merged collection of data from the worker processes.
    """

    num_instances = -1

    def __init__(
        self,
        store: torch.distributed.TCPStore,
        world_size: typing.Optional[int] = None,
        is_master_: typing.Optional[bool] = None,
        rank: typing.Optional[int] = None,
        identifier: typing.Optional[str] = None,
    ):
        DictStore.num_instances += 1
        name = self.__class__.__name__
        identifier = f"{name}/{DictStore.num_instances}" if identifier is None else identifier
        self._store = torch.distributed.PrefixStore(identifier, store)
        self._operation = -1
        self._world_size = get_world_size() if world_size is None else world_size
        self._is_master = is_master() if is_master_ is None else is_master_
        self._rank = get_rank() if rank is None else rank
        self.data: DictStoreDataCollection = {}

    async def _get(self, key: str) -> DictStoreData:
        """
        NOTE: Learn about JSONs compact encoding, here: https://docs.python.org/3/library/json.html
        """
        result = json.loads(gzip.decompress(bytes.fromhex(self._store.get(key).decode())).decode())
        assert self._store.delete_key(key)
        return result

    async def _gets(self, keys: typing.List[str]) -> typing.List[DictStoreData]:
        tasks = tuple(self._get(k) for k in keys)
        return typing.cast(typing.List[DictStoreData], await asyncio.gather(*tasks))

    def _set(self, values: DictStoreData):
        encoded = gzip.compress(json.dumps(values, separators=(",", ":")).encode()).hex()
        self._store.set(f"/{self._rank}/{self._operation}", encoded)

    def _update(self, data: typing.List[DictStoreData]):
        """Shallow update `self.data` with `data`."""
        update = collections.defaultdict(list)
        for dict_ in data:
            for key, value in dict_.items():
                update[key].append(value)
        for key in set(itertools.chain(update.keys(), self.data.keys())):
            group = tuple(update[key]) if key in update else tuple()
            if key not in self.data:
                self.data[key] = [tuple() for _ in range(self._operation)]
            self.data[key].append(group)

    def update(self, data: DictStoreData):
        """Shallow update the master process `self.data` with `data`."""
        self._operation += 1
        if self._is_master:
            ranks = [i for i in range(self._world_size) if i != get_master_rank()]
            keys = [f"/{i}/{self._operation}" for i in ranks]
            self._store.wait(keys)
            self._update([data] + asyncio.run(self._gets(keys)))
        else:
            self._set(data)
