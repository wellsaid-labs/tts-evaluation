# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import collections
import gzip
import itertools
import json
import logging
import typing

import torch
import torch.cuda
import torch.distributed
import torch.multiprocessing
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

    NOTE: We use `torch.distributed.gather_object` instead of `torch.distributed.TCPStore` because
    of these issues:
    https://github.com/pytorch/pytorch/issues/53872
    https://github.com/pytorch/pytorch/issues/53840

    Args:
        data: On the master process, this is a merged collection of data from the worker processes.
    """

    def __init__(self):
        self.data: DictStoreDataCollection = {}
        self._operation = -1

    @staticmethod
    def _decode(encoded: str) -> DictStoreData:
        """
        NOTE: Learn about JSONs compact encoding, here: https://docs.python.org/3/library/json.html
        """
        return json.loads(gzip.decompress(bytes.fromhex(encoded)).decode())

    @staticmethod
    def _encode(values: DictStoreData) -> str:
        return gzip.compress(json.dumps(values, separators=(",", ":")).encode()).hex()

    def _gather(self, data: DictStoreData) -> typing.List[DictStoreData]:
        outputs = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(outputs, self._encode(data))
        return [self._decode(typing.cast(str, o)) for o in outputs]

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
        merged = self._gather(data)
        if is_master():
            self._update(merged)
