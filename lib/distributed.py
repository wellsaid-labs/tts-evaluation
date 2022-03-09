# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import gc
import itertools
import logging
import pickle
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


class DictStore:
    """DictStore gathers `dict`s from workers on master.

    TODO: Look into other compression algorithms like Zstandard:
    https://www.lucidchart.com/techblog/2019/12/06/json-compression-alternative-binary-formats-and-compression-methods/

    NOTE: We use `torch.distributed.gather_object` instead of `torch.distributed.TCPStore` because
    of these issues:
    https://github.com/pytorch/pytorch/issues/53872
    https://github.com/pytorch/pytorch/issues/53840
    Also, `TCPStore` only allows for integer values.

    NOTE: Speed up `pickle.loads` with this approach:
    https://stackoverflow.com/questions/2766685/how-can-i-speed-up-unpickling-large-objects-if-i-have-plenty-of-ram

    Args:
        data: On the master process, this is a merged collection of data from the worker processes.
    """

    def __init__(self):
        self.data: typing.Dict[typing.Any, typing.List[typing.Sequence]] = {}
        self._operation = -1

    @staticmethod
    def _decode(encoded: str) -> typing.Dict:
        return pickle.loads(bytes.fromhex(encoded))

    @staticmethod
    def _encode(values: typing.Dict) -> str:
        return pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL).hex()

    def _gather(self, data: typing.Dict) -> typing.List[str]:
        outputs = [None for _ in range(get_world_size())]
        torch.distributed.all_gather_object(outputs, self._encode(data))
        return typing.cast(typing.List[str], outputs)

    def _update(self, data: typing.List[typing.Dict]):
        """Shallow update `self.data` with `data`."""
        update_keys = set(itertools.chain(*tuple(d.keys() for d in data)))
        data_keys = set(self.data.keys())
        new_keys = update_keys - data_keys
        self.data.update({k: [tuple()] * self._operation for k in new_keys})
        for key in data_keys - update_keys:
            self.data[key].append(tuple())
        for key in update_keys:
            self.data[key].append([d[key] for d in data if key in d])

    def update(self, data: typing.Dict):
        """Shallow update the master process `self.data` with `data`."""
        self._operation += 1
        gathered = self._gather(data)
        if is_master():
            gc.disable()
            decoded = [self._decode(typing.cast(str, o)) for o in gathered]
            gc.enable()
            self._update(decoded)
