# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import logging
import typing
import typing_extensions

import torch

import lib

logger = logging.getLogger(__name__)


def is_initialized() -> bool:
    """ Return `True` if distributed mode is initialized. """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_master_rank() -> typing_extensions.Literal[0]:
    """ Returns the rank of the master processs. """
    return 0


def is_master() -> bool:
    """ Returns `True` if distributed isn't initialized or if this process is the master process.
    """
    if not is_initialized():
        return True
    return torch.distributed.get_rank() == get_master_rank()


def assert_synced(value: float, message: str = ''):
    """ Assert that `value` is the same between master and worker nodes.

    NOTE: The `value` is split into digits to support large numbers like 128-bit hashes.

    Args:
        value: Value to check.
        message: Assert message.
    """
    torch_ = torch.cuda if torch.cuda.is_available() else torch
    if is_master():
        length = torch_.LongTensor([len(str(value))])  # type: ignore
    else:
        length = torch_.LongTensor(1)  # type: ignore
    torch.distributed.broadcast(length, src=get_master_rank())
    length = length.item()
    assert len(str(value)) == length, message
    value_tensor = torch_.LongTensor([int(d) for d in str(value)])  # type: ignore
    if is_master():
        torch.distributed.broadcast(value_tensor, src=get_master_rank())
        master_value = value_tensor
    else:
        master_value = torch_.LongTensor(length)  # type: ignore
        torch.distributed.broadcast(master_value, src=get_master_rank())
    assert torch.equal(master_value, value_tensor), message


def spawn(*args, **kwargs):
    """ `torch.multiprocessing.spawn` wrapper.

    NOTE (michael): Without an assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    crashes in a nondescript way.
    """
    num_cuda_devices = torch.cuda.device_count()
    assert num_cuda_devices > 0, 'Unable to find CUDA devices.'
    torch.multiprocessing.spawn(*args, nprocs=num_cuda_devices, **kwargs)


class DistributedAverage(lib.utils.Average):
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
        torch.distributed.reduce(packed, dst=get_master_rank())
        self.post_sync_total_value, self.post_sync_total_count = tuple(packed.tolist())
        self.last_update_value = (self.post_sync_total_value - last_post_sync_total_value) / (
            self.post_sync_total_count - last_post_sync_total_count)
        return self
