# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import logging
import typing

import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.nn.functional

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
    if not is_initialized():
        return get_master_rank()
    return torch.distributed.get_rank()


def is_master() -> bool:
    """Returns `True` if distributed isn't initialized or if this process is the master process."""
    if not is_initialized():
        return True
    return torch.distributed.get_rank() == get_master_rank()


def get_world_size() -> int:
    return torch.distributed.get_world_size() if is_initialized() else 1


def spawn(*args, nprocs=None, **kwargs):
    """`torch.multiprocessing.spawn` wrapper.

    NOTE (michael): Without an assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    crashes in a nondescript way.
    """
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0, "Unable to find CUDA devices."
        nprocs = torch.cuda.device_count() if nprocs is None else nprocs
    return torch.multiprocessing.spawn(*args, nprocs=nprocs, **kwargs)  # type: ignore
