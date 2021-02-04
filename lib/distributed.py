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


_default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce(value: float, device: torch.device, dst: int = get_master_rank(), **kwargs) -> float:
    """Reduce `value` from all processes via a reduction operation
    like `torch.distributed.ReduceOp.SUM`."""
    if not is_initialized():
        return value
    packed = torch.tensor([value], dtype=torch.float, device=device)
    torch.distributed.reduce(packed, dst=dst, **kwargs)
    return typing.cast(float, packed.item())


def all_gather(value: float, device: torch.device, **kwargs) -> typing.List[float]:
    """ Gather `value` from all processes into a `list`. """
    if not is_initialized():
        return [value]
    world_size = torch.distributed.get_world_size()
    return_ = [torch.zeros(1, device=device, dtype=torch.float) for _ in range(world_size)]
    tensor = torch.tensor([value], device=device, dtype=torch.float)
    torch.distributed.all_gather(return_, tensor, **kwargs)
    return [typing.cast(float, t.item()) for t in return_]


def gather_list(
    values: typing.List[float], device: torch.device, **kwargs
) -> typing.List[typing.List[float]]:
    """Gather `values` from all processes into a `list` on the `dst` process.

    TODO: Support `typing.List[int]`.
    """
    if not is_initialized():
        return [values]
    lengths = [int(l) for l in all_gather(len(values), device=device, **kwargs)]
    max_ = max(lengths)
    return_ = [torch.zeros(max_, device=device, dtype=torch.float) for _ in lengths]
    tensor = torch.tensor(values, device=device, dtype=torch.float)
    tensor = torch.nn.functional.pad(tensor, [0, max_ - len(values)])
    # NOTE: `ProcessGroupNCCL` does not support `gather`, so we need to use `all_gather`.
    torch.distributed.all_gather(return_, tensor, **kwargs)
    return [t.tolist()[:l] for t, l in zip(return_, lengths)]


def spawn(*args, nprocs=None, **kwargs):
    """`torch.multiprocessing.spawn` wrapper.

    NOTE (michael): Without an assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    crashes in a nondescript way.
    """
    if torch.cuda.is_available():
        assert torch.cuda.device_count() > 0, "Unable to find CUDA devices."
        nprocs = torch.cuda.device_count() if nprocs is None else nprocs
    return torch.multiprocessing.spawn(*args, nprocs=nprocs, **kwargs)  # type: ignore
