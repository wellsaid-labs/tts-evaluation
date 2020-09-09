import logging

import torch

logger = logging.getLogger(__name__)


def is_initialized():
    """ Return ``True`` if distributed is mode is initialized. """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_master_rank():
    """ Returns the rank of the master processs. """
    return 0


def is_master():
    """
    Returns:
        bool: ``True`` if master process of distributed process or if
        ``is_initialized()`` is ``False``.
    """
    if not is_initialized():
        return True

    return torch.distributed.get_rank() == get_master_rank()


def assert_synced(value, message='', type_=torch.cuda):
    """ Assert that `value` is the same between master and worker nodes.

    Args:
        value (number): Value to check.
        message (str): Assert message.
        type_ (any): Default tensor type to use.
    """
    # NOTE: The `value` is split into digits to support large numbers like 128-bit hashes.
    if is_master():
        length = type_.LongTensor([len(str(value))])
    else:
        length = type_.LongTensor(1)
    torch.distributed.broadcast(length, src=get_master_rank())
    length = length.item()
    assert len(str(value)) == length, message

    value = type_.LongTensor([int(d) for d in str(value)])
    if is_master():
        torch.distributed.broadcast(value, src=get_master_rank())
        master_value = value
    else:
        master_value = type_.LongTensor(length)
        torch.distributed.broadcast(master_value, src=get_master_rank())

    assert torch.equal(master_value, value), message


def spawn(*args, **kwargs):
    """ Wrapper for `torch.multiprocessing.spawn`.
    """
    num_cuda_devices = torch.cuda.device_count()
    # NOTE (michael): Without this assert, when `nprocs` is zero, `torch.multiprocessing.spawn`
    # crashes in a nondescript way.
    assert num_cuda_devices > 0, 'Unable to find CUDA devices.'
    torch.multiprocessing.spawn(
        *args,
        nprocs=num_cuda_devices,
        **kwargs,
    )
