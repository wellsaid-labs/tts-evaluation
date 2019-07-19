import logging
import random

from torchnlp.utils import shuffle as do_deterministic_shuffle

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


def distribute_batch_sampler(batch_sampler, batch_size, device, type_=torch.cuda):
    """ Split the batch sampler iterable between processes.

    NOTE: ``distribute_batch_sampler`` requires reading all samples in ``batch_sampler``. This
    decision was made to avoid any extra synchronization.

    Requires:
        - Distributed was initialized: ``is_initialized() == True``
        - The node device was set via: ``torch.cuda.set_device()``

    Args:
        batch_sampler (iterable or None): An iterable that returns batched dataset indicies. None
            for workers and iterable for master.
        batch_size (int): Note that batch size must be divisable by world size.
        device (torch.device)
        type_ (any): Default tensor type to use.

    Returns:
        (iterable): Batched dataset indicies reserved for this device. Each process has now
            a batch size of ``batch_size / world_size``.
    """
    world_size = torch.distributed.get_world_size()

    if is_master():
        batches = list(batch_sampler)
        # batches [num_batches, batch_size]
        batches = type_.LongTensor(batches)  # Compatible with ``torch.distributed.broadcast``

        # Batch sample invariants
        assert batches.shape[1] == batch_size
        assert len(batches.shape) == 2

        num_batches = batches.shape[0]
        num_batches = type_.LongTensor([num_batches])
    else:
        num_batches = type_.LongTensor(1)

    torch.distributed.broadcast(num_batches, src=get_master_rank())

    num_batches = int(num_batches)
    data_size = (num_batches, world_size, int(batch_size / world_size))
    if is_master():
        # Split data per process
        batches = batches.view(*data_size)
        torch.distributed.broadcast(batches, src=get_master_rank())
    else:
        batches = type_.LongTensor(*data_size)
        torch.distributed.broadcast(batches, src=get_master_rank())

    # LEARN MORE: https://github.com/pytorch/pytorch/issues/11734
    # ``device.index`` could be ``None`` and cause a silent failure.
    assert isinstance(device.index, int), 'device.index must be specified'
    return batches[:, device.index].cpu().tolist()


def random_shuffle(list_, type_=torch.cuda, random_seed=None):
    """ Shuffle randomly the same way across multiple process.

    Within a distributed setup each process has its own random generator and those random number
    generators might in different positions. This module will shuffle the same way in processes
    with out-of-sync number generators.

    Args:
        list_ (list)
        type_ (any, optional): Default tensor type to use.
        random_seed (int, optiona): If provided the shuffle is deterministic based on the seed
            instead of the global generator state.
    """
    if random_seed is not None:
        do_deterministic_shuffle(list_, random_seed=random_seed)
        return

    if not is_initialized():
        random.shuffle(list_)
        return

    # Broadcast a random seed from master
    random_seed = type_.LongTensor(1)
    if is_master():
        random_seed[0] = random.randint(1, 2**31)
    torch.distributed.broadcast(random_seed, src=get_master_rank())
    random_seed = int(random_seed)

    do_deterministic_shuffle(list_, random_seed=random_seed)
