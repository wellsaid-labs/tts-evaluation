import logging

import torch

logger = logging.getLogger(__name__)


def in_use():
    """ Return if distributed is mode in use. """
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_master_rank():
    """ Returns the rank of the default master processs. The master process is the process that
    does one off tasks like saving checkpoints.
    """
    return 0


def is_master():
    """ Returns ``True`` if master process of distributed program """
    return torch.distributed.get_rank() == get_master_rank()


def broadcast_string(string, type_=torch.cuda):
    """ Broadcast from master a string to other processes.

    Args:
        string (str)
        type_ (any): Default tensor type to use.

    Returns:
        (str): String sent by master.
    """
    string = list(string)
    string = [ord(c) for c in string]  # Convert string into an integer representation

    if is_master():
        len_string = type_.LongTensor([len(string)])
    else:
        len_string = type_.LongTensor(1)

    torch.distributed.broadcast(len_string, src=get_master_rank())
    len_string = int(len_string)

    if is_master():
        string = type_.LongTensor(string)
    else:
        string = type_.LongTensor(len_string)

    torch.distributed.broadcast(string, src=get_master_rank())
    string = string.cpu().tolist()
    return ''.join([chr(c) for c in string])


def distribute_batch_sampler(batch_sampler, batch_size, device, type_=torch.cuda):
    """ Split the batch sampler iterable between processes.

    NOTE: For the purposes of step speed, this function does not compute the batches iteratively.
    This avoids extra synchronizations.

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
