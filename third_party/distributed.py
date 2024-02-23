import pickle

import torch
import torch.cuda
from torch.distributed.distributed_c10d import (
    _check_for_nccl_backend,
    _rank_not_in_group,
    _validate_output_list_for_rank,
    _warn_not_in_group,
    all_gather,
    gather,
    get_rank,
    get_world_size,
)


def _object_to_tensor(obj, device):
    bytes_ = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    byte_storage = torch.ByteStorage.from_buffer(bytes_)
    byte_tensor = torch.tensor(byte_storage, dtype=torch.uint8, device=device)
    local_size = torch.tensor(
        [byte_tensor.numel()], dtype=torch.long, device=device
    )
    return byte_tensor, local_size


def gather_object(obj, object_gather_list=None, dst=0, group=None):
    """
    Gathers picklable objects from the whole group in a single process.
    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank. (default is 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user's responsiblity to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    Example::
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
                gather_objects[dist.get_rank()],
                output if dist.get_rank() == 0 else None,
                dst=0
            )
        >>> # On rank 0
        >>> output
        ['foo', 12, {1: 2}]
    """
    if _rank_not_in_group(group):  # type: ignore
        _warn_not_in_group("gather_object")
        return

    # Ensure object_gather_list is specified appopriately.
    my_rank = get_rank()
    _validate_output_list_for_rank(my_rank, dst, object_gather_list)
    group_size = get_world_size(group=group)
    is_nccl = _check_for_nccl_backend(group)
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if is_nccl
        else torch.device("cpu")
    )
    cpu = torch.device("cpu")

    input_tensor, local_size = _object_to_tensor(obj, device)

    # NOTE: Gather all local sizes. This is so that we can find the max size, and index until the
    # correct size when deserializing the tensors.
    object_sizes_tensor = torch.zeros(
        group_size, dtype=torch.long, device=device
    )
    object_size_list = [
        object_sizes_tensor[i : i + 1] for i in range(group_size)
    ]

    # NOTE: Allgather tensor sizes. An all-gather is needed here despite this being a gather, since
    # each rank needs to broadcast a tensor of the same (maximal) size.
    all_gather(object_size_list, local_size, group=group)

    if object_sizes_tensor.device == cpu:
        object_sizes_tensor = object_sizes_tensor.to(cpu)
    object_sizes = object_sizes_tensor.tolist()
    max_object_size = int(max(object_sizes))

    # TODO: Does it make sense to resize them afterwards? Or does it make sense just to...
    # create them in the correct size initially?
    input_tensor.resize_(max_object_size)

    if my_rank != dst:
        gather(input_tensor, gather_list=None, dst=dst, group=group)
        return

    shape = (group_size, max_object_size)
    coalesced_output_tensor = torch.empty(
        shape, dtype=torch.uint8, device=device
    )
    gather_list = [coalesced_output_tensor[i] for i in range(group_size)]
    gather(input_tensor, gather_list=gather_list, dst=dst, group=group)
    if coalesced_output_tensor.device != cpu:
        coalesced_output_tensor = coalesced_output_tensor.to(cpu)
    coalesced_bytes = coalesced_output_tensor.view(-1).numpy().tobytes()
    assert object_gather_list is not None
    for i in range(group_size):
        # NOTE: Apply this potentially? https://github.com/pytorch/pytorch/issues/19143
        bytes_ = coalesced_bytes[
            max_object_size * i : max_object_size * (i + 1)
        ]
        object_gather_list[i] = pickle.loads(bytes_)
