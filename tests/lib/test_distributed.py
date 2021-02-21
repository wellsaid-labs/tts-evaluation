import functools

import torch
import torch.distributed
import torch.multiprocessing

import lib


def init_process_group(rank, nprocs, backend="gloo", init_method="tcp://127.0.0.1:23456"):
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=nprocs, rank=rank
    )


def _is_initialized(rank, nprocs):
    """ Helper function for `test_is_initialized*`. """
    init_process_group(rank=rank, nprocs=nprocs)
    assert lib.distributed.is_initialized()


def test_is_initialized():
    """ Test `lib.distributed.is_initialized` returns `True` if distributed is initialized."""
    nprocs = 2
    torch.multiprocessing.spawn(functools.partial(_is_initialized, nprocs=nprocs), nprocs=nprocs)


def test_is_initialized__not_initialized():
    """ Test `lib.distributed.is_initialized` returns `False` if distributed isn't initialized."""
    assert not lib.distributed.is_initialized()


def _is_master(rank, nprocs):
    """ Helper function for `test_is_master`. """
    init_process_group(rank=rank, nprocs=nprocs)
    assert lib.distributed.is_master() == (rank == lib.distributed.get_master_rank())


def test_is_master():
    """ Test `lib.distributed.is_master` differentiates master and worker processes. """
    nprocs = 2
    torch.multiprocessing.spawn(functools.partial(_is_master, nprocs=nprocs), nprocs=nprocs)


def test_is_master__not_initialized():
    """ Test `lib.distributed.is_master` returns `True` if distributed is initialized. """
    assert lib.distributed.is_master()
