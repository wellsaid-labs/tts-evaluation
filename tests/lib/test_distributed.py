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
    """Helper function for `test_is_initialized*`."""
    init_process_group(rank=rank, nprocs=nprocs)
    assert lib.distributed.is_initialized()


def test_is_initialized():
    """Test `lib.distributed.is_initialized` returns `True` if distributed is initialized."""
    nprocs = 2
    torch.multiprocessing.spawn(functools.partial(_is_initialized, nprocs=nprocs), nprocs=nprocs)


def test_is_initialized__not_initialized():
    """Test `lib.distributed.is_initialized` returns `False` if distributed isn't initialized."""
    assert not lib.distributed.is_initialized()


def _is_master(rank, nprocs):
    """Helper function for `test_is_master`."""
    init_process_group(rank=rank, nprocs=nprocs)
    assert lib.distributed.is_master() == (rank == lib.distributed.get_master_rank())


def test_is_master():
    """Test `lib.distributed.is_master` differentiates master and worker processes."""
    nprocs = 2
    torch.multiprocessing.spawn(functools.partial(_is_master, nprocs=nprocs), nprocs=nprocs)


def test_is_master__not_initialized():
    """Test `lib.distributed.is_master` returns `True` if distributed is initialized."""
    assert lib.distributed.is_master()


def _dict_store_helper(rank, nprocs, backend="gloo", init_method="tcp://127.0.0.1:23456"):
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=nprocs, rank=rank
    )
    assert lib.distributed.is_initialized()
    store = lib.distributed.DictStore()

    if rank == 0:
        store.update({"a": 1, "b": 1})
        assert store.data == {
            "a": [(1, 2)],
            "b": [(1,)],
            "c": [(2,)],
        }

        store.update({"b": 1, "c": 1})
        assert store.data == {
            "a": [(1, 2), tuple()],
            "b": [(1,), (1,)],
            "c": [(2,), (1, 2)],
            "d": [tuple(), (2,)],
        }

        store.update({"a": "a", "b": None})
        assert store.data == {
            "a": [(1, 2), tuple(), ("a",)],
            "b": [(1,), (1,), (None,)],
            "c": [(2,), (1, 2), (["c"],)],
            "d": [tuple(), (2,), ({"d": "d"},)],
        }

    if rank == 1:
        store.update({"a": 2, "c": 2})
        assert store.data == {}

        store.update({"c": 2, "d": 2})
        assert store.data == {}

        store.update({"c": ["c"], "d": {"d": "d"}})
        assert store.data == {}


def test_dict_store():
    """Test `lib.distributed.DictStore` gathers data onto master."""
    nprocs = 2
    partial = functools.partial(_dict_store_helper, nprocs=nprocs)
    torch.multiprocessing.spawn(partial, nprocs=nprocs)
