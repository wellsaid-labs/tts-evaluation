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


def test_dict_store():
    """Test `lib.distributed.DictStore` gathers data onto master."""
    make_store = functools.partial(torch.distributed.TCPStore, "127.0.0.1", 29500, 1)
    main_store = lib.distributed.DictStore(make_store(is_master=True), 2, True, 0)
    store = lib.distributed.DictStore(make_store(is_master=False), 2, False, 1)

    store.update({"a": 2, "c": 2})
    main_store.update({"a": 1, "b": 1})
    assert main_store.data == {
        "a": [(1, 2)],
        "b": [(1,)],
        "c": [(2,)],
    }

    store.update({"c": 2, "d": 2})
    main_store.update({"b": 1, "c": 1})
    assert main_store.data == {
        "a": [(1, 2), tuple()],
        "b": [(1,), (1,)],
        "c": [(2,), (1, 2)],
        "d": [tuple(), (2,)],
    }

    store.update({"c": ["c"], "d": {"d": "d"}})
    main_store.update({"a": "a", "b": None})
    assert main_store.data == {
        "a": [(1, 2), tuple(), ("a",)],
        "b": [(1,), (1,), (None,)],
        "c": [(2,), (1, 2), (["c"],)],
        "d": [tuple(), (2,), ({"d": "d"},)],
    }

    # NOTE: Store data has been garbage collected.
    assert main_store._store.num_keys() == 1


def test_dict_store__order_of_operations():
    """Test `lib.distributed.DictStore` respects the order of operations."""
    make_store = functools.partial(torch.distributed.TCPStore, "127.0.0.1", 29500, 1)
    main_store = lib.distributed.DictStore(make_store(is_master=True), 2, True, 0)
    store = lib.distributed.DictStore(make_store(is_master=False), 2, False, 1)

    store.update({"a": 2, "c": 2})
    store.update({"a": 4, "c": 4})
    main_store.update({"a": 1, "b": 1})
    main_store.update({"a": 3, "b": 3})
    assert main_store.data == {
        "a": [(1, 2), (3, 4)],
        "b": [(1,), (3,)],
        "c": [(2,), (4,)],
    }
