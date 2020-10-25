import functools

import pytest
import torch

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


def _assert_synced(rank, nprocs, master_value, worker_value):
    """ Helper function for `test_assert_synced*`. """
    init_process_group(rank=rank, nprocs=nprocs)
    value = master_value if lib.distributed.is_master() else worker_value
    lib.distributed.assert_synced(value)


def test_assert_synced():
    """ Test `lib.distributed.assert_synced` doesn't error if the values do match. """
    nprocs = 2
    partial = functools.partial(_assert_synced, nprocs=nprocs, master_value=1, worker_value=1)
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_assert_synced__multiple_digits():
    """Test `lib.distributed.assert_synced` doesn't error if the values do match and have
    multiple digits."""
    nprocs = 2
    partial = functools.partial(_assert_synced, nprocs=nprocs, master_value=123, worker_value=123)
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_assert_synced__no_match():
    """ Test `lib.distributed.assert_synced` errors if the values don't match. """
    nprocs = 2
    partial = functools.partial(_assert_synced, nprocs=nprocs, master_value=1, worker_value=2)
    with pytest.raises(Exception):
        torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_assert_synced__no_match__wrong_length():
    """ Test `lib.distributed.assert_synced` errors if the values have different lengths. """
    nprocs = 2
    partial = functools.partial(_assert_synced, nprocs=nprocs, master_value=12, worker_value=123)
    with pytest.raises(Exception):
        torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_assert_synced__single_process():
    """ Test `lib.distributed.assert_synced` doesn't error if there is only a single process. """
    nprocs = 1
    partial = functools.partial(_assert_synced, nprocs=nprocs, master_value=12, worker_value=123)
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def assert_(
    rank,
    nprocs,
    callable_,
    expected,
    backend="gloo",
    init_method="tcp://127.0.0.1:23456",
    test_workers=False,
):
    """ `assert_` the result of a distributed operation. """
    init_process_group(rank=rank, nprocs=nprocs)
    return_ = callable_(rank)
    if test_workers or lib.distributed.is_master():
        assert return_ == expected


def test_reduce():
    """ Test `lib.distributed.reduce_` reduces values from work processes to the main process. """
    nprocs = 3
    partial = functools.partial(
        assert_, nprocs=nprocs, callable_=lib.distributed.reduce_, expected=3
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_gather():
    """ Test `lib.distributed.gather` gathers values from work processes to the main process. """
    nprocs = 2
    partial = functools.partial(
        assert_, nprocs=nprocs, callable_=lib.distributed.gather, expected=list(range(nprocs))
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_all_gather():
    """ Test `lib.distributed.all_gather` gathers values from work processes. """
    nprocs = 2
    partial = functools.partial(
        assert_,
        nprocs=nprocs,
        callable_=lib.distributed.all_gather,
        expected=list(range(nprocs)),
        test_workers=True,
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def _gather_list(rank):
    """ Helper function for `test_gather_list`. """
    return lib.distributed.gather_list([rank] * rank)


def test_gather_list():
    """ Test `lib.distributed.gather_list` gathers list from work processes to the main process. """
    nprocs = 3
    partial = functools.partial(
        assert_, nprocs=nprocs, callable_=_gather_list, expected=[[], [1], [2, 2]]
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)
