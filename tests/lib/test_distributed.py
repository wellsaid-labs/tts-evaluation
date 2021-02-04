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
    """ Test `lib.distributed.reduce` reduces values from work processes to the main process. """
    nprocs = 3
    partial = functools.partial(lib.distributed.reduce, device=torch.device("cpu"))
    partial = functools.partial(assert_, nprocs=nprocs, callable_=partial, expected=3)
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_reduce_identity():
    """ Test `lib.distributed.reduce` functions outside of a distributed environment. """
    assert lib.distributed.reduce(0, torch.device("cpu")) == 0


def test_all_gather():
    """ Test `lib.distributed.all_gather` gathers values from work processes. """
    nprocs = 2
    partial = functools.partial(
        assert_,
        nprocs=nprocs,
        callable_=functools.partial(lib.distributed.all_gather, device=torch.device("cpu")),
        expected=list(range(nprocs)),
        test_workers=True,
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_all_gather__identity():
    """ Test `lib.distributed.all_gather` functions outside of a distributed environment. """
    assert lib.distributed.all_gather(0, torch.device("cpu")) == [0]


def _gather_list(rank):
    """ Helper function for `test_gather_list`. """
    return lib.distributed.gather_list([rank] * rank, torch.device("cpu"))


def test_gather_list():
    """ Test `lib.distributed.gather_list` gathers list from work processes to the main process. """
    nprocs = 3
    partial = functools.partial(
        assert_, nprocs=nprocs, callable_=_gather_list, expected=[[], [1], [2, 2]]
    )
    torch.multiprocessing.spawn(partial, nprocs=nprocs)


def test_gather_list__identity():
    """ Test `lib.distributed.gather_list` functions outside of a distributed environment. """
    assert lib.distributed.gather_list([0], torch.device("cpu")) == [[0]]
