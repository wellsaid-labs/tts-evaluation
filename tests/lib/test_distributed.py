import enum
import functools
import math
import random
import tempfile
import time
import typing
from functools import partial
from unittest import mock

import pytest
import torch
import torch.distributed
import torch.nn
from torch.multiprocessing.spawn import spawn

import lib
from lib.distributed import ListedDict, NumeralizePadEmbed


def init_process_group(rank, nprocs, backend="gloo", init_method="tcp://127.0.0.1:23456"):
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=nprocs, rank=rank
    )
    assert lib.distributed.is_initialized()


def _is_initialized(rank, nprocs):
    """Helper function for `test_is_initialized*`."""
    init_process_group(rank=rank, nprocs=nprocs)
    assert lib.distributed.is_initialized()


def test_is_initialized():
    """Test `lib.distributed.is_initialized` returns `True` if distributed is initialized."""
    nprocs = 2
    spawn(functools.partial(_is_initialized, nprocs=nprocs), nprocs=nprocs)


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
    spawn(functools.partial(_is_master, nprocs=nprocs), nprocs=nprocs)


def test_is_master__not_initialized():
    """Test `lib.distributed.is_master` returns `True` if distributed is initialized."""
    assert lib.distributed.is_master()


def test_listed_dict():
    """Test `ListedDict` on it's basic operations."""
    data: ListedDict[str, int] = ListedDict()
    data.append([{"a": 1}, {"a": 2}, {"b": 11}, {"b": 22}])
    data.append([{"a": 2}, {"a": 3}])
    assert data["a"] == [[1, 2], [2, 3]]
    assert data[-1:]["a"] == [[2, 3]]
    assert sorted(list(data[-1:].keys())) == ["a", "b"]
    assert sorted(list(data.keys())) == ["a", "b"]
    assert sorted(list(data)) == ["a", "b"]
    assert "a" in data
    assert "b" in data
    assert "c" not in data
    assert len(data) == 2


def _dict_store_helper(rank, nprocs):
    init_process_group(rank, nprocs)
    store = lib.distributed.DictStore()
    unravel = lambda d: {k: d[k] for k in d}
    if rank == 0:
        store.update({"a": 1, "b": 1})
        assert unravel(store.data) == {
            "a": [[1, 2]],
            "b": [[1]],
            "c": [[2]],
        }

        store.update({"b": 1, "c": 1})
        assert unravel(store.data) == {
            "a": [[1, 2], []],
            "b": [[1], [1]],
            "c": [[2], [1, 2]],
            "d": [[], [2]],
        }

        store.update({"a": "a", "b": None})
        assert unravel(store.data) == {
            "a": [[1, 2], [], ["a"]],
            "b": [[1], [1], [None]],
            "c": [[2], [1, 2], [["c"]]],
            "d": [[], [2], [{"d": "d"}]],
        }

    if rank == 1:
        store.update({"a": 2, "c": 2})
        assert unravel(store.data) == {}

        store.update({"c": 2, "d": 2})
        assert unravel(store.data) == {}

        store.update({"c": ["c"], "d": {"d": "d"}})
        assert unravel(store.data) == {}

    assert [k in store.key_cache for k in ("a", "b", "c", "d")]


def test_dict_store():
    """Test `lib.distributed.DictStore` gathers data onto master."""
    nprocs = 2
    partial = functools.partial(_dict_store_helper, nprocs=nprocs)
    spawn(partial, nprocs=nprocs)


def _dict_store__speed_helper(rank, nprocs):
    init_process_group(rank, nprocs)
    store = lib.distributed.DictStore()
    data = {(i, str(i)): random.random() for i in range(3000)}
    event = time.perf_counter()
    for _ in range(5):
        store.update(data)
    timing = time.perf_counter() - event
    assert timing < 0.25, timing


def test_dict_store__speed():
    """Test `lib.distributed.DictStore` is fast based on a realistic workload."""
    nprocs = 4
    partial = functools.partial(_dict_store__speed_helper, nprocs=nprocs)
    spawn(partial, nprocs=nprocs)


def _dict_store__update_vocab_helper(rank, nprocs):
    init_process_group(rank, nprocs)

    store = lib.distributed.DictStore(cache_keys=False)
    [store.update({"a": 2}) for _ in range(10)]
    assert store.sync_every == 1

    store = lib.distributed.DictStore(cache_keys=True)
    [store.update({"a": 2}) for _ in range(10)]
    assert store.sync_every == 8


def test_dict_store__update_vocab():
    """Test `lib.distributed.DictStore` syncs less often if the vocab isn't updated."""
    nprocs = 4
    partial = functools.partial(_dict_store__update_vocab_helper, nprocs=nprocs)
    spawn(partial, nprocs=nprocs)


def test_numeralize_pad_embed__1d():
    """Test `NumeralizePadEmbed` in a basic training case with a 1-dimensional input."""
    model = NumeralizePadEmbed(100, 16)
    initial_vocab = model.vocab.copy()
    embedded, mask = model(["a"])
    assert torch.equal(embedded, model.embed(torch.tensor([2])))
    assert torch.equal(mask, torch.tensor([True]))
    assert model.vocab == {**initial_vocab, "a": 2}
    assert len(model._new_tokens) == 0


def test_numeralize_pad_embed__2d():
    """Test `NumeralizePadEmbed` in a basic training case with a 2-dimensional input."""
    model = NumeralizePadEmbed(100, 16)
    initial_vocab = model.vocab.copy()
    embedded, mask = model([["a"]])
    assert torch.equal(embedded, model.embed(torch.tensor([[2]])))
    assert torch.equal(mask, torch.tensor([[True]]))
    assert model.vocab == {**initial_vocab, "a": 2}
    assert len(model._new_tokens) == 0


def test_numeralize_pad_embed__no_updates():
    """Test `NumeralizePadEmbed` that timed updated have no impact on non-distributed training."""
    model = NumeralizePadEmbed(100, 16)
    initial_vocab = model.vocab.copy()
    model.update_every = 100
    embedded, mask = model([["a"]])
    embedded, mask = model([["b"]])
    assert torch.equal(embedded, model.embed(torch.tensor([[3]])))
    assert torch.equal(mask, torch.tensor([[True]]))
    assert model.vocab == {**initial_vocab, "a": 2, "b": 3}


def test_numeralize_pad_embed__padding():
    """Test `NumeralizePadEmbed` pads and masks the output correctly."""
    model = NumeralizePadEmbed(100, 16)
    initial_vocab = model.vocab.copy()

    embedded, mask = model([["a"]])
    assert torch.equal(embedded, model.embed(torch.tensor([[2]])))
    assert torch.equal(mask, torch.tensor([[True]]))

    embedded, mask = model([["a"], ["a", "b"]])
    assert torch.equal(embedded, model.embed(torch.tensor([[2, 2], [model.pad_idx, 3]])))
    assert torch.equal(mask, torch.tensor([[True, True], [False, True]]))

    assert model.vocab == {**initial_vocab, "a": 2, "b": 3}


def test_numeralize_pad_embed__allow_unk_on_eval():
    """Test `NumeralizePadEmbed` handles unknown tokens during evaluation and doesn't update
    vocab."""
    model = NumeralizePadEmbed(100, 16, allow_unk_on_eval=False)
    initial_vocab = model.vocab.copy()

    model.eval()
    with pytest.raises(KeyError):
        model([["a"]])
    assert model._unk_tokens == set()
    model.allow_unk_on_eval = True

    embedded, mask = model([["a"]])
    assert model._unk_tokens == {"a"}
    assert torch.equal(embedded, model.embed(torch.tensor([[model.unk_idx]])))
    assert torch.equal(mask, torch.tensor([[True]]))
    assert model.vocab == initial_vocab
    assert len(model._new_tokens) == 0

    model.train()
    model([[]])
    assert model._unk_tokens == set()


def test_numeralize_pad_embed__zero_length():
    """Test `NumeralizePadEmbed` can handle a zero length sequence."""
    model = NumeralizePadEmbed(100, 16)
    model.train(mode=False)
    embedded, mask = model([[]])
    assert embedded.shape == (0, 1, 16)
    assert mask.shape == (0, 1)


def test_numeralize_pad_embed__upate_tokens():
    """Test `NumeralizePadEmbed` update tokens can add/update new tokens and embeddings."""
    embedding_size = 16
    model = NumeralizePadEmbed(100, embedding_size)
    initial_vocab = model.vocab.copy()

    # Add new token
    model.update_tokens(["a"])
    assert model.vocab == {**initial_vocab, "a": 2}

    # Add new embedding
    embedding = torch.rand((1, embedding_size))
    model.update_tokens(["b"], embedding)
    assert model.vocab == {**initial_vocab, "a": 2, "b": 3}
    assert torch.allclose(model.weight[model.vocab["b"]], embedding)

    # Update existing embedding
    embedding = torch.rand((1, embedding_size))
    model.update_tokens(["a"], embedding)
    assert model.vocab == {**initial_vocab, "a": 2, "b": 3}
    assert torch.allclose(model.weight[model.vocab["a"]], embedding)


def test_numeralize_pad_embed__too_many_tokens():
    """Test `NumeralizePadEmbed` errors if too many tokens have been registered."""
    model = NumeralizePadEmbed(1, 16)
    model([["a"]])
    with pytest.raises(ValueError):
        model([["b"]])


def _init_numeralize_pad_embed(rank, nprocs, file_name, *args, **kwargs):
    """Initialize various objects for testing the `NumeralizePadEmbed` in a distributed
    context."""
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{file_name}", world_size=nprocs, rank=rank
    )
    model = NumeralizePadEmbed(*args, **kwargs)
    initial_vocab = model.vocab.copy()
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return initial_vocab, model, optimizer


def _spawn_helper(func, nprocs=2):
    """Spawn multiple processes for testing."""
    file_name = tempfile.mkstemp()[1]
    partial_ = partial(func, nprocs=nprocs, file_name=file_name)
    spawn(partial_, nprocs=nprocs)


def _numeralize_pad_embed__distributed_helper(rank, nprocs, file_name):
    initial_vocab, model, optimizer = _init_numeralize_pad_embed(rank, nprocs, file_name, 100, 16)
    for i in range(3):
        model = model.train(mode=True)
        optimizer.zero_grad()
        input_ = [list(range(100 * rank, 100 * rank + i + 1))]
        out, _ = model(input_)
        out.sum().backward()
        optimizer.step()
        model = model.train(mode=False)
        model(input_)
    expected = {**initial_vocab, 0: 2, 100: 3, 1: 4, 101: 5, 2: 6, 102: 7}
    assert typing.cast(NumeralizePadEmbed, model.module).vocab == expected
    assert len(typing.cast(NumeralizePadEmbed, model.module)._new_tokens) == 0


def test_numeralize_pad_embed__distributed():
    """Test `NumeralizePadEmbed` in a basic distributed training case."""
    _spawn_helper(_numeralize_pad_embed__distributed_helper)


def _numeralize_pad_embed__distributed_duplicate_tokens_helper(rank, nprocs, file_name):
    initial_vocab, model, optimizer = _init_numeralize_pad_embed(rank, nprocs, file_name, 100, 16)
    model = model.train(mode=True)
    optimizer.zero_grad()
    out, _ = model([list(range(4))])
    out.sum().backward()
    optimizer.step()
    expected = {**initial_vocab, 0: 2, 1: 3, 2: 4, 3: 5}
    assert typing.cast(NumeralizePadEmbed, model.module).vocab == expected
    assert len(typing.cast(NumeralizePadEmbed, model.module)._new_tokens) == 0


def test_numeralize_pad_embed__distributed_duplicate_tokens():
    """Test `NumeralizePadEmbed` syncs devices correctly which submit the same new token."""
    _spawn_helper(_numeralize_pad_embed__distributed_duplicate_tokens_helper)


def _numeralize_pad_embed__distributed_updates_helper(rank, nprocs, file_name):
    _, model, optimizer = _init_numeralize_pad_embed(rank, nprocs, file_name, 100, 16)
    side_effect = torch.distributed.all_gather_object
    with mock.patch("lib.utils.torch.distributed.all_gather_object") as all_gather_mock:
        all_gather_mock.side_effect = lambda *a, **k: side_effect(*a, **k)
        assert all_gather_mock.call_count == 0
        for i, update_every in zip(range(10), [1, 2, 2, 4, 4, 4, 4, 8, 8, 8]):
            model([["a"]])[0].sum().backward()
            assert all_gather_mock.call_count == math.log2(update_every) + 1
            assert typing.cast(NumeralizePadEmbed, model.module).update_every == update_every
            optimizer.step()


def test_numeralize_pad_embed__distributed_updates():
    """Test `NumeralizePadEmbed` calls `torch.distributed.all_gather_object` the right number of
    times."""
    _spawn_helper(_numeralize_pad_embed__distributed_updates_helper)


class NotSortable(enum.Enum):
    A = 1
    B = 2
    C = 3
    D = 4


def _numeralize_pad_embed__distributed_not_sortable(rank, nprocs, file_name):
    _, model, _ = _init_numeralize_pad_embed(rank, nprocs, file_name, 100, 16)
    random.seed(123 + rank)
    module = typing.cast(NumeralizePadEmbed, model.module)
    input_ = [random.choice(list(NotSortable)) for _ in range(100)]
    model(input_)
    outputs = [None for _ in range(lib.distributed.get_world_size())]
    torch.distributed.all_gather_object(outputs, module.vocab)
    assert all(module.vocab == o for o in outputs)


def test_numeralize_pad_embed__distributed_not_sortable():
    """Test `NumeralizePadEmbed` handles unsortable items that have different sorting from process
    to process. This is a regression test to ensure unsortable items are put in the vocab in the
    same order between different processes.
    """
    _spawn_helper(_numeralize_pad_embed__distributed_not_sortable, nprocs=4)
