from collections import Counter
from unittest import mock

import pytest
import random
import torch

from torch.utils.data.sampler import BatchSampler
from torch.utils.data.sampler import SequentialSampler

from src.environment import fork_rng
from src.environment import fork_rng_wrap
from src.environment import set_seed
from src.samplers import BalancedSampler
from src.samplers import BucketBatchSampler
from src.samplers import DeterministicSampler
from src.samplers import DistributedBatchSampler
from src.samplers import get_number_of_elements
from src.samplers import OomBatchSampler
from src.samplers import RepeatSampler


def test_repeat_sampler():
    sampler = RepeatSampler([1])
    iterator = iter(sampler)
    assert next(iterator) == 1
    assert next(iterator) == 1
    assert next(iterator) == 1


def test_oom_batch_sampler():
    data = list(range(-4, 14))
    sampler = SequentialSampler(data)
    batch_sampler = BatchSampler(sampler, 4, False)
    oom_sampler = OomBatchSampler(batch_sampler, lambda i: data[i], num_batches=3)
    list_ = list(oom_sampler)
    # The largest batches are first
    assert [data[i] for i in list_[0]] == [8, 9, 10, 11]
    assert [data[i] for i in list_[1]] == [12, 13]
    assert [data[i] for i in list_[2]] == [4, 5, 6, 7]
    assert len(list_) == 5


def test_get_number_of_elements():
    assert get_number_of_elements([torch.randn(5, 5), torch.randn(4, 4)]) == 41


@mock.patch('src.samplers.src.distributed.is_initialized', return_value=True)
def test_distributed_batch_sampler(_):
    sampler = SequentialSampler(list(range(15)))
    batch_sampler = BatchSampler(sampler, 10, False)

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=0)
    assert list(distributed_sampler) == [[0, 4, 8], [10, 14]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=1)
    assert list(distributed_sampler) == [[1, 5, 9], [11]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=2)
    assert list(distributed_sampler) == [[2, 6], [12]]
    assert len(distributed_sampler) == 2

    distributed_sampler = DistributedBatchSampler(batch_sampler, num_replicas=4, rank=3)
    assert list(distributed_sampler) == [[3, 7], [13]]
    assert len(distributed_sampler) == 2


@fork_rng_wrap(seed=123)
def test_bucket_batch_sampler():
    sampler = SequentialSampler(list(range(10)))
    batch_sampler = BucketBatchSampler(
        sampler, batch_size=3, drop_last=False, bucket_size_multiplier=2)
    assert len(batch_sampler) == 4
    assert list(batch_sampler) == [[0, 1, 2], [3, 4, 5], [9], [6, 7, 8]]


def test_bucket_batch_sampler__drop_last():
    sampler = SequentialSampler(list(range(10)))
    batch_sampler = BucketBatchSampler(
        sampler, batch_size=3, drop_last=True, bucket_size_multiplier=2)
    assert len(batch_sampler) == 3
    assert len(list(iter(batch_sampler))) == 3


# NOTE: `fork_rng_wrap` to ensure the tests never randomly fail due to an rare sampling.
@fork_rng_wrap(seed=123)
def test_balanced_sampler():
    data = ['a', 'a', 'b', 'b', 'b', 'c']
    num_samples = 10000
    sampler = BalancedSampler(data, replacement=True, num_samples=num_samples)
    assert len(sampler) == num_samples
    samples = [data[i] for i in sampler]
    assert len(samples) == num_samples
    counts = Counter(samples)
    assert counts['a'] / num_samples == pytest.approx(.33, 0.2)
    assert counts['b'] / num_samples == pytest.approx(.33, 0.2)
    assert counts['c'] / num_samples == pytest.approx(.33, 0.2)


@fork_rng_wrap(seed=123)
def test_balanced_sampler__weighted():
    data = [('a', 0), ('a', 1), ('a', 2), ('b', 2), ('c', 1)]
    num_samples = 10000
    sampler = BalancedSampler(
        data,
        replacement=True,
        num_samples=num_samples,
        get_weight=lambda e: e[1],
        get_class=lambda e: e[0])
    samples = [data[i] for i in sampler]
    counts = Counter(samples)
    assert counts[('a', 2)] / num_samples == pytest.approx(.22, 0.2)
    assert counts[('a', 1)] / num_samples == pytest.approx(.11, 0.2)
    assert counts[('a', 0)] / num_samples == 0.0
    assert counts[('b', 2)] / num_samples == pytest.approx(.33, 0.2)
    assert counts[('c', 1)] / num_samples == pytest.approx(.33, 0.2)


def test_deterministic_sampler__nondeterministic_iter():
    with fork_rng(seed=123):
        data = [random.randint(1, 100) for i in range(100)]

    sampler = DeterministicSampler(BalancedSampler(data), random_seed=123)
    assert len(sampler) == len(data)
    samples = [data[i] for i in sampler]
    assert samples[:10] == [3, 35, 99, 43, 67, 82, 66, 68, 100, 14]

    # NOTE: Each iteration is new sample from `sampler`; however, the entire sequence of iterations
    # is deterministic based on the `random_seed=123`
    new_samples = [data[i] for i in sampler]
    assert samples != new_samples


def test_deterministic_sampler__nondeterministic_next():

    class _Sampler():

        def __iter__(self):
            for _ in range(100):
                yield random.randint(1, 100)

    sampler = DeterministicSampler(_Sampler(), random_seed=123)
    assert list(sampler)[:10] == [7, 35, 12, 99, 53, 35, 14, 5, 49, 69]


def test_deterministic_sampler__side_effects():
    """ Ensure that the sampler does not affect random generation after it's finished. """
    set_seed(123)
    pre_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    sampler = DeterministicSampler(list(range(10)))
    list(iter(sampler))

    post_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    set_seed(123)
    assert pre_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
    assert post_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]


@fork_rng_wrap(seed=123)
def test_balanced_sampler__nondeterministic():
    data = [random.randint(1, 100) for i in range(100)]
    sampler = BalancedSampler(data)
    samples = [data[i] for i in sampler]
    new_samples = [data[i] for i in sampler]
    assert new_samples != samples


# TODO: Remove
# def test_balance_list():
#     balanced = balance_list(['a', 'a', 'b', 'b', 'c'])
#     assert len(balanced) == 3
#     assert len(set(balanced)) == 3

# def test_balance_list__determinism():
#     """ Test to ensure that `balance_list` is deterministic when `random_seed` is provided. """
#     random_ = random.Random(123)
#     list_ = [(i, random_.choice('abcde')) for i in range(99)]
#     balanced = balance_list(list_, get_class=lambda i: i[1], random_seed=123)
#     assert len(balanced) == 70
#     count = Counter([e[1] for e in balanced])
#     assert len(set(count.values())) == 1  # Ensure that the list is balanced.
#     assert [e[0] for e in balanced[:10]] == [7, 33, 62, 51, 14, 50, 19, 73, 56, 21]

# def test_balance_list__nondeterminism():
#     """ Test to ensure that `balance_list` is not deterministic when `random_seed` is not
#     provided.
#     """
#     random_ = random.Random(123)
#     list_ = [(i, random_.choice('abcde')) for i in range(10000)]
#     balanced = balance_list(list_, get_class=lambda i: i[1])

#     list_other = deepcopy(list_)
#     balanced_other = balance_list(list_other, get_class=lambda i: i[1])
#     # NOTE: This test should fail one in 10000 times.
#     assert balanced_other[0][0] != balanced[0][0]

# def test_balance_list__get_weight():
#     list_ = [(1, 'a'), (1, 'a'), (2, 'b')]
#     balanced = balance_list(list_, get_class=lambda i: i[1], get_weight=lambda i: i[0])
#     assert len(balanced) == 3
