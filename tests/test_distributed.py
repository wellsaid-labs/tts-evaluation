from unittest import mock

import pytest
import torch

from src.distributed import distribute_batch_sampler
from src.distributed import get_master_rank
from src.distributed import is_initialized
from src.distributed import is_master
from src.distributed import random_shuffle
from src.distributed import assert_synced


@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('src.distributed.torch.distributed')
def test_assert_synced__master(mock_distributed, _):
    mock_distributed.broadcast.return_value = None
    assert_synced(123, '', type_=torch)


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('src.distributed.torch.distributed')
def test_assert_synced__worker(mock_distributed, _):

    def mock_broadcast_side_effect(tensor, src):
        if tensor.shape[0] == 1:
            tensor.copy_(torch.tensor([3]))
        else:
            tensor.copy_(torch.tensor([1, 2, 3]))

    mock_distributed.broadcast.side_effect = mock_broadcast_side_effect
    assert_synced(123, '', type_=torch)


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('src.distributed.torch.distributed')
def test_assert_synced__worker__wrong_length(mock_distributed, _):

    def mock_broadcast_side_effect(tensor, src):
        if tensor.shape[0] == 1:
            tensor.copy_(torch.tensor([3]))
        else:
            tensor.copy_(torch.tensor([1, 2, 3]))

    mock_distributed.broadcast.side_effect = mock_broadcast_side_effect

    with pytest.raises(AssertionError):
        assert_synced(12, type_=torch)


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('src.distributed.torch.distributed')
def test_assert_synced__worker__wrong_value(mock_distributed, _):

    def mock_broadcast_side_effect(tensor, src):
        if tensor.shape[0] == 1:
            tensor.copy_(torch.tensor([3]))
        else:
            tensor.copy_(torch.tensor([1, 2, 3]))

    mock_distributed.broadcast.side_effect = mock_broadcast_side_effect

    with pytest.raises(AssertionError):
        assert_synced(124, type_=torch)


def test_random_shuffle__deterministic():
    list_ = list(range(1000))
    random_shuffle(list_, random_seed=123)
    assert list_[:10] == [178, 830, 641, 410, 255, 470, 433, 580, 257, 450]


def test_random_shuffle__non_distributed__non_deterministic():
    list_ = list(range(1000))
    random_shuffle(list_)

    list_other = list(range(1000))
    random_shuffle(list_)

    assert list_ != list_other


@mock.patch('src.distributed.is_initialized', return_value=True)
@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('src.distributed.torch.distributed')
def test_random_shuffle__distributed__non_deterministic(mock_distributed, _, __):
    mock_distributed.broadcast.return_value = None

    list_ = list(range(1000))
    random_shuffle(list_, type_=torch)

    list_other = list(range(1000))
    random_shuffle(list_, type_=torch)

    assert list_ != list_other


@mock.patch('src.distributed.torch.distributed')
def test_is_master(mock_distributed):
    mock_distributed.get_rank.return_value = get_master_rank()
    assert is_master()


def test_is_master__not_initialized():
    # Defaults to ``is_master() == True``
    assert not is_initialized()
    assert is_master()


@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('src.distributed.torch.distributed')
def test_distribute_batch_sampler_master(mock_distributed, _):
    world_size = 2
    batch_sampler = [[1, 2, 3, 4], [5, 6, 7, 8]]
    batch_size = 4
    mock_distributed.get_world_size.return_value = world_size
    mock_distributed.broadcast.return_value = None
    updated_batch_sampler = distribute_batch_sampler(batch_sampler, batch_size,
                                                     torch.device('cpu:0'), torch)
    assert updated_batch_sampler == [[1, 2], [5, 6]]


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('src.distributed.torch.distributed')
def test_distribute_batch_sampler_worker(mock_distributed, _):
    world_size = 2
    batch_size = 4
    device = torch.device('cpu:0')
    mock_distributed.get_world_size.return_value = world_size
    batch_sampler = [[1, 2, 3, 4], [5, 6, 7, 8]]
    batch_sampler = torch.tensor(batch_sampler).view(
        len(batch_sampler), int(batch_size / world_size), world_size)

    def mock_broadcast_side_effect(tensor, src):
        if len(tensor.shape) == 1:
            tensor.fill_(2)  # num_batches
        if len(tensor.shape) == 3:
            tensor.copy_(batch_sampler)  # batches

    mock_distributed.broadcast.side_effect = mock_broadcast_side_effect
    updated_batch_sampler = distribute_batch_sampler(batch_sampler, batch_size, device, torch)
    assert updated_batch_sampler == [[1, 2], [5, 6]]


def mock_func(a):
    return a**2
