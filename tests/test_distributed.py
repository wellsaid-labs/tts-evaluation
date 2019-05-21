from unittest import mock

import torch

from src.distributed import broadcast_string
from src.distributed import distribute_batch_sampler
from src.distributed import get_master_rank
from src.distributed import is_master
from src.distributed import is_initialized
from src.distributed import map_multiprocess


@mock.patch('torch.distributed')
def test_is_master(mock_distributed):
    mock_distributed.get_rank.return_value = get_master_rank()
    assert is_master()


def test_is_master__not_initialized():
    # Defaults to ``is_master() == True``
    assert not is_initialized()
    assert is_master()


@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('torch.distributed')
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
@mock.patch('torch.distributed')
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


@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('torch.distributed')
def test_broadcast_string(mock_distributed, __):
    mock_distributed.broadcast.return_value = None
    string = "this is a test !@#()(!@#/.'"
    assert string == broadcast_string(string, torch)


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('torch.distributed')
def test_broadcast_string_worker(mock_distributed, __):
    string = "this is a test !@#()(!@#/.'"
    string_tensor = list(string)
    string_tensor = [ord(c) for c in string]
    string_tensor = torch.tensor(string_tensor)

    def mock_broadcast_side_effect(tensor, src):
        if tensor.shape[0] == 1:
            tensor.fill_(len(string))  # num_batches
        else:
            tensor.copy_(string_tensor)  # batches

    mock_distributed.broadcast.side_effect = mock_broadcast_side_effect

    assert string == broadcast_string(string, torch)


def mock_func(a):
    return a**2


def test_map_multiprocess():
    expected = [1, 4, 9]
    processed = map_multiprocess([1, 2, 3], mock_func)
    assert expected == processed


@mock.patch('src.distributed.is_master', return_value=False)
def test_map_multiprocess_master(_):
    expected = [1, 4, 9]
    processed = map_multiprocess([1, 2, 3], mock_func)
    assert expected == processed
