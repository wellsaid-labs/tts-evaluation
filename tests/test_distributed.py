from unittest import mock

import torch

from src.distributed import broadcast_string
from src.distributed import distribute_batch_sampler
from src.distributed import get_master_rank
from src.distributed import is_master


@mock.patch('torch.distributed.get_rank', return_value=get_master_rank())
def test_is_master(_):
    assert is_master()


@mock.patch('torch.distributed.broadcast', return_value=None)
@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('torch.distributed.get_world_size', return_value=2)
def test_distribute_batch_sampler_master(_, __, ___):
    batch_sampler = [[1, 2, 3, 4], [5, 6, 7, 8]]
    batch_size = 4
    updated_batch_sampler = distribute_batch_sampler(batch_sampler, batch_size,
                                                     torch.device('cpu:0'))
    assert updated_batch_sampler == [[1, 2], [5, 6]]


@mock.patch('torch.distributed.broadcast')
@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('torch.distributed.get_world_size', return_value=2)
def test_distribute_batch_sampler_worker(mock_get_world_size, __, mock_broadcast):
    world_size = 2
    batch_size = 4
    device = torch.device('cpu:0')
    mock_get_world_size.return_value = world_size
    batch_sampler = [[1, 2, 3, 4], [5, 6, 7, 8]]
    batch_sampler = torch.tensor(batch_sampler).view(
        len(batch_sampler), int(batch_size / world_size), world_size)

    def mock_broadcast_side_effect(tensor, src):
        if len(tensor.shape) == 1:
            tensor.fill_(2)  # num_batches
        if len(tensor.shape) == 3:
            tensor.copy_(batch_sampler)  # batches

    mock_broadcast.side_effect = mock_broadcast_side_effect
    updated_batch_sampler = distribute_batch_sampler(batch_sampler, batch_size, device)
    assert updated_batch_sampler == [[1, 2], [5, 6]]


@mock.patch('src.distributed.is_master', return_value=True)
@mock.patch('torch.distributed.broadcast', return_value=None)
def test_broadcast_string(_, __):
    string = "this is a test !@#()(!@#/.'"
    device = torch.device('cpu:0')
    assert string == broadcast_string(string, device)


@mock.patch('src.distributed.is_master', return_value=False)
@mock.patch('torch.distributed.broadcast')
def test_broadcast_string_worker(mock_broadcast, __):
    string = "this is a test !@#()(!@#/.'"
    string_tensor = list(string)
    string_tensor = [ord(c) for c in string]
    string_tensor = torch.tensor(string_tensor)

    device = torch.device('cpu:0')

    def mock_broadcast_side_effect(tensor, src):
        if tensor.shape[0] == 1:
            tensor.fill_(len(string))  # num_batches
        else:
            tensor.copy_(string_tensor)  # batches

    mock_broadcast.side_effect = mock_broadcast_side_effect

    assert string == broadcast_string(string, device)
