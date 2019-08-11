from unittest import mock

import pytest
import torch

from src.distributed import get_master_rank
from src.distributed import is_initialized
from src.distributed import is_master
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


@mock.patch('src.distributed.torch.distributed')
def test_is_master(mock_distributed):
    mock_distributed.get_rank.return_value = get_master_rank()
    assert is_master()


def test_is_master__not_initialized():
    # Defaults to ``is_master() == True``
    assert not is_initialized()
    assert is_master()


def mock_func(a):
    return a**2
