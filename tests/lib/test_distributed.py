from unittest import mock

import pytest
import torch

import lib


def test_is_master__not_initialized():
    """Test `lib.distributed.is_master` and `lib.distributed.is_initialized`  base case on a
    machine without CUDA.
    """
    assert not torch.cuda.is_available()
    assert not lib.distributed.is_initialized()
    assert lib.distributed.is_master()


@mock.patch("lib.distributed.is_initialized", return_value=True)
@mock.patch("lib.distributed.torch.distributed")
def test_is_master(mock_distributed, _):
    """ Test `lib.distributed.is_master` if distributed is initialized. """
    mock_distributed.get_rank.return_value = lib.distributed.get_master_rank()
    assert lib.distributed.is_master()


@mock.patch("lib.distributed.is_master", return_value=True)
@mock.patch("lib.distributed.torch.distributed")
def test_assert_synced__master(mock_distributed, _):
    """ Test `lib.distributed.assert_synced` passes for master process. """
    mock_distributed.broadcast.return_value = None
    lib.distributed.assert_synced(123, "")


def _mock_broadcast_side_effect(tensor, src):
    """ Mock broadcast of `length` 3 and value `123`. """
    if tensor.shape[0] == 1:
        tensor.copy_(torch.tensor([3]))
    else:
        tensor.copy_(torch.tensor([1, 2, 3]))


@mock.patch("lib.distributed.is_master", return_value=False)
@mock.patch("lib.distributed.torch.distributed")
def test_assert_synced__worker(mock_distributed, _):
    """ Test `lib.distributed.assert_synced` passes for worker process with the correct value. """
    mock_distributed.broadcast.side_effect = _mock_broadcast_side_effect
    lib.distributed.assert_synced(123, "")


@mock.patch("lib.distributed.is_master", return_value=False)
@mock.patch("lib.distributed.torch.distributed")
def test_assert_synced__worker__wrong_value(mock_distributed, _):
    """ Test `lib.distributed.assert_synced` failes for worker process with the wrong value. """
    mock_distributed.broadcast.side_effect = _mock_broadcast_side_effect
    with pytest.raises(AssertionError):
        lib.distributed.assert_synced(124)


@mock.patch("lib.distributed.is_master", return_value=False)
@mock.patch("lib.distributed.torch.distributed")
def test_assert_synced__worker__wrong_length(mock_distributed, _):
    """ Test `lib.distributed.assert_synced` failes for worker process with the wrong value. """
    mock_distributed.broadcast.side_effect = _mock_broadcast_side_effect
    with pytest.raises(AssertionError):
        lib.distributed.assert_synced(12)


def test_spawn():
    """ Test `lib.distributed.spawn` base case on a machine without CUDA. """
    assert not torch.cuda.is_available()
    with pytest.raises(AssertionError):
        lib.distributed.spawn()


@mock.patch("torch.distributed")
def test_distributed_average_metric(mock_distributed):
    """ Test `DistributedAverage` is able to track the average afer a `sync`. """
    mock_distributed.reduce.return_value = None
    metric = lib.distributed.DistributedAverage()
    assert metric.last_update_value is None
    metric.update(torch.tensor([0.5]), torch.tensor(3))
    metric.update(0.25, 2)
    assert metric.last_update_value == 0.25
    assert metric.sync().last_update_value == 0.4
    assert metric.reset() == 0.4
    assert metric.last_update_value is None
