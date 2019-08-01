import torch

from src.environment import TEST_DATA_PATH
from src.utils.on_disk_tensor import OnDiskTensor

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'utils'


def test_on_disk_tensor():
    original = torch.rand(4, 10)
    tensor = OnDiskTensor.from_tensor(str(TEST_DATA_PATH_LOCAL / 'tensor.npy'), original)
    assert tensor.shape == original.shape
    assert tensor.shape == original.shape  # Smoke test caching
    assert torch.equal(tensor.to_tensor(), original)
    assert tensor.exists()
    tensor.unlink()


def test_on_disk_tensor_eq():
    assert OnDiskTensor(str(TEST_DATA_PATH_LOCAL / 'tensor.npy')) == OnDiskTensor(
        str(TEST_DATA_PATH_LOCAL / 'tensor.npy'))
    assert OnDiskTensor(str(TEST_DATA_PATH_LOCAL / 'other_tensor.npy')) != OnDiskTensor(
        str(TEST_DATA_PATH_LOCAL / 'tensor.npy'))


def test_on_disk_tensor_hash():
    assert hash(OnDiskTensor(str(TEST_DATA_PATH_LOCAL / 'tensor.npy'))) == hash(
        OnDiskTensor(str(TEST_DATA_PATH_LOCAL / 'tensor.npy')))
