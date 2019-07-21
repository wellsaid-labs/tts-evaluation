import torch

from src.utils.on_disk_tensor import OnDiskTensor


def test_on_disk_tensor():
    original = torch.rand(4, 10)
    tensor = OnDiskTensor.from_tensor('tests/_test_data/tensor.npy', original)
    assert tensor.shape == original.shape
    assert tensor.shape == original.shape  # Smoke test caching
    assert torch.equal(tensor.to_tensor(), original)
    assert tensor.exists()
    tensor.unlink()


def test_on_disk_tensor_eq():
    assert OnDiskTensor('tests/_test_data/tensor.npy') == OnDiskTensor(
        'tests/_test_data/tensor.npy')
    assert OnDiskTensor('tests/_test_data/other_tensor.npy') != OnDiskTensor(
        'tests/_test_data/tensor.npy')


def test_on_disk_tensor_hash():
    assert hash(OnDiskTensor('tests/_test_data/tensor.npy')) == hash(
        OnDiskTensor('tests/_test_data/tensor.npy'))
