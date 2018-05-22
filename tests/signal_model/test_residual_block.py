import torch

from src.signal_model.residual_block import ResidualBlock


def test_residual_block():
    hidden_size = 64
    batch_size = 2
    num_samples = 69
    skip_size = 32
    input_ = torch.FloatTensor(hidden_size, batch_size, num_samples)
    conditional = torch.FloatTensor(hidden_size * 2, batch_size, num_samples)

    block = ResidualBlock(hidden_size=hidden_size, skip_size=skip_size, kernel_size=2)
    out, skip = block(input_, conditional)

    assert out.shape == (hidden_size, batch_size, num_samples)
    assert skip.shape == (batch_size, skip_size, num_samples)

    # Smoke test back prop
    (out.sum() + skip.sum()).backward()

    # Test ``kernel_size`` = 3
    print(input_.size())
    print(conditional.size())
    block = ResidualBlock(hidden_size=hidden_size, skip_size=skip_size, kernel_size=3)
    out, skip = block(input_, conditional)

    assert out.shape == (hidden_size, batch_size, num_samples)
    assert skip.shape == (batch_size, skip_size, num_samples)

    # Smoke test back prop
    (out.sum() + skip.sum()).backward()
