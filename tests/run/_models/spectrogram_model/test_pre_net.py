import torch

import run


def test_pre_net():
    """Test `run._models.spectrogram_model.pre_net.PreNet` handles a basic case."""
    num_frames = 32
    batch_size = 5
    hidden_size = 64
    num_frame_channels = 60
    pre_net = run._models.spectrogram_model.pre_net.PreNet(
        hidden_size=hidden_size, num_frame_channels=num_frame_channels, num_layers=2, dropout=0.5
    )
    input_ = torch.randn(num_frames, batch_size, num_frame_channels)
    output, hidden_state = pre_net(input_)
    assert output.dtype == torch.float
    assert output.shape == (num_frames, batch_size, hidden_size)
    assert isinstance(hidden_state, tuple)
    output.sum().backward()
