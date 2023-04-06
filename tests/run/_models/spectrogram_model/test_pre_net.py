import torch

import run


def test_pre_net():
    """Test `run._models.spectrogram_model.pre_net.PreNet` handles a basic case."""
    num_frames = 32
    batch_size = 5
    size = 64
    num_frame_channels = 60
    seq_meta_embed_size = 128
    pre_net = run._models.spectrogram_model.pre_net.PreNet(
        size=size,
        num_frame_channels=num_frame_channels,
        seq_meta_embed_size=seq_meta_embed_size,
        num_layers=2,
        dropout=0.5,
    )
    input_ = torch.randn(num_frames, batch_size, num_frame_channels)
    seq_metadata = torch.randn(batch_size, seq_meta_embed_size)
    output = pre_net(input_, seq_metadata)
    assert output.dtype == torch.float
    assert output.shape == (num_frames, batch_size, size)
    output.sum().backward()