import torch

import lib.spectrogram_model.pre_net


def test_pre_net():
    """Test `lib.spectrogram_model.pre_net.PreNet` handles a basic case."""
    num_frames = 32
    batch_size = 5
    size = 64
    num_frame_channels = 60
    speaker_embedding_size = 128
    pre_net = lib.spectrogram_model.pre_net.PreNet(
        size=size,
        num_frame_channels=num_frame_channels,
        speaker_embedding_size=speaker_embedding_size,
        num_layers=2,
        dropout=0.5,
    )
    input_ = torch.randn(num_frames, batch_size, num_frame_channels)
    speaker = torch.randn(batch_size, speaker_embedding_size)
    output = pre_net(input_, speaker)
    assert output.dtype == torch.float
    assert output.shape == (num_frames, batch_size, size)
    output.sum().backward()