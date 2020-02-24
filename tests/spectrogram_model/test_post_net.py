import torch

from src.spectrogram_model.post_net import PostNet


def test_post_net():
    num_frames = 32
    batch_size = 5
    frame_channels = 60
    pre_net = PostNet(frame_channels=frame_channels)

    # NOTE: spectrogram frames are around the range of 0 to 1
    input_ = torch.rand(batch_size, frame_channels, num_frames)
    mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    output = pre_net(input_, mask)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (batch_size, frame_channels, num_frames)

    output.sum().backward()
