import torch

from src.spectrogram_model.post_net import PostNet


def test_post_net():
    num_frames = 32
    batch_size = 5
    frame_channels = 60
    post_net = PostNet(frame_channels=frame_channels)

    # NOTE: spectrogram frames are around the range of 0 to 1
    input_ = torch.rand(num_frames, batch_size, frame_channels)
    mask = torch.ones(batch_size, num_frames, dtype=torch.bool)
    output = post_net(input_, mask)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (num_frames, batch_size, frame_channels)

    output.sum().backward()
