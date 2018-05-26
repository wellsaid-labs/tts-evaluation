import torch

from src.feature_model.post_net import PostNet


def test_post_net():
    num_frames = 32
    batch_size = 5
    frame_channels = 60
    pre_net = PostNet(frame_channels=frame_channels)

    # NOTE: spectrogram frames are around the range of 0 to 1
    input_ = torch.autograd.Variable(
        torch.FloatTensor(batch_size, frame_channels, num_frames).uniform_(0, 1))
    output = pre_net(input_)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (batch_size, frame_channels, num_frames)
