import torch

from src.spectrogram_model.pre_net import PreNet


def test_pre_net():
    num_frames = 32
    batch_size = 5
    hidden_size = 64
    frame_channels = 60
    pre_net = PreNet(hidden_size=hidden_size, frame_channels=frame_channels)

    # NOTE: spectrogram frames are around the range of 0 to 1
    input_ = torch.rand(num_frames, batch_size, frame_channels)
    output = pre_net(input_)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (num_frames, batch_size, hidden_size)

    output.sum().backward()
