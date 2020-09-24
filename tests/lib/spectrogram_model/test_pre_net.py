import torch

import lib.spectrogram_model.pre_net


def test_pre_net():
    """ Test `lib.spectrogram_model.pre_net.PreNet` handles a basic case. """
    num_frames = 32
    batch_size = 5
    hidden_size = 64
    frame_channels = 60
    pre_net = lib.spectrogram_model.pre_net.PreNet(
        hidden_size=hidden_size, frame_channels=frame_channels, num_layers=2, dropout=0.5)
    input_ = torch.randn(num_frames, batch_size, frame_channels)
    output = pre_net(input_)
    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (num_frames, batch_size, hidden_size)
    output.sum().backward()
