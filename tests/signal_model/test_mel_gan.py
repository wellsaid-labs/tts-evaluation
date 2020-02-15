import torch

from src.signal_model.mel_gan import Generator


def test_mel_gan():
    batch_size = 4
    num_frames = 8
    frame_channels = 128

    generator = Generator(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    out = generator(spectrogram)

    assert out.shape == (batch_size, generator.scale_factor * num_frames)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()


def test_mel_gan__no_batch():
    num_frames = 9  # Test odd number of frames
    frame_channels = 128

    generator = Generator(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    out = generator(spectrogram)

    assert out.shape == (generator.scale_factor * num_frames,)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()
