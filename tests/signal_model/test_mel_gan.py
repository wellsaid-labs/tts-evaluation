import torch

from src.signal_model.mel_gan import Generator


def test_mel_gan():
    batch_size = 4
    num_frames = 8
    frame_channels = 18

    generator = Generator(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    out = generator(spectrogram)

    assert out.shape == (
        batch_size,
        generator.hop_length * num_frames,
    )

    out.sum().backward()


def test_mel_gan__no_batch():
    num_frames = 8
    frame_channels = 18

    generator = Generator(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    out = generator(spectrogram)

    assert out.shape == (generator.hop_length * num_frames,)

    out.sum().backward()
