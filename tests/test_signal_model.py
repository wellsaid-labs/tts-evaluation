import torch

from src.signal_model import SignalModel
from src.signal_model import SpectrogramDiscriminator
from src.signal_model import trim


def test_spectrogram_discriminator():
    batch_size = 4
    num_frames = 16
    fft_length = 1024
    num_mel_bins = 128
    discriminator = SpectrogramDiscriminator(fft_length, num_mel_bins)
    spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_mel_spectrogram = torch.randn(batch_size, num_frames, num_mel_bins)
    output = discriminator(spectrogram, db_spectrogram, db_mel_spectrogram)
    assert output.shape == (batch_size,)
    output.sum().backward()


def test_trim():
    a, b = trim(torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3]), dim=0)
    assert torch.equal(a, torch.tensor([2, 3]))
    assert torch.equal(b, torch.tensor([2, 3]))


def test_signal_model():
    batch_size = 4
    num_frames = 8
    frame_channels = 128

    model = SignalModel(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    out = model(spectrogram)

    assert out.shape == (batch_size, model.upscale_factor * num_frames)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()


def test_signal_model__no_batch():
    num_frames = 9  # Test odd number of frames
    frame_channels = 128

    model = SignalModel(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    out = model(spectrogram)

    assert out.shape == (model.upscale_factor * num_frames,)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()
