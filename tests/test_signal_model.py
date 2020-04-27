import itertools

import numpy
import torch

from src.signal_model import generate_waveform
from src.signal_model import SignalModel
from src.signal_model import SpectrogramDiscriminator


def test_generate_waveform():
    """ Test if incremental generation produces the same output as none-incremental. """
    batch_size = 2
    num_frames = 53
    frame_channels = 6

    model = SignalModel(input_size=frame_channels, ratios=[2, 2], max_channel_size=32, padding=13)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (batch_size, model.upscale_factor * num_frames)

    for i in itertools.chain([1, 26, 27, 53]):
        generated = torch.cat(list(generate_waveform(model, spectrogram.split(i, dim=1))), dim=1)
        assert generated.shape == (batch_size, model.upscale_factor * num_frames)
        numpy.testing.assert_almost_equal(immediate.detach().numpy(), generated.detach().numpy())


def test_generate_waveform_small():
    """ Test if incremental generation produces the same output as none-incremental. """
    batch_size = 2
    num_frames = 1
    frame_channels = 6

    model = SignalModel(input_size=frame_channels, ratios=[2, 2], max_channel_size=32, padding=13)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (batch_size, model.upscale_factor * num_frames)

    generated = torch.cat(list(generate_waveform(model, spectrogram.split(1, dim=1))), dim=1)
    assert generated.shape == (batch_size, model.upscale_factor * num_frames)
    numpy.testing.assert_almost_equal(immediate.detach().numpy(), generated.detach().numpy())


def test_generate_waveform__no_batch_dim():
    """ Test if incremental generation produces the same output as none-incremental. """
    num_frames = 37
    frame_channels = 8
    split_size = 26

    model = SignalModel(input_size=frame_channels, ratios=[2, 2], max_channel_size=32, padding=13)
    spectrogram = torch.randn([num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (model.upscale_factor * num_frames,)

    generated = torch.cat(list(generate_waveform(model, spectrogram.split(split_size))))
    assert generated.shape == (model.upscale_factor * num_frames,)
    numpy.testing.assert_almost_equal(immediate.detach().numpy(), generated.detach().numpy())


def test_generate_waveform__padding_invariance():
    batch_size = 2
    num_frames = 27
    frame_channels = 6
    padding = 7
    split_size = 26

    model = SignalModel(input_size=frame_channels, ratios=[2, 2], max_channel_size=32, padding=13)

    spectrogram = torch.randn([batch_size, num_frames + padding * 2, frame_channels])
    mask = torch.cat(
        [
            torch.zeros([batch_size, padding]),
            torch.ones([batch_size, num_frames]),
            torch.zeros([batch_size, padding])
        ],
        dim=1,
    ).bool()

    immediate = model(spectrogram[:, padding:-padding])

    generated = generate_waveform(model, spectrogram.split(split_size, dim=1),
                                  mask.split(split_size, dim=1))
    generated = torch.cat(list(generated), dim=1)

    # Ensure padded output is zero
    assert generated[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert generated[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0

    # Ensure not padding output isn't affected.
    numpy.testing.assert_almost_equal(
        generated[:,
                  padding * model.upscale_factor:-padding * model.upscale_factor].detach().numpy(),
        immediate.detach().numpy())


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


def test_signal_model__batch_invariance():
    batch_size = 4
    num_frames = 8
    frame_channels = 128

    model = SignalModel(input_size=frame_channels)
    batched_spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    batched_out = model(batched_spectrogram)

    out = model(batched_spectrogram[0])

    numpy.testing.assert_almost_equal(batched_out[0].detach().numpy(), out.detach().numpy())


def test_signal_model__padding_invariance():
    batch_size = 4
    num_frames = 8
    frame_channels = 128
    padding = 3

    model = SignalModel(input_size=frame_channels)

    spectrogram = torch.randn([batch_size, num_frames + padding * 2, frame_channels])
    mask = torch.cat(
        [
            torch.zeros([batch_size, padding]),
            torch.ones([batch_size, num_frames]),
            torch.zeros([batch_size, padding])
        ],
        dim=1,
    ).bool()
    padded_out = model(spectrogram, mask)
    out = model(spectrogram[:, padding:-padding])

    # Ensure padded output is zero
    assert padded_out[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert padded_out[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0

    # Ensure not padding output isn't affected.
    numpy.testing.assert_almost_equal(
        padded_out[:,
                   padding * model.upscale_factor:-padding * model.upscale_factor].detach().numpy(),
        out.detach().numpy())
