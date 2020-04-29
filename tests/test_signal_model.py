import numpy
import torch

from src.signal_model import generate_waveform
from src.signal_model import L1L2Loss
from src.signal_model import SignalModel
from src.signal_model import SpectrogramDiscriminator
from src.signal_model import trim


def test_l1_l2_loss():
    loss = L1L2Loss()
    input_ = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input_, target)
    output.backward()


def _get_small_signal_model(*args, **kwargs):
    return SignalModel(*args, hidden_size=2, max_channel_size=8, ratios=[2], padding=12, **kwargs)


def test_generate_waveform():
    """ Test if incremental generation produces the same output as none-incremental. """
    batch_size = 4
    num_frames = 8
    frame_channels = 16

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    immediate = model(spectrogram)
    incremental = generate_waveform(model, spectrogram, split_size=4, generator=False)
    other_incremental = generate_waveform(model, spectrogram, split_size=2, generator=True)
    other_incremental = torch.cat(list(other_incremental), dim=1)

    assert immediate.shape == (batch_size, model.upscale_factor * num_frames)
    assert incremental.shape == (batch_size, model.upscale_factor * num_frames)
    assert other_incremental.shape == (batch_size, model.upscale_factor * num_frames)

    numpy.testing.assert_almost_equal(immediate.detach().numpy(), incremental.detach().numpy())
    numpy.testing.assert_almost_equal(immediate.detach().numpy(),
                                      other_incremental.detach().numpy())


def test_generate_waveform__no_batch_dim():
    """ Test if incremental generation produces the same output as none-incremental. """
    num_frames = 8
    frame_channels = 16

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    immediate = model(spectrogram)
    incremental = generate_waveform(model, spectrogram, split_size=4, generator=False)
    other_incremental = generate_waveform(model, spectrogram, split_size=2, generator=True)
    other_incremental = torch.cat(list(other_incremental), dim=-1)

    assert immediate.shape == (model.upscale_factor * num_frames)
    assert incremental.shape == (model.upscale_factor * num_frames)
    assert other_incremental.shape == (model.upscale_factor * num_frames)

    numpy.testing.assert_almost_equal(immediate.detach().numpy(), incremental.detach().numpy())
    numpy.testing.assert_almost_equal(immediate.detach().numpy(),
                                      other_incremental.detach().numpy())


def test_generate_waveform__padding_invariance():
    batch_size = 4
    num_frames = 8
    frame_channels = 16
    padding = 3

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames + padding * 2, frame_channels])
    mask = torch.cat(
        [
            torch.zeros([batch_size, padding]),
            torch.ones([batch_size, num_frames]),
            torch.zeros([batch_size, padding])
        ],
        dim=1,
    ).bool()
    padded_out = generate_waveform(model, spectrogram, mask, split_size=4, generator=False)
    out = model(spectrogram[:, padding:-padding])

    # Ensure padded output is zero
    assert padded_out[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert padded_out[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0

    # Ensure not padding output isn't affected.
    numpy.testing.assert_almost_equal(
        padded_out[:,
                   padding * model.upscale_factor:-padding * model.upscale_factor].detach().numpy(),
        out.detach().numpy())


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
    frame_channels = 16

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    out = model(spectrogram)

    assert out.shape == (batch_size, model.upscale_factor * num_frames)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()


def test_signal_model__no_batch():
    num_frames = 9  # Test odd number of frames
    frame_channels = 16

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    out = model(spectrogram)

    assert out.shape == (model.upscale_factor * num_frames,)

    assert out.max() <= 1.0
    assert out.min() >= -1.0

    out.sum().backward()


def test_signal_model__batch_invariance():
    batch_size = 4
    num_frames = 8
    frame_channels = 16

    model = _get_small_signal_model(input_size=frame_channels)
    batched_spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    batched_out = model(batched_spectrogram)

    out = model(batched_spectrogram[0])

    numpy.testing.assert_almost_equal(batched_out[0].detach().numpy(), out.detach().numpy())


def test_signal_model__padding_invariance():
    batch_size = 4
    num_frames = 8
    frame_channels = 16
    padding = 3

    model = _get_small_signal_model(input_size=frame_channels)

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
