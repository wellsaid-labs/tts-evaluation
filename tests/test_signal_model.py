import itertools

import torch

from src.signal_model import generate_waveform
from src.signal_model import has_weight_norm
from src.signal_model import L1L2Loss
from src.signal_model import SignalModel
from src.signal_model import SpectrogramDiscriminator
from tests._utils import assert_almost_equal


def test_has_weight_norm():
    module = torch.nn.Linear(20, 40)
    torch.nn.utils.weight_norm(module, name='weight')

    assert has_weight_norm(module)

    torch.nn.utils.remove_weight_norm(module, name='weight')
    assert not has_weight_norm(module)


def test_l1_l2_loss():
    loss = L1L2Loss()
    input_ = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input_, target)
    output.backward()


def _get_small_signal_model(*args, **kwargs):
    return SignalModel(*args, hidden_size=2, max_channel_size=8, ratios=[2], **kwargs)


def test_signal_model__excess_padding():
    """ Test if the model is able to output the correct size regardless of the parameters. """
    frame_channels = 4
    for i in range(1, 3):
        for j in range(1, 3):
            for k in range(1, 3):
                model = SignalModel(
                    input_size=frame_channels, hidden_size=2, max_channel_size=8, ratios=[i] * j)
                spectrogram = torch.randn([k, frame_channels])
                assert model(spectrogram).shape == (model.upscale_factor * k,)


def test_signal_model__train_mode():
    """ Basic test is `train` mode can be applied multiple times. """
    model = _get_small_signal_model()
    model.train()
    model.train()
    model.eval()
    model.eval()


def test_generate_waveform():
    """ Test if incremental generation produces the same output as none-incremental. """
    batch_size = 2
    num_frames = 53
    frame_channels = 6

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (batch_size, model.upscale_factor * num_frames)

    for i in itertools.chain([1, 26, 27, 53]):
        generated = torch.cat(list(generate_waveform(model, spectrogram.split(i, dim=1))), dim=1)
        assert generated.shape == (batch_size, model.upscale_factor * num_frames)
        assert_almost_equal(immediate, generated)


def test_generate_waveform_small():
    """ Test if incremental generation produces the same output as none-incremental. """
    batch_size = 2
    num_frames = 1
    frame_channels = 6

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (batch_size, model.upscale_factor * num_frames)

    generated = torch.cat(list(generate_waveform(model, spectrogram.split(1, dim=1))), dim=1)
    assert generated.shape == (batch_size, model.upscale_factor * num_frames)
    assert_almost_equal(immediate, generated)


def test_generate_waveform__no_batch_dim():
    """ Test if incremental generation produces the same output as none-incremental. """
    num_frames = 37
    frame_channels = 8
    split_size = 26

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    immediate = model(spectrogram)
    assert immediate.shape == (model.upscale_factor * num_frames,)

    generated = torch.cat(list(generate_waveform(model, spectrogram.split(split_size))))
    assert generated.shape == (model.upscale_factor * num_frames,)
    assert_almost_equal(immediate, generated)


def test_generate_waveform__padding_invariance():
    batch_size = 2
    num_frames = 27
    frame_channels = 6
    padding = 7
    split_size = 26

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

    immediate = model(spectrogram[:, padding:-padding])

    generated = generate_waveform(model, spectrogram.split(split_size, dim=1),
                                  mask.split(split_size, dim=1))
    generated = torch.cat(list(generated), dim=1)

    # Ensure padded output is zero
    assert generated[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert generated[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0

    # Ensure not padding output isn't affected.
    assert_almost_equal(
        generated[:, padding * model.upscale_factor:-padding * model.upscale_factor], immediate)


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

    assert_almost_equal(batched_out[0], out)


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
    assert_almost_equal(
        padded_out[:, padding * model.upscale_factor:-padding * model.upscale_factor], out)
