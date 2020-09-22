import itertools
import math

import torch

from tests import _utils

import lib


def test_l1_l2_loss():
    """ Test `lib.signal_model.L1L2Loss` is differentiable. """
    loss = lib.signal_model.L1L2Loss()
    input_ = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input_, target)
    output.backward()


def test__interpolate_and_concat():
    """ Test `lib.signal_model._InterpolateAndConcat` trims and concats. """
    module = lib.signal_model._InterpolateAndConcat(size=1, scale_factor=2)
    concat = torch.arange(0, 3, dtype=torch.float).view(1, 1, 3)
    tensor = torch.ones(1, 1, 4)
    output = module(tensor, concat)
    _utils.assert_almost_equal(output, torch.tensor([[tensor.tolist()[0][0], [0, 1, 1, 2]]]))


def test__interpolate_and_mask():
    """ Test `lib.signal_model._InterpolateAndMask` trims and masks. """
    module = lib.signal_model._InterpolateAndMask(scale_factor=2)
    mask = torch.tensor([0, 1, 0], dtype=torch.float).view(1, 1, 3)
    tensor = torch.full((1, 1, 4), 2, dtype=torch.float)
    output = module(tensor, mask)
    _utils.assert_almost_equal(output, torch.tensor([[[0, 2, 2, 0]]]))


def test__pixel_shuffle_1d():
    """ Test `lib.signal_model._PixelShuffle1d` reshapes the input correctly. """
    module = lib.signal_model._PixelShuffle1d(upscale_factor=4)
    tensor = torch.arange(0, 12).view(1, 3, 4).transpose(1, 2)
    output = module(tensor)
    _utils.assert_almost_equal(output, torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]))


def test__block():
    """ Test `lib.signal_model._Block` is differentiable and outputs the right shape. """
    in_channels = 4
    num_frames = 4
    batch_size = 3
    out_channels = 2
    upscale_factor = 2
    input_scale = 4
    module = lib.signal_model._Block(in_channels, out_channels, upscale_factor, input_scale)
    padding = math.ceil(module.padding_required) * input_scale * 2
    excess_padding = (math.ceil(module.padding_required) -
                      module.padding_required) * input_scale * 2
    output = module(
        torch.randn(batch_size, in_channels, num_frames + padding),
        torch.randn(batch_size, 1, num_frames // input_scale + 2),
        torch.randn(batch_size, in_channels, num_frames // input_scale + 2))
    assert output.shape == (batch_size, out_channels,
                            num_frames * upscale_factor + excess_padding * upscale_factor)
    output.sum().backward()


def test__has_weight_norm():
    """ Test `lib.signal_model._has_weight_norm` detects `torch.nn.utils.weight_norm`. """
    module = torch.nn.Linear(20, 40)
    torch.nn.utils.weight_norm(module, name='weight')
    assert lib.signal_model._has_weight_norm(module)
    torch.nn.utils.remove_weight_norm(module, name='weight')
    assert not lib.signal_model._has_weight_norm(module)


def _get_small_signal_model(input_size=6, hidden_size=2, ratios=[2], max_channel_size=8, mu=255):
    return lib.signal_model.SignalModel(
        input_size=input_size,
        hidden_size=hidden_size,
        max_channel_size=max_channel_size,
        ratios=ratios,
        mu=mu)


def test_signal_model():
    """ Test `lib.signal_model.SignalModel` output is the right shape, in range and differentiable.
    """
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


def test_signal_model__no_batch__odd():
    """ Test `lib.signal_model.SignalModel` can handle an input without a batch dimension, and
    an odd number of frames. """
    num_frames = 9
    frame_channels = 16
    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    out = model(spectrogram)
    assert out.shape == (model.upscale_factor * num_frames,)
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    out.sum().backward()


def test_signal_model__batch_invariance():
    """ Test `lib.signal_model.SignalModel` output doesn't vary with batch size. """
    batch_size = 4
    num_frames = 8
    frame_channels = 16
    model = _get_small_signal_model(input_size=frame_channels)
    batched_spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    batched_out = model(batched_spectrogram)
    out = model(batched_spectrogram[0])
    _utils.assert_almost_equal(batched_out[0], out)


def test_signal_model__padding_invariance():
    """ Test `lib.signal_model.SignalModel` output doesn't vary with masked padding, and the
    output is masked. """
    batch_size = 4
    num_frames = 8
    frame_channels = 16
    padding = 3
    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames + padding * 2, frame_channels])
    mask = torch.cat([
        torch.zeros([batch_size, padding]),
        torch.ones([batch_size, num_frames]),
        torch.zeros([batch_size, padding])
    ], 1).bool()
    padded_out = model(spectrogram, mask)
    out = model(spectrogram[:, padding:-padding])
    # NOTE: Ensure the output is masked.
    assert padded_out[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert padded_out[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0
    # NOTE: Ensure the output isn't affected by padding.
    _utils.assert_almost_equal(
        padded_out[:, padding * model.upscale_factor:-padding * model.upscale_factor], out)


def test_signal_model__shape():
    """ Test `lib.signal_model.SignalModel` output is the correct shape given different
    scale factors and input sizes. Particularly, this tests the `padding` and `excess_padding`
    implementations. """
    frame_channels = 4
    for i, j, input_size in itertools.product(range(1, 4), range(1, 3), range(1, 4)):
        model = lib.signal_model.SignalModel(
            input_size=frame_channels, hidden_size=2, max_channel_size=4, ratios=[i] * j, mu=255)
        spectrogram = torch.randn([input_size, frame_channels])
        assert model(spectrogram).shape == (model.upscale_factor * input_size,)


def test_train():
    """ Test `lib.signal_model.SignalModel.train` executes. """
    model = _get_small_signal_model()
    model.train()
    model.train()
    model.eval()
    model.eval()


def test_spectrogram_discriminator():
    """ Test `lib.signal_model.SpectrogramDiscriminator` output is the right shape and
    differentiable. """
    batch_size = 4
    num_frames = 16
    fft_length = 1024
    num_mel_bins = 128
    discriminator = lib.signal_model.SpectrogramDiscriminator(fft_length, num_mel_bins, 16)
    spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_mel_spectrogram = torch.randn(batch_size, num_frames, num_mel_bins)
    output = discriminator(spectrogram, db_spectrogram, db_mel_spectrogram)
    assert output.shape == (batch_size,)
    output.sum().backward()


def test_generate_waveform():
    """ Test `lib.signal_model.generate_waveform` is consistent with `lib.signal_model.SignalModel`
    given different spectrogram generators.
    """
    batch_size = 2
    num_frames = 53
    frame_channels = 6

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames, frame_channels])
    output = model(spectrogram)
    assert output.shape == (batch_size, model.upscale_factor * num_frames)

    for i in itertools.chain([1, 26, 27, 53]):
        generated = torch.cat(
            list(lib.signal_model.generate_waveform(model, spectrogram.split(i, dim=1))), dim=1)
        assert generated.shape == (batch_size, model.upscale_factor * num_frames)
        _utils.assert_almost_equal(output, generated)


def test_generate_waveform__no_batch_dim():
    """ Test `lib.signal_model.generate_waveform` is consistent with `lib.signal_model.SignalModel`
    given no batch dimension.
    """
    num_frames = 37
    frame_channels = 8
    split_size = 26

    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([num_frames, frame_channels])
    output = model(spectrogram)
    assert output.shape == (model.upscale_factor * num_frames,)

    generated = torch.cat(
        list(lib.signal_model.generate_waveform(model, spectrogram.split(split_size))))
    assert generated.shape == (model.upscale_factor * num_frames,)
    _utils.assert_almost_equal(output, generated)


def test_generate_waveform__padding_invariance():
    """ Test `lib.signal_model.generate_waveform` output doesn't vary with masked padding, and the
    output is masked. """
    batch_size = 2
    num_frames = 27
    frame_channels = 6
    padding = 7
    split_size = 26
    model = _get_small_signal_model(input_size=frame_channels)
    spectrogram = torch.randn([batch_size, num_frames + padding * 2, frame_channels])
    mask = torch.cat([
        torch.zeros([batch_size, padding]),
        torch.ones([batch_size, num_frames]),
        torch.zeros([batch_size, padding])
    ], 1).bool()
    immediate = model(spectrogram[:, padding:-padding])
    generated = lib.signal_model.generate_waveform(model, spectrogram.split(split_size, dim=1),
                                                   mask.split(split_size, dim=1))
    generated = torch.cat(list(generated), dim=1)
    # NOTE: Ensure the output is masked.
    assert generated[:, :padding * model.upscale_factor].abs().sum().item() == 0.0
    assert generated[:, -padding * model.upscale_factor:].abs().sum().item() == 0.0
    # NOTE: Ensure the output isn't affected by padding.
    _utils.assert_almost_equal(
        generated[:, padding * model.upscale_factor:-padding * model.upscale_factor], immediate)
