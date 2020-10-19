import itertools
import math
import typing

import torch
from torchnlp.random import fork_rng
from torchnlp.utils import lengths_to_mask

import lib
from lib.signal_model import generate_waveform
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=5)


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
    excess_padding = (
        (math.ceil(module.padding_required) - module.padding_required) * input_scale * 2
    )
    output = module(
        torch.randn(batch_size, in_channels, num_frames + padding),
        torch.randn(batch_size, 1, num_frames // input_scale + 2),
        torch.randn(batch_size, in_channels, num_frames // input_scale + 2),
    )
    assert output.shape == (
        batch_size,
        out_channels,
        num_frames * upscale_factor + excess_padding * upscale_factor,
    )
    output.sum().backward()


def test__has_weight_norm():
    """ Test `lib.signal_model._has_weight_norm` detects `torch.nn.utils.weight_norm`. """
    module = torch.nn.Linear(20, 40)
    torch.nn.utils.weight_norm(module, name="weight")
    assert lib.signal_model._has_weight_norm(module)
    torch.nn.utils.remove_weight_norm(module, name="weight")
    assert not lib.signal_model._has_weight_norm(module)


def _make_small_signal_model(
    batch_size: int = 4,
    num_frames: int = 8,
    num_frame_channels: int = 6,
    hidden_size: int = 2,
    ratios: typing.List[int] = [2],
    max_channel_size: int = 8,
    mu: int = 255,
) -> typing.Tuple[lib.signal_model.SignalModel, torch.Tensor, typing.Tuple[int, int, int]]:
    """ Make `lib.signal_model.SignalModel` and it's inputs for testing."""
    model = lib.signal_model.SignalModel(
        input_size=num_frame_channels,
        hidden_size=hidden_size,
        max_channel_size=max_channel_size,
        ratios=ratios,
        mu=mu,
    )
    spectrogram = torch.randn([batch_size, num_frames, num_frame_channels])
    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    for name, parameter in model.named_parameters():
        if parameter.std() == 0:
            torch.nn.init.normal_(parameter)
    return model, spectrogram, (batch_size, num_frames, num_frame_channels)


def test_signal_model():
    """Test `lib.signal_model.SignalModel` output is the right shape, in range and differentiable.
    """
    model, spectrogram, (batch_size, num_frames, _) = _make_small_signal_model()
    out = model(spectrogram)
    assert out.shape == (batch_size, model.upscale_factor * num_frames)
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    out.sum().backward()


def test_signal_model__no_batch__odd():
    """Test `lib.signal_model.SignalModel` can handle an input without a batch dimension, and
    an odd number of frames."""
    model, spectrogram, (_, num_frames, _) = _make_small_signal_model(num_frames=9, batch_size=1)
    out = model(spectrogram.squeeze(0))
    assert out.shape == (model.upscale_factor * num_frames,)
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    out.sum().backward()


def test_signal_model__batch_invariance():
    """ Test `lib.signal_model.SignalModel` output doesn't vary with batch size. """
    model, spectrogram, (_, num_frames, _) = _make_small_signal_model()
    _utils.assert_almost_equal(model(spectrogram)[0], model(spectrogram[0]))


def test_signal_model__padding_invariance():
    """Test `lib.signal_model.SignalModel` output doesn't vary with masked padding, and the
    output is masked."""
    model, _, (batch_size, num_frames, num_frame_channels) = _make_small_signal_model()
    padding = 3
    spectrogram = torch.randn([batch_size, num_frames + padding * 2, num_frame_channels])
    mask = torch.cat(
        [
            torch.zeros([batch_size, padding]),
            torch.ones([batch_size, num_frames]),
            torch.zeros([batch_size, padding]),
        ],
        1,
    ).bool()
    padded_out = model(spectrogram, mask)
    out = model(spectrogram[:, padding:-padding])
    # NOTE: Ensure the output is masked.
    assert padded_out[:, : padding * model.upscale_factor].abs().sum().item() == 0.0
    assert padded_out[:, -padding * model.upscale_factor :].abs().sum().item() == 0.0
    # NOTE: Ensure the output isn't affected by padding.
    _utils.assert_almost_equal(
        padded_out[:, padding * model.upscale_factor : -padding * model.upscale_factor],
        out,
    )


def test_signal_model__shape():
    """Test `lib.signal_model.SignalModel` output is the correct shape given different
    scale factors and input sizes. Particularly, this tests the `padding` and `excess_padding`
    implementations."""
    frame_channels = 4
    for i, j, input_size in itertools.product(range(1, 4), range(1, 3), range(1, 4)):
        model = lib.signal_model.SignalModel(
            input_size=frame_channels,
            hidden_size=2,
            max_channel_size=4,
            ratios=[i] * j,
            mu=255,
        )
        spectrogram = torch.randn([input_size, frame_channels])
        assert model(spectrogram).shape == (model.upscale_factor * input_size,)


def test_train():
    """ Test `lib.signal_model.SignalModel.train` executes. """
    model, *_ = _make_small_signal_model()
    model.train()
    model.train()
    model.eval()
    model.eval()


def test_spectrogram_discriminator():
    """Test `lib.signal_model.SpectrogramDiscriminator` output is the right shape and
    differentiable."""
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
    """Test `lib.signal_model.generate_waveform` is consistent with `lib.signal_model.SignalModel`
    given different spectrogram generators.
    """
    model, spectrogram, (batch_size, num_frames, _) = _make_small_signal_model(
        num_frames=53, batch_size=2
    )
    output = model(spectrogram)
    assert output.shape == (batch_size, model.upscale_factor * num_frames)
    for i in itertools.chain([1, 26, 27, 53]):
        generated = torch.cat(list(generate_waveform(model, spectrogram.split(i, dim=1))), dim=1)
        assert generated.shape == (batch_size, model.upscale_factor * num_frames)
        _utils.assert_almost_equal(output, generated)


def test_generate_waveform__no_batch_dim():
    """Test `lib.signal_model.generate_waveform` is consistent with `lib.signal_model.SignalModel`
    given no batch dimension.
    """
    split_size = 26
    model, spectrogram, (_, num_frames, _) = _make_small_signal_model(num_frames=37, batch_size=1)
    output = model(spectrogram.squeeze(0))
    assert output.shape == (model.upscale_factor * num_frames,)
    generated = torch.cat(list(generate_waveform(model, spectrogram.squeeze(0).split(split_size))))
    assert generated.shape == (model.upscale_factor * num_frames,)
    _utils.assert_almost_equal(output, generated)


def test_generate_waveform__padding_invariance():
    """Test `lib.signal_model.generate_waveform` output doesn't vary with masked padding, and the
    output is masked."""
    padding = 7
    split_size = 26
    model, _, (batch_size, num_frames, num_frame_channels) = _make_small_signal_model(
        num_frames=27, batch_size=2
    )
    spectrogram = torch.randn([batch_size, num_frames + padding * 2, num_frame_channels])
    mask = torch.cat(
        [
            torch.zeros([batch_size, padding]),
            torch.ones([batch_size, num_frames]),
            torch.zeros([batch_size, padding]),
        ],
        1,
    ).bool()
    immediate = model(spectrogram[:, padding:-padding])
    generator = generate_waveform(
        model, spectrogram.split(split_size, dim=1), mask.split(split_size, dim=1)
    )
    generated = torch.cat(list(generator), dim=1)
    # NOTE: Ensure the output is masked.
    assert generated[:, : padding * model.upscale_factor].abs().sum().item() == 0.0
    assert generated[:, -padding * model.upscale_factor :].abs().sum().item() == 0.0
    # NOTE: Ensure the output isn't affected by padding.
    _utils.assert_almost_equal(
        generated[:, padding * model.upscale_factor : -padding * model.upscale_factor],
        immediate,
    )


_expected_parameters = {
    "pre_net.1.bias": torch.tensor(-1.547447),
    "pre_net.1.weight_g": torch.tensor(2.000000),
    "pre_net.1.weight_v": torch.tensor(-0.366229),
    "pre_net.2.weight": torch.tensor(-1.373768),
    "pre_net.2.bias": torch.tensor(-0.467871),
    "network.0.shortcut.0.bias": torch.tensor(2.412678),
    "network.0.shortcut.0.weight_g": torch.tensor(2.133929),
    "network.0.shortcut.0.weight_v": torch.tensor(0.596350),
    "network.0.block.1.bias": torch.tensor(0.369434),
    "network.0.block.1.weight_g": torch.tensor(1.278345),
    "network.0.block.1.weight_v": torch.tensor(-0.149943),
    "network.0.block.4.bias": torch.tensor(0.899690),
    "network.0.block.4.weight_g": torch.tensor(2.0),
    "network.0.block.4.weight_v": torch.tensor(2.384378),
    "network.0.block.8.bias": torch.tensor(-2.756881),
    "network.0.block.8.weight_g": torch.tensor(2.0),
    "network.0.block.8.weight_v": torch.tensor(0.758507),
    "network.0.other_block.1.bias": torch.tensor(-0.033680),
    "network.0.other_block.1.weight_g": torch.tensor(2.0),
    "network.0.other_block.1.weight_v": torch.tensor(-0.704928),
    "network.0.other_block.4.bias": torch.tensor(2.051252),
    "network.0.other_block.4.weight_g": torch.tensor(2.0),
    "network.0.other_block.4.weight_v": torch.tensor(0.717614),
    "network.0.other_block.7.bias": torch.tensor(-0.531909),
    "network.0.other_block.7.weight_g": torch.tensor(-1.106336),
    "network.0.other_block.7.weight_v": torch.tensor(2.906337),
    "network.1.shortcut.0.bias": torch.tensor(2.877002),
    "network.1.shortcut.0.weight_g": torch.tensor(-0.786183),
    "network.1.shortcut.0.weight_v": torch.tensor(1.792434),
    "network.1.block.1.bias": torch.tensor(-0.885676),
    "network.1.block.1.weight_g": torch.tensor(2.0),
    "network.1.block.1.weight_v": torch.tensor(1.211417),
    "network.1.block.4.bias": torch.tensor(0.920205),
    "network.1.block.4.weight_g": torch.tensor(2.0),
    "network.1.block.4.weight_v": torch.tensor(-0.727726),
    "network.1.block.8.bias": torch.tensor(0.203688),
    "network.1.block.8.weight_g": torch.tensor(2.809510),
    "network.1.block.8.weight_v": torch.tensor(1.425151),
    "network.1.other_block.1.bias": torch.tensor(0.414940),
    "network.1.other_block.1.weight_g": torch.tensor(-2.601039),
    "network.1.other_block.1.weight_v": torch.tensor(-0.505602),
    "network.1.other_block.4.bias": torch.tensor(-1.247683),
    "network.1.other_block.4.weight_g": torch.tensor(2.0),
    "network.1.other_block.4.weight_v": torch.tensor(-1.676837),
    "network.1.other_block.7.bias": torch.tensor(1.187213),
    "network.1.other_block.7.weight_g": torch.tensor(2.0),
    "network.1.other_block.7.weight_v": torch.tensor(2.517972),
    "network.2.shortcut.0.bias": torch.tensor(-1.397728),
    "network.2.shortcut.0.weight_g": torch.tensor(2.808054),
    "network.2.shortcut.0.weight_v": torch.tensor(-2.561306),
    "network.2.block.1.bias": torch.tensor(-1.973773),
    "network.2.block.1.weight_g": torch.tensor(2.000000),
    "network.2.block.1.weight_v": torch.tensor(-0.059381),
    "network.2.block.4.bias": torch.tensor(-2.988898),
    "network.2.block.4.weight_g": torch.tensor(4.0),
    "network.2.block.4.weight_v": torch.tensor(0.056639),
    "network.2.block.8.bias": torch.tensor(-0.005530),
    "network.2.block.8.weight_g": torch.tensor(2.0),
    "network.2.block.8.weight_v": torch.tensor(2.141014),
    "network.2.other_block.1.bias": torch.tensor(-0.423700),
    "network.2.other_block.1.weight_g": torch.tensor(-1.399166),
    "network.2.other_block.1.weight_v": torch.tensor(1.765871),
    "network.2.other_block.4.bias": torch.tensor(-0.839966),
    "network.2.other_block.4.weight_g": torch.tensor(2.000000),
    "network.2.other_block.4.weight_v": torch.tensor(0.504495),
    "network.2.other_block.7.bias": torch.tensor(-1.460063),
    "network.2.other_block.7.weight_g": torch.tensor(0.077671),
    "network.2.other_block.7.weight_v": torch.tensor(1.384603),
    "network.3.bias": torch.tensor(0.990304),
    "network.3.weight_g": torch.tensor(0.834376),
    "network.3.weight_v": torch.tensor(1.858381),
    "network.6.bias": torch.tensor(1.355464),
    "network.6.weight_g": torch.tensor(0.213282),
    "network.6.weight_v": torch.tensor(1.856596),
    "condition.bias": torch.tensor(-1.527744),
    "condition.weight_g": torch.tensor(-0.499057),
    "condition.weight_v": torch.tensor(0.597046),
}

_expected_grads = {
    "pre_net.1.bias": torch.tensor(-2.156012e-07),
    "pre_net.1.weight_g": torch.tensor(0.018762),
    "pre_net.1.weight_v": torch.tensor(0.001269),
    "pre_net.2.weight": torch.tensor(0.619135),
    "pre_net.2.bias": torch.tensor(0.494207),
    "network.0.shortcut.0.bias": torch.tensor(-0.337951),
    "network.0.shortcut.0.weight_g": torch.tensor(-0.276910),
    "network.0.shortcut.0.weight_v": torch.tensor(-0.359326),
    "network.0.block.1.bias": torch.tensor(-0.004609),
    "network.0.block.1.weight_g": torch.tensor(-0.010500),
    "network.0.block.1.weight_v": torch.tensor(-0.009534),
    "network.0.block.4.bias": torch.tensor(0.264169),
    "network.0.block.4.weight_g": torch.tensor(-0.042547),
    "network.0.block.4.weight_v": torch.tensor(-0.096912),
    "network.0.block.8.bias": torch.tensor(-0.337951),
    "network.0.block.8.weight_g": torch.tensor(0.068765),
    "network.0.block.8.weight_v": torch.tensor(-0.386223),
    "network.0.other_block.1.bias": torch.tensor(0.127988),
    "network.0.other_block.1.weight_g": torch.tensor(0.024156),
    "network.0.other_block.1.weight_v": torch.tensor(-0.185204),
    "network.0.other_block.4.bias": torch.tensor(-0.442171),
    "network.0.other_block.4.weight_g": torch.tensor(0.074128),
    "network.0.other_block.4.weight_v": torch.tensor(-0.574519),
    "network.0.other_block.7.bias": torch.tensor(-0.268222),
    "network.0.other_block.7.weight_g": torch.tensor(-0.059678),
    "network.0.other_block.7.weight_v": torch.tensor(-0.485142),
    "network.1.shortcut.0.bias": torch.tensor(0.048052),
    "network.1.shortcut.0.weight_g": torch.tensor(0.725731),
    "network.1.shortcut.0.weight_v": torch.tensor(0.747637),
    "network.1.block.1.bias": torch.tensor(-0.030226),
    "network.1.block.1.weight_g": torch.tensor(0.039441),
    "network.1.block.1.weight_v": torch.tensor(0.083798),
    "network.1.block.4.bias": torch.tensor(-0.405699),
    "network.1.block.4.weight_g": torch.tensor(-0.015950),
    "network.1.block.4.weight_v": torch.tensor(0.067614),
    "network.1.block.8.bias": torch.tensor(0.048052),
    "network.1.block.8.weight_g": torch.tensor(-0.298613),
    "network.1.block.8.weight_v": torch.tensor(2.785087),
    "network.1.other_block.1.bias": torch.tensor(0.021341),
    "network.1.other_block.1.weight_g": torch.tensor(-0.022065),
    "network.1.other_block.1.weight_v": torch.tensor(-0.166280),
    "network.1.other_block.4.bias": torch.tensor(-0.301493),
    "network.1.other_block.4.weight_g": torch.tensor(0.006783),
    "network.1.other_block.4.weight_v": torch.tensor(-0.176757),
    "network.1.other_block.7.bias": torch.tensor(0.028061),
    "network.1.other_block.7.weight_g": torch.tensor(-0.098151),
    "network.1.other_block.7.weight_v": torch.tensor(0.129992),
    "network.2.shortcut.0.bias": torch.tensor(0.700525),
    "network.2.shortcut.0.weight_g": torch.tensor(-1.744184),
    "network.2.shortcut.0.weight_v": torch.tensor(6.802945),
    "network.2.block.1.bias": torch.tensor(-0.080132),
    "network.2.block.1.weight_g": torch.tensor(0.185340),
    "network.2.block.1.weight_v": torch.tensor(-0.513411),
    "network.2.block.4.bias": torch.tensor(-0.105657),
    "network.2.block.4.weight_g": torch.tensor(-0.044303),
    "network.2.block.4.weight_v": torch.tensor(-1.171456),
    "network.2.block.8.bias": torch.tensor(0.700525),
    "network.2.block.8.weight_g": torch.tensor(-0.299234),
    "network.2.block.8.weight_v": torch.tensor(0.529226),
    "network.2.other_block.1.bias": torch.tensor(0.020188),
    "network.2.other_block.1.weight_g": torch.tensor(-0.020140),
    "network.2.other_block.1.weight_v": torch.tensor(0.296558),
    "network.2.other_block.4.bias": torch.tensor(-0.005036),
    "network.2.other_block.4.weight_g": torch.tensor(0.040998),
    "network.2.other_block.4.weight_v": torch.tensor(-0.021219),
    "network.2.other_block.7.bias": torch.tensor(0.694215),
    "network.2.other_block.7.weight_g": torch.tensor(-0.085698),
    "network.2.other_block.7.weight_v": torch.tensor(-0.768474),
    "network.3.bias": torch.tensor(8.689142),
    "network.3.weight_g": torch.tensor(-13.373070),
    "network.3.weight_v": torch.tensor(-9.279540),
    "network.6.bias": torch.tensor(46.350098),
    "network.6.weight_g": torch.tensor(6.817425),
    "network.6.weight_v": torch.tensor(-129.049683),
    "condition.bias": torch.tensor(-0.028309),
    "condition.weight_g": torch.tensor(-0.044978),
    "condition.weight_v": torch.tensor(0.001236),
}


def test_signal_model__version():
    """ Test `lib.signal_model.SignalModel` has not changed since it was last tested. """
    with fork_rng(123):
        (
            model,
            _,
            (batch_size, num_frames, num_frame_channels),
        ) = _make_small_signal_model(batch_size=16)
        spectrogram = torch.randn(batch_size, num_frames + model.padding * 2, num_frame_channels)
        spectrogram_length = torch.randint(
            model.padding + 1, num_frames + model.padding * 2, (batch_size,)
        )
        spectrogram_length[-1] = num_frames + model.padding * 2
        spectrogram_mask = lengths_to_mask(spectrogram_length)

        signal = model(spectrogram, spectrogram_mask, pad_input=False)

        for name, parameter in model.named_parameters():
            assert_almost_equal(_expected_parameters[name], parameter.sum())
        assert_almost_equal(
            signal.sum(dim=-1),
            torch.tensor(
                [
                    2.451438,
                    2.457014,
                    0.666661,
                    2.451525,
                    2.451438,
                    1.592244,
                    1.278022,
                    1.278021,
                    2.451413,
                    2.451438,
                    2.454954,
                    2.451438,
                    2.451446,
                    2.457036,
                    2.451438,
                    2.456430,
                ]
            ),
        )

        signal.sum().backward()

        assert_almost_equal(signal.sum(), torch.tensor(34.251957))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert_almost_equal(_expected_grads[name], parameter.grad.sum())
        assert_almost_equal(torch.randn(1), torch.tensor(-0.465967))
