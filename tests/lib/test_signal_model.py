import itertools
import math
import typing
from unittest import mock

import torch
import torch.nn
from torch.nn import Embedding
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm
from torchnlp.random import fork_rng
from torchnlp.utils import lengths_to_mask

import lib
from lib.signal_model import SignalModel, generate_waveform
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=4)


def test__interpolate_and_concat():
    """Test `lib.signal_model._InterpolateAndConcat` trims and concats."""
    module = lib.signal_model._InterpolateAndConcat(size=1, scale_factor=2)
    concat = torch.arange(0, 3, dtype=torch.float).view(1, 1, 3)
    tensor = torch.ones(1, 1, 4)
    output = module(tensor, concat)
    assert_almost_equal(output, torch.tensor([[tensor.tolist()[0][0], [0, 1, 1, 2]]]))


def test__interpolate_and_mask():
    """Test `lib.signal_model._InterpolateAndMask` trims and masks."""
    module = lib.signal_model._InterpolateAndMask(scale_factor=2)
    mask = torch.tensor([0, 1, 0], dtype=torch.float).view(1, 1, 3)
    tensor = torch.full((1, 1, 4), 2, dtype=torch.float)
    output = module(tensor, mask)
    assert_almost_equal(output, torch.tensor([[[0, 2, 2, 0]]]))


def test__pixel_shuffle_1d():
    """Test `lib.signal_model._PixelShuffle1d` reshapes the input correctly."""
    module = lib.signal_model._PixelShuffle1d(upscale_factor=4)
    tensor = torch.arange(0, 12).view(1, 3, 4).transpose(1, 2)
    output = module(tensor)
    assert_almost_equal(output, torch.tensor([[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]]))


def test__block():
    """Test `lib.signal_model._Block` is differentiable and outputs the right shape."""
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
    """Test `lib.signal_model._has_weight_norm` detects `torch.nn.utils.weight_norm`."""
    module = torch.nn.Linear(20, 40)
    weight_norm(module, name="weight")
    assert lib.signal_model._has_weight_norm(module)
    remove_weight_norm(module, name="weight")
    assert not lib.signal_model._has_weight_norm(module)


class _Config(typing.NamedTuple):
    max_seq_meta_values: typing.Tuple[int, int] = (8, 10)
    batch_size: int = 4
    num_frames: int = 8
    frame_size: int = 6
    hidden_size: int = 2
    ratios: typing.List[int] = [2]
    max_channel_size: int = 8
    seq_meta_embed_size: int = 8
    mu: int = 255
    padding: int = 0


_ModelInputs = typing.Tuple[torch.Tensor, typing.List[typing.Tuple[int, int]], torch.Tensor]


def _make_small_signal_model(config: _Config) -> typing.Tuple[SignalModel, _ModelInputs]:
    """Make `SignalModel` and it's inputs for testing."""
    model = SignalModel(
        max_seq_meta_values=config.max_seq_meta_values,
        seq_meta_embed_size=config.seq_meta_embed_size,
        frame_size=config.frame_size,
        hidden_size=config.hidden_size,
        ratios=config.ratios,
        max_channel_size=config.max_channel_size,
        mu=config.mu,
    )
    speaker = torch.randint(0, config.max_seq_meta_values[0], (config.batch_size,)).tolist()
    session = torch.randint(0, config.max_seq_meta_values[1], (config.batch_size,)).tolist()
    seq_metadata = list(zip(speaker, session))
    num_frames = config.num_frames + config.padding * 2
    spectrogram = torch.randn([config.batch_size, num_frames, config.frame_size])
    mask = [
        torch.zeros([config.batch_size, config.padding]),
        torch.ones([config.batch_size, config.num_frames]),
        torch.zeros([config.batch_size, config.padding]),
    ]
    mask = torch.cat(mask, 1).bool()

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    for _, parameter in model.named_parameters():
        if parameter.std() == 0:
            torch.nn.init.normal_(parameter)

    return model, (spectrogram, seq_metadata, mask)


def test_signal_model():
    """Test `SignalModel` output is the right shape, in range and differentiable."""
    config = _Config()
    model, inputs = _make_small_signal_model(config)
    out = model(*inputs)
    assert out.shape == (config.batch_size, model.upscale_factor * config.num_frames)
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    out.sum().backward()


def test_signal_model__odd():
    """Test `SignalModel` can handle an input with an odd number of frames."""
    config = _Config(num_frames=9, batch_size=1)
    model, inputs = _make_small_signal_model(config)
    out = model(*inputs)
    assert out.shape == (config.batch_size, model.upscale_factor * config.num_frames)
    assert out.max() <= 1.0
    assert out.min() >= -1.0
    out.sum().backward()


def test_signal_model__batch_invariance():
    """Test `SignalModel` output doesn't vary with batch size."""
    model, inputs = _make_small_signal_model(_Config())
    first = tuple(i[0:1] for i in inputs)
    first = typing.cast(typing.Tuple[torch.Tensor, typing.List[typing.Tuple[int, int]]], first)
    assert_almost_equal(model(*inputs)[0:1], model(*first))


def test_signal_model__padding_invariance():
    """Test `SignalModel` output doesn't vary with masked padding, and the output is masked."""
    config = _Config(padding=3)
    model, inputs = _make_small_signal_model(config)

    padded_out = model(*inputs)
    out = model(inputs[0][:, config.padding : -config.padding], inputs[1])
    padding_len = model.upscale_factor * config.padding

    # NOTE: Ensure the output is masked.
    assert padded_out[:, :padding_len].abs().sum().item() == 0.0
    assert padded_out[:, -padding_len:].abs().sum().item() == 0.0

    # NOTE: Ensure the output isn't affected by padding.
    assert_almost_equal(padded_out[:, padding_len:-padding_len], out)


def test_signal_model__shape():
    """Test `SignalModel` output is the correct shape given different scale factors and input sizes.
    Particularly, this tests the `padding` and `excess_padding` implementations."""
    for i, j, num_frames in itertools.product(range(1, 4), range(1, 3), range(1, 4)):
        config = _Config(
            max_seq_meta_values=(10, 12),
            frame_size=4,
            ratios=[i] * j,
            max_channel_size=4,
            seq_meta_embed_size=16,
            num_frames=num_frames,
        )
        model, inputs = _make_small_signal_model(config)
        assert model(*inputs).shape == (config.batch_size, model.upscale_factor * config.num_frames)


def test_train():
    """Test `SignalModel.train` executes."""
    model, *_ = _make_small_signal_model(_Config())
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
    max_seq_meta_values = (12, 10)
    discriminator = lib.signal_model.SpectrogramDiscriminator(
        fft_length, num_mel_bins, max_seq_meta_values, 12, 16
    )
    spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_spectrogram = torch.randn(batch_size, num_frames, fft_length // 2 + 1)
    db_mel_spectrogram = torch.randn(batch_size, num_frames, num_mel_bins)
    speaker = torch.randint(0, max_seq_meta_values[0], (batch_size,)).tolist()
    session = torch.randint(0, max_seq_meta_values[1], (batch_size,)).tolist()
    seq_metadata = list(zip(speaker, session))
    output = discriminator(spectrogram, db_spectrogram, db_mel_spectrogram, seq_metadata)
    assert output.shape == (batch_size,)
    output.sum().backward()


def test_generate_waveform():
    """Test `lib.signal_model.generate_waveform` is consistent with `SignalModel`
    given different spectrogram generators.
    """
    config = _Config(num_frames=53, batch_size=2)
    model, inputs = _make_small_signal_model(config)
    output = model(*inputs)
    assert output.shape == (config.batch_size, model.upscale_factor * config.num_frames)
    for i in itertools.chain([1, 26, 27, 53]):
        generator = generate_waveform(model, inputs[0].split(i, dim=1), inputs[1])
        generated = torch.cat(list(generator), dim=1)
        assert generated.shape == (config.batch_size, model.upscale_factor * config.num_frames)
        assert_almost_equal(output, generated)


def test_generate_waveform__padding_invariance():
    """Test `lib.signal_model.generate_waveform` output doesn't vary with masked padding, and the
    output is masked."""
    split_size = 26
    config = _Config(num_frames=27, batch_size=2, padding=7)
    model, (spectrogram, seq_metadata, mask) = _make_small_signal_model(config)

    immediate = model(spectrogram[:, config.padding : -config.padding], seq_metadata)
    splits = spectrogram.split(split_size, dim=1)
    generator = generate_waveform(model, splits, seq_metadata, mask.split(split_size, dim=1))
    generated = torch.cat(list(generator), dim=1)
    padding_len = model.upscale_factor * config.padding

    # NOTE: Ensure the output is masked.
    assert generated[:, :padding_len].abs().sum().item() == 0.0
    assert generated[:, -padding_len:].abs().sum().item() == 0.0

    # NOTE: Ensure the output isn't affected by padding.
    assert_almost_equal(generated[:, padding_len:-padding_len], immediate)


_expected_parameters = {
    "encoder.embed_metadata.0.weight": torch.tensor(-0.574029),
    "encoder.embed_metadata.1.weight": torch.tensor(0.777365),
    "pre_net.1.bias": torch.tensor(0.389551),
    "pre_net.1.weight_g": torch.tensor(2.000000),
    "pre_net.1.weight_v": torch.tensor(1.008913),
    "pre_net.2.weight": torch.tensor(0.980255),
    "pre_net.2.bias": torch.tensor(1.415027),
    "network.0.shortcut.0.bias": torch.tensor(-0.143234),
    "network.0.shortcut.0.weight_g": torch.tensor(2.000000),
    "network.0.shortcut.0.weight_v": torch.tensor(-1.787402),
    "network.0.block.1.bias": torch.tensor(-1.263540),
    "network.0.block.1.weight_g": torch.tensor(2.000000),
    "network.0.block.1.weight_v": torch.tensor(2.424097),
    "network.0.block.4.bias": torch.tensor(-0.189329),
    "network.0.block.4.weight_g": torch.tensor(2.000000),
    "network.0.block.4.weight_v": torch.tensor(-0.275666),
    "network.0.block.8.bias": torch.tensor(-2.300243),
    "network.0.block.8.weight_g": torch.tensor(1.442972),
    "network.0.block.8.weight_v": torch.tensor(-2.289390),
    "network.0.other_block.1.bias": torch.tensor(-0.765345),
    "network.0.other_block.1.weight_g": torch.tensor(2.000000),
    "network.0.other_block.1.weight_v": torch.tensor(2.196348),
    "network.0.other_block.4.bias": torch.tensor(-1.368577),
    "network.0.other_block.4.weight_g": torch.tensor(2.000000),
    "network.0.other_block.4.weight_v": torch.tensor(-1.501767),
    "network.0.other_block.7.bias": torch.tensor(0.790266),
    "network.0.other_block.7.weight_g": torch.tensor(2.000000),
    "network.0.other_block.7.weight_v": torch.tensor(0.562463),
    "network.1.shortcut.0.bias": torch.tensor(0.895958),
    "network.1.shortcut.0.weight_g": torch.tensor(1.310875),
    "network.1.shortcut.0.weight_v": torch.tensor(-0.262897),
    "network.1.block.1.bias": torch.tensor(-1.379132),
    "network.1.block.1.weight_g": torch.tensor(2.000000),
    "network.1.block.1.weight_v": torch.tensor(-0.842446),
    "network.1.block.4.bias": torch.tensor(0.407032),
    "network.1.block.4.weight_g": torch.tensor(2.000000),
    "network.1.block.4.weight_v": torch.tensor(2.141014),
    "network.1.block.8.bias": torch.tensor(2.652335),
    "network.1.block.8.weight_g": torch.tensor(2.000000),
    "network.1.block.8.weight_v": torch.tensor(1.014063),
    "network.1.other_block.1.bias": torch.tensor(-0.808023),
    "network.1.other_block.1.weight_g": torch.tensor(2.000000),
    "network.1.other_block.1.weight_v": torch.tensor(-0.180521),
    "network.1.other_block.4.bias": torch.tensor(-0.260156),
    "network.1.other_block.4.weight_g": torch.tensor(2.021936),
    "network.1.other_block.4.weight_v": torch.tensor(1.384603),
    "network.1.other_block.7.bias": torch.tensor(0.373084),
    "network.1.other_block.7.weight_g": torch.tensor(2.000000),
    "network.1.other_block.7.weight_v": torch.tensor(1.344408),
    "network.2.shortcut.0.bias": torch.tensor(2.734252),
    "network.2.shortcut.0.weight_g": torch.tensor(2.759658),
    "network.2.shortcut.0.weight_v": torch.tensor(1.413496),
    "network.2.block.1.bias": torch.tensor(1.007614),
    "network.2.block.1.weight_g": torch.tensor(2.000000),
    "network.2.block.1.weight_v": torch.tensor(1.617141),
    "network.2.block.4.bias": torch.tensor(1.338760),
    "network.2.block.4.weight_g": torch.tensor(4.000000),
    "network.2.block.4.weight_v": torch.tensor(1.427749),
    "network.2.block.8.bias": torch.tensor(-1.078366),
    "network.2.block.8.weight_g": torch.tensor(2.000000),
    "network.2.block.8.weight_v": torch.tensor(2.550148),
    "network.2.other_block.1.bias": torch.tensor(-0.989810),
    "network.2.other_block.1.weight_g": torch.tensor(1.404594),
    "network.2.other_block.1.weight_v": torch.tensor(-1.414033),
    "network.2.other_block.4.bias": torch.tensor(-3.240095),
    "network.2.other_block.4.weight_g": torch.tensor(2.000000),
    "network.2.other_block.4.weight_v": torch.tensor(2.699658),
    "network.2.other_block.7.bias": torch.tensor(1.564060),
    "network.2.other_block.7.weight_g": torch.tensor(2.000000),
    "network.2.other_block.7.weight_v": torch.tensor(1.841271),
    "network.3.bias": torch.tensor(-0.579897),
    "network.3.weight_g": torch.tensor(-2.105088),
    "network.3.weight_v": torch.tensor(1.271017),
    "network.6.bias": torch.tensor(0.263355),
    "network.6.weight_g": torch.tensor(-0.738752),
    "network.6.weight_v": torch.tensor(1.744133),
    "condition.bias": torch.tensor(1.185164),
    "condition.weight_g": torch.tensor(-2.169669),
    "condition.weight_v": torch.tensor(-1.653843),
}

_expected_grads = {
    "encoder.embed_metadata.0.weight": torch.tensor(-0.005022),
    "encoder.embed_metadata.1.weight": torch.tensor(0.121203),
    "pre_net.1.bias": torch.tensor(-0.000001),
    "pre_net.1.weight_g": torch.tensor(-0.167925),
    "pre_net.1.weight_v": torch.tensor(0.089933),
    "pre_net.2.weight": torch.tensor(-0.337646),
    "pre_net.2.bias": torch.tensor(-0.007025),
    "network.0.shortcut.0.bias": torch.tensor(-0.053820),
    "network.0.shortcut.0.weight_g": torch.tensor(0.470699),
    "network.0.shortcut.0.weight_v": torch.tensor(0.017525),
    "network.0.block.1.bias": torch.tensor(0.008283),
    "network.0.block.1.weight_g": torch.tensor(0.018947),
    "network.0.block.1.weight_v": torch.tensor(-0.010229),
    "network.0.block.4.bias": torch.tensor(0.046683),
    "network.0.block.4.weight_g": torch.tensor(0.013338),
    "network.0.block.4.weight_v": torch.tensor(0.165964),
    "network.0.block.8.bias": torch.tensor(-0.053820),
    "network.0.block.8.weight_g": torch.tensor(0.054566),
    "network.0.block.8.weight_v": torch.tensor(-0.196224),
    "network.0.other_block.1.bias": torch.tensor(-0.000762),
    "network.0.other_block.1.weight_g": torch.tensor(-0.000485),
    "network.0.other_block.1.weight_v": torch.tensor(0.001860),
    "network.0.other_block.4.bias": torch.tensor(-0.016758),
    "network.0.other_block.4.weight_g": torch.tensor(-0.001833),
    "network.0.other_block.4.weight_v": torch.tensor(0.005866),
    "network.0.other_block.7.bias": torch.tensor(-0.052985),
    "network.0.other_block.7.weight_g": torch.tensor(0.001660),
    "network.0.other_block.7.weight_v": torch.tensor(0.036058),
    "network.1.shortcut.0.bias": torch.tensor(0.016888),
    "network.1.shortcut.0.weight_g": torch.tensor(0.764580),
    "network.1.shortcut.0.weight_v": torch.tensor(0.755654),
    "network.1.block.1.bias": torch.tensor(0.117015),
    "network.1.block.1.weight_g": torch.tensor(0.232476),
    "network.1.block.1.weight_v": torch.tensor(0.017425),
    "network.1.block.4.bias": torch.tensor(0.069414),
    "network.1.block.4.weight_g": torch.tensor(0.098588),
    "network.1.block.4.weight_v": torch.tensor(0.163732),
    "network.1.block.8.bias": torch.tensor(0.016888),
    "network.1.block.8.weight_g": torch.tensor(0.143607),
    "network.1.block.8.weight_v": torch.tensor(-0.282731),
    "network.1.other_block.1.bias": torch.tensor(0.079360),
    "network.1.other_block.1.weight_g": torch.tensor(0.093194),
    "network.1.other_block.1.weight_v": torch.tensor(0.072368),
    "network.1.other_block.4.bias": torch.tensor(-0.040673),
    "network.1.other_block.4.weight_g": torch.tensor(0.075578),
    "network.1.other_block.4.weight_v": torch.tensor(-0.045973),
    "network.1.other_block.7.bias": torch.tensor(0.093232),
    "network.1.other_block.7.weight_g": torch.tensor(0.126495),
    "network.1.other_block.7.weight_v": torch.tensor(0.322689),
    "network.2.shortcut.0.bias": torch.tensor(-0.024796),
    "network.2.shortcut.0.weight_g": torch.tensor(0.784628),
    "network.2.shortcut.0.weight_v": torch.tensor(-0.652103),
    "network.2.block.1.bias": torch.tensor(0.035368),
    "network.2.block.1.weight_g": torch.tensor(0.266133),
    "network.2.block.1.weight_v": torch.tensor(0.452084),
    "network.2.block.4.bias": torch.tensor(-0.069046),
    "network.2.block.4.weight_g": torch.tensor(0.224250),
    "network.2.block.4.weight_v": torch.tensor(-1.719246),
    "network.2.block.8.bias": torch.tensor(-0.024796),
    "network.2.block.8.weight_g": torch.tensor(0.262480),
    "network.2.block.8.weight_v": torch.tensor(0.334629),
    "network.2.other_block.1.bias": torch.tensor(0.019532),
    "network.2.other_block.1.weight_g": torch.tensor(-0.088311),
    "network.2.other_block.1.weight_v": torch.tensor(-0.045489),
    "network.2.other_block.4.bias": torch.tensor(0.042766),
    "network.2.other_block.4.weight_g": torch.tensor(0.093596),
    "network.2.other_block.4.weight_v": torch.tensor(0.099916),
    "network.2.other_block.7.bias": torch.tensor(-0.015736),
    "network.2.other_block.7.weight_g": torch.tensor(0.094327),
    "network.2.other_block.7.weight_v": torch.tensor(-0.440499),
    "network.3.bias": torch.tensor(0.218567),
    "network.3.weight_g": torch.tensor(-0.650937),
    "network.3.weight_v": torch.tensor(-2.625804),
    "network.6.bias": torch.tensor(6.847976),
    "network.6.weight_g": torch.tensor(0.412710),
    "network.6.weight_v": torch.tensor(-1.729314),
    "condition.bias": torch.tensor(-0.002321),
    "condition.weight_g": torch.tensor(0.209711),
    "condition.weight_v": torch.tensor(-0.306043),
}

_expected_signal = [
    0.157694,
    0.060988,
    0.144178,
    0.132551,
    0.132334,
    0.098022,
    0.132578,
    0.132578,
    0.136925,
    0.094106,
    0.163105,
    0.132575,
    0.073113,
    0.135382,
    0.132185,
    0.131859,
]
_expected_signal = torch.tensor(_expected_signal)


def _side_effect(num_embeddings: int, *args, padding_idx=None, **kwargs):
    """Side-effect used in `_make_backward_compatible_model` for creating the `Embedding`.

    TODO: Remove and update `test_signal_model__version` values.
    """
    default_tokens = len(lib.utils.NumeralizePadEmbed._Tokens)
    return Embedding(num_embeddings - default_tokens, *args, padding_idx=None, **kwargs)


def _make_backward_compatible_model(config: _Config):
    """Set `Embedding` in a backward compatible way so `test_signal_model__version` passes.

    TODO: Remove and update `test_signal_model__version` values.
    """
    with mock.patch("lib.utils.torch.nn.Embedding") as module:
        module.side_effect = _side_effect
        model, *other = _make_small_signal_model(config)

    for embed, max_values in zip(model.encoder.embed_metadata, config.max_seq_meta_values):
        embed = typing.cast(lib.utils.NumeralizePadEmbed, embed)
        embed.vocab.update({i: i for i in range(max_values)})
        embed.num_embeddings = len(embed.vocab)

    return (model, *other)


def test_signal_model__version():
    """Test `SignalModel` has not changed since it was last tested."""
    torch.set_printoptions(precision=6, linewidth=100)

    with fork_rng(123):
        config = _Config(batch_size=16)
        model, (_, seq_metadata, _) = _make_backward_compatible_model(config)
        padded_num_frames = config.num_frames + model.padding * 2
        spectrogram = torch.randn(config.batch_size, padded_num_frames, config.frame_size)
        spectrogram_len = torch.randint(model.padding + 1, padded_num_frames, (config.batch_size,))
        spectrogram_len[-1] = padded_num_frames
        spectrogram_mask = lengths_to_mask(spectrogram_len)

        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(-0.27735))

        signal = model(spectrogram, seq_metadata, spectrogram_mask, pad_input=False)

        _utils.print_params("_expected_parameters", model.named_parameters())
        for name, parameter in model.named_parameters():
            assert_almost_equal(_expected_parameters[name], parameter.sum())

        print("Signal", signal.sum(dim=-1))
        assert_almost_equal(signal.sum(dim=-1), _expected_signal)

        signal.sum().backward()

        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _utils.print_params("_expected_grads", grads)
        for name, grad in grads:
            assert_almost_equal(_expected_grads[name], grad.sum())

        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.555366))
