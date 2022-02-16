import itertools
import math
import random
import types
import typing
from unittest import mock

import hparams
import pytest
import torch
import torch.nn
from hparams import HParams
from torch.nn import Embedding
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchnlp.random import fork_rng
from torchnlp.utils import lengths_to_mask

import lib
from lib.spectrogram_model import Inputs, Mode, SpectrogramModel
from lib.spectrogram_model.attention import Attention
from lib.spectrogram_model.containers import AttentionHiddenState, DecoderHiddenState, Encoded
from lib.spectrogram_model.decoder import Decoder
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=5)


class _Config(typing.NamedTuple):
    max_tokens: int = 17
    max_seq_meta_values: typing.Tuple[int, int] = (3, 5)
    num_frame_channels: int = 6
    batch_size: int = 5
    max_frames: int = 5
    max_num_tokens: int = 6
    max_tokens_index: int = 0

    @property
    def max_frames_per_token(self) -> float:
        return self.max_frames / self.max_num_tokens


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    hparams.clear_config()


def _make_spectrogram_model(
    config: _Config,
    seq_meta_embed_size: int = 8,
    output_scalar: float = 1.2,
    stop_threshold: float = 0.5,
    dropout: float = 0.5,
    window_length: int = 3,
    stop_token_eps: float = 1e-10,
) -> SpectrogramModel:
    """Make `spectrogram_model.SpectrogramModel` for testing."""
    hparams_config = {
        lib.spectrogram_model.encoder.Encoder.__init__: HParams(
            seq_meta_embed_dropout=dropout,
            out_size=16,
            hidden_size=16,
            num_convolution_layers=2,
            convolution_filter_size=3,
            lstm_layers=1,
            dropout=dropout,
        ),
        lib.spectrogram_model.decoder.Decoder.__init__: HParams(
            pre_net_size=16,
            lstm_hidden_size=16,
            encoder_output_size=16,
            stop_net_dropout=dropout,
        ),
        lib.spectrogram_model.pre_net.PreNet.__init__: HParams(num_layers=1, dropout=dropout),
        lib.spectrogram_model.attention.Attention.__init__: HParams(
            hidden_size=4,
            convolution_filter_size=3,
            dropout=dropout,
            window_length=window_length,
            avg_frames_per_token=1.0,
        ),
    }
    hparams.add_config(hparams_config)
    model = SpectrogramModel(
        max_tokens=config.max_tokens,
        max_seq_meta_values=config.max_seq_meta_values,
        seq_meta_embed_size=seq_meta_embed_size,
        num_frame_channels=config.num_frame_channels,
        max_frames_per_token=config.max_frames_per_token,
        output_scalar=output_scalar,
        stop_threshold=stop_threshold,
        stop_token_eps=stop_token_eps,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    [torch.nn.init.normal_(p) for p in model.parameters() if p.std() == 0]

    return model


def _make_inputs(
    config: _Config,
) -> typing.Tuple[Inputs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make `spectrogram_model.SpectrogramModel` inputs for testing."""
    long_ = torch.long

    # NOTE: `1` and `transpose(0, 1)` is set for backwards compatibility so that same random numbers
    # are generated.
    # TODO: Remove and update `test_spectrogram_model__version` values.
    tokens_size = (config.max_num_tokens, config.batch_size)
    tokens = torch.randint(1, config.max_tokens, tokens_size).transpose(0, 1).tolist()
    speakers = torch.randint(0, config.max_seq_meta_values[0], (config.batch_size,)).tolist()
    sessions = torch.randint(0, config.max_seq_meta_values[1], (config.batch_size,)).tolist()

    num_tokens = torch.randint(1, config.max_num_tokens, (config.batch_size,), dtype=long_)
    # NOTE: Ensure at least one sequence is `max_num_tokens`.
    num_tokens[config.max_tokens_index] = config.max_num_tokens
    for i in range(config.batch_size):
        tokens[i] = tokens[i][: num_tokens[i]]

    inputs = Inputs(tokens, list(zip(speakers, sessions)))

    target_frames = torch.randn(config.max_frames, config.batch_size, config.num_frame_channels)
    target_lengths = torch.randint(1, config.max_frames, (config.batch_size,), dtype=long_)
    target_lengths[-1] = config.max_frames  # NOTE: Ensure at least one sequence is `max_frames`.
    target_mask = lengths_to_mask(target_lengths).transpose(0, 1)  # [num_frames, batch_size]

    return inputs, num_tokens, target_frames, target_mask, target_lengths


def _logit(x: torch.Tensor) -> torch.Tensor:
    """Learn more: https://github.com/pytorch/pytorch/issues/37060

    Example:
        >>> torch.sigmoid(_logit(torch.tensor(0.5)))
        tensor(0.5000)
        >>> torch.sigmoid(_logit(torch.tensor(0.25)))
        tensor(0.2500)
        >>> torch.sigmoid(_logit(torch.tensor(0.9)))
        tensor(0.9000)
    """
    return torch.log(x) - torch.log1p(-x)


def _rand_logit(*shape: int, offset=0) -> torch.Tensor:
    """`_logit(torch.rand(*shape))` where the random distribution for each index is independent.

    Args:
        *shape: The shape of the returned tensor.
        offset: Offset the `torch` random number generator by executing it `offset` times.

    Example:
        >>> from torchnlp.random import set_seed
        >>> set_seed(123); _make_size_invariant_random_tensor(1)
        tensor([0.4721])
        >>> set_seed(123); _make_size_invariant_random_tensor(2)
        tensor([0.4721, 1.2948])
        >>> set_seed(123); _make_size_invariant_random_tensor(3)
        tensor([ 0.4721,  1.2948, -0.0914])
    """
    with fork_rng(random.randint(0, 2 ** 16)):
        return_ = torch.zeros(*shape)
        for i in range(offset):
            torch.rand(1)
        for index in itertools.product(*tuple(range(i) for i in shape)):
            return_[index] = _logit(torch.rand(1))
    return return_


def _get_index_first_nonzero(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Get index of the first nonzero value."""
    # Learn more:
    # https://discuss.pytorch.org/t/first-nonzero-index/24769/7
    non_zero = tensor > 0
    value, index = ((non_zero.float().cumsum(dim=dim) == 1) & non_zero).max(dim=dim)
    index[value == 0] = -1
    return index


def _mock_model(model: SpectrogramModel) -> typing.Callable[[int], None]:
    """Mock `model` such that...
    - `stop_token` output is invariant to the batch size and sequence length.
    - `stop_token` output is sampled from a uniform distribution.
    - `hidden_state.attention_hidden_state.window_start` updates incrementally.

    NOTE: Without this mock, a randomly initialized model might never stop generating or it might
    stop generating immediately.
    """
    _decoder_forward = model.decoder.forward
    _attention_forward = model.decoder.attention.forward
    window_length = model.decoder.attention.window_length
    offset = 0

    def set_stop_token_rand_offset(new_offset):
        nonlocal offset
        offset = new_offset

    def attention_forward(
        self: Attention,
        encoded: Encoded,
        query: torch.Tensor,
        hidden_state: AttentionHiddenState,
        token_skip_warning: int,
    ):
        window_start = hidden_state.window_start
        cumulative_alignment_padding = self.cumulative_alignment_padding
        slice_ = slice(cumulative_alignment_padding, -cumulative_alignment_padding)
        first_token = hidden_state.cumulative_alignment[:, slice_].sum() == 0
        context, alignment, hidden_state = _attention_forward(
            encoded, query, hidden_state, token_skip_warning
        )
        # NOTE: On the first iteration, `window_start` should not advance because it needs to
        # focus on the first token.
        window_start = (
            window_start.zero_()
            if first_token
            else torch.clamp(torch.min(window_start + 1, encoded.num_tokens - window_length), min=0)
        )
        return context, alignment, hidden_state._replace(window_start=window_start)

    model.decoder.attention.forward = types.MethodType(attention_forward, model.decoder.attention)

    def decoder_forward(self: Decoder, *args, hidden_state: DecoderHiddenState, **kwargs):
        out = _decoder_forward(*args, hidden_state=hidden_state, **kwargs)  # type: ignore
        iter_ = range(out.stop_tokens.shape[0])
        tokens = torch.stack([_rand_logit(out.stop_tokens.shape[1], offset=offset) for _ in iter_])
        return out._replace(stop_tokens=tokens)

    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)

    return set_stop_token_rand_offset


def test_spectrogram_model():
    """Test `spectrogram_model.SpectrogramModel` handles a basic case."""
    with fork_rng(123):
        config = _Config(batch_size=1)
        inputs, *_ = _make_inputs(config)
        model = _make_spectrogram_model(config)
        _mock_model(model)

        frames, stop_tokens, alignments, lengths, reached_max = model(
            inputs, mode=Mode.INFER, use_tqdm=True
        )

        assert frames.dtype == torch.float
        assert frames.shape == (lengths.max(), config.batch_size, model.num_frame_channels)
        assert stop_tokens.dtype == torch.float
        assert stop_tokens.shape == (lengths.max(), config.batch_size)
        assert alignments.dtype == torch.float
        assert alignments.shape == (lengths.max(), config.batch_size, config.max_num_tokens)
        assert lengths.shape == (1, config.batch_size)
        for i, length in enumerate(lengths[0].tolist()):
            assert length > 0
            assert length <= config.max_frames
            thresholded = torch.sigmoid(stop_tokens[length - 1, i]) >= model.stop_threshold
            assert thresholded or reached_max[:, i]
        assert reached_max.dtype == torch.bool
        assert reached_max.sum().item() >= 0


def test_spectrogram_model__train():
    """Test `spectrogram_model.SpectrogramModel` handles a basic training case."""
    config = _Config()
    inputs, _, target_frames, target_mask, _ = _make_inputs(config)
    model = _make_spectrogram_model(config)
    _mock_model(model)

    preds = model(inputs, target_frames, target_mask=target_mask)

    assert preds.frames.dtype == torch.float
    assert preds.frames.shape == (config.max_frames, config.batch_size, config.num_frame_channels)
    assert preds.stop_tokens.dtype == torch.float
    assert preds.stop_tokens.shape == (config.max_frames, config.batch_size)
    assert preds.alignments.dtype == torch.float
    assert preds.alignments.shape == (config.max_frames, config.batch_size, config.max_num_tokens)
    (preds.frames.sum() + preds.stop_tokens.sum()).backward()


def test_spectrogram_model__reached_max_all():
    """Test `spectrogram_model.SpectrogramModel` handles `reached_max`."""
    config = _Config(batch_size=32)
    inputs, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)

    # NOTE: Make sure that stop-token is not predicted; therefore, reaching `max_frames_per_token`.
    weight = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].weight)
    torch.nn.init.constant_(weight, -math.inf)
    bias = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].bias)
    torch.nn.init.constant_(bias, -math.inf)

    preds = model(inputs, mode=Mode.INFER)

    assert preds.frames.dtype == torch.float
    assert preds.frames.shape == (config.max_frames, config.batch_size, config.num_frame_channels)
    assert preds.stop_tokens.dtype == torch.float
    assert preds.stop_tokens.shape == (config.max_frames, config.batch_size)
    assert preds.alignments.dtype == torch.float
    assert preds.alignments.shape == (config.max_frames, config.batch_size, config.max_num_tokens)
    assert preds.lengths.shape == (1, config.batch_size)
    assert preds.reached_max.dtype == torch.bool
    assert preds.reached_max.sum().item() == config.batch_size


def test_spectrogram_model__is_stop():
    """Test `spectrogram_model.SpectrogramModel._is_stop` basic cases."""
    config = _Config()
    model = _make_spectrogram_model(config, window_length=3, stop_threshold=0.5)
    tensor = torch.tensor
    _is_stop = lambda a, b, c, d: model._is_stop(_logit(tensor(a)), tensor(b), tensor(c), tensor(d))
    # NOTE: For example, test that this handles a scenario where the window intersects the boundary
    # and `stop_token` is above threshold.
    assert _is_stop(1.0, 8, 5, False)[0]
    assert not _is_stop(1.0, 8, 4, False)[0]
    assert not _is_stop(0.25, 8, 5, False)[0]
    assert _is_stop(0.25, 8, 4, True)[0]


def test_spectrogram_model__stop():
    """Test `spectrogram_model.SpectrogramModel` `stop_tokens` is consistent with `lengths`,
    `window_start`, `window_length` and masking."""
    with fork_rng(123):
        config = _Config(batch_size=16, max_frames=8)
        inputs, num_tokens, *_ = _make_inputs(config)
        window_length = 3
        model = _make_spectrogram_model(config, window_length=window_length)
        _mock_model(model)

        preds = model(inputs, mode=Mode.INFER)

        max_lengths = (num_tokens.float() * config.max_frames_per_token).long()
        max_lengths = torch.clamp(max_lengths, min=1)
        threshold = torch.sigmoid(preds.stop_tokens) >= model.stop_threshold
        for i in range(config.batch_size):  # NOTE: Only stop if the window includes the last token.
            min_index = torch.clamp_min(num_tokens[i] - window_length, 0).item()
            min_index = typing.cast(int, min_index)
            threshold[:min_index, i] = False
        stopped_index = _get_index_first_nonzero(threshold)
        stopped_index[stopped_index == -1] = max_lengths[stopped_index == -1] - 1
        expected_length = torch.min(stopped_index + 1, max_lengths)
        assert_almost_equal(preds.lengths.squeeze(0), expected_length)

        for i in range(config.batch_size):
            assert preds.frames[typing.cast(int, preds.lengths[:, i].item()) :, i].sum() == 0


def test_spectrogram_model__infer_train():
    """Test `spectrogram_model.SpectrogramModel` outputs for train and infer are consistent."""
    config = _Config()
    inputs, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        preds = model(inputs, mode=Mode.INFER)

    with fork_rng(seed=123):
        aligned_preds = model(
            inputs,
            target_frames=preds.frames,
            target_mask=lengths_to_mask(preds.lengths).transpose(0, 1),
            mode=Mode.FORWARD,
        )

    assert_almost_equal(preds.frames, aligned_preds.frames)
    assert_almost_equal(preds.stop_tokens, aligned_preds.stop_tokens)
    assert_almost_equal(preds.alignments, aligned_preds.alignments)


def _set_embedding_vocab(model: SpectrogramModel, config: _Config):
    """Update `model` vocab so it can be run in inference mode."""
    model.encoder.embed_token.update_tokens(list(range(config.max_tokens)))
    for i, max_values in enumerate(config.max_seq_meta_values):
        embedding = typing.cast(lib.utils.PaddingAndLazyEmbedding, model.encoder.embed_metadata[i])
        embedding.update_tokens(list(range(max_values)))


def test_spectrogram_model__infer_generate():
    """Test `spectrogram_model.SpectrogramModel` outputs for infer and generate are consistent."""
    config = _Config()
    inputs, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        _set_embedding_vocab(model, config)
        preds = model.eval()(inputs, mode=Mode.INFER)

    for i in [1, 8, 11]:
        with fork_rng(seed=123):
            generator = model(inputs, mode=Mode.GENERATE, split_size=i)
            generated = tuple(zip(*list(generator)))

        assert_almost_equal(preds.frames, torch.cat(generated[0]))
        assert_almost_equal(preds.stop_tokens, torch.cat(generated[1]))
        assert_almost_equal(preds.alignments, torch.cat(generated[2]))
        assert_almost_equal(preds.lengths, generated[3][-1])
        assert_almost_equal(preds.reached_max, generated[4][-1])


# NOTE: The random generator for dropout varies based on the tensor size; therefore, it's
# dependent on the `BatchSize` and we need to disable it. For example:
# >>> import torch
# >>> torch.manual_seed(123)
# >>> batch_dropout = torch.nn.functional.dropout(torch.ones(5, 5))
# >>> torch.manual_seed(123)
# >>> dropout = torch.nn.functional.dropout(torch.ones(5))
# >>> batch_dropout[0] != dropout


def test_spectrogram_model__infer_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` infer ouput is batch and padding invariant."""
    config = _Config()
    inputs, num_tokens, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    set_stop_token_rand_offset = _mock_model(model)

    with fork_rng(seed=123):
        _set_embedding_vocab(model, config)
        batch_preds = model.eval()(inputs, mode=Mode.INFER)

    for i in range(config.batch_size):
        set_stop_token_rand_offset(i)
        num_tokens_ = typing.cast(int, num_tokens[i].item())
        with fork_rng(seed=123):
            inputs_ = inputs._replace(
                tokens=[t[:num_tokens_] for t in inputs.tokens][i : i + 1],
                metadata=inputs.metadata[i : i + 1],
            )
            preds = model(inputs_, mode=Mode.INFER)

        length = typing.cast(int, batch_preds.lengths[0, i].item())
        assert_almost_equal(preds.reached_max, batch_preds.reached_max[:, i : i + 1])
        assert_almost_equal(preds.frames, batch_preds.frames[:length, i : i + 1])
        assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i : i + 1])
        batch_preds_alignments = batch_preds.alignments[:length, i : i + 1, :num_tokens_]
        assert_almost_equal(preds.alignments, batch_preds_alignments)
        assert_almost_equal(preds.lengths, batch_preds.lengths[:, i : i + 1])


def test_spectrogram_model__train_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` train ouput is batch and padding invariant.
    Additionally, this tests inputting a tensor without a batch dimension."""
    config = _Config(batch_size=5)
    batch_inputs, _, target_frames, target_mask, target_lengths = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)
    i = config.max_tokens_index
    padding = 3
    num_tokens = config.max_num_tokens - padding
    batch_inputs.tokens[i] = batch_inputs.tokens[i][:num_tokens]
    target_lengths[i] = config.max_frames - padding

    with fork_rng(seed=123):
        target_mask = lengths_to_mask(target_lengths).transpose(0, 1)
        batch_preds = model(batch_inputs, target_frames=target_frames, target_mask=target_mask)
        (batch_preds.frames[:, i].sum() + batch_preds.stop_tokens[:, i].sum()).backward()
        batch_grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    length = typing.cast(int, target_lengths[i].item())
    inputs = batch_inputs._replace(
        tokens=[t[:num_tokens] for t in batch_inputs.tokens][i : i + 1],
        metadata=batch_inputs.metadata[i : i + 1],
    )

    with fork_rng(seed=123):
        target_mask = lengths_to_mask(length).transpose(0, 1)
        target_frames = target_frames[:length, i : i + 1]
        preds = model(inputs, target_frames=target_frames, target_mask=target_mask)
        (preds.frames.sum() + preds.stop_tokens.sum()).backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    assert_almost_equal(preds.frames, batch_preds.frames[:length, i : i + 1])
    assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i : i + 1])
    assert_almost_equal(preds.alignments, batch_preds.alignments[:length, i : i + 1, :num_tokens])
    [assert_almost_equal(r, e) for r, e in zip(grad, batch_grad)]


_expected_parameters = {
    "encoder.embed_metadata.0.weight": torch.tensor(-3.281343),
    "encoder.embed_metadata.1.weight": torch.tensor(0.318184),
    "encoder.embed_token.weight": torch.tensor(-4.785396),
    "encoder.embed.0.weight": torch.tensor(0.664301),
    "encoder.embed.0.bias": torch.tensor(-0.198331),
    "encoder.embed.2.weight": torch.tensor(-1.639882),
    "encoder.embed.2.bias": torch.tensor(4.312972),
    "encoder.conv_layers.0.1.weight": torch.tensor(-1.119497),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.490040),
    "encoder.conv_layers.1.1.weight": torch.tensor(2.456295),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.185969),
    "encoder.norm_layers.0.weight": torch.tensor(1.655902),
    "encoder.norm_layers.0.bias": torch.tensor(-3.722785),
    "encoder.norm_layers.1.weight": torch.tensor(-7.264867),
    "encoder.norm_layers.1.bias": torch.tensor(-2.107303),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(0.409841),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(5.275919),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(-1.311591),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-0.637413),
    "encoder.lstm.rnn_layers.0.0.initial_hidden_state": torch.tensor(1.588171),
    "encoder.lstm.rnn_layers.0.0.initial_cell_state": torch.tensor(-1.368738),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-2.351552),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-4.956273),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(-1.248557),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(-1.170437),
    "encoder.lstm.rnn_layers.0.1.initial_hidden_state": torch.tensor(2.101204),
    "encoder.lstm.rnn_layers.0.1.initial_cell_state": torch.tensor(4.115802),
    "encoder.lstm_norm.weight": torch.tensor(3.985575),
    "encoder.lstm_norm.bias": torch.tensor(-6.604763),
    "encoder.project_out.1.weight": torch.tensor(-2.717630),
    "encoder.project_out.1.bias": torch.tensor(0.226285),
    "encoder.project_out.2.weight": torch.tensor(5.298065),
    "encoder.project_out.2.bias": torch.tensor(3.595932),
    "decoder.initial_state.0.weight": torch.tensor(-0.311898),
    "decoder.initial_state.0.bias": torch.tensor(-0.189921),
    "decoder.initial_state.2.weight": torch.tensor(0.960531),
    "decoder.initial_state.2.bias": torch.tensor(-0.015286),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-4.600914),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(0.042175),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(2.474819),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(1.660940),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(10.139328),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(1.214987),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(1.377370),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(1.508063),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(2.194119),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(-0.949346),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(-7.278845),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-7.797914),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(0.398802),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-0.777656),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(0.557445),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(-5.485558),
    "decoder.attention.alignment_conv.weight": torch.tensor(-1.962989),
    "decoder.attention.alignment_conv.bias": torch.tensor(-0.437283),
    "decoder.attention.project_query.weight": torch.tensor(-0.806193),
    "decoder.attention.project_query.bias": torch.tensor(0.339169),
    "decoder.attention.project_scores.1.weight": torch.tensor(0.543799),
    "decoder.linear_out.weight": torch.tensor(0.763836),
    "decoder.linear_out.bias": torch.tensor(0.063751),
    "decoder.linear_stop_token.1.weight": torch.tensor(1.014468),
    "decoder.linear_stop_token.1.bias": torch.tensor(-0.203153),
}

_expected_grads = {
    "encoder.embed_metadata.0.weight": torch.tensor(-7.907637),
    "encoder.embed_metadata.1.weight": torch.tensor(-11.760118),
    "encoder.embed_token.weight": torch.tensor(-3.497558),
    "encoder.embed.0.weight": torch.tensor(-14.660476),
    "encoder.embed.0.bias": torch.tensor(1.748332),
    "encoder.embed.2.weight": torch.tensor(2.589019),
    "encoder.embed.2.bias": torch.tensor(-0.990012),
    "encoder.conv_layers.0.1.weight": torch.tensor(-6.184843),
    "encoder.conv_layers.0.1.bias": torch.tensor(-1.638364),
    "encoder.conv_layers.1.1.weight": torch.tensor(-15.923275),
    "encoder.conv_layers.1.1.bias": torch.tensor(-2.013168),
    "encoder.norm_layers.0.weight": torch.tensor(7.147020),
    "encoder.norm_layers.0.bias": torch.tensor(-3.396744),
    "encoder.norm_layers.1.weight": torch.tensor(-4.972385),
    "encoder.norm_layers.1.bias": torch.tensor(0.294768),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(-28.238379),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(-0.522231),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(1.671067),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(1.671067),
    "encoder.lstm.rnn_layers.0.0.initial_hidden_state": torch.tensor(-0.577697),
    "encoder.lstm.rnn_layers.0.0.initial_cell_state": torch.tensor(-0.550854),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-25.273197),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(0.215798),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(1.894700),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(1.894700),
    "encoder.lstm.rnn_layers.0.1.initial_hidden_state": torch.tensor(0.136555),
    "encoder.lstm.rnn_layers.0.1.initial_cell_state": torch.tensor(0.478012),
    "encoder.lstm_norm.weight": torch.tensor(6.709639),
    "encoder.lstm_norm.bias": torch.tensor(3.587944),
    "encoder.project_out.1.weight": torch.tensor(-0.000019),
    "encoder.project_out.1.bias": torch.tensor(0.000000),
    "encoder.project_out.2.weight": torch.tensor(11.672823),
    "encoder.project_out.2.bias": torch.tensor(19.259249),
    "decoder.initial_state.0.weight": torch.tensor(-0.061283),
    "decoder.initial_state.0.bias": torch.tensor(0.164277),
    "decoder.initial_state.2.weight": torch.tensor(-2.805097),
    "decoder.initial_state.2.bias": torch.tensor(-0.309708),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-0.162394),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(0.000772),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(0.048216),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(0.003243),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(1.538284),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-0.367154),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(-0.012093),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-0.012093),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(0.007095),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(-0.297339),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(19.436945),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(5.100532),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(2.016759),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(2.016759),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(-0.185272),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(2.316336),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.062329),
    "decoder.attention.alignment_conv.bias": torch.tensor(-0.036579),
    "decoder.attention.project_query.weight": torch.tensor(0.053658),
    "decoder.attention.project_query.bias": torch.tensor(-0.036579),
    "decoder.attention.project_scores.1.weight": torch.tensor(0.423372),
    "decoder.linear_out.weight": torch.tensor(16.593655),
    "decoder.linear_out.bias": torch.tensor(-9.847878),
    "decoder.linear_stop_token.1.weight": torch.tensor(-2.556858),
    "decoder.linear_stop_token.1.bias": torch.tensor(3.230159),
}


# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
_expected_frames = [
    [0.410448, 0.380882, 0.407706, 0.435772, 0.489131, 0.557615, 0.000000, 0.000000],
    [-2.895171, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.058867, -0.066061, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.821297, -0.717726, -0.601379, -0.494431, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.190225, 0.211858, 0.258358, 0.324704, 0.361927, 0.418569, 0.406192, 0.432240],
]
_expected_frames = torch.tensor(_expected_frames)
_eps = 1.000001e-10
# NOTE: For first and last sequence, the `stop_token` always predicts `_eps` because the
# `window_start` largely stays at zero and the the number of tokens is larger than the window
# length.
_expected_stop_tokens = [
    [_eps, _eps, _eps, _eps, _eps, _eps, _eps, _eps],
    [0.5033216, 0.4968749, 0.4149721, 0.4112923, 0.5192890, 0.5012044, 0.4654292, 0.4587282],
    [0.4141655, 0.4636760, 0.4464000, 0.5012614, 0.4427690, 0.3828269, 0.4381701, 0.4134280],
    [0.4401205, 0.4932628, 0.4238498, 0.5533317, 0.4784267, 0.5046895, 0.5398313, 0.5192513],
    [_eps, _eps, _eps, _eps, _eps, _eps, _eps, _eps],
]
_expected_stop_tokens = torch.tensor(_expected_stop_tokens)
_expected_alignments = [
    [3.054298, 2.351486, 2.594216, 0.000000, 0.000000, 0.000000],
    [4.961336, 3.038664, 0.000000, 0.000000, 0.000000, 0.000000],
    [4.776866, 3.223135, 0.000000, 0.000000, 0.000000, 0.000000],
    [3.158964, 2.293172, 2.547863, 0.000000, 0.000000, 0.000000],
    [2.878742, 2.453774, 2.667484, 0.000000, 0.000000, 0.000000],
]
_expected_alignments = torch.tensor(_expected_alignments)


def _side_effect(config: _Config, num_embeddings: int, *args, padding_idx=None, **kwargs):
    """Side-effect used in `_make_backward_compatible_model` for creating the `Embedding`.

    TODO: Remove and update `test_spectrogram_model__version` values.
    """
    assert all(config.max_tokens != n for n in config.max_seq_meta_values)
    default_tokens = len(lib.utils.PaddingAndLazyEmbedding._Tokens)
    padding_idx = padding_idx if num_embeddings == (config.max_tokens + default_tokens) else None
    return Embedding(num_embeddings - default_tokens, *args, padding_idx=padding_idx, **kwargs)


def _make_backward_compatible_model(config: _Config, stop_threshold=0.5):
    """Set `Embedding` in a backward compatible way so `test_spectrogram_model__version` passes.

    TODO: Remove and update `test_spectrogram_model__version` values.
    """

    with mock.patch("lib.utils.torch.nn.Embedding") as module:
        module.side_effect = lambda *a, **k: _side_effect(config, *a, **k)
        model = _make_spectrogram_model(config, stop_threshold=stop_threshold, stop_token_eps=_eps)

    model.encoder.embed_token.vocab.update({i: i for i in range(config.max_tokens)})
    model.encoder.embed_token.num_embeddings = len(model.encoder.embed_token.vocab)
    for embed, max_values in zip(model.encoder.embed_metadata, config.max_seq_meta_values):
        embed = typing.cast(lib.utils.PaddingAndLazyEmbedding, embed)
        embed.vocab.update({i: i for i in range(max_values)})
        embed.num_embeddings = len(embed.vocab)

    return model


def test_spectrogram_model__version():
    """Test `spectrogram_model.SpectrogramModel` has not changed since it was last tested."""
    torch.set_printoptions(precision=6, linewidth=100)

    with fork_rng(123):
        # TODO: Remove `max_tokens_index` and update `test_spectrogram_model__version` values.
        config = _Config(max_frames=8, max_tokens_index=-1)
        inputs, _, target_frames, target_mask, _ = _make_inputs(config)
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.162034))

    with fork_rng(123):
        model = _make_backward_compatible_model(config)
        with torch.no_grad():
            preds = model(inputs, mode=Mode.INFER)

        _utils.print_params("_expected_parameters", model.named_parameters())
        for name, parameter in model.named_parameters():
            assert_almost_equal(_expected_parameters[name], parameter.sum())
        print("Frames", preds.frames.sum(dim=-1).transpose(0, 1))
        assert_almost_equal(preds.frames.sum(dim=-1).transpose(0, 1), _expected_frames)
        print("Stop Tokens", torch.sigmoid(preds.stop_tokens.transpose(0, 1)))
        assert_almost_equal(torch.sigmoid(preds.stop_tokens.transpose(0, 1)), _expected_stop_tokens)
        print("Alignments", preds.alignments.sum(dim=0))
        assert_almost_equal(preds.alignments.sum(dim=0), _expected_alignments)
        print("Lengths", preds.lengths.squeeze(0))
        assert_almost_equal(preds.lengths.squeeze(0), torch.tensor([6, 1, 2, 4, 8]))
        print("Reached Max", preds.reached_max.squeeze(0))
        expected = torch.tensor([True, False, True, True, True])
        assert_almost_equal(preds.reached_max.squeeze(0), expected)

    with fork_rng(seed=123):
        target_frames = preds.frames
        target_mask = lengths_to_mask(preds.lengths).transpose(0, 1)
        preds = model(inputs, target_frames, target_mask=target_mask)

        spectrogram_loss = mse_loss(preds.frames, target_frames, reduction="none")
        spectrogram_loss *= target_mask.unsqueeze(2)
        target = torch.zeros(preds.frames.shape[0], config.batch_size)
        stop_token_loss = binary_cross_entropy_with_logits(
            preds.stop_tokens, target, reduction="none"
        )
        stop_token_loss *= target_mask
        (spectrogram_loss.sum() + stop_token_loss.sum()).backward()

        print("Spectrogram Loss", spectrogram_loss.sum())
        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(43.443901))
        print("Stop Token Loss", stop_token_loss.sum())
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(4.343839))
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _utils.print_params("_expected_grads", grads)
        for name, grad in grads:
            assert_almost_equal(_expected_grads[name], grad.sum())
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.3122985))
