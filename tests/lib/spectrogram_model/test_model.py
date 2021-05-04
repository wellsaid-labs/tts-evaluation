import itertools
import math
import random
import types
import typing

import hparams
import pytest
import torch
import torch.nn
from hparams import HParams
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchnlp.random import fork_rng
from torchnlp.utils import lengths_to_mask

import lib
from lib.spectrogram_model import Mode, Params, SpectrogramModel
from lib.spectrogram_model.attention import Attention, AttentionHiddenState
from lib.spectrogram_model.decoder import Decoder, DecoderHiddenState
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=5)


class _Config(typing.NamedTuple):
    vocab_size: int = 17
    num_speakers: int = 3
    num_sessions: int = 5
    num_frame_channels: int = 6
    batch_size: int = 5
    max_frames: int = 5
    max_num_tokens: int = 6
    padding_index: int = 0

    @property
    def max_frames_per_token(self) -> float:
        return self.max_frames / self.max_num_tokens


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    hparams.clear_config()


def _make_spectrogram_model(
    config: _Config,
    speaker_embedding_size: int = 8,
    output_scalar: float = 1.2,
    stop_threshold: float = 0.5,
    dropout: float = 0.5,
    padding_index: int = 0,
    window_length: int = 3,
) -> SpectrogramModel:
    """ Make `spectrogram_model.SpectrogramModel` for testing."""
    hparams_config = {
        lib.spectrogram_model.encoder.Encoder.__init__: HParams(
            out_size=16,
            hidden_size=16,
            num_convolution_layers=2,
            convolution_filter_size=3,
            lstm_layers=1,
            dropout=dropout,
            padding_index=padding_index,
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
        ),
    }
    hparams.add_config(hparams_config)
    model = SpectrogramModel(
        vocab_size=config.vocab_size,
        num_speakers=config.num_speakers,
        num_sessions=config.num_sessions,
        speaker_embedding_size=speaker_embedding_size,
        num_frame_channels=config.num_frame_channels,
        max_frames_per_token=config.max_frames_per_token,
        output_scalar=output_scalar,
        speaker_embed_dropout=dropout,
        stop_threshold=stop_threshold,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    [torch.nn.init.normal_(p) for p in model.parameters() if p.std() == 0]

    return model


def _make_inputs(config: _Config) -> typing.Tuple[Params, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Make `spectrogram_model.SpectrogramModel` inputs for testing."""
    long_ = torch.long
    size = (config.max_num_tokens, config.batch_size)
    tokens = torch.randint(config.padding_index + 1, config.vocab_size, size, dtype=long_)
    speaker = torch.randint(0, config.num_speakers, (1, config.batch_size), dtype=long_)
    session = torch.randint(0, config.num_sessions, (1, config.batch_size), dtype=long_)

    num_tokens = torch.randint(1, config.max_num_tokens, (config.batch_size,), dtype=long_)
    # NOTE: Ensure at least one sequence is `max_num_tokens`.
    num_tokens[-1] = config.max_num_tokens

    target_frames = torch.randn(config.max_frames, config.batch_size, config.num_frame_channels)
    target_lengths = torch.randint(1, config.max_frames, (config.batch_size,), dtype=long_)
    target_lengths[-1] = config.max_frames  # NOTE: Ensure at least one sequence is `max_frames`.
    target_mask = lengths_to_mask(target_lengths).transpose(0, 1)  # [num_frames, batch_size]

    return Params(tokens, speaker, session, num_tokens), target_frames, target_mask, target_lengths


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
    """ Get index of the first nonzero value. """
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
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        query: torch.Tensor,
        hidden_state: AttentionHiddenState,
        token_skip_warning: int,
    ):
        window_start = hidden_state.window_start
        cumulative_alignment_padding = self.cumulative_alignment_padding
        slice_ = slice(cumulative_alignment_padding, -cumulative_alignment_padding)
        first_token = hidden_state.cumulative_alignment[:, slice_].sum() == 0
        context, alignment, hidden_state = _attention_forward(
            tokens, tokens_mask, num_tokens, query, hidden_state, token_skip_warning
        )
        # NOTE: On the first iteration, `window_start` should not advance because it needs to
        # focus on the first token.
        window_start = (
            window_start.zero_()
            if first_token
            else torch.clamp(torch.min(window_start + 1, num_tokens - window_length), min=0)
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
    """ Test `spectrogram_model.SpectrogramModel` handles a basic case. """
    with fork_rng(123):
        config = _Config(batch_size=1)
        params, *_ = _make_inputs(config)
        model = _make_spectrogram_model(config)
        _mock_model(model)

        frames, stop_tokens, alignments, lengths, reached_max = model(
            params, mode=Mode.INFER, use_tqdm=True
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
    """ Test `spectrogram_model.SpectrogramModel` handles a basic training case. """
    config = _Config()
    params, target_frames, target_mask, _ = _make_inputs(config)
    model = _make_spectrogram_model(config)
    _mock_model(model)

    preds = model(params, target_frames, target_mask=target_mask)

    assert preds.frames.dtype == torch.float
    assert preds.frames.shape == (config.max_frames, config.batch_size, config.num_frame_channels)
    assert preds.stop_tokens.dtype == torch.float
    assert preds.stop_tokens.shape == (config.max_frames, config.batch_size)
    assert preds.alignments.dtype == torch.float
    assert preds.alignments.shape == (config.max_frames, config.batch_size, config.max_num_tokens)
    (preds.frames.sum() + preds.stop_tokens.sum()).backward()


def test_spectrogram_model__reached_max_all():
    """ Test `spectrogram_model.SpectrogramModel` handles `reached_max`. """
    config = _Config(batch_size=32)
    params, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)

    # NOTE: Make sure that stop-token is not predicted; therefore, reaching `max_frames_per_token`.
    weight = typing.cast(torch.nn.Parameter, model.decoder.linear_stop_token[-1].weight)
    torch.nn.init.constant_(weight, -math.inf)
    bias = typing.cast(torch.nn.Parameter, model.decoder.linear_stop_token[-1].bias)
    torch.nn.init.constant_(bias, -math.inf)

    preds = model(params, mode=Mode.INFER)

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
    """ Test `spectrogram_model.SpectrogramModel._is_stop` basic cases. """
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
        params, *_ = _make_inputs(config)
        window_length = 3
        model = _make_spectrogram_model(config, window_length=window_length)
        _mock_model(model)

        preds = model(params, mode=Mode.INFER)

        max_lengths = (params.num_tokens.float() * config.max_frames_per_token).long()
        max_lengths = torch.clamp(max_lengths, min=1)
        threshold = torch.sigmoid(preds.stop_tokens) >= model.stop_threshold
        for i in range(config.batch_size):  # NOTE: Only stop if the window includes the last token.
            min_index = torch.clamp_min(params.num_tokens[i] - window_length, 0).item()
            min_index = typing.cast(int, min_index)
            threshold[:min_index, i] = False
        stopped_index = _get_index_first_nonzero(threshold)
        stopped_index[stopped_index == -1] = max_lengths[stopped_index == -1] - 1
        expected_length = torch.min(stopped_index + 1, max_lengths)
        assert_almost_equal(preds.lengths.squeeze(0), expected_length)

        for i in range(config.batch_size):
            assert preds.frames[typing.cast(int, preds.lengths[:, i].item()) :, i].sum() == 0


def test_spectrogram_model__infer_train():
    """ Test `spectrogram_model.SpectrogramModel` outputs for train and infer are consistent. """
    config = _Config()
    params, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        preds = model(params, mode=Mode.INFER)

    with fork_rng(seed=123):
        aligned_preds = model(
            params,
            target_frames=preds.frames,
            target_mask=lengths_to_mask(preds.lengths).transpose(0, 1),
            mode=Mode.FORWARD,
        )

    assert_almost_equal(preds.frames, aligned_preds.frames)
    assert_almost_equal(preds.stop_tokens, aligned_preds.stop_tokens)
    assert_almost_equal(preds.alignments, aligned_preds.alignments)


def test_spectrogram_model__infer_generate():
    """ Test `spectrogram_model.SpectrogramModel` outputs for infer and generate are consistent. """
    config = _Config()
    params, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        preds = model.eval()(params, mode=Mode.INFER)

    for i in [1, 8, 11]:
        with fork_rng(seed=123):
            generator = model(params, mode=Mode.GENERATE, split_size=i)
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
    """ Test `spectrogram_model.SpectrogramModel` infer ouput is batch and padding invariant. """
    config = _Config()
    params, *_ = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    set_stop_token_rand_offset = _mock_model(model)

    with fork_rng(seed=123):
        batch_preds = model.eval()(params, mode=Mode.INFER)

    for i in range(config.batch_size):
        set_stop_token_rand_offset(i)
        num_tokens_ = typing.cast(int, params.num_tokens[i].item())
        with fork_rng(seed=123):
            params_ = params._replace(
                tokens=params.tokens[:num_tokens_, i : i + 1],
                speaker=params.speaker[:, i : i + 1],
                session=params.session[:, i : i + 1],
                num_tokens=None,
            )
            preds = model(params_, mode=Mode.INFER)

        length = typing.cast(int, batch_preds.lengths[0, i].item())
        assert_almost_equal(preds.reached_max, batch_preds.reached_max[:, i : i + 1])
        assert_almost_equal(preds.frames, batch_preds.frames[:length, i : i + 1])
        assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i : i + 1])
        assert_almost_equal(
            preds.alignments, batch_preds.alignments[:length, i : i + 1, :num_tokens_]
        )
        assert_almost_equal(preds.lengths, batch_preds.lengths[:, i : i + 1])


def test_spectrogram_model__train_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` train ouput is batch and padding invariant.
    Additionally, this tests inputting a tensor without a batch dimension."""
    config = _Config(batch_size=5)
    batch_params, target_frames, target_mask, target_lengths = _make_inputs(config)
    model = _make_spectrogram_model(config, dropout=0)
    _mock_model(model)
    i = 0
    padding = 3
    batch_params.num_tokens[i] = config.max_num_tokens - padding
    target_lengths[i] = config.max_frames - padding

    with fork_rng(seed=123):
        target_mask = lengths_to_mask(target_lengths).transpose(0, 1)
        batch_preds = model(batch_params, target_frames=target_frames, target_mask=target_mask)
        (batch_preds.frames[:, i].sum() + batch_preds.stop_tokens[:, i].sum()).backward()
        batch_grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    num_tokens = typing.cast(int, batch_params.num_tokens[i].item())
    length = typing.cast(int, target_lengths[i].item())
    params = batch_params._replace(
        tokens=batch_params.tokens[:num_tokens, i],
        speaker=batch_params.speaker[:, i],
        session=batch_params.session[:, i],
        num_tokens=batch_params.num_tokens[i],
    )

    with fork_rng(seed=123):
        preds = model(
            params,
            target_frames=target_frames[:length, i],
            target_mask=lengths_to_mask(length).transpose(0, 1),
        )
        (preds.frames.sum() + preds.stop_tokens.sum()).backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    assert_almost_equal(preds.frames, batch_preds.frames[:length, i])
    assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i])
    assert_almost_equal(preds.alignments, batch_preds.alignments[:length, i, :num_tokens])
    [assert_almost_equal(r, e) for r, e in zip(grad, batch_grad)]


_expected_parameters = {
    "embed_speaker.weight": torch.tensor(-3.281343),
    "embed_session.weight": torch.tensor(0.318184),
    "encoder.embed_token.weight": torch.tensor(-4.785396),
    "encoder.embed.0.weight": torch.tensor(0.664301),
    "encoder.embed.0.bias": torch.tensor(-0.198331),
    "encoder.embed.2.weight": torch.tensor(3.939656),
    "encoder.embed.2.bias": torch.tensor(-7.579462),
    "encoder.conv_layers.0.1.weight": torch.tensor(-1.119497),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.490040),
    "encoder.conv_layers.1.1.weight": torch.tensor(2.456295),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.185969),
    "encoder.norm_layers.0.weight": torch.tensor(-8.844258),
    "encoder.norm_layers.0.bias": torch.tensor(-5.434916),
    "encoder.norm_layers.1.weight": torch.tensor(-5.422180),
    "encoder.norm_layers.1.bias": torch.tensor(4.086608),
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
    "encoder.lstm_norm.weight": torch.tensor(-2.368324),
    "encoder.lstm_norm.bias": torch.tensor(1.298576),
    "encoder.project_out.1.weight": torch.tensor(-2.717630),
    "encoder.project_out.1.bias": torch.tensor(0.226285),
    "encoder.project_out.2.weight": torch.tensor(1.860385),
    "encoder.project_out.2.bias": torch.tensor(-3.161848),
    "decoder.initial_state.0.weight": torch.tensor(-0.311898),
    "decoder.initial_state.0.bias": torch.tensor(-0.189921),
    "decoder.initial_state.2.weight": torch.tensor(0.960531),
    "decoder.initial_state.2.bias": torch.tensor(-0.015286),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-9.554928),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(-0.229378),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(6.837050),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(3.909478),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(8.431663),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-3.706831),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(0.619256),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(2.228600),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(7.790916),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(-10.041410),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(-2.215471),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-8.092188),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(-1.975391),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-0.461659),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(3.538294),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(-2.747256),
    "decoder.attention.alignment_conv.weight": torch.tensor(-1.361115),
    "decoder.attention.alignment_conv.bias": torch.tensor(1.202135),
    "decoder.attention.project_query.weight": torch.tensor(0.222881),
    "decoder.attention.project_query.bias": torch.tensor(-0.355874),
    "decoder.attention.project_scores.1.weight": torch.tensor(0.864701),
    "decoder.linear_out.weight": torch.tensor(-0.863789),
    "decoder.linear_out.bias": torch.tensor(-0.048714),
    "decoder.linear_stop_token.1.weight": torch.tensor(0.699064),
    "decoder.linear_stop_token.1.bias": torch.tensor(-0.196400),
}

_expected_grads = {
    "embed_speaker.weight": torch.tensor(1.275005),
    "embed_session.weight": torch.tensor(-5.430242),
    "encoder.embed_token.weight": torch.tensor(-0.067350),
    "encoder.embed.0.weight": torch.tensor(-0.213939),
    "encoder.embed.0.bias": torch.tensor(-1.186311),
    "encoder.embed.2.weight": torch.tensor(4.092452),
    "encoder.embed.2.bias": torch.tensor(-0.054729),
    "encoder.conv_layers.0.1.weight": torch.tensor(-4.192872),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.031791),
    "encoder.conv_layers.1.1.weight": torch.tensor(-7.586918),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.140693),
    "encoder.norm_layers.0.weight": torch.tensor(-2.640352),
    "encoder.norm_layers.0.bias": torch.tensor(-0.176361),
    "encoder.norm_layers.1.weight": torch.tensor(5.162328),
    "encoder.norm_layers.1.bias": torch.tensor(-0.302757),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(22.260273),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(-4.308381),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(2.217513),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(2.217513),
    "encoder.lstm.rnn_layers.0.0.initial_hidden_state": torch.tensor(0.001022),
    "encoder.lstm.rnn_layers.0.0.initial_cell_state": torch.tensor(1.394379),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-9.801765),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-0.574815),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(-0.907552),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(-0.907552),
    "encoder.lstm.rnn_layers.0.1.initial_hidden_state": torch.tensor(0.089072),
    "encoder.lstm.rnn_layers.0.1.initial_cell_state": torch.tensor(-0.334629),
    "encoder.lstm_norm.weight": torch.tensor(2.498835),
    "encoder.lstm_norm.bias": torch.tensor(1.249718),
    "encoder.project_out.1.weight": torch.tensor(-0.000000),
    "encoder.project_out.1.bias": torch.tensor(-0.000000),
    "encoder.project_out.2.weight": torch.tensor(16.302975),
    "encoder.project_out.2.bias": torch.tensor(15.883661),
    "decoder.initial_state.0.weight": torch.tensor(0.253010),
    "decoder.initial_state.0.bias": torch.tensor(-0.026042),
    "decoder.initial_state.2.weight": torch.tensor(1.354814),
    "decoder.initial_state.2.bias": torch.tensor(0.209956),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(0.574012),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(-0.333455),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(0.116931),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(0.004417),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(2.416901),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(1.627989),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(-0.859889),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-0.859889),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(-0.058048),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(-0.034265),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(16.567486),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-1.191152),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(-5.196106),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-5.196106),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(0.156329),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(0.754755),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.011130),
    "decoder.attention.alignment_conv.bias": torch.tensor(0.034588),
    "decoder.attention.project_query.weight": torch.tensor(0.001342),
    "decoder.attention.project_query.bias": torch.tensor(0.034588),
    "decoder.attention.project_scores.1.weight": torch.tensor(-0.024561),
    "decoder.linear_out.weight": torch.tensor(14.579857),
    "decoder.linear_out.bias": torch.tensor(-15.973321),
    "decoder.linear_stop_token.1.weight": torch.tensor(-3.994295),
    "decoder.linear_stop_token.1.bias": torch.tensor(8.405702),
}


# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
_expected_frames = [
    [-1.149209, -1.094164, -1.018154, -0.953568, -0.958324, -0.970162, 0.000000, 0.000000],
    [-0.447528, -0.328897, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-1.706318, -1.610223, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-2.444543, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.662563, -0.527245, -0.458826, -0.414914, -0.368667, -0.322661, -0.286051, -0.281697],
]
_expected_frames = torch.tensor(_expected_frames)
_expected_stop_tokens = [
    [0.436714, 0.442603, 0.412385, 0.486228, 0.433828, 0.453362, 0.442403, 0.391349],
    [0.468914, 0.450384, 0.444624, 0.468269, 0.424533, 0.395018, 0.414366, 0.404041],
    [0.480666, 0.469176, 0.479263, 0.496875, 0.518305, 0.409967, 0.412097, 0.412305],
    [0.561519, 0.424608, 0.403607, 0.430372, 0.440361, 0.402844, 0.421830, 0.424811],
    [0.549809, 0.479079, 0.483532, 0.456837, 0.371691, 0.456284, 0.403644, 0.396222],
]
_expected_stop_tokens = torch.tensor(_expected_stop_tokens)
_expected_alignments = [
    [2.745575, 2.577180, 2.677245, 0.000000, 0.000000, 0.000000],
    [4.023643, 3.976357, 0.000000, 0.000000, 0.000000, 0.000000],
    [4.188869, 3.811131, 0.000000, 0.000000, 0.000000, 0.000000],
    [2.761335, 2.573435, 2.665230, 0.000000, 0.000000, 0.000000],
    [2.755137, 2.594827, 2.650036, 0.000000, 0.000000, 0.000000],
]
_expected_alignments = torch.tensor(_expected_alignments)


def _print_params(label: str, params: typing.Iterable[typing.Tuple[str, torch.Tensor]]):
    print(label + " = {")
    for name, parameter in params:
        print(f'    "{name}": torch.tensor({parameter.sum().item():.6f}),')
    print("}")


def test_spectrogram_model__version():
    """ Test `spectrogram_model.SpectrogramModel` has not changed since it was last tested. """
    torch.set_printoptions(precision=6, linewidth=100)

    with fork_rng(123):
        config = _Config(max_frames=8)
        params, target_frames, target_mask, _ = _make_inputs(config)
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.162034))

    with fork_rng(123):
        model = _make_spectrogram_model(config, stop_threshold=0.5)
        with torch.no_grad():
            preds = model(params, mode=Mode.INFER)

        _print_params("_expected_parameters", model.named_parameters())
        for name, parameter in model.named_parameters():
            assert_almost_equal(_expected_parameters[name], parameter.sum())
        print("Frames", preds.frames.sum(dim=-1).transpose(0, 1))
        assert_almost_equal(preds.frames.sum(dim=-1).transpose(0, 1), _expected_frames)
        print("Stop Tokens", torch.sigmoid(preds.stop_tokens.transpose(0, 1)))
        assert_almost_equal(torch.sigmoid(preds.stop_tokens.transpose(0, 1)), _expected_stop_tokens)
        print("Alignments", preds.alignments.sum(dim=0))
        assert_almost_equal(preds.alignments.sum(dim=0), _expected_alignments)
        print("Lengths", preds.lengths.squeeze(0))
        assert_almost_equal(preds.lengths.squeeze(0), torch.tensor([6, 2, 2, 1, 8]))
        print("Reached Max", preds.reached_max.squeeze(0))
        expected = torch.tensor([True, True, True, False, True])
        assert_almost_equal(preds.reached_max.squeeze(0), expected)

    with fork_rng(seed=123):
        target_frames = preds.frames
        target_mask = lengths_to_mask(preds.lengths).transpose(0, 1)
        preds = model(params, target_frames, target_mask=target_mask)

        spectrogram_loss = mse_loss(preds.frames, target_frames, reduction="none")
        spectrogram_loss *= target_mask.unsqueeze(2)
        target = torch.zeros(preds.frames.shape[0], config.batch_size)
        stop_token_loss = binary_cross_entropy_with_logits(
            preds.stop_tokens, target, reduction="none"
        )
        stop_token_loss *= target_mask
        (spectrogram_loss.sum() + stop_token_loss.sum()).backward()

        print("Spectrogram Loss", spectrogram_loss.sum())
        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(30.047640))
        print("Stop Token Loss", stop_token_loss.sum())
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(11.177251))
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _print_params("_expected_grads", grads)
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert_almost_equal(_expected_grads[name], parameter.grad.sum())
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.3122985))
