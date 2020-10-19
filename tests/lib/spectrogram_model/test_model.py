import itertools
import math
import random
import types
import typing

import hparams
import pytest
import torch
from hparams import HParams
from torchnlp.random import fork_rng

import lib
from lib.spectrogram_model import SpectrogramModel
from lib.spectrogram_model.attention import (
    LocationRelativeAttention,
    LocationRelativeAttentionHiddenState,
)
from lib.spectrogram_model.decoder import AutoregressiveDecoder, AutoregressiveDecoderHiddenState
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=5)


class _Inputs(typing.NamedTuple):
    tokens: torch.Tensor
    speaker: torch.Tensor
    num_tokens: torch.Tensor
    target_stop_token: torch.Tensor
    target_frames: torch.Tensor
    target_lengths: torch.Tensor


class _InputParameters(typing.NamedTuple):
    vocab_size: int
    num_speakers: int
    num_frame_channels: int
    max_frames_per_token: float
    batch_size: int
    max_frames: int
    max_num_tokens: int


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    hparams.clear_config()


def _make_spectrogram_model(
    params: _InputParameters,
    speaker_embedding_size: int = 8,
    output_scalar: float = 1.2,
    stop_threshold: float = 0.5,
    dropout: float = 0.5,
    padding_index: int = 0,
    window_length: int = 3,
) -> SpectrogramModel:
    """ Make `spectrogram_model.SpectrogramModel` for testing."""
    hparams.add_config(
        {
            lib.spectrogram_model.encoder.Encoder.__init__: HParams(
                out_size=16,
                hidden_size=16,
                num_convolution_layers=2,
                convolution_filter_size=3,
                lstm_layers=1,
                dropout=dropout,
                padding_index=padding_index,
            ),
            lib.spectrogram_model.decoder.AutoregressiveDecoder.__init__: HParams(
                pre_net_size=16,
                lstm_hidden_size=16,
                encoder_output_size=16,
                stop_net_dropout=dropout,
            ),
            lib.spectrogram_model.pre_net.PreNet.__init__: HParams(num_layers=1, dropout=dropout),
            lib.spectrogram_model.attention.LocationRelativeAttention.__init__: HParams(
                hidden_size=4,
                convolution_filter_size=3,
                dropout=dropout,
                window_length=window_length,
            ),
        }
    )

    model = SpectrogramModel(
        vocab_size=params.vocab_size,
        num_speakers=params.num_speakers,
        speaker_embedding_size=speaker_embedding_size,
        num_frame_channels=params.num_frame_channels,
        max_frames_per_token=params.max_frames_per_token,
        output_scalar=output_scalar,
        speaker_embed_dropout=dropout,
        stop_threshold=stop_threshold,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    for name, parameter in model.named_parameters():
        if parameter.std() == 0:
            torch.nn.init.normal_(parameter)

    return model


def _make_inputs(
    vocab_size: int = 17,
    num_speakers: int = 3,
    num_frame_channels: int = 6,
    batch_size: int = 5,
    max_num_tokens: int = 6,
    padding_index: int = 0,
    max_frames: int = 5,
) -> typing.Tuple[_Inputs, _InputParameters]:
    """ Make `spectrogram_model.SpectrogramModel` inputs for testing."""
    long_ = torch.long
    tokens = torch.randint(padding_index + 1, vocab_size, (max_num_tokens, batch_size), dtype=long_)
    speaker = torch.randint(0, num_speakers, (1, batch_size), dtype=long_)

    num_tokens = torch.randint(1, max_num_tokens, (batch_size,), dtype=long_)
    num_tokens[-1] = max_num_tokens  # NOTE: Ensure at least one sequence is `max_num_tokens`.

    max_frames_per_token = max_frames / max_num_tokens

    target_frames = torch.randn(max_frames, batch_size, num_frame_channels)
    target_lengths = torch.randint(1, max_frames, (batch_size,), dtype=long_)
    target_lengths[-1] = max_frames  # NOTE: Ensure at least one sequence is `max_frames`.

    target_stop_token = torch.zeros(max_frames, batch_size)
    target_stop_token[-1] = 1.0

    return (
        _Inputs(
            tokens,
            speaker,
            num_tokens,
            target_stop_token,
            target_frames,
            target_lengths,
        ),
        _InputParameters(
            vocab_size,
            num_speakers,
            num_frame_channels,
            max_frames_per_token,
            batch_size,
            max_frames,
            max_num_tokens,
        ),
    )


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
        self: LocationRelativeAttention,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        query: torch.Tensor,
        hidden_state: LocationRelativeAttentionHiddenState,
        token_skip_warning: int,
    ):
        window_start = hidden_state.window_start
        cumulative_alignment_padding = self.cumulative_alignment_padding
        first_token = (
            hidden_state.cumulative_alignment[
                :, cumulative_alignment_padding:-cumulative_alignment_padding
            ].sum()
            == 0
        )
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

    model.decoder.attention.forward = types.MethodType(  # type: ignore
        attention_forward, model.decoder.attention
    )

    def decoder_forward(
        self: AutoregressiveDecoder, *args, hidden_state: AutoregressiveDecoderHiddenState, **kwargs
    ):
        frames, stop_token, alignments, updated_hidden_state = _decoder_forward(  # type: ignore
            *args, hidden_state=hidden_state, **kwargs
        )
        stop_token = torch.stack(
            [_rand_logit(stop_token.shape[1], offset=offset) for i in range(stop_token.shape[0])]
        )
        return frames, stop_token, alignments, updated_hidden_state

    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)  # type: ignore

    return set_stop_token_rand_offset


def test_spectrogram_model():
    """ Test `spectrogram_model.SpectrogramModel` handles a basic case. """
    with fork_rng(123):
        (tokens, speaker, num_tokens, *_), params = _make_inputs(batch_size=1)
        model = _make_spectrogram_model(params)
        _mock_model(model)

        frames, stop_tokens, alignments, lengths, reached_max = model(
            tokens, speaker, num_tokens=num_tokens, mode="infer", use_tqdm=True
        )

        assert frames.dtype == torch.float
        assert frames.shape == (
            lengths.max(),
            params.batch_size,
            model.num_frame_channels,
        )
        assert stop_tokens.dtype == torch.float
        assert stop_tokens.shape == (lengths.max(), params.batch_size)
        assert alignments.dtype == torch.float
        assert alignments.shape == (
            lengths.max(),
            params.batch_size,
            params.max_num_tokens,
        )
        assert lengths.shape == (1, params.batch_size)
        for i, length in enumerate(lengths[0].tolist()):
            assert length > 0
            assert length <= params.max_frames
            assert (
                torch.sigmoid(stop_tokens[length - 1, i]) >= model.stop_threshold
                or reached_max[:, i]
            )
        assert reached_max.dtype == torch.bool
        assert reached_max.sum().item() >= 0


def test_spectrogram_model__train():
    """ Test `spectrogram_model.SpectrogramModel` handles a basic training case. """
    (
        tokens,
        speaker,
        num_tokens,
        target_stop_token,
        target_frames,
        target_lengths,
    ), params = _make_inputs()
    model = _make_spectrogram_model(params)
    _mock_model(model)

    frames, stop_token, alignment, spectrogram_loss, stop_token_loss = model(
        tokens,
        speaker,
        target_frames,
        target_stop_token,
        num_tokens=num_tokens,
        target_lengths=target_lengths,
    )

    assert frames.dtype == torch.float
    assert frames.shape == (
        params.max_frames,
        params.batch_size,
        params.num_frame_channels,
    )

    assert stop_token.dtype == torch.float
    assert stop_token.shape == (params.max_frames, params.batch_size)

    assert alignment.dtype == torch.float
    assert alignment.shape == (
        params.max_frames,
        params.batch_size,
        params.max_num_tokens,
    )

    assert spectrogram_loss.dtype == torch.float
    assert spectrogram_loss.shape == (
        params.max_frames,
        params.batch_size,
        params.num_frame_channels,
    )

    assert stop_token_loss.dtype == torch.float
    assert stop_token_loss.shape == (params.max_frames, params.batch_size)

    (spectrogram_loss.sum() + stop_token_loss.sum()).backward()


def test_spectrogram_model__is_stop():
    """ Test `spectrogram_model.SpectrogramModel._is_stop` basic cases. """
    _, params = _make_inputs()
    model = _make_spectrogram_model(params, window_length=3, stop_threshold=0.5)
    tensor = torch.tensor
    _is_stop = lambda a, b, c, d: model._is_stop(_logit(tensor(a)), tensor(b), tensor(c), tensor(d))
    # NOTE: For example, test that this handles a scenario where the window intersects the boundary
    # and `stop_token` is above threshold.
    assert _is_stop(1.0, 8, 5, False)
    assert not _is_stop(1.0, 8, 4, False)
    assert not _is_stop(0.25, 8, 5, False)
    assert _is_stop(0.25, 8, 4, True)


def test_spectrogram_model__stop():
    """Test `spectrogram_model.SpectrogramModel` `stop_tokens` is consistent with `lengths`,
    `window_start`, `window_length` and masking."""
    with fork_rng(123):
        (tokens, speaker, num_tokens, *_), params = _make_inputs(batch_size=16, max_frames=8)
        window_length = 3
        model = _make_spectrogram_model(params, window_length=window_length)
        _mock_model(model)

        frames, stop_tokens, alignments, lengths, reached_max = model(
            tokens, speaker, num_tokens=num_tokens, mode="infer"
        )

        max_lengths = torch.clamp((num_tokens.float() * params.max_frames_per_token).long(), min=1)
        threshold = torch.sigmoid(stop_tokens) >= model.stop_threshold
        for i in range(params.batch_size):  # NOTE: Only stop if the window includes the last token.
            min_index = typing.cast(int, torch.clamp_min(num_tokens[i] - window_length, 0).item())
            threshold[:min_index, i] = False
        stopped_index = _get_index_first_nonzero(threshold)
        stopped_index[stopped_index == -1] = max_lengths[stopped_index == -1] - 1
        expected_length = torch.min(stopped_index + 1, max_lengths)
        assert_almost_equal(lengths.squeeze(0), expected_length)

        for i in range(params.batch_size):
            assert frames[typing.cast(int, lengths[:, i].item()) :, i].sum() == 0


def test_spectrogram_model__infer_train():
    """ Test `spectrogram_model.SpectrogramModel` outputs for train and infer are consistent. """
    (tokens, speaker, num_tokens, *_), params = _make_inputs()
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        frames, stop_token, alignment, lengths, *_ = model(
            tokens, speaker, num_tokens=num_tokens, mode="infer"
        )

    with fork_rng(seed=123):
        aligned_frames, aligned_stop_token, aligned_alignment, *_ = model(
            tokens,
            speaker,
            num_tokens=num_tokens,
            target_frames=frames,
            target_lengths=lengths,
            target_stop_token=torch.zeros(frames.shape[0], params.batch_size),
            mode="forward",
        )

    assert_almost_equal(frames, aligned_frames)
    assert_almost_equal(stop_token, aligned_stop_token)
    assert_almost_equal(alignment, aligned_alignment)


def test_spectrogram_model__infer_generate():
    """ Test `spectrogram_model.SpectrogramModel` outputs for infer and generate are consistent. """
    (tokens, speaker, num_tokens, target_stop_token, *_), params = _make_inputs()
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        frames, stop_token, alignment, lengths, reached_max = model.eval()(
            tokens, speaker, num_tokens=num_tokens, mode="infer"
        )

    for i in [1, 8, 11]:
        with fork_rng(seed=123):
            generator = model(tokens, speaker, num_tokens=num_tokens, mode="generate", split_size=i)
            generated = tuple(zip(*list(generator)))

        assert_almost_equal(frames, torch.cat(generated[0]))
        assert_almost_equal(stop_token, torch.cat(generated[1]))
        assert_almost_equal(alignment, torch.cat(generated[2]))
        assert_almost_equal(lengths, generated[3][-1])
        assert_almost_equal(reached_max, generated[4][-1])


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
    (tokens, speaker, num_tokens, target_stop_token, *_), params = _make_inputs()
    model = _make_spectrogram_model(params, dropout=0)
    set_stop_token_rand_offset = _mock_model(model)

    with fork_rng(seed=123):
        (
            batch_frames,
            batch_stop_token,
            batch_alignment,
            batch_lengths,
            batch_reached_max,
        ) = model.eval()(tokens, speaker, num_tokens=num_tokens, mode="infer")

    for i in range(params.batch_size):
        set_stop_token_rand_offset(i)
        num_tokens_ = typing.cast(int, num_tokens[i].item())
        with fork_rng(seed=123):
            frames, stop_token, alignment, lengths, reached_max = model(
                tokens[:num_tokens_, i : i + 1], speaker[:, i : i + 1], mode="infer"
            )
        length = typing.cast(int, batch_lengths[0, i].item())
        assert_almost_equal(reached_max, batch_reached_max[:, i : i + 1])
        assert_almost_equal(frames, batch_frames[:length, i : i + 1])
        assert_almost_equal(stop_token, batch_stop_token[:length, i : i + 1])
        assert_almost_equal(alignment, batch_alignment[:length, i : i + 1, :num_tokens_])
        assert_almost_equal(lengths, batch_lengths[:, i : i + 1])


def test_spectrogram_model__train_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` train ouput is batch and padding invariant.
    Additionally, this tests inputting a tensor without a batch dimension."""
    (
        tokens,
        speaker,
        num_tokens,
        target_stop_token,
        target_frames,
        target_lengths,
    ), params = _make_inputs(batch_size=5)
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)
    i = 0
    padding = 3
    num_tokens[i] = params.max_num_tokens - padding
    target_lengths[i] = params.max_frames - padding

    with fork_rng(seed=123):
        batch_frames, batch_stop_token, batch_alignment, *_ = model(
            tokens,
            speaker,
            target_frames,
            target_stop_token,
            num_tokens=num_tokens,
            target_lengths=target_lengths,
        )
        (batch_frames[:, i].sum() + batch_stop_token[:, i].sum()).backward()
        batch_grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    num_tokens_ = typing.cast(int, num_tokens[i].item())
    length = typing.cast(int, target_lengths[i].item())

    with fork_rng(seed=123):
        frames, stop_token, alignment, *_ = model(
            tokens[:num_tokens_, i],
            speaker[:, i],
            target_frames[:length, i],
            target_stop_token[:length, i],
            num_tokens=num_tokens[i],
            target_lengths=target_lengths[i],
        )
        (frames.sum() + stop_token.sum()).backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    assert_almost_equal(frames, batch_frames[:length, i])
    assert_almost_equal(stop_token, batch_stop_token[:length, i])
    assert_almost_equal(alignment, batch_alignment[:length, i, :num_tokens_])
    [assert_almost_equal(r, e) for r, e in zip(grad, batch_grad)]


def test_spectrogram_model__filter_reached_max():
    """ Test `spectrogram_model.SpectrogramModel` `filter_reached_max` filters outputs. """
    with fork_rng(seed=123):
        (tokens, speaker, num_tokens, *_), params = _make_inputs(batch_size=32)
        model = _make_spectrogram_model(params, dropout=0)
        _mock_model(model)

        frames, stop_tokens, alignments, lengths, reached_max = model(
            tokens,
            speaker,
            num_tokens=num_tokens,
            filter_reached_max=True,
            mode="infer",
        )

        num_reached_max = reached_max.sum().item()
        max_length = lengths.max().item()

        assert frames.dtype == torch.float
        assert frames.shape == (
            max_length,
            params.batch_size - num_reached_max,
            params.num_frame_channels,
        )

        assert stop_tokens.dtype == torch.float
        assert stop_tokens.shape == (max_length, params.batch_size - num_reached_max)

        assert alignments.dtype == torch.float
        assert alignments.shape == (
            max_length,
            params.batch_size - num_reached_max,
            params.max_num_tokens,
        )

        assert lengths.shape == (1, params.batch_size - num_reached_max)

        for length in lengths[0].tolist():
            assert length > 0
            assert length <= max_length

        assert reached_max.dtype == torch.bool
        assert reached_max.sum().item() >= 0


def test_spectrogram_model__filter_reached_max_all():
    """ Test `spectrogram_model.SpectrogramModel` `filter_reached_max` filters all outputs. """
    (tokens, speaker, num_tokens, *_), params = _make_inputs(batch_size=32)
    model = _make_spectrogram_model(params, dropout=0)

    # NOTE: Make sure that stop-token is not predicted; therefore, reaching `max_frames_per_token`.
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].bias, -math.inf)

    frames, stop_tokens, alignments, lengths, reached_max = model(
        tokens, speaker, num_tokens=num_tokens, filter_reached_max=True, mode="infer"
    )

    assert frames.dtype == torch.float
    assert frames.shape == (params.max_frames, 0, params.num_frame_channels)

    assert stop_tokens.dtype == torch.float
    assert stop_tokens.shape == (params.max_frames, 0)

    assert alignments.dtype == torch.float
    assert alignments.shape == (params.max_frames, 0, params.max_num_tokens)

    assert lengths.shape == (1, 0)

    assert reached_max.dtype == torch.bool
    assert reached_max.sum().item() == params.batch_size


_expected_parameters = {
    "embed_speaker.0.weight": torch.tensor(0.39949676),
    "encoder.embed_token.0.weight": torch.tensor(7.67798615),
    "encoder.embed_token.1.weight": torch.tensor(0.29746649),
    "encoder.embed_token.1.bias": torch.tensor(-5.00974178),
    "encoder.conv_layers.0.1.weight": torch.tensor(-8.05314255),
    "encoder.conv_layers.0.1.bias": torch.tensor(0.04011588),
    "encoder.conv_layers.1.1.weight": torch.tensor(-1.72997069),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.03732613),
    "encoder.norm_layers.0.weight": torch.tensor(-0.88882411),
    "encoder.norm_layers.0.bias": torch.tensor(-1.26923680),
    "encoder.norm_layers.1.weight": torch.tensor(-5.34419727),
    "encoder.norm_layers.1.bias": torch.tensor(-2.39172292),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(2.04774165),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(3.54227114),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(-0.43791330),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-0.35210660),
    "encoder.lstm.rnn_layers.0.0.initial_hidden_state": torch.tensor(2.69637227),
    "encoder.lstm.rnn_layers.0.0.initial_cell_state": torch.tensor(0.36667824),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(1.36758196),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-4.15125179),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(-1.31241310),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(-0.96104062),
    "encoder.lstm.rnn_layers.0.1.initial_hidden_state": torch.tensor(-0.43204951),
    "encoder.lstm.rnn_layers.0.1.initial_cell_state": torch.tensor(-1.69320869),
    "encoder.lstm_norm.weight": torch.tensor(1.91737902),
    "encoder.lstm_norm.bias": torch.tensor(0.47159982),
    "encoder.project_out.1.weight": torch.tensor(2.46367121),
    "encoder.project_out.1.bias": torch.tensor(0.06434536),
    "encoder.project_out.2.weight": torch.tensor(2.07165432),
    "encoder.project_out.2.bias": torch.tensor(-3.94069958),
    "decoder.initial_state.0.weight": torch.tensor(0.37878996),
    "decoder.initial_state.0.bias": torch.tensor(0.28067344),
    "decoder.initial_state.2.weight": torch.tensor(-0.96706909),
    "decoder.initial_state.2.bias": torch.tensor(-0.05505154),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(2.43696499),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(-0.28135747),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(2.73182106),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(-0.58952391),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(3.33185148),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-1.35779607),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(-0.44079229),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-1.38366318),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(-0.22750753),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(-2.49000144),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(7.15912151),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-9.60962296),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(-2.07527757),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-0.52936530),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(0.79439640),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(-1.64558387),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.82357860),
    "decoder.attention.alignment_conv.bias": torch.tensor(0.05188113),
    "decoder.attention.project_query.weight": torch.tensor(0.14241329),
    "decoder.attention.project_query.bias": torch.tensor(0.39978328),
    "decoder.attention.project_scores.1.weight": torch.tensor(-0.14765823),
    "decoder.linear_out.weight": torch.tensor(-1.12889552),
    "decoder.linear_out.bias": torch.tensor(0.22626500),
    "decoder.linear_stop_token.1.weight": torch.tensor(-0.44179803),
    "decoder.linear_stop_token.1.bias": torch.tensor(0.03798738),
}

_expected_grads = {
    "embed_speaker.0.weight": torch.tensor(16.26010895),
    "encoder.embed_token.0.weight": torch.tensor(4.73111868e-07),
    "encoder.embed_token.1.weight": torch.tensor(-7.50992203),
    "encoder.embed_token.1.bias": torch.tensor(-0.46228945),
    "encoder.conv_layers.0.1.weight": torch.tensor(-40.49258041),
    "encoder.conv_layers.0.1.bias": torch.tensor(7.58469152),
    "encoder.conv_layers.1.1.weight": torch.tensor(12.29766273),
    "encoder.conv_layers.1.1.bias": torch.tensor(-3.14681864),
    "encoder.norm_layers.0.weight": torch.tensor(-4.18798256),
    "encoder.norm_layers.0.bias": torch.tensor(10.21989822),
    "encoder.norm_layers.1.weight": torch.tensor(8.40873146),
    "encoder.norm_layers.1.bias": torch.tensor(1.26906466),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(-0.11435124),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(-0.11637684),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(0.29493862),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(0.29493865),
    "encoder.lstm.rnn_layers.0.0.initial_hidden_state": torch.tensor(0.33917564),
    "encoder.lstm.rnn_layers.0.0.initial_cell_state": torch.tensor(0.04230714),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-2.07285643),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-0.13213179),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(0.49486268),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(0.49486274),
    "encoder.lstm.rnn_layers.0.1.initial_hidden_state": torch.tensor(0.06051822),
    "encoder.lstm.rnn_layers.0.1.initial_cell_state": torch.tensor(0.06602637),
    "encoder.lstm_norm.weight": torch.tensor(0.22697496),
    "encoder.lstm_norm.bias": torch.tensor(9.15591049),
    "encoder.project_out.1.weight": torch.tensor(-1.43051147e-06),
    "encoder.project_out.1.bias": torch.tensor(-2.38418579e-07),
    "encoder.project_out.2.weight": torch.tensor(20.02828026),
    "encoder.project_out.2.bias": torch.tensor(-6.23300076),
    "decoder.initial_state.0.weight": torch.tensor(-1.37541485),
    "decoder.initial_state.0.bias": torch.tensor(0.21989432),
    "decoder.initial_state.2.weight": torch.tensor(2.75009632),
    "decoder.initial_state.2.bias": torch.tensor(0.36180872),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-0.00957000),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(0.39604229),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(-0.28310370),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(0.64580381),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(-2.39013195),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(0.18025517),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(0.56573170),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(0.56573170),
    "decoder.lstm_layer_one.initial_hidden_state": torch.tensor(-0.06125612),
    "decoder.lstm_layer_one.initial_cell_state": torch.tensor(0.05205773),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(16.85721016),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(0.76760024),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(-3.87068844),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-3.87068915),
    "decoder.lstm_layer_two.initial_hidden_state": torch.tensor(-0.74548197),
    "decoder.lstm_layer_two.initial_cell_state": torch.tensor(-0.25080174),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.14329046),
    "decoder.attention.alignment_conv.bias": torch.tensor(0.01829420),
    "decoder.attention.project_query.weight": torch.tensor(0.00397818),
    "decoder.attention.project_query.bias": torch.tensor(0.01829420),
    "decoder.attention.project_scores.1.weight": torch.tensor(-0.04756508),
    "decoder.linear_out.weight": torch.tensor(154.67251587),
    "decoder.linear_out.bias": torch.tensor(-29.51614761),
    "decoder.linear_stop_token.1.weight": torch.tensor(-5.31875134),
    "decoder.linear_stop_token.1.bias": torch.tensor(7.06315708),
}

# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
# fmt: off
_expected_frames = torch.tensor([
    [-0.917694, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.978204, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.454064, -0.721204, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.706687, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.050533, -0.499279, -0.647046, -0.504565, -0.468152, -0.727498, -0.527175, -0.627889]
])
_expected_stop_tokens = torch.tensor([
    [0.505230, 0.546093, 0.473365, 0.506375, 0.524675, 0.534680, 0.574927, 0.582296],
    [0.529479, 0.493175, 0.542568, 0.525175, 0.549298, 0.514473, 0.581858, 0.501836],
    [0.559182, 0.565931, 0.573525, 0.515562, 0.574869, 0.574278, 0.494139, 0.519157],
    [0.450524, 0.531667, 0.526494, 0.501388, 0.537756, 0.473344, 0.557139, 0.512733],
    [0.551814, 0.533680, 0.554990, 0.541771, 0.550752, 0.621639, 0.611772, 0.605627]
])
# fmt: on


def test_spectrogram_model__version():
    """ Test `spectrogram_model.SpectrogramModel` has not changed since it was last tested. """
    with fork_rng(123):
        (
            tokens,
            speaker,
            num_tokens,
            target_stop_token,
            target_frames,
            target_lengths,
        ), params = _make_inputs(max_frames=8)
        assert_almost_equal(torch.randn(1), torch.tensor(-0.16081724))

    with fork_rng(123):
        model = _make_spectrogram_model(params, stop_threshold=0.5)
        with torch.no_grad():
            frames, stop_tokens, alignments, lengths, reached_max = model(
                tokens, speaker, num_tokens=num_tokens, mode="infer"
            )

        for name, parameter in model.named_parameters():
            assert_almost_equal(_expected_parameters[name], parameter.sum())
        assert_almost_equal(frames.sum(dim=-1).transpose(0, 1), _expected_frames)
        assert_almost_equal(torch.sigmoid(stop_tokens.transpose(0, 1)), _expected_stop_tokens)
        assert_almost_equal(
            alignments.sum(dim=0),
            torch.tensor(
                [
                    [3.920979, 4.079021, 0.000000, 0.000000, 0.000000, 0.000000],
                    [3.813822, 4.186178, 0.000000, 0.000000, 0.000000, 0.000000],
                    [0.983837, 2.622251, 2.689949, 1.703963, 0.000000, 0.000000],
                    [8.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
                    [1.317250, 2.238706, 2.735094, 1.375616, 0.333333, 0.000000],
                ]
            ),
        )
        assert_almost_equal(lengths.squeeze(0), torch.tensor([1, 1, 2, 1, 8]))
        assert_almost_equal(reached_max.squeeze(0), torch.tensor([False, False, False, True, True]))

    with fork_rng(seed=123):
        frames, stop_token, alignment, spectrogram_loss, stop_token_loss = model(
            tokens,
            speaker,
            num_tokens=num_tokens,
            target_frames=frames,
            target_lengths=lengths,
            target_stop_token=torch.zeros(frames.shape[0], params.batch_size),
            mode="forward",
        )

        (spectrogram_loss.sum() + stop_token_loss.sum()).backward()

        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(21.64362144))
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(10.23205566))
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                assert_almost_equal(_expected_grads[name], parameter.grad.sum())
        assert_almost_equal(torch.randn(1), torch.tensor(0.3122985))
