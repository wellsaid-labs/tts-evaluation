import dataclasses
import itertools
import math
import random
import types
import typing

import config as cf
import torch
import torch.nn
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.utils.rnn import pad_sequence
from torchnlp.random import fork_rng

import run
from lib.distributed import NumeralizePadEmbed
from lib.utils import lengths_to_mask
from run._models.spectrogram_model.attention import Attention
from run._models.spectrogram_model.containers import (
    AttentionHiddenState,
    DecoderHiddenState,
    Encoded,
    Preds,
)
from run._models.spectrogram_model.decoder import Decoder
from run._models.spectrogram_model.model import Inputs, Mode, SpectrogramModel
from tests import _utils

assert_almost_equal = lambda *a, **k: _utils.assert_almost_equal(*a, **k, decimal=5)


class Params(typing.NamedTuple):
    max_tokens: int = 17
    max_seq_meta_vals: typing.Tuple[int, int] = (3, 5)
    num_frame_channels: int = 6
    batch_size: int = 5
    max_frames: int = 5
    max_num_tokens: int = 6
    max_tokens_index: int = 0
    max_word_vector_size: int = 20
    max_seq_vector_size: int = 2
    max_anno_vector_size: int = 1
    annos: typing.Tuple[str, str] = ("anno_embed", "anno_mask")

    @property
    def max_frames_per_token(self) -> float:
        return self.max_frames / self.max_num_tokens

    @property
    def token_embed_idx(self):
        idx = self.max_anno_vector_size
        return {
            self.annos[0]: slice(0, idx),
            self.annos[1]: slice(idx, idx + 1),
            "word_vector": slice(idx + 1, idx + 1 + self.max_word_vector_size),
        }


def _make_spectrogram_model(
    params: Params,
    seq_embed_size: int = 8,
    anno_embed_size: int = 3,
    output_scalar: float = 1.2,
    stop_threshold: float = 0.5,
    dropout: float = 0.5,
    window_length: int = 3,
    stop_token_eps: float = 1e-10,
) -> SpectrogramModel:
    """Make `spectrogram_model.SpectrogramModel` for testing."""
    config = {
        run._models.spectrogram_model.encoder.Encoder: cf.Args(
            seq_meta_embed_dropout=dropout,
            out_size=16,
            hidden_size=16,
            num_conv_layers=2,
            conv_filter_size=3,
            lstm_layers=1,
            dropout=dropout,
            token_meta_embed_size=0,
            anno_embed_size=anno_embed_size,
        ),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(
            pre_net_size=16,
            lstm_hidden_size=16,
            encoder_out_size=16,
            stop_net_dropout=dropout,
            stop_net_hidden_size=16,
        ),
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(num_layers=1, dropout=dropout),
        run._models.spectrogram_model.attention.Attention: cf.Args(
            hidden_size=4,
            conv_filter_size=3,
            dropout=dropout,
            window_length=window_length,
            avg_frames_per_token=1.0,
        ),
        torch.nn.LayerNorm: cf.Args(eps=1e-05),
    }
    cf.add(config, overwrite=True)
    model = SpectrogramModel(
        max_tokens=params.max_tokens,
        max_seq_meta_vals=params.max_seq_meta_vals,
        max_token_meta_vals=tuple(),
        max_word_vector_size=params.max_word_vector_size,
        max_seq_vector_size=params.max_seq_vector_size,
        max_anno_vector_size=params.max_anno_vector_size,
        annos=[params.annos],
        seq_embed_size=seq_embed_size,
        num_frame_channels=params.num_frame_channels,
        output_scalar=output_scalar,
        stop_threshold=stop_threshold,
        stop_token_eps=stop_token_eps,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    [torch.nn.init.normal_(p) for p in model.parameters() if p.std() == 0]

    return model


def _make_inputs(
    params: Params,
) -> typing.Tuple[Inputs, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make `spectrogram_model.SpectrogramModel` inputs for testing."""
    long_ = torch.long

    # NOTE: `1` and `transpose(0, 1)` is set for backwards compatibility so that same random numbers
    # are generated.
    # TODO: Remove and update `test_spectrogram_model__version` values.
    tokens_size = (params.max_num_tokens, params.batch_size)
    tokens = torch.randint(1, params.max_tokens, tokens_size).transpose(0, 1).tolist()
    speakers = torch.randint(0, params.max_seq_meta_vals[0], (params.batch_size,)).tolist()
    sessions = torch.randint(0, params.max_seq_meta_vals[1], (params.batch_size,)).tolist()

    num_tokens = torch.randint(1, params.max_num_tokens, (params.batch_size,), dtype=long_)
    # NOTE: Ensure at least one sequence is `max_num_tokens`.
    num_tokens[params.max_tokens_index] = params.max_num_tokens
    for i in range(params.batch_size):
        tokens[i] = tokens[i][: num_tokens[i]]

    token_vector_size = params.max_word_vector_size + params.max_anno_vector_size
    token_vectors = torch.randn(params.batch_size, params.max_num_tokens, token_vector_size)
    token_vectors = [token_vectors[i, : len(t)] for i, t in enumerate(tokens)]
    token_vectors = [torch.cat((t, torch.ones(t.shape[0], 1)), dim=1) for t in token_vectors]
    seq_vectors = torch.randn(params.batch_size, params.max_seq_vector_size)
    max_audio_len = (params.max_frames_per_token * num_tokens).ceil()

    inputs = Inputs(
        tokens=tokens,
        seq_meta=list(zip(speakers, sessions)),
        token_meta=[[] for _ in range(params.batch_size)],
        seq_vectors=seq_vectors,
        token_vector_idx=params.token_embed_idx,
        token_vectors=pad_sequence(token_vectors, batch_first=True),
        slices=[slice(0, int(n)) for n in num_tokens],
        max_audio_len=max_audio_len,
    )

    target_frames = torch.randn(params.max_frames, params.batch_size, params.num_frame_channels)
    target_lengths = torch.randint(1, params.max_frames, (params.batch_size,), dtype=long_)
    target_lengths[-1] = params.max_frames  # NOTE: Ensure at least one sequence is `max_frames`.
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
    with fork_rng(random.randint(0, 2**16)):
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
        cum_alignment_padding = self.cum_alignment_padding
        slice_ = slice(cum_alignment_padding, -cum_alignment_padding)
        first_token = hidden_state.cum_alignment[:, slice_].sum() == 0
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


def _check_preds(params: Params, model: SpectrogramModel, num_tokens: torch.Tensor, preds: Preds):
    """Check invariants for `preds`."""
    max_frames = params.max_frames if model.training else preds.num_frames.max()
    assert max_frames <= params.max_frames
    assert preds.frames.dtype == torch.float
    assert preds.frames.shape == (max_frames, params.batch_size, model.num_frame_channels)
    assert preds.stop_tokens.dtype == torch.float
    assert preds.stop_tokens.shape == (max_frames, params.batch_size)
    assert preds.alignments.dtype == torch.float
    assert preds.alignments.shape == (max_frames, params.batch_size, params.max_num_tokens)
    assert preds.num_frames.dtype == torch.long
    assert preds.num_frames.shape == (params.batch_size,)
    for i, num_frames in enumerate(preds.num_frames.tolist()):
        assert num_frames > 0
        assert num_frames <= max_frames
        probability = torch.sigmoid(preds.stop_tokens[num_frames - 1, i])
        thresholded = probability >= model.stop_threshold
        if model.training:
            assert num_frames < params.max_frames_per_token * num_tokens[i] or preds.reached_max[i]
        else:
            assert thresholded or preds.reached_max[i]
    assert preds.frames_mask.dtype == torch.bool
    assert preds.frames_mask.shape == (params.batch_size, max_frames)
    assert preds.num_tokens.dtype == torch.long
    assert preds.num_tokens.shape == (params.batch_size,)
    assert preds.tokens_mask.dtype == torch.bool
    assert preds.tokens_mask.shape == (params.batch_size, params.max_num_tokens)
    assert preds.reached_max.dtype == torch.bool
    assert preds.reached_max.shape == (params.batch_size,)


def test_spectrogram_model():
    """Test `spectrogram_model.SpectrogramModel` handles a basic case."""
    with fork_rng(123):
        params = Params(batch_size=1)
        inputs, num_tokens, *_ = _make_inputs(params)
        model = _make_spectrogram_model(params).eval()
        _mock_model(model)
        _set_embedding_vocab(model, params)
        preds = model(inputs, mode=Mode.INFER, use_tqdm=True)
        _check_preds(params, model, num_tokens, preds)


def test_spectrogram_model__train():
    """Test `spectrogram_model.SpectrogramModel` handles a basic training case."""
    params = Params()
    inputs, num_tokens, target_frames, target_mask, _ = _make_inputs(params)
    model = _make_spectrogram_model(params)
    _mock_model(model)
    preds = model(inputs, target_frames, target_mask=target_mask)
    _check_preds(params, model, num_tokens, preds)
    (preds.frames.sum() + preds.stop_tokens.sum()).backward()


def test_spectrogram_model__reached_max_all():
    """Test `spectrogram_model.SpectrogramModel` handles `reached_max`."""
    params = Params(batch_size=32)
    inputs, num_tokens, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)

    # NOTE: Make sure that stop-token is not predicted; therefore, reaching `max_frames_per_token`.
    weight = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].weight)
    torch.nn.init.constant_(weight, -math.inf)
    bias = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].bias)
    torch.nn.init.constant_(bias, -math.inf)

    preds = model(inputs, mode=Mode.INFER)
    _check_preds(params, model, num_tokens, preds)
    assert preds.reached_max.sum().item() == params.batch_size


def test_spectrogram_model__is_stop():
    """Test `spectrogram_model.SpectrogramModel._is_stop` basic cases."""
    params = Params()
    model = _make_spectrogram_model(params, window_length=3, stop_threshold=0.5)
    tensor = torch.tensor
    _is_stop = lambda a, b, c, d: model._is_stop(_logit(tensor(a)), tensor(b), tensor(c), tensor(d))
    # NOTE: For example, test that this handles a scenario where the window intersects the boundary
    # and `stop_token` is above threshold.
    assert _is_stop(1.0, 8, 6, False)[0]
    assert not _is_stop(1.0, 8, 5, False)[0]
    assert not _is_stop(0.25, 8, 6, False)[0]
    assert _is_stop(0.25, 8, 5, True)[0]


def test_spectrogram_model__stop():
    """Test `spectrogram_model.SpectrogramModel` `stop_tokens` is consistent with `lengths`,
    `window_start`, `window_length` and masking."""
    with fork_rng(123):
        params = Params(batch_size=16, max_frames=8)
        inputs, num_tokens, *_ = _make_inputs(params)
        window_length = 3
        model = _make_spectrogram_model(params, window_length=window_length)
        _mock_model(model)

        preds = model(inputs, mode=Mode.INFER)

        max_lengths = inputs.max_audio_len
        threshold = torch.sigmoid(preds.stop_tokens) >= model.stop_threshold
        for i in range(params.batch_size):  # NOTE: Only stop if the window includes the last token.
            min_index = torch.clamp_min(num_tokens[i] - window_length, 0).item()
            min_index = typing.cast(int, min_index)
            threshold[:min_index, i] = False
        stopped_index = _get_index_first_nonzero(threshold)
        stopped_index[stopped_index == -1] = max_lengths[stopped_index == -1] - 1
        expected_length = torch.min(stopped_index + 1, max_lengths)
        assert_almost_equal(preds.num_frames, expected_length)

        for i in range(params.batch_size):
            assert preds.frames[typing.cast(int, preds.num_frames[i].item()) :, i].sum() == 0


def test_spectrogram_model__infer_train():
    """Test `spectrogram_model.SpectrogramModel` outputs for train and infer are consistent."""
    params = Params()
    inputs, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        preds = model(inputs, mode=Mode.INFER)

    with fork_rng(seed=123):
        target_mask = preds.frames_mask.transpose(0, 1)
        aligned_preds = model(inputs, preds.frames, target_mask, mode=Mode.FORWARD)

    assert_almost_equal(preds.frames, aligned_preds.frames)
    assert_almost_equal(preds.stop_tokens, aligned_preds.stop_tokens)
    assert_almost_equal(preds.alignments, aligned_preds.alignments)


def _set_embedding_vocab(model: SpectrogramModel, params: Params):
    """Update `model` vocab so it can be run in inference mode."""
    model.encoder.embed_token.update_tokens(list(range(params.max_tokens)))
    for i, max_values in enumerate(params.max_seq_meta_vals):
        embedding = typing.cast(NumeralizePadEmbed, model.encoder.embed_seq_meta[i])
        embedding.update_tokens(list(range(max_values)))


def test_spectrogram_model__infer_generate():
    """Test `spectrogram_model.SpectrogramModel` outputs for infer and generate are consistent."""
    params = Params()
    inputs, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)

    with fork_rng(seed=123):
        _set_embedding_vocab(model, params)
        preds = model.eval()(inputs, mode=Mode.INFER)

    for i in [1, 8, 11]:
        with fork_rng(seed=123):
            generated = list(model(inputs, mode=Mode.GENERATE, split_size=i))

        num_frames = torch.stack([g.num_frames for g in generated]).sum(dim=0)
        assert_almost_equal(preds.frames, torch.cat([g.frames for g in generated]))
        assert_almost_equal(preds.stop_tokens, torch.cat([g.stop_tokens for g in generated]))
        assert_almost_equal(preds.alignments, torch.cat([g.alignments for g in generated]))
        assert_almost_equal(preds.num_frames, num_frames)
        assert_almost_equal(preds.frames_mask, torch.cat([g.frames_mask for g in generated], dim=1))
        assert_almost_equal(preds.num_tokens, generated[-1].num_tokens)
        assert_almost_equal(preds.tokens_mask, generated[-1].tokens_mask)
        assert_almost_equal(preds.reached_max, generated[-1].reached_max)


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
    params = Params()
    inputs, num_tokens, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    set_stop_token_rand_offset = _mock_model(model)

    with fork_rng(seed=123):
        _set_embedding_vocab(model, params)
        batch_preds = model.eval()(inputs, mode=Mode.INFER)

    for i in range(params.batch_size):
        set_stop_token_rand_offset(i)
        num_tokens_ = typing.cast(int, num_tokens[i].item())
        with fork_rng(seed=123):
            preds = model(inputs[i], mode=Mode.INFER)

        length = typing.cast(int, batch_preds.num_frames[i].item())
        assert_almost_equal(preds.reached_max, batch_preds.reached_max[i : i + 1])
        assert_almost_equal(preds.frames, batch_preds.frames[:length, i : i + 1])
        assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i : i + 1])
        batch_preds_alignments = batch_preds.alignments[:length, i : i + 1, :num_tokens_]
        assert_almost_equal(preds.alignments, batch_preds_alignments)
        assert_almost_equal(preds.num_frames, batch_preds.num_frames[i : i + 1])


def test_spectrogram_model__train_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` train ouput is batch and padding invariant.
    Additionally, this tests inputting a tensor without a batch dimension."""
    params = Params(batch_size=5)
    batch_inputs, _, target_frames, target_mask, target_lengths = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)
    i = params.max_tokens_index
    padding = 3
    num_tokens = params.max_num_tokens - padding
    batch_inputs.tokens[i] = batch_inputs.tokens[i][:num_tokens]
    max_tokens = max(len(seq) for seq in batch_inputs.tokens)
    for j in range(batch_inputs.num_token_meta):
        batch_inputs.token_meta[i][j][:num_tokens]
    slice_ = batch_inputs.slices[i]
    batch_inputs.slices[i] = slice(slice_.start, min(num_tokens, slice_.stop))
    token_vectors = batch_inputs.token_vectors[:, :max_tokens]
    token_vectors[i][num_tokens:].fill_(0)
    batch_inputs = dataclasses.replace(batch_inputs, token_vectors=token_vectors)
    target_lengths[i] = params.max_frames - padding

    with fork_rng(seed=123):
        target_mask = lengths_to_mask(target_lengths).transpose(0, 1)
        batch_preds = model(batch_inputs, target_frames=target_frames, target_mask=target_mask)
        (batch_preds.frames[:, i].sum() + batch_preds.stop_tokens[:, i].sum()).backward()
        batch_grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    length = typing.cast(int, target_lengths[i].item())

    with fork_rng(seed=123):
        target_mask = target_mask[:length, i : i + 1]
        target_frames = target_frames[:length, i : i + 1]
        preds = model(batch_inputs[i], target_frames=target_frames, target_mask=target_mask)
        (preds.frames.sum() + preds.stop_tokens.sum()).backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    assert_almost_equal(preds.frames, batch_preds.frames[:length, i : i + 1])
    assert_almost_equal(preds.stop_tokens, batch_preds.stop_tokens[:length, i : i + 1])
    assert_almost_equal(preds.alignments, batch_preds.alignments[:length, i : i + 1, :num_tokens])
    [assert_almost_equal(r, e) for r, e in zip(grad, batch_grad)]


_expected_parameters = {
    "encoder.embed_seq_metadata.0.weight": torch.tensor(-3.827646),
    "encoder.embed_seq_metadata.1.weight": torch.tensor(-1.871155),
    "encoder.embed_token.weight": torch.tensor(-36.022980),
    "encoder.embed_anno.0.weight": torch.tensor(-3.370965),
    "encoder.embed_anno.0.bias": torch.tensor(2.010341),
    "encoder.embed_anno.2.weight": torch.tensor(-0.352856),
    "encoder.embed_anno.2.bias": torch.tensor(-0.226936),
    "encoder.embed_anno.4.weight": torch.tensor(-5.261642),
    "encoder.embed_anno.4.bias": torch.tensor(0.262012),
    "encoder.embed.0.weight": torch.tensor(-0.809235),
    "encoder.embed.0.bias": torch.tensor(-0.833671),
    "encoder.embed.2.weight": torch.tensor(0.551945),
    "encoder.embed.2.bias": torch.tensor(0.481612),
    "encoder.embed.4.weight": torch.tensor(1.382087),
    "encoder.embed.4.bias": torch.tensor(0.933656),
    "encoder.conv_layers.0.1.weight": torch.tensor(13.946832),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.505374),
    "encoder.conv_layers.1.1.weight": torch.tensor(-4.201049),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.364832),
    "encoder.norm_layers.0.weight": torch.tensor(5.420585),
    "encoder.norm_layers.0.bias": torch.tensor(-3.638918),
    "encoder.norm_layers.1.weight": torch.tensor(5.298835),
    "encoder.norm_layers.1.bias": torch.tensor(-7.071516),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(-6.716073),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(0.385707),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(1.562320),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-0.298599),
    "encoder.lstm.rnn_layers.0.0.init_hidden_state": torch.tensor(-0.986030),
    "encoder.lstm.rnn_layers.0.0.init_cell_state": torch.tensor(5.761682),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(0.858555),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-2.350116),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(-2.192030),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(-0.840875),
    "encoder.lstm.rnn_layers.0.1.init_hidden_state": torch.tensor(1.022181),
    "encoder.lstm.rnn_layers.0.1.init_cell_state": torch.tensor(-0.790193),
    "encoder.lstm_norm.weight": torch.tensor(-2.621604),
    "encoder.lstm_norm.bias": torch.tensor(5.736939),
    "encoder.project_out.1.weight": torch.tensor(2.072137),
    "encoder.project_out.1.bias": torch.tensor(0.229817),
    "encoder.project_out.2.weight": torch.tensor(3.967675),
    "encoder.project_out.2.bias": torch.tensor(-4.416607),
    "decoder.init_state.0.weight": torch.tensor(-2.975752),
    "decoder.init_state.0.bias": torch.tensor(0.344184),
    "decoder.init_state.2.weight": torch.tensor(3.919436),
    "decoder.init_state.2.bias": torch.tensor(0.121071),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-5.477797),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(1.224363),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(-6.607403),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(7.299874),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(-16.328350),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-1.505205),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(-2.548468),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-0.134175),
    "decoder.lstm_layer_one.init_hidden_state": torch.tensor(-6.031653),
    "decoder.lstm_layer_one.init_cell_state": torch.tensor(2.228779),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(-5.870141),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-0.399409),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(0.221064),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(-0.213854),
    "decoder.lstm_layer_two.init_hidden_state": torch.tensor(-0.944610),
    "decoder.lstm_layer_two.init_cell_state": torch.tensor(4.049337),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.244437),
    "decoder.attention.alignment_conv.bias": torch.tensor(1.149926),
    "decoder.attention.project_query.weight": torch.tensor(-1.341796),
    "decoder.attention.project_query.bias": torch.tensor(0.364319),
    "decoder.attention.project_scores.1.weight": torch.tensor(0.572894),
    "decoder.linear_out.weight": torch.tensor(3.133666),
    "decoder.linear_out.bias": torch.tensor(0.008901),
    "decoder.linear_stop_token.1.weight": torch.tensor(-1.682154),
    "decoder.linear_stop_token.1.bias": torch.tensor(-0.042868),
    "decoder.linear_stop_token.3.weight": torch.tensor(0.620160),
    "decoder.linear_stop_token.3.bias": torch.tensor(-0.041649),
}

_expected_grads = {
    "encoder.embed_seq_metadata.0.weight": torch.tensor(2.247766),
    "encoder.embed_seq_metadata.1.weight": torch.tensor(6.535187),
    "encoder.embed_token.weight": torch.tensor(-1.449149),
    "encoder.embed_anno.0.weight": torch.tensor(-3.654608),
    "encoder.embed_anno.0.bias": torch.tensor(-3.673264),
    "encoder.embed_anno.2.weight": torch.tensor(41.643044),
    "encoder.embed_anno.2.bias": torch.tensor(6.719882),
    "encoder.embed_anno.4.weight": torch.tensor(7.758373),
    "encoder.embed_anno.4.bias": torch.tensor(-1.472111),
    "encoder.embed.0.weight": torch.tensor(-6.232493),
    "encoder.embed.0.bias": torch.tensor(0.118486),
    "encoder.embed.2.weight": torch.tensor(9.502970),
    "encoder.embed.2.bias": torch.tensor(2.536233),
    "encoder.embed.4.weight": torch.tensor(-4.506941),
    "encoder.embed.4.bias": torch.tensor(4.538265),
    "encoder.conv_layers.0.1.weight": torch.tensor(-1.302409),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.096138),
    "encoder.conv_layers.1.1.weight": torch.tensor(-6.959103),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.274330),
    "encoder.norm_layers.0.weight": torch.tensor(2.081227),
    "encoder.norm_layers.0.bias": torch.tensor(-2.425773),
    "encoder.norm_layers.1.weight": torch.tensor(-11.688414),
    "encoder.norm_layers.1.bias": torch.tensor(-0.731841),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(40.779400),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(0.099935),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(-1.752003),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-1.752003),
    "encoder.lstm.rnn_layers.0.0.init_hidden_state": torch.tensor(-0.088266),
    "encoder.lstm.rnn_layers.0.0.init_cell_state": torch.tensor(-0.680689),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-7.809122),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(0.220291),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(0.332669),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(0.332669),
    "encoder.lstm.rnn_layers.0.1.init_hidden_state": torch.tensor(0.005065),
    "encoder.lstm.rnn_layers.0.1.init_cell_state": torch.tensor(0.060816),
    "encoder.lstm_norm.weight": torch.tensor(-8.555500),
    "encoder.lstm_norm.bias": torch.tensor(0.941254),
    "encoder.project_out.1.weight": torch.tensor(-0.000011),
    "encoder.project_out.1.bias": torch.tensor(0.000000),
    "encoder.project_out.2.weight": torch.tensor(5.708027),
    "encoder.project_out.2.bias": torch.tensor(9.798748),
    "decoder.init_state.0.weight": torch.tensor(-6.419244),
    "decoder.init_state.0.bias": torch.tensor(0.409415),
    "decoder.init_state.2.weight": torch.tensor(13.271113),
    "decoder.init_state.2.bias": torch.tensor(1.928933),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(0.064790),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(0.052363),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(-0.085744),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(-0.171775),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(0.728407),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(0.667925),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(-0.284277),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-0.284277),
    "decoder.lstm_layer_one.init_hidden_state": torch.tensor(0.026772),
    "decoder.lstm_layer_one.init_cell_state": torch.tensor(-0.064871),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(2.427370),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(3.100259),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(0.356638),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(0.356638),
    "decoder.lstm_layer_two.init_hidden_state": torch.tensor(-0.298672),
    "decoder.lstm_layer_two.init_cell_state": torch.tensor(-0.097097),
    "decoder.attention.alignment_conv.weight": torch.tensor(-2.043309),
    "decoder.attention.alignment_conv.bias": torch.tensor(0.213110),
    "decoder.attention.project_query.weight": torch.tensor(-0.183178),
    "decoder.attention.project_query.bias": torch.tensor(0.213110),
    "decoder.attention.project_scores.1.weight": torch.tensor(0.928229),
    "decoder.linear_out.weight": torch.tensor(8.644670),
    "decoder.linear_out.bias": torch.tensor(-9.016104),
    "decoder.linear_stop_token.1.weight": torch.tensor(-5.242250),
    "decoder.linear_stop_token.1.bias": torch.tensor(0.319173),
    "decoder.linear_stop_token.3.weight": torch.tensor(9.020034),
    "decoder.linear_stop_token.3.bias": torch.tensor(1.754208),
}

# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
_expected_frames = [
    [-0.612119, -0.658185, -0.699279, -0.868697, -0.785768, -0.769109, -0.789424, -0.802131],
    [-1.791631, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.750081, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.626863, -0.592974, -0.553276, -0.607893, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.227751, 0.267221, 0.260314, 0.192665, 0.169256, 0.000000, 0.000000, 0.000000],
]
_expected_frames = torch.tensor(_expected_frames)
_eps = 1.000001e-10
# NOTE: For first, the `stop_token` always predicts `_eps` because the `window_start` largely stays
# at zero and the the number of tokens is larger than the window length.
_expected_stop_tokens = [
    [_eps, _eps, _eps, _eps, _eps, _eps, _eps, _eps],
    [0.5201635, 0.4931442, 0.5028796, 0.5211118, 0.5496151, 0.5292537, 0.5159689, 0.5018378],
    [0.6141419, 0.5777515, 0.5278518, 0.5470251, 0.6061662, 0.5895721, 0.6242852, 0.5449392],
    [_eps, _eps, 0.4628496, 0.5619906, 0.5336012, 0.5389556, 0.5327610, 0.4974324],
    [_eps, _eps, _eps, _eps, 0.6027632, 0.6267574, 0.6156967, 0.5213081],
]
_expected_stop_tokens = torch.tensor(_expected_stop_tokens)
_expected_alignments = [
    [1.349304, 2.345946, 1.835844, 1.250977, 0.325121, 0.000000],
    [2.489095, 2.674609, 0.000000, 0.000000, 0.000000, 0.000000],
    [2.611588, 2.780289, 0.000000, 0.000000, 0.000000, 0.000000],
    [1.294550, 2.564388, 1.904388, 0.000000, 0.000000, 0.000000],
    [1.619465, 2.644195, 2.053463, 1.055696, 0.000000, 0.000000],
]
_expected_alignments = torch.tensor(_expected_alignments)


def test_spectrogram_model__version():
    """Test `spectrogram_model.SpectrogramModel` has not changed since it was last tested.

    TODO: This test won't pass on M1 MacBooks until PyTorch fixes some bugs:
    https://github.com/pytorch/pytorch/issues/84030
    """
    torch.set_printoptions(precision=6, linewidth=100)

    with fork_rng(123):
        params = Params(max_frames=8)
        inputs, _, target_frames, _, _ = _make_inputs(params)
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.503481))

    with fork_rng(123):
        model = _make_spectrogram_model(params)
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
        print("Num Frames", preds.num_frames)
        assert_almost_equal(preds.num_frames, torch.tensor([8, 1, 1, 4, 5]))
        print("Reached Max", preds.reached_max)
        expected = torch.tensor([True, False, False, True, True])
        assert_almost_equal(preds.reached_max, expected)

    with fork_rng(seed=123):
        target_frames = preds.frames
        targets_mask = preds.frames_mask.transpose(0, 1)
        preds = model(inputs, target_frames, targets_mask)

        spectrogram_loss = mse_loss(preds.frames, target_frames, reduction="none")
        spectrogram_loss *= targets_mask.unsqueeze(2)
        target = torch.zeros(preds.frames.shape[0], params.batch_size)
        stop_token_loss = binary_cross_entropy_with_logits(
            preds.stop_tokens, target, reduction="none"
        )
        stop_token_loss *= targets_mask
        (spectrogram_loss.sum() + stop_token_loss.sum()).backward()

        print("Spectrogram Loss", spectrogram_loss.sum())
        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(21.917810))
        print("Stop Token Loss", stop_token_loss.sum())
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(2.644117))
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _utils.print_params("_expected_grads", grads)
        for name, grad in grads:
            assert_almost_equal(_expected_grads[name], grad.sum())
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(-0.387737))
