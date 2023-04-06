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
    max_token_meta_values: typing.Tuple[int] = (7,)
    num_frame_channels: int = 6
    batch_size: int = 5
    max_frames: int = 5
    max_num_tokens: int = 9
    max_tokens_indices: typing.Tuple[int, ...] = (0, 1)
    min_tokens_indices: typing.Tuple[int, ...] = (2,)
    max_word_vector_size: int = 11
    word_vector_size: int = 7
    max_seq_vector_size: int = 4
    seq_vector_size: int = 2
    max_anno_vector_size: int = 10
    annos: typing.Tuple[typing.Tuple[str, str], ...] = (
        ("anno_vector_a", "anno_mask_a"),
        ("anno_vector_b", "anno_mask_b"),
    )

    @property
    def max_frames_per_token(self) -> float:
        return self.max_frames / self.max_num_tokens

    @property
    def token_vector_idx(self):
        annos = [i for a in self.annos for i in a]
        indicies = {a: slice(i, i + 1) for i, a in enumerate(annos)}
        slice_ = slice(len(annos), len(annos) + self.word_vector_size)
        return {**indicies, "word_vector": slice_}


def _make_spectrogram_model(
    params: Params,
    encoder_hidden_size: int = 16,
    output_scalar: float = 1.2,
    stop_threshold: float = 0.5,
    dropout: float = 0.5,
    window_len: int = 3,
    stop_token_eps: float = 1e-10,
    clamp_output: bool = False,
) -> SpectrogramModel:
    """Make `spectrogram_model.SpectrogramModel` for testing."""
    config = {
        run._models.spectrogram_model.encoder.Encoder: cf.Args(num_layers=2, conv_filter_size=3),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(hidden_size=16),
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(dropout=dropout),
        run._models.spectrogram_model.attention.Attention: cf.Args(
            conv_filter_size=3, window_len=window_len, avg_frames_per_token=1.0
        ),
        torch.nn.LayerNorm: cf.Args(eps=1e-05),
    }
    cf.add(config, overwrite=True)

    model = SpectrogramModel(
        max_tokens=params.max_tokens,
        max_seq_meta_vals=params.max_seq_meta_vals,
        max_token_meta_vals=params.max_token_meta_values,
        max_word_vector_size=params.max_word_vector_size,
        max_seq_vector_size=params.max_seq_vector_size,
        max_anno_vector_size=params.max_anno_vector_size,
        annos=params.annos,
        encoder_hidden_size=encoder_hidden_size,
        num_frame_channels=params.num_frame_channels,
        output_scalar=output_scalar,
        stop_threshold=stop_threshold,
        stop_token_eps=stop_token_eps,
        clamp_output=clamp_output,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    [torch.nn.init.normal_(p) for p in model.parameters() if p.std() == 0]

    return model


def _make_inputs(params: Params):
    """Make `spectrogram_model.SpectrogramModel` inputs for testing."""
    batch_size = (params.batch_size,)
    seq_size = (params.batch_size, params.max_num_tokens)
    tokens = torch.randint(1, params.max_tokens, seq_size).tolist()
    seq_meta_a = torch.randint(0, params.max_seq_meta_vals[0], batch_size).tolist()
    seq_meta_b = torch.randint(0, params.max_seq_meta_vals[1], batch_size).tolist()
    seq_vectors = torch.randn(params.batch_size, params.seq_vector_size)
    token_meta_a = torch.randint(0, params.max_token_meta_values[0], seq_size).tolist()
    num_tokens = torch.randint(1, params.max_num_tokens, batch_size, dtype=torch.long)
    word_vector_mask = torch.rand((*seq_size, 1)) < 0.5
    word_vector = torch.randn(*seq_size, params.word_vector_size) * word_vector_mask
    anno_vector_a_mask = torch.rand(seq_size) < 0.5
    anno_vector_a = torch.randn(*seq_size) * anno_vector_a_mask
    anno_vector_b_mask = torch.rand(seq_size) < 0.5
    anno_vector_b = torch.randn(*seq_size) * anno_vector_b_mask
    anno_tensors = (anno_vector_a, anno_vector_a_mask, anno_vector_b, anno_vector_b_mask)
    anno_vector = torch.stack(anno_tensors, dim=2)
    token_vector = torch.cat((anno_vector, word_vector), dim=2)
    slice_lengths = [random.randint(max(int(n) - 3, 1), int(n)) for n in num_tokens]
    slice_starts = [random.randint(0, int(n) - l) for n, l in zip(num_tokens, slice_lengths)]
    slices = [slice(s, s + l) for s, l in zip(slice_starts, slice_lengths)]
    target_frames = torch.randn(params.max_frames, params.batch_size, params.num_frame_channels)
    target_lengths = torch.randint(1, params.max_frames, batch_size, dtype=torch.long)

    # NOTE: Ensure at least one sequence is `max_num_tokens`.
    for idx in params.max_tokens_indices:
        if idx < params.batch_size:
            num_tokens[idx] = params.max_num_tokens
            slices[idx] = slice(0, params.max_num_tokens)
            target_lengths[idx] = params.max_frames

    # NOTE: Ensure at least one sequence has only 1 token.
    for idx in params.min_tokens_indices:
        if idx < params.batch_size:
            num_tokens[idx] = 1
            slices[idx] = slice(0, 1)
            target_lengths[idx] = 1

    for i in range(params.batch_size):
        tokens[i] = tokens[i][: num_tokens[i]]
        token_meta_a[i] = token_meta_a[i][: num_tokens[i]]
        token_vector[i, num_tokens[i] :].fill_(0)

    num_sliced_tokens = torch.tensor([s.stop - s.start for s in slices])
    max_audio_len = (params.max_frames_per_token * num_sliced_tokens).ceil().long()
    target_mask = lengths_to_mask(target_lengths).transpose(0, 1)  # [num_frames, batch_size]

    inputs = Inputs(
        tokens=tokens,
        seq_meta=list(zip(seq_meta_a, seq_meta_b)),
        token_meta=[[m] for m in token_meta_a],
        seq_vectors=seq_vectors,
        token_vector_idx=params.token_vector_idx,
        token_vectors=token_vector,
        slices=slices,
        max_audio_len=max_audio_len,
    )

    return inputs, num_tokens, num_sliced_tokens, target_frames, target_mask, target_lengths


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
        tensor([0.4721, 1.2948, -0.0914])
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
    _attention_forward = model.decoder.attn_rnn.attn.forward
    window_len = model.decoder.attn_rnn.attn.window_len
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
        slice_ = slice(self.padding, -self.padding)
        first_token = hidden_state.cum_alignment[:, slice_].sum() == 0
        context, alignment, hidden_state = _attention_forward(
            encoded, query, hidden_state, token_skip_warning
        )
        # NOTE: On the first iteration, `window_start` should not advance because it needs to
        # focus on the first token.
        window_start_ = torch.min(window_start + 1, encoded.num_tokens - window_len)
        window_start = window_start.zero_() if first_token else torch.clamp(window_start_, min=0)
        return context, alignment, dataclasses.replace(hidden_state, window_start=window_start)

    model.decoder.attn_rnn.attn.forward = types.MethodType(
        attention_forward, model.decoder.attn_rnn.attn
    )

    def decoder_forward(self: Decoder, *args, hidden_state: DecoderHiddenState, **kwargs):
        out = _decoder_forward(*args, hidden_state=hidden_state, **kwargs)  # type: ignore
        iter_ = range(out.stop_tokens.shape[0])
        tokens = torch.stack([_rand_logit(out.stop_tokens.shape[1], offset=offset) for _ in iter_])
        return out._replace(stop_tokens=tokens)

    model.decoder.forward = types.MethodType(decoder_forward, model.decoder)

    return set_stop_token_rand_offset


def _check_preds(
    params: Params,
    model: SpectrogramModel,
    num_sliced_tokens: torch.Tensor,
    preds: Preds,
):
    """Check invariants for `preds`."""
    preds.check_invariants()
    max_frames = params.max_frames if model.training else preds.num_frames.max()
    assert max_frames <= params.max_frames
    for i, num_frames in enumerate(preds.num_frames.tolist()):
        assert num_frames > 0
        assert num_frames <= max_frames
        probability = torch.sigmoid(preds.stop_tokens[num_frames - 1, i])
        thresholded = probability >= model.stop_threshold
        if model.training:
            max_frames_per_token = params.max_frames_per_token
            assert num_frames < max_frames_per_token * num_sliced_tokens[i] or preds.reached_max[i]
        else:
            assert thresholded or preds.reached_max[i]


def test_spectrogram_model():
    """Test `spectrogram_model.SpectrogramModel` handles a basic case."""
    with fork_rng(123):
        params = Params(batch_size=1)
        inputs, _, num_sliced_tokens, *_ = _make_inputs(params)
        model = _make_spectrogram_model(params).eval()
        _mock_model(model)
        _set_embedding_vocab(model, params)
        preds = model(inputs, mode=Mode.INFER, use_tqdm=True)
        _check_preds(params, model, num_sliced_tokens, preds)


def test_spectrogram_model__train():
    """Test `spectrogram_model.SpectrogramModel` handles a basic training case."""
    params = Params()
    inputs, _, num_sliced_tokens, target_frames, target_mask, _ = _make_inputs(params)
    model = _make_spectrogram_model(params, clamp_output=True)
    _mock_model(model)
    preds = model(inputs, target_frames, target_mask=target_mask)
    _check_preds(params, model, num_sliced_tokens, preds)
    assert model.output_max.max() == target_frames.max() / model.output_scalar
    assert model.output_min.min() == target_frames.min() / model.output_scalar
    (preds.frames.sum() + preds.stop_tokens.sum()).backward()


def test_spectrogram_model__reached_max_all():
    """Test `spectrogram_model.SpectrogramModel` handles `reached_max`."""
    params = Params(batch_size=32)
    inputs, _, num_sliced_tokens, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)

    # NOTE: Make sure that stop-token is not predicted; therefore, reaching `max_frames_per_token`.
    weight = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].weight)
    torch.nn.init.constant_(weight, -math.inf)
    bias = typing.cast(torch.nn.parameter.Parameter, model.decoder.linear_stop_token[-1].bias)
    torch.nn.init.constant_(bias, -math.inf)

    preds = model(inputs, mode=Mode.INFER)
    _check_preds(params, model, num_sliced_tokens, preds)
    assert preds.reached_max.sum().item() == params.batch_size


def test_spectrogram_model__is_stop():
    """Test `spectrogram_model.SpectrogramModel._is_stop` basic cases."""
    params = Params()
    model = _make_spectrogram_model(params, window_len=3, stop_threshold=0.5)
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
    `window_start`, `window_len` and masking."""
    with fork_rng(123):
        params = Params(batch_size=16, max_frames=8)
        inputs, _, num_sliced_tokens, *_ = _make_inputs(params)
        window_len = 3
        model = _make_spectrogram_model(params, window_len=window_len)
        _mock_model(model)

        preds = model(inputs, mode=Mode.INFER)

        max_lengths = inputs.max_audio_len
        threshold = torch.sigmoid(preds.stop_tokens) >= model.stop_threshold
        for i in range(params.batch_size):  # NOTE: Only stop if the window includes the last token.
            min_index = torch.clamp_min(num_sliced_tokens[i] - window_len, 0).item()
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
    inputs, _, num_sliced_tokens, *_ = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    set_stop_token_rand_offset = _mock_model(model)

    with fork_rng(seed=123):
        _set_embedding_vocab(model, params)
        batch_preds = model.eval()(inputs, mode=Mode.INFER)

    for i in range(params.batch_size):
        set_stop_token_rand_offset(i)
        with fork_rng(seed=123):
            preds = model(inputs[i], mode=Mode.INFER)

        assert_almost_equal(preds.reached_max, batch_preds[i : i + 1].reached_max)
        assert_almost_equal(preds.frames, batch_preds[i : i + 1].frames)
        assert_almost_equal(preds.stop_tokens, batch_preds[i : i + 1].stop_tokens)
        assert_almost_equal(preds.alignments, batch_preds[i : i + 1].alignments)
        assert_almost_equal(preds.num_frames, batch_preds[i : i + 1].num_frames)


def test_spectrogram_model__train_batch_padding_invariance():
    """Test `spectrogram_model.SpectrogramModel` train ouput is batch and padding invariant.
    Additionally, this tests inputting a tensor without a batch dimension."""
    params = Params(batch_size=5)
    batch_inputs, _, _, target_frames, target_mask, target_lengths = _make_inputs(params)
    model = _make_spectrogram_model(params, dropout=0)
    _mock_model(model)

    idx = params.max_tokens_indices[0]
    padding = 2
    assert params.max_num_tokens > padding * 3
    num_tokens = params.max_num_tokens - padding
    batch_inputs.tokens[idx] = batch_inputs.tokens[idx][:num_tokens]
    batch_inputs.token_vectors[idx, num_tokens:].fill_(0)
    for j in range(batch_inputs.num_token_meta):
        batch_inputs.token_meta[idx][j] = batch_inputs.token_meta[idx][j][:num_tokens]
    slice_ = slice(padding, num_tokens - padding)
    num_sliced_tokens = slice_.stop - slice_.start
    batch_inputs.slices[idx] = slice_
    batch_inputs.max_audio_len[idx] = params.max_frames_per_token * num_sliced_tokens
    batch_inputs = dataclasses.replace(batch_inputs)

    target_lengths[idx] = params.max_frames - padding
    target_mask = lengths_to_mask(target_lengths).transpose(0, 1)

    with fork_rng(seed=123):
        batch_preds = model(batch_inputs, target_frames=target_frames, target_mask=target_mask)
        (batch_preds.frames[:, idx].sum() + batch_preds.stop_tokens[:, idx].sum()).backward()
        batch_grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    num_frames = typing.cast(int, target_lengths[idx].item())
    inputs = batch_inputs[idx]
    target_mask = target_mask[:num_frames, idx : idx + 1]
    target_frames = target_frames[:num_frames, idx : idx + 1]

    with fork_rng(seed=123):
        preds = model(inputs, target_frames=target_frames, target_mask=target_mask)
        (preds.frames.sum() + preds.stop_tokens.sum()).backward()
        grad = [p.grad for p in model.parameters() if p.grad is not None]
        model.zero_grad()

    assert_almost_equal(preds.frames, batch_preds[idx : idx + 1].frames)
    assert_almost_equal(preds.stop_tokens, batch_preds[idx : idx + 1].stop_tokens)
    assert_almost_equal(preds.alignments, batch_preds[idx : idx + 1].alignments)
    [assert_almost_equal(r, e) for r, e in zip(grad, batch_grad)]


_expected_parameters = {
    "encoder.embed_seq_meta.0.embed.weight": torch.tensor(16.943052),
    "encoder.embed_seq_meta.1.embed.weight": torch.tensor(5.409315),
    "encoder.embed_seq_vector.weight": torch.tensor(1.958080),
    "encoder.embed_seq_vector.bias": torch.tensor(-0.596941),
    "encoder.embed_token.embed.weight": torch.tensor(-7.434854),
    "encoder.embed_token_meta.0.embed.weight": torch.tensor(-13.164258),
    "encoder.embed_word_vec.weight": torch.tensor(-0.219317),
    "encoder.embed_word_vec.bias": torch.tensor(0.251124),
    "encoder.norm_embed.weight": torch.tensor(-2.418278),
    "encoder.norm_embed.bias": torch.tensor(-2.455902),
    "encoder.embed_annos.weight": torch.tensor(0.081344),
    "encoder.embed_annos.bias": torch.tensor(1.285097),
    "encoder.pre_net.0.1.weight": torch.tensor(-1.260459),
    "encoder.pre_net.0.1.bias": torch.tensor(-0.382393),
    "encoder.pre_net.1.1.weight": torch.tensor(-1.222551),
    "encoder.pre_net.1.1.bias": torch.tensor(-0.565423),
    "encoder.blocks.0.norm.weight": torch.tensor(0.653282),
    "encoder.blocks.0.norm.bias": torch.tensor(4.182391),
    "encoder.blocks.0.lstm.weight_ih_l0": torch.tensor(-1.739493),
    "encoder.blocks.0.lstm.weight_hh_l0": torch.tensor(-4.145900),
    "encoder.blocks.0.lstm.bias_ih_l0": torch.tensor(0.652574),
    "encoder.blocks.0.lstm.bias_hh_l0": torch.tensor(-0.179724),
    "encoder.blocks.0.lstm.init_hidden_state": torch.tensor(-1.262706),
    "encoder.blocks.0.lstm.init_cell_state": torch.tensor(4.626403),
    "encoder.blocks.0.ff_norm.weight": torch.tensor(-0.041233),
    "encoder.blocks.0.ff_norm.bias": torch.tensor(3.236254),
    "encoder.blocks.0.ff.proj.weight": torch.tensor(4.980308),
    "encoder.blocks.0.ff.proj.bias": torch.tensor(1.961943),
    "encoder.blocks.0.ff.out.weight": torch.tensor(0.204000),
    "encoder.blocks.0.ff.out.bias": torch.tensor(-0.071591),
    "encoder.blocks.1.norm.weight": torch.tensor(2.795757),
    "encoder.blocks.1.norm.bias": torch.tensor(-6.156712),
    "encoder.blocks.1.lstm.weight_ih_l0": torch.tensor(1.433082),
    "encoder.blocks.1.lstm.weight_hh_l0": torch.tensor(-4.329137),
    "encoder.blocks.1.lstm.bias_ih_l0": torch.tensor(-0.781742),
    "encoder.blocks.1.lstm.bias_hh_l0": torch.tensor(-0.194515),
    "encoder.blocks.1.lstm.init_hidden_state": torch.tensor(-0.540895),
    "encoder.blocks.1.lstm.init_cell_state": torch.tensor(-3.302565),
    "encoder.blocks.1.ff_norm.weight": torch.tensor(-1.193851),
    "encoder.blocks.1.ff_norm.bias": torch.tensor(-0.808373),
    "encoder.blocks.1.ff.proj.weight": torch.tensor(-6.975581),
    "encoder.blocks.1.ff.proj.bias": torch.tensor(1.617217),
    "encoder.blocks.1.ff.out.weight": torch.tensor(-1.744055),
    "encoder.blocks.1.ff.out.bias": torch.tensor(-0.158968),
    "encoder.out.weight": torch.tensor(-3.959515),
    "encoder.out.bias": torch.tensor(-1.373152),
    "decoder.init_state.0.weight": torch.tensor(-1.840380),
    "decoder.init_state.0.bias": torch.tensor(0.145609),
    "decoder.init_state.2.weight": torch.tensor(-3.298804),
    "decoder.init_state.2.bias": torch.tensor(0.223963),
    "decoder.pre_net.encode.weight_ih_l0": torch.tensor(1.066760),
    "decoder.pre_net.encode.weight_hh_l0": torch.tensor(-2.127719),
    "decoder.pre_net.encode.bias_ih_l0": torch.tensor(-0.735991),
    "decoder.pre_net.encode.bias_hh_l0": torch.tensor(0.385362),
    "decoder.pre_net.encode.init_hidden_state": torch.tensor(5.968543),
    "decoder.pre_net.encode.init_cell_state": torch.tensor(4.274887),
    "decoder.pre_net.out.1.weight": torch.tensor(2.158975),
    "decoder.pre_net.out.1.bias": torch.tensor(6.258615),
    "decoder.attn_rnn.lstm.weight_ih": torch.tensor(3.227081),
    "decoder.attn_rnn.lstm.weight_hh": torch.tensor(2.475742),
    "decoder.attn_rnn.lstm.bias_ih": torch.tensor(-0.379075),
    "decoder.attn_rnn.lstm.bias_hh": torch.tensor(-1.337197),
    "decoder.attn_rnn.lstm.init_hidden_state": torch.tensor(1.553586),
    "decoder.attn_rnn.lstm.init_cell_state": torch.tensor(3.268224),
    "decoder.attn_rnn.attn.alignment_conv.weight": torch.tensor(-2.089337),
    "decoder.attn_rnn.attn.alignment_conv.bias": torch.tensor(2.710139),
    "decoder.attn_rnn.attn.project_query.weight": torch.tensor(0.562974),
    "decoder.attn_rnn.attn.project_query.bias": torch.tensor(-0.635165),
    "decoder.attn_rnn.attn.project_scores.weight": torch.tensor(0.374247),
    "decoder.linear_stop_token.1.weight": torch.tensor(-0.518477),
    "decoder.linear_stop_token.1.bias": torch.tensor(0.105098),
    "decoder.linear_stop_token.3.weight": torch.tensor(-1.250005),
    "decoder.linear_stop_token.3.bias": torch.tensor(0.062122),
    "decoder.lstm_out.weight_ih_l0": torch.tensor(17.202736),
    "decoder.lstm_out.weight_hh_l0": torch.tensor(-2.934480),
    "decoder.lstm_out.bias_ih_l0": torch.tensor(1.368614),
    "decoder.lstm_out.bias_hh_l0": torch.tensor(0.442680),
    "decoder.lstm_out.init_hidden_state": torch.tensor(4.561096),
    "decoder.lstm_out.init_cell_state": torch.tensor(6.946077),
    "decoder.linear_out.0.weight": torch.tensor(1.415857),
    "decoder.linear_out.0.bias": torch.tensor(-0.893772),
    "decoder.linear_out.2.weight": torch.tensor(1.002060),
    "decoder.linear_out.2.bias": torch.tensor(0.137269),
}

_expected_grads = {
    "encoder.embed_seq_meta.0.embed.weight": torch.tensor(-0.066338),
    "encoder.embed_seq_meta.1.embed.weight": torch.tensor(-0.066338),
    "encoder.embed_seq_vector.weight": torch.tensor(0.082776),
    "encoder.embed_seq_vector.bias": torch.tensor(-0.066338),
    "encoder.embed_token.embed.weight": torch.tensor(-0.017105),
    "encoder.embed_token_meta.0.embed.weight": torch.tensor(0.035062),
    "encoder.embed_word_vec.weight": torch.tensor(0.327195),
    "encoder.embed_word_vec.bias": torch.tensor(0.120810),
    "encoder.norm_embed.weight": torch.tensor(0.001755),
    "encoder.norm_embed.bias": torch.tensor(0.005967),
    "encoder.embed_annos.weight": torch.tensor(-0.026988),
    "encoder.embed_annos.bias": torch.tensor(-0.052589),
    "encoder.pre_net.0.1.weight": torch.tensor(-0.220374),
    "encoder.pre_net.0.1.bias": torch.tensor(-0.030864),
    "encoder.pre_net.1.1.weight": torch.tensor(-0.000000),
    "encoder.pre_net.1.1.bias": torch.tensor(-0.000000),
    "encoder.blocks.0.norm.weight": torch.tensor(0.040254),
    "encoder.blocks.0.norm.bias": torch.tensor(0.001197),
    "encoder.blocks.0.lstm.weight_ih_l0": torch.tensor(0.428562),
    "encoder.blocks.0.lstm.weight_hh_l0": torch.tensor(0.112785),
    "encoder.blocks.0.lstm.bias_ih_l0": torch.tensor(0.200992),
    "encoder.blocks.0.lstm.bias_hh_l0": torch.tensor(0.200992),
    "encoder.blocks.0.lstm.init_hidden_state": torch.tensor(-0.014722),
    "encoder.blocks.0.lstm.init_cell_state": torch.tensor(-0.000284),
    "encoder.blocks.0.ff_norm.weight": torch.tensor(-0.035857),
    "encoder.blocks.0.ff_norm.bias": torch.tensor(0.057972),
    "encoder.blocks.0.ff.proj.weight": torch.tensor(1.182941),
    "encoder.blocks.0.ff.proj.bias": torch.tensor(0.341060),
    "encoder.blocks.0.ff.out.weight": torch.tensor(-0.000000),
    "encoder.blocks.0.ff.out.bias": torch.tensor(-0.000000),
    "encoder.blocks.1.norm.weight": torch.tensor(-0.082201),
    "encoder.blocks.1.norm.bias": torch.tensor(0.091741),
    "encoder.blocks.1.lstm.weight_ih_l0": torch.tensor(1.316781),
    "encoder.blocks.1.lstm.weight_hh_l0": torch.tensor(0.094078),
    "encoder.blocks.1.lstm.bias_ih_l0": torch.tensor(-0.194272),
    "encoder.blocks.1.lstm.bias_hh_l0": torch.tensor(-0.194272),
    "encoder.blocks.1.lstm.init_hidden_state": torch.tensor(-0.016610),
    "encoder.blocks.1.lstm.init_cell_state": torch.tensor(-0.035311),
    "encoder.blocks.1.ff_norm.weight": torch.tensor(-0.006801),
    "encoder.blocks.1.ff_norm.bias": torch.tensor(-0.011287),
    "encoder.blocks.1.ff.proj.weight": torch.tensor(-0.214220),
    "encoder.blocks.1.ff.proj.bias": torch.tensor(-0.032677),
    "encoder.blocks.1.ff.out.weight": torch.tensor(-0.000000),
    "encoder.blocks.1.ff.out.bias": torch.tensor(-0.000000),
    "encoder.out.weight": torch.tensor(1.020916),
    "encoder.out.bias": torch.tensor(1.051214),
    "decoder.init_state.0.weight": torch.tensor(0.189184),
    "decoder.init_state.0.bias": torch.tensor(-0.010299),
    "decoder.init_state.2.weight": torch.tensor(0.147908),
    "decoder.init_state.2.bias": torch.tensor(0.293262),
    "decoder.pre_net.encode.weight_ih_l0": torch.tensor(0.130525),
    "decoder.pre_net.encode.weight_hh_l0": torch.tensor(-4.105129),
    "decoder.pre_net.encode.bias_ih_l0": torch.tensor(-1.432487),
    "decoder.pre_net.encode.bias_hh_l0": torch.tensor(-1.432488),
    "decoder.pre_net.encode.init_hidden_state": torch.tensor(-0.125631),
    "decoder.pre_net.encode.init_cell_state": torch.tensor(-0.303121),
    "decoder.pre_net.out.1.weight": torch.tensor(-0.177572),
    "decoder.pre_net.out.1.bias": torch.tensor(2.784616),
    "decoder.attn_rnn.lstm.weight_ih": torch.tensor(0.138957),
    "decoder.attn_rnn.lstm.weight_hh": torch.tensor(-0.037687),
    "decoder.attn_rnn.lstm.bias_ih": torch.tensor(-0.021458),
    "decoder.attn_rnn.lstm.bias_hh": torch.tensor(-0.021458),
    "decoder.attn_rnn.lstm.init_hidden_state": torch.tensor(0.073046),
    "decoder.attn_rnn.lstm.init_cell_state": torch.tensor(0.042865),
    "decoder.attn_rnn.attn.alignment_conv.weight": torch.tensor(-0.062420),
    "decoder.attn_rnn.attn.alignment_conv.bias": torch.tensor(0.020186),
    "decoder.attn_rnn.attn.project_query.weight": torch.tensor(-0.008898),
    "decoder.attn_rnn.attn.project_query.bias": torch.tensor(0.020186),
    "decoder.attn_rnn.attn.project_scores.weight": torch.tensor(0.543210),
    "decoder.linear_stop_token.1.weight": torch.tensor(-26.615719),
    "decoder.linear_stop_token.1.bias": torch.tensor(-4.563582),
    "decoder.linear_stop_token.3.weight": torch.tensor(37.938263),
    "decoder.linear_stop_token.3.bias": torch.tensor(7.429334),
    "decoder.lstm_out.weight_ih_l0": torch.tensor(3.578972),
    "decoder.lstm_out.weight_hh_l0": torch.tensor(0.279987),
    "decoder.lstm_out.bias_ih_l0": torch.tensor(0.513433),
    "decoder.lstm_out.bias_hh_l0": torch.tensor(0.513433),
    "decoder.lstm_out.init_hidden_state": torch.tensor(-0.006695),
    "decoder.lstm_out.init_cell_state": torch.tensor(-0.037581),
    "decoder.linear_out.0.weight": torch.tensor(96.716957),
    "decoder.linear_out.0.bias": torch.tensor(11.800383),
    "decoder.linear_out.2.weight": torch.tensor(41.281086),
    "decoder.linear_out.2.bias": torch.tensor(6.859480),
}

# fmt: off
# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
_expected_frames = [
    [0.183752, 1.263783, 0.199781, 0.180702, 0.517118, 0.212449, -0.028916, 0.002912,
     0.265127, 0.919176, 0.896476, 0.482225, 0.289585],
    [-0.368306, 0.530756, 0.544356, -0.394061, 0.297002, 0.331881, 0.257461, 1.034235,
     0.403439, 0.211838, -0.186634, 0.919892, -0.087571],
    [-0.338226, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.515775, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.368340, 0.530839, -0.307036, 0.343490, -0.338309, 0.523675, 0.451867, 0.291560,
     0.419088, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.310652, 0.991464, 0.344260, 0.745572, 0.698335, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.419979, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.093053, 0.388644, 0.576656, 0.425759, 0.334013, 0.381638, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [-0.412254, 0.183834, 0.745491, 0.282927, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.532618, 0.043094, -0.454403, -0.066790, 0.507632, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
]
_expected_frames = torch.tensor(_expected_frames)
# NOTE: For first, the `stop_token` always predicts `_e` because the `window_start` largely stays
# at zero and the the number of tokens is larger than the window length.
_expected_stop_tokens = [
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10],
    [5.321714e-01, 4.596499e-01, 4.261609e-01, 4.970221e-01, 4.883769e-01, 4.768900e-01,
     5.123591e-01, 5.069470e-01, 5.980458e-01, 6.112827e-01, 4.309723e-01, 4.888828e-01,
     5.710340e-01],
    [5.310367e-01, 4.092691e-01, 2.978090e-01, 5.097395e-01, 5.316379e-01, 5.086237e-01,
     4.216758e-01, 4.298418e-01, 4.224053e-01, 5.799737e-01, 4.692875e-01, 4.130270e-01,
     3.984516e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10],
    [4.290667e-01, 4.141870e-01, 4.219839e-01, 3.779531e-01, 4.659240e-01, 4.300356e-01,
     4.380225e-01, 5.496593e-01, 4.252779e-01, 5.273933e-01, 4.084761e-01, 5.602357e-01,
     3.986839e-01],
    [5.220850e-01, 4.824966e-01, 4.969140e-01, 5.737752e-01, 5.758968e-01, 4.623605e-01,
     5.255774e-01, 5.266117e-01, 5.025638e-01, 5.024447e-01, 4.906588e-01, 5.561344e-01,
     4.958811e-01],
    [1.000001e-10, 1.000001e-10, 4.643380e-01, 4.572891e-01, 4.905357e-01, 4.647149e-01,
     4.850212e-01, 4.393467e-01, 4.432347e-01, 5.412685e-01, 4.294289e-01, 4.272121e-01,
     4.232338e-01],
    [1.000001e-10, 1.000001e-10, 3.882686e-01, 5.215589e-01, 4.624801e-01, 5.913907e-01,
     4.645633e-01, 4.939559e-01, 4.764597e-01, 4.003231e-01, 3.691745e-01, 5.222153e-01,
     4.719032e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10]
]
_expected_stop_tokens = torch.tensor(_expected_stop_tokens)
_expected_alignments = [
    [3.351704, 3.407894, 0.992016, 1.017579, 1.006448, 0.679579, 0.399382, 0.000000, 0.000000],
    [0.638879, 2.319791, 3.363157, 3.010255, 1.213306, 1.025604, 0.804627, 0.345509, 0.000000],
    [5.705660, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [4.500727, 4.553255, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [3.892905, 3.668438, 1.062833, 1.284568, 0.763787, 0.000000, 0.000000, 0.000000, 0.000000],
    [2.730651, 4.887575, 4.362044, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [3.810284, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [0.944268, 1.212941, 3.770082, 4.431189, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [3.828095, 4.338525, 3.425868, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
    [5.616264, 4.049974, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]
]
_expected_alignments = torch.tensor(_expected_alignments)
# fmt: on


def test_spectrogram_model__version():
    """Test `spectrogram_model.SpectrogramModel` has not changed since it was last tested.

    TODO: This test won't pass on M1 MacBooks until PyTorch fixes some bugs:
    https://github.com/pytorch/pytorch/issues/84030
    """
    torch.set_printoptions(precision=6, linewidth=100)

    with fork_rng(123):
        params = Params(max_frames=13, batch_size=10)
        inputs, _, _, target_frames, _, _ = _make_inputs(params)
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(-0.28274))

    with fork_rng(123456):
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
        assert_almost_equal(preds.num_frames, torch.tensor([13, 13, 1, 1, 9, 5, 1, 6, 4, 5]))
        print("Reached Max", preds.reached_max)
        expected = torch.tensor([True, True, False, False, True, True, False, True, False, True])
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
        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(11.564344))
        print("Stop Token Loss", stop_token_loss.sum())
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(10.039842))
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _utils.print_params("_expected_grads", grads)
        for name, grad in grads:
            assert_almost_equal(_expected_grads[name], grad.sum())
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(0.704654))
