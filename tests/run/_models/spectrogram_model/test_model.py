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
    output_min: float = -1.0
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
) -> SpectrogramModel:
    """Make `spectrogram_model.SpectrogramModel` for testing."""
    config = {
        run._models.spectrogram_model.encoder.Encoder: cf.Args(num_layers=2, conv_filter_size=3),
        run._models.spectrogram_model.decoder.Decoder: cf.Args(hidden_size=16),
        run._models.spectrogram_model.pre_net.PreNet: cf.Args(num_layers=1, dropout=dropout),
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
        output_min=params.output_min,
        stop_threshold=stop_threshold,
        stop_token_eps=stop_token_eps,
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
    target_frames = torch.clamp(target_frames, min=params.output_min)
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
    model = _make_spectrogram_model(params)
    _mock_model(model)
    preds = model(inputs, target_frames, target_mask=target_mask)
    _check_preds(params, model, num_sliced_tokens, preds)
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
    "encoder.embed_seq_meta.0.weight": torch.tensor(1.852018),
    "encoder.embed_seq_meta.1.weight": torch.tensor(9.608484),
    "encoder.embed_token_meta.0.weight": torch.tensor(1.268590),
    "encoder.embed_token.weight": torch.tensor(-19.686558),
    "encoder.embed_anno.0.weight": torch.tensor(-0.515247),
    "encoder.embed_anno.0.bias": torch.tensor(-0.444562),
    "encoder.embed_anno.2.weight": torch.tensor(-0.173128),
    "encoder.embed_anno.2.bias": torch.tensor(-0.034205),
    "encoder.embed_anno.4.weight": torch.tensor(-4.253194),
    "encoder.embed_anno.4.bias": torch.tensor(1.322851),
    "encoder.embed.0.weight": torch.tensor(-1.117358),
    "encoder.embed.0.bias": torch.tensor(-0.124495),
    "encoder.embed.2.weight": torch.tensor(-1.802696),
    "encoder.embed.2.bias": torch.tensor(-0.167022),
    "encoder.embed.4.weight": torch.tensor(3.417824),
    "encoder.embed.4.bias": torch.tensor(-2.517295),
    "encoder.conv_layers.0.1.weight": torch.tensor(-0.958008),
    "encoder.conv_layers.0.1.bias": torch.tensor(-0.102842),
    "encoder.conv_layers.1.1.weight": torch.tensor(-2.953365),
    "encoder.conv_layers.1.1.bias": torch.tensor(0.022060),
    "encoder.norm_layers.0.weight": torch.tensor(4.577718),
    "encoder.norm_layers.0.bias": torch.tensor(-0.761458),
    "encoder.norm_layers.1.weight": torch.tensor(3.238918),
    "encoder.norm_layers.1.bias": torch.tensor(2.790632),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(-3.652131),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(1.398460),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(1.733211),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-1.098471),
    "encoder.lstm.rnn_layers.0.0.init_hidden_state": torch.tensor(2.117465),
    "encoder.lstm.rnn_layers.0.0.init_cell_state": torch.tensor(4.676821),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(11.619355),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(-3.790668),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(-1.488367),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(0.223798),
    "encoder.lstm.rnn_layers.0.1.init_hidden_state": torch.tensor(-2.553659),
    "encoder.lstm.rnn_layers.0.1.init_cell_state": torch.tensor(-1.680955),
    "encoder.lstm_norm.weight": torch.tensor(-4.309731),
    "encoder.lstm_norm.bias": torch.tensor(-3.387039),
    "encoder.project_out.1.weight": torch.tensor(0.518785),
    "encoder.project_out.1.bias": torch.tensor(-0.004707),
    "encoder.project_out.2.weight": torch.tensor(-4.151481),
    "encoder.project_out.2.bias": torch.tensor(0.099431),
    "decoder.init_state.0.weight": torch.tensor(-4.552614),
    "decoder.init_state.0.bias": torch.tensor(-0.061705),
    "decoder.init_state.2.weight": torch.tensor(-3.814329),
    "decoder.init_state.2.bias": torch.tensor(-1.229276),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(0.328073),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(-0.518337),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(-4.130414),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(3.660183),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(-2.638722),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-0.150317),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(1.041038),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(-0.374360),
    "decoder.lstm_layer_one.init_hidden_state": torch.tensor(0.614631),
    "decoder.lstm_layer_one.init_cell_state": torch.tensor(4.569544),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(-0.418454),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(6.479007),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(-2.019958),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(1.541070),
    "decoder.lstm_layer_two.init_hidden_state": torch.tensor(5.815588),
    "decoder.lstm_layer_two.init_cell_state": torch.tensor(9.060997),
    "decoder.attention.alignment_conv.weight": torch.tensor(1.035867),
    "decoder.attention.alignment_conv.bias": torch.tensor(-0.528715),
    "decoder.attention.project_query.weight": torch.tensor(2.014140),
    "decoder.attention.project_query.bias": torch.tensor(0.099615),
    "decoder.attention.project_scores.1.weight": torch.tensor(-0.234625),
    "decoder.linear_out.weight": torch.tensor(2.222932),
    "decoder.linear_out.bias": torch.tensor(-0.137687),
    "decoder.linear_stop_token.1.weight": torch.tensor(1.959174),
    "decoder.linear_stop_token.1.bias": torch.tensor(-0.677653),
    "decoder.linear_stop_token.3.weight": torch.tensor(0.763183),
    "decoder.linear_stop_token.3.bias": torch.tensor(0.145512),
}

_expected_grads = {
    "encoder.embed_seq_meta.0.weight": torch.tensor(-29.919094),
    "encoder.embed_seq_meta.1.weight": torch.tensor(-79.813225),
    "encoder.embed_token_meta.0.weight": torch.tensor(-2.782639),
    "encoder.embed_token.weight": torch.tensor(2.383842),
    "encoder.embed_anno.0.weight": torch.tensor(4.227918),
    "encoder.embed_anno.0.bias": torch.tensor(-4.887265),
    "encoder.embed_anno.2.weight": torch.tensor(70.545288),
    "encoder.embed_anno.2.bias": torch.tensor(36.659855),
    "encoder.embed_anno.4.weight": torch.tensor(-11.038326),
    "encoder.embed_anno.4.bias": torch.tensor(8.870480),
    "encoder.embed.0.weight": torch.tensor(-71.652191),
    "encoder.embed.0.bias": torch.tensor(-5.248698),
    "encoder.embed.2.weight": torch.tensor(73.600769),
    "encoder.embed.2.bias": torch.tensor(16.179394),
    "encoder.embed.4.weight": torch.tensor(-5.704032),
    "encoder.embed.4.bias": torch.tensor(-1.720087),
    "encoder.conv_layers.0.1.weight": torch.tensor(19.459681),
    "encoder.conv_layers.0.1.bias": torch.tensor(3.175056),
    "encoder.conv_layers.1.1.weight": torch.tensor(-0.802662),
    "encoder.conv_layers.1.1.bias": torch.tensor(-1.064252),
    "encoder.norm_layers.0.weight": torch.tensor(13.825806),
    "encoder.norm_layers.0.bias": torch.tensor(-13.439140),
    "encoder.norm_layers.1.weight": torch.tensor(-42.703293),
    "encoder.norm_layers.1.bias": torch.tensor(3.497506),
    "encoder.lstm.rnn_layers.0.0.weight_ih_l0": torch.tensor(-9.971325),
    "encoder.lstm.rnn_layers.0.0.weight_hh_l0": torch.tensor(0.751405),
    "encoder.lstm.rnn_layers.0.0.bias_ih_l0": torch.tensor(-1.579176),
    "encoder.lstm.rnn_layers.0.0.bias_hh_l0": torch.tensor(-1.579176),
    "encoder.lstm.rnn_layers.0.0.init_hidden_state": torch.tensor(-0.092618),
    "encoder.lstm.rnn_layers.0.0.init_cell_state": torch.tensor(-0.281054),
    "encoder.lstm.rnn_layers.0.1.weight_ih_l0": torch.tensor(-10.558510),
    "encoder.lstm.rnn_layers.0.1.weight_hh_l0": torch.tensor(0.546101),
    "encoder.lstm.rnn_layers.0.1.bias_ih_l0": torch.tensor(1.104011),
    "encoder.lstm.rnn_layers.0.1.bias_hh_l0": torch.tensor(1.104011),
    "encoder.lstm.rnn_layers.0.1.init_hidden_state": torch.tensor(0.342603),
    "encoder.lstm.rnn_layers.0.1.init_cell_state": torch.tensor(0.508274),
    "encoder.lstm_norm.weight": torch.tensor(-17.635391),
    "encoder.lstm_norm.bias": torch.tensor(5.752861),
    "encoder.project_out.1.weight": torch.tensor(-0.000010),
    "encoder.project_out.1.bias": torch.tensor(0.000004),
    "encoder.project_out.2.weight": torch.tensor(-21.025402),
    "encoder.project_out.2.bias": torch.tensor(28.131565),
    "decoder.init_state.0.weight": torch.tensor(-2.648980),
    "decoder.init_state.0.bias": torch.tensor(-0.903231),
    "decoder.init_state.2.weight": torch.tensor(24.273449),
    "decoder.init_state.2.bias": torch.tensor(2.582769),
    "decoder.pre_net.layers.0.0.weight": torch.tensor(-5.161791),
    "decoder.pre_net.layers.0.0.bias": torch.tensor(-1.811457),
    "decoder.pre_net.layers.0.2.weight": torch.tensor(-2.433525),
    "decoder.pre_net.layers.0.2.bias": torch.tensor(-0.411948),
    "decoder.lstm_layer_one.weight_ih": torch.tensor(-14.825934),
    "decoder.lstm_layer_one.weight_hh": torch.tensor(-0.833654),
    "decoder.lstm_layer_one.bias_ih": torch.tensor(2.163858),
    "decoder.lstm_layer_one.bias_hh": torch.tensor(2.163858),
    "decoder.lstm_layer_one.init_hidden_state": torch.tensor(0.259958),
    "decoder.lstm_layer_one.init_cell_state": torch.tensor(0.875450),
    "decoder.lstm_layer_two.weight_ih_l0": torch.tensor(-85.576233),
    "decoder.lstm_layer_two.weight_hh_l0": torch.tensor(-1.953106),
    "decoder.lstm_layer_two.bias_ih_l0": torch.tensor(1.107665),
    "decoder.lstm_layer_two.bias_hh_l0": torch.tensor(1.107665),
    "decoder.lstm_layer_two.init_hidden_state": torch.tensor(-0.495610),
    "decoder.lstm_layer_two.init_cell_state": torch.tensor(-1.247253),
    "decoder.attention.alignment_conv.weight": torch.tensor(-0.969161),
    "decoder.attention.alignment_conv.bias": torch.tensor(0.288607),
    "decoder.attention.project_query.weight": torch.tensor(-0.600842),
    "decoder.attention.project_query.bias": torch.tensor(0.288607),
    "decoder.attention.project_scores.1.weight": torch.tensor(-1.757438),
    "decoder.linear_out.weight": torch.tensor(53.109726),
    "decoder.linear_out.bias": torch.tensor(-54.563400),
    "decoder.linear_stop_token.1.weight": torch.tensor(-16.366018),
    "decoder.linear_stop_token.1.bias": torch.tensor(2.052802),
    "decoder.linear_stop_token.3.weight": torch.tensor(14.696085),
    "decoder.linear_stop_token.3.bias": torch.tensor(4.593737),
}

# fmt: off
# NOTE: `test_spectrogram_model__version` tests the model accross multiple cases: one frame,
# multiple frames, and max frames.
_expected_frames = [
    [1.366075, 1.245820, 1.046142, 0.858820, 0.816668, 0.743831, 0.819121, 1.133023,
     1.259643, 1.204182, 1.160060, 1.050212],
    [2.653476, 2.369536, 2.352416, 2.279980, 2.238275, 2.082993, 2.071112, 2.020142,
     2.044245, 2.040308, 2.027943, 1.964504],
    [0.375175, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000],
    [1.378470, 1.257034, 1.803847, 1.820856, 1.994045, 2.132876, 2.101607, 2.028780,
     2.145771, 2.229941, 2.076566, 0.000000],
    [1.527878, 1.394192, 1.702764, 1.652411, 1.389725, 1.342866, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000],
    [2.258687, 1.955694, 2.321157, 2.353877, 2.370153, 2.397995, 2.449710, 2.438354,
     0.000000, 0.000000, 0.000000, 0.000000],
    [1.481529, 1.272664, 1.025689, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000],
    [0.621059, 1.035297, 1.078354, 1.854820, 2.199958, 2.191419, 2.193977, 1.832925,
     0.000000, 0.000000, 0.000000, 0.000000],
    [0.047801, -0.050726, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000, 0.000000, 0.000000],
    [0.229594, 0.162643, -0.067939, -0.553439, -0.781329, -0.980638, -0.998462, -1.063241,
     0.000000, 0.000000, 0.000000, 0.000000]
]
_expected_frames = torch.tensor(_expected_frames)
# NOTE: For first, the `stop_token` always predicts `_e` because the `window_start` largely stays
# at zero and the the number of tokens is larger than the window length.
_expected_stop_tokens = [
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10],
    [6.244421e-01, 5.727453e-01, 6.267647e-01, 6.121179e-01, 6.279113e-01, 6.334319e-01,
     5.954402e-01, 5.853063e-01, 5.746242e-01, 5.874828e-01, 5.459400e-01, 5.466809e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 5.424259e-01,
     5.785822e-01, 6.454517e-01, 5.604892e-01, 5.773947e-01, 5.413802e-01, 6.879032e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 5.857375e-01, 6.492345e-01, 5.392531e-01],
    [1.000001e-10, 1.000001e-10, 6.934602e-01, 7.317717e-01, 6.126302e-01, 4.913065e-01,
     5.253308e-01, 5.496082e-01, 5.310274e-01, 4.809340e-01, 6.426164e-01, 4.557088e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 7.313493e-01, 5.528712e-01, 7.301527e-01, 6.137644e-01, 6.180637e-01],
    [1.000001e-10, 6.698847e-01, 7.360702e-01, 6.912283e-01, 5.621150e-01, 5.999660e-01,
     5.553227e-01, 5.823148e-01, 5.631109e-01, 5.482294e-01, 6.312528e-01, 6.071500e-01],
    [1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10,
     1.000001e-10, 1.000001e-10, 1.000001e-10, 1.000001e-10, 5.599211e-01, 5.575398e-01]
]
_expected_stop_tokens = torch.tensor(_expected_stop_tokens)
_expected_alignments = [
    [0.960883, 1.852487, 1.671617, 1.771577, 1.002615, 1.599182, 1.373143, 1.135456, 0.000000,
     0.000000, 0.000000],
    [1.931316, 2.188295, 2.800028, 2.172840, 1.894981, 0.394679, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000],
    [3.201679, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000],
    [1.179194, 1.584349, 2.378683, 2.081485, 2.022616, 1.072133, 0.732867, 0.324633, 0.000000,
     0.000000, 0.000000],
    [0.879053, 1.263301, 1.416670, 1.421922, 2.446049, 2.049813, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000],
    [0.923566, 1.920799, 1.970317, 2.439686, 2.306876, 1.220054, 0.563905, 0.000000, 0.000000,
     0.000000, 0.000000],
    [0.919206, 1.537302, 2.564836, 2.630302, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000],
    [0.950684, 1.205644, 1.862708, 1.628725, 2.235595, 1.830989, 1.587425, 0.000000, 0.000000,
     0.000000, 0.000000],
    [0.986724, 3.219990, 3.139766, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
     0.000000, 0.000000],
    [0.869615, 1.670807, 2.509629, 2.598578, 2.468219, 1.078097, 0.170666, 0.000000, 0.000000,
     0.000000, 0.000000]
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
        assert_almost_equal(val, torch.tensor(-0.273233))

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
        assert_almost_equal(preds.num_frames, torch.tensor([12, 12, 1, 11, 6, 8, 3, 8, 2, 8]))
        print("Reached Max", preds.reached_max)
        expected = torch.tensor([True, True, True, True, False, True, False, True, False, True])
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
        assert_almost_equal(spectrogram_loss.sum(), torch.tensor(133.995667))
        print("Stop Token Loss", stop_token_loss.sum())
        assert_almost_equal(stop_token_loss.sum(), torch.tensor(6.975104))
        grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
        _utils.print_params("_expected_grads", grads)
        for name, grad in grads:
            assert_almost_equal(_expected_grads[name], grad.sum())
        val = torch.randn(1)
        print("Rand", val)
        assert_almost_equal(val, torch.tensor(-1.050120))
