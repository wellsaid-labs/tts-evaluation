import typing
from functools import partial

import torch
from torchnlp.random import fork_rng

import lib.spectrogram_model.encoder
from tests import _utils

assert_almost_equal = partial(_utils.assert_almost_equal, decimal=5)


def test__roll():
    """ Test `lib.spectrogram_model.encoder._roll` to roll given a simple tensor. """
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    result = lib.spectrogram_model.encoder._roll(tensor, shift=torch.tensor(4), dim=0)
    assert torch.equal(result, torch.tensor([3, 4, 5, 6, 1, 2]))


def test__roll__2d():
    """Test `lib.spectrogram_model.encoder._roll` to roll given a 2d `tensor` with variable
    `shift`. Furthermore, this tests a negative `dim`."""
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    result = lib.spectrogram_model.encoder._roll(
        tensor, shift=torch.tensor([0, 1, 2]), dim=-1
    )
    assert torch.equal(result, torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test__roll__transpose():
    """Test `lib.spectrogram_model.encoder._roll` to roll given a transposed 2d `tensor`.
     `lib.spectrogram_model.encoder._roll` should return consistent results regardless of the
    dimension and ordering of the data."""
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]).transpose(0, 1)
    result = lib.spectrogram_model.encoder._roll(
        tensor, shift=torch.tensor([0, 1, 2]), dim=0
    ).transpose(0, 1)
    assert torch.equal(result, torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test__roll__3d():
    """ Test `lib.spectrogram_model.encoder._roll` to roll given a 3d `tensor` and 2d `start`. """
    tensor = torch.tensor([1, 2, 3]).view(1, 3, 1).expand(4, 3, 4)
    shift = torch.arange(0, 16).view(4, 4)
    result = lib.spectrogram_model.encoder._roll(tensor, shift=shift, dim=1)
    assert shift[0, 0] == 0
    assert torch.equal(result[0, :, 0], torch.tensor([1, 2, 3]))
    assert shift[0, 1] == 1
    assert torch.equal(result[0, :, 1], torch.tensor([3, 1, 2]))
    assert shift[0, 2] == 2
    assert torch.equal(result[0, :, 2], torch.tensor([2, 3, 1]))
    assert shift[0, 3] == 3
    assert torch.equal(result[0, :, 3], torch.tensor([1, 2, 3]))
    assert shift[1, 0] == 4
    assert torch.equal(result[1, :, 0], torch.tensor([3, 1, 2]))
    assert shift[1, 1] == 5
    assert torch.equal(result[1, :, 1], torch.tensor([2, 3, 1]))


def _make_rnn(
    input_size: int = 4, hidden_size: int = 5, num_layers: int = 2, **kwargs
) -> lib.spectrogram_model.encoder._RightMaskedBiRNN:
    """ Make `encoder._RightMaskedBiRNN` for testing."""
    return lib.spectrogram_model.encoder._RightMaskedBiRNN(
        input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, **kwargs
    )


def _make_rnn_inputs(
    module: lib.spectrogram_model.encoder._RightMaskedBiRNN,
    batch_size: int = 2,
    seq_len: int = 3,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """ Make `encoder._RightMaskedBiRNN` inputs for testing."""
    tokens = torch.randn(seq_len, batch_size, module.input_size)
    tokens_mask = torch.ones(seq_len, batch_size, dtype=torch.bool)
    return tokens, tokens_mask


def test__right_masked_bi_rnn__lstm():
    """ Test `encoder._RightMaskedBiRNN` is consistent with `torch.nn.LSTM`. """
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(rnn_class=torch.nn.LSTM)
    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    result = masked_bi_rnn(tokens, tokens_mask, num_tokens)
    assert torch.equal(lstm(tokens)[0], masked_bi_rnn(tokens, tokens_mask, num_tokens))
    result.sum().backward()


def test__right_masked_bi_rnn__gru():
    """ Test `encoder._RightMaskedBiRNN` is consistent with `torch.nn.GRU`. """
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(rnn_class=torch.nn.GRU)
    with fork_rng(123):
        gru = torch.nn.GRU(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    assert torch.equal(gru(tokens)[0], masked_bi_rnn(tokens, tokens_mask, num_tokens))


def test__right_masked_bi_rnn__jagged_mask():
    """Test `encoder._RightMaskedBiRNN` ignores the jagged padding and is consistent with
    `torch.nn.LSTM`."""
    input_size = 1
    hidden_size = 5

    # tokens [batch_size, seq_len] → [seq_len, batch_size, input_size]
    tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]], dtype=torch.float32)
    tokens = tokens.transpose(0, 1).unsqueeze(2)
    # tokens_mask [batch_size, seq_len] → [seq_len, batch_size]
    tokens_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool
    ).transpose(0, 1)
    num_tokens = tokens_mask.sum(dim=0)

    masked_bi_rnn = lib.spectrogram_model.encoder._RightMaskedBiRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        rnn_class=torch.nn.LSTM,
    )
    result = masked_bi_rnn(tokens, tokens_mask, num_tokens)

    forward_lstm, backward_lstm = masked_bi_rnn.rnn_layers[0]
    expected = torch.zeros(tokens.shape[0], tokens.shape[1], hidden_size * 2)
    expected[:, :, :hidden_size] = forward_lstm(tokens)[0].masked_fill(
        ~tokens_mask.unsqueeze(2), 0
    )
    expected[:3, 0:1, hidden_size:] = backward_lstm(tokens[:3, 0:1].flip(0))[0].flip(0)
    expected[:2, 1:2, hidden_size:] = backward_lstm(tokens[:2, 1:2].flip(0))[0].flip(0)
    assert_almost_equal(result, expected)


def test__right_masked_bi_rnn__multilayer_mask():
    """Test `encoder._RightMaskedBiRNN` ignores the multiple layers of padding and is consistent
    with  `torch.nn.LSTM`."""
    with fork_rng(123):
        masked_bi_rnn = _make_rnn(num_layers=3, rnn_class=torch.nn.LSTM)
    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=masked_bi_rnn.input_size,
            hidden_size=masked_bi_rnn.hidden_size,
            num_layers=masked_bi_rnn.num_layers,
            bidirectional=True,
        )
    padding_len = 2
    tokens, tokens_mask = _make_rnn_inputs(masked_bi_rnn)
    num_tokens = tokens_mask.sum(dim=0)
    padded_tokens = torch.cat(
        [tokens, torch.zeros(padding_len, *tokens.shape[1:3])], dim=0
    )
    padded_tokens_mask = torch.cat(
        [tokens_mask, torch.zeros(padding_len, tokens_mask.shape[1], dtype=torch.bool)],
        dim=0,
    )

    expected = lstm(tokens)[0]
    result = masked_bi_rnn(padded_tokens, padded_tokens_mask, num_tokens)
    assert_almost_equal(expected, result[:-padding_len])
    assert result[-padding_len:].sum().item() == 0
    assert result[:-padding_len].sum().item() != 0


def _make_encoder(
    vocab_size=10,
    speaker_embedding_size=6,
    out_size=8,
    hidden_size=8,
    num_convolution_layers=2,
    convolution_filter_size=5,
    lstm_layers=2,
    dropout=0.5,
    padding_index=0,
    batch_size=4,
    num_tokens=5,
):
    """ Make `encoder.Encoder` and it's inputs for testing."""
    encoder = lib.spectrogram_model.encoder.Encoder(
        vocab_size=vocab_size,
        speaker_embedding_size=speaker_embedding_size,
        out_size=out_size,
        hidden_size=hidden_size,
        num_convolution_layers=num_convolution_layers,
        convolution_filter_size=convolution_filter_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
        padding_index=padding_index,
    )

    # NOTE: Ensure modules like `LayerNorm` perturbs the input instead of being just an identity.
    for name, parameter in encoder.named_parameters():
        if parameter.std() == 0:
            torch.nn.init.normal_(parameter)

    tokens = torch.randint(1, vocab_size, (batch_size, num_tokens))
    tokens_mask = torch.ones(batch_size, num_tokens, dtype=torch.bool)
    speaker = torch.randn(batch_size, speaker_embedding_size)
    return encoder, (tokens, tokens_mask, speaker), (num_tokens, batch_size, out_size)


def test_encoder():
    """ Test `encoder.Encoder` handles a basic case. """
    (
        module,
        (tokens, tokens_mask, speaker),
        (num_tokens, batch_size, out_size),
    ) = _make_encoder()
    output = module(tokens, tokens_mask, tokens_mask.sum(dim=1), speaker)
    assert output.dtype == torch.float
    assert output.shape == (num_tokens, batch_size, out_size)
    output.sum().backward()


def test_encoder_filter_size():
    """ Test `encoder.Encoder` handles different filter sizes. """
    for filter_size in [1, 3, 5]:
        (
            module,
            (tokens, tokens_mask, speaker),
            (num_tokens, batch_size, out_size),
        ) = _make_encoder(convolution_filter_size=filter_size)
        output = module(tokens, tokens_mask, tokens_mask.sum(dim=1), speaker)
        assert output.shape == (num_tokens, batch_size, out_size)
        output.sum().backward()


def test_encoder_padding_invariance():
    """ Test `encoder.Encoder` is consistent regardless of the padding. """
    (
        module,
        (tokens, tokens_mask, speaker),
        (num_tokens, batch_size, out_size),
    ) = _make_encoder(dropout=0)
    expected = module(tokens, tokens_mask, tokens_mask.sum(dim=1), speaker)
    expected.sum().backward()
    expected_grad = [p.grad for p in module.parameters() if p.grad is not None]
    module.zero_grad()
    for padding_len in range(1, 10):
        padding = torch.zeros(batch_size, padding_len)
        padded_tokens = torch.cat([tokens, padding.long()], dim=1)
        padded_tokens_mask = torch.cat([tokens_mask, padding.bool()], dim=1)
        result = module(
            padded_tokens, padded_tokens_mask, tokens_mask.sum(dim=1), speaker
        )
        result.sum().backward()
        result_grad = [p.grad for p in module.parameters() if p.grad is not None]
        module.zero_grad()
        assert_almost_equal(result[:-padding_len], expected, decimal=5)
        [assert_almost_equal(r, e) for r, e in zip(result_grad, expected_grad)]
