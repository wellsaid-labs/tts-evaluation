import numpy
import torch

from torchnlp.random import fork_rng

from src.spectrogram_model.encoder import Encoder
from src.spectrogram_model.encoder import RightMaskedBiLSTM
from src.spectrogram_model.encoder import roll
from tests._utils import assert_almost_equal


def test_roll():
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert torch.equal(
        roll(tensor, shift=torch.tensor([0, 1, 2]), dim=-1),
        torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test_roll__larger_shift():
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    assert torch.equal(roll(tensor, shift=torch.tensor(4), dim=0), torch.tensor([3, 4, 5, 6, 1, 2]))


def test_roll__transpose():
    tensor = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]]).transpose(0, 1)
    result = roll(tensor, shift=torch.tensor([0, 1, 2]), dim=0).transpose(0, 1)
    assert torch.equal(result, torch.tensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test_roll__3d():
    tensor = torch.tensor([1, 2, 3]).view(1, 3, 1).expand(4, 3, 4)
    shift = torch.arange(0, 16).view(4, 4)
    result = roll(tensor, shift=shift, dim=1)

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


def test_right_masked_bi_lstm__identity():
    """ Test if `RightMaskedBiLSTM` is equal to an bidirectional LSTM without masking. """
    batch_size = 2
    seq_len = 3
    input_size = 4
    lstm_hidden_size = 5
    num_layers = 2

    tokens = torch.randn(seq_len, batch_size, input_size)
    tokens_mask = torch.ones(seq_len, batch_size, dtype=torch.bool)

    with fork_rng(123):
        masked_bi_lstm = RightMaskedBiLSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)

    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            bidirectional=True)

    result = masked_bi_lstm(tokens, tokens_mask)
    expected, _ = lstm(tokens)
    assert torch.equal(expected, result)


def test_right_masked_bi_lstm__uneven_mask():
    """ Test if `RightMaskedBiLSTM` is able to handle an uneven LSTM mask on the end. """
    input_size = 1
    lstm_hidden_size = 5

    # tokens [batch_size, seq_len] → [seq_len, batch_size, 1]
    tokens = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]],
                          dtype=torch.float32).transpose(0, 1).unsqueeze(2)

    # backward_tokens [batch_size, seq_len] → [seq_len, batch_size, 1]
    backward_tokens = torch.tensor([[3, 2, 1, 0, 0], [2, 1, 0, 0, 0]],
                                   dtype=torch.float32).transpose(0, 1).unsqueeze(2)

    # tokens_mask [batch_size, seq_len] → [seq_len, batch_size]
    tokens_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool).transpose(0, 1)

    with fork_rng(123):
        masked_bi_lstm = RightMaskedBiLSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)

    forward_lstm, backward_lstm = masked_bi_lstm.lstm_layers[0]
    tokens_mask_expanded = tokens_mask.unsqueeze(2)
    expected_forward_pass = forward_lstm(tokens)[0].masked_fill(~tokens_mask_expanded, 0)
    expected_backward_pass = backward_lstm(backward_tokens)[0].masked_fill(~tokens_mask_expanded, 0)

    result = masked_bi_lstm(tokens, tokens_mask)

    assert_almost_equal((expected_forward_pass.sum() + expected_backward_pass.sum()), result.sum())


def test_right_masked_bi_lstm__multiple_masked_layers():
    """ Test if `RightMaskedBiLSTM` is able to handle a mask over multiple layers. """
    batch_size = 2
    seq_len = 3
    input_size = 4
    lstm_hidden_size = 5
    padding_len = 2
    num_layers = 3

    tokens = torch.randn(seq_len, batch_size, input_size)
    padded_tokens = torch.cat([tokens, torch.zeros(padding_len, batch_size, input_size)], dim=0)

    tokens_mask = torch.ones(seq_len, batch_size, dtype=torch.bool)
    padded_tokens_mask = torch.cat(
        [tokens_mask, torch.zeros(padding_len, batch_size, dtype=torch.bool)], dim=0)

    with fork_rng(123):
        masked_bi_lstm = RightMaskedBiLSTM(
            input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers)

    with fork_rng(123):
        lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            bidirectional=True)

    expected, _ = lstm(tokens)
    result = masked_bi_lstm(padded_tokens, padded_tokens_mask)

    assert_almost_equal(expected.sum(), result.sum(), decimal=4)
    assert torch.equal(expected, result[:-padding_len])
    assert result[-padding_len:].sum().item() == 0
    assert result.sum().item() != 0


def test_right_masked_bi_lstm__backwards():
    """ Test if `MaskedBackwardLSTM` can compute a gradient. """
    batch_size = 2
    seq_len = 3
    input_size = 4
    lstm_hidden_size = 5
    tokens = torch.randn(seq_len, batch_size, input_size)
    tokens_mask = torch.ones(seq_len, batch_size, 1).bool()

    masked_bi_lstm = RightMaskedBiLSTM(
        input_size=input_size, hidden_size=lstm_hidden_size, num_layers=1)

    masked_bi_lstm(tokens, tokens_mask).sum().backward()


encoder_params = {
    'batch_size': 4,
    'num_tokens': 5,
    'vocab_size': 10,
    'hidden_size': 16,
    'out_dim': 8,
}


def test_encoder():
    encoder = Encoder(
        encoder_params['vocab_size'],
        out_dim=encoder_params['out_dim'],
        hidden_size=encoder_params['hidden_size'])

    # NOTE: 1-index to avoid using 0 typically associated with padding
    tokens = torch.randint(1, encoder_params['vocab_size'],
                           (encoder_params['batch_size'], encoder_params['num_tokens']))
    tokens_mask = torch.ones(
        encoder_params['batch_size'], encoder_params['num_tokens'], dtype=torch.bool)

    output = encoder(tokens, tokens_mask)

    assert output.type() == 'torch.FloatTensor'
    assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                            encoder_params['out_dim'])

    # Smoke test backward
    output.sum().backward()


def test_encoder_filter_size():
    for filter_size in [1, 3, 5]:
        encoder = Encoder(
            encoder_params['vocab_size'],
            out_dim=encoder_params['out_dim'],
            hidden_size=encoder_params['hidden_size'],
            convolution_filter_size=filter_size)

        # NOTE: 1-index to avoid using 0 typically associated with padding
        tokens = torch.randint(1, encoder_params['vocab_size'],
                               (encoder_params['batch_size'], encoder_params['num_tokens']))
        tokens_mask = torch.ones(
            encoder_params['batch_size'], encoder_params['num_tokens'], dtype=torch.bool)

        output = encoder(tokens, tokens_mask)

        assert output.type() == 'torch.FloatTensor'
        assert output.shape == (encoder_params['num_tokens'], encoder_params['batch_size'],
                                encoder_params['out_dim'])


def test_encoder_padding_invariance():
    """ Ensure that the encoder results are not affected by padding. """

    encoder = Encoder(
        encoder_params['vocab_size'],
        out_dim=encoder_params['out_dim'],
        hidden_size=encoder_params['hidden_size'],
        num_convolution_layers=2,
        lstm_layers=2,
        dropout=0)

    # Ensure `LayerNorm` perturbs the input instead of being just an identity.
    for module in encoder.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.uniform_(module.weight)
            torch.nn.init.uniform_(module.bias)

    tokens = torch.randint(1, encoder_params['vocab_size'],
                           (encoder_params['batch_size'], encoder_params['num_tokens']))
    tokens_mask = torch.ones(
        encoder_params['batch_size'], encoder_params['num_tokens'], dtype=torch.bool)

    expected = None
    expected_grad = None
    for padding_len in range(10):
        padding = torch.zeros(encoder_params['batch_size'], padding_len)
        padded_tokens = torch.cat([tokens, padding.long()], dim=1)
        padded_tokens_mask = torch.cat([tokens_mask, padding.bool()], dim=1)

        result = encoder(padded_tokens, padded_tokens_mask).sum()
        result.backward()
        result_grad = sum([p.grad.sum() for p in encoder.parameters() if p.grad is not None])
        encoder.zero_grad()

        if expected is None and expected_grad is None:
            expected = result
            expected_grad = result_grad.detach().numpy()
        else:
            assert_almost_equal(result, expected, decimal=5)
            numpy.isclose(result_grad.detach().numpy(), expected_grad)
