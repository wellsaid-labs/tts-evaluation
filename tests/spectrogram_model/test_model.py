import math

from hparams import add_config
from hparams import HParams
from torchnlp.random import fork_rng

import torch
import numpy

from src.spectrogram_model import SpectrogramModel


def test_spectrogram_model_inference__batch_size_sensitivity():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 7
    padding_len = 2

    # Ensure that the model computes the full length
    add_config({'src.spectrogram_model.model.SpectrogramModel._infer': HParams(stop_threshold=1.0)})

    # NOTE: The random generator for dropout varies based on the tensor size; therefore, it's
    # dependent on the `BatchSize` and we need to disable it. For example:
    # >>> import torch
    # >>> torch.manual_seed(123)
    # >>> batch_dropout = torch.nn.functional.dropout(torch.ones(5, 5))
    # >>> torch.manual_seed(123)
    # >>> dropout = torch.nn.functional.dropout(torch.ones(5))
    # >>> batch_dropout[0] != dropout
    add_config({'src.spectrogram_model.pre_net.PreNet.__init__': HParams(dropout=0.0)})

    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens).eval()

    # Ensure `LayerNorm` perturbs the input instead of being just an identity.
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.uniform_(module.weight)
            torch.nn.init.uniform_(module.bias)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    input_[-padding_len:, 0] = 0

    speaker = torch.randint(1, num_speakers, (1, batch_size))

    batched_num_tokens = torch.randint(1, num_tokens, (batch_size,))
    batched_num_tokens[1] = num_tokens  # NOTE: One of the lengths must be at max length
    batched_num_tokens[0] = num_tokens - padding_len

    # frames [num_frames, batch_size, frame_channels]
    # frames_with_residual [num_frames, batch_size, frame_channels]
    # stop_token [num_frames, batch_size]
    # alignment [num_frames, batch_size, num_tokens]
    with fork_rng(seed=123):
        (batched_frames, batched_frames_with_residual, batched_stop_token, batched_alignment,
         batched_lengths, batched_reached_max) = model(
             input_, speaker, num_tokens=batched_num_tokens)
        assert batched_reached_max.sum() == batch_size

    with fork_rng(seed=123):
        frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
            input_[:-padding_len, :1], speaker[:, :1])
        assert reached_max

    assert_almost_equal = lambda a, b: numpy.testing.assert_almost_equal(
        a.detach().numpy(), b.detach().numpy(), decimal=5)

    assert_almost_equal(frames, batched_frames[:lengths[0], :1])
    assert_almost_equal(frames_with_residual, batched_frames_with_residual[:lengths[0], :1])
    assert_almost_equal(stop_token, batched_stop_token[:lengths[0], :1])
    assert_almost_equal(alignment, batched_alignment[:lengths[0], :1, :int(batched_num_tokens[0])])
    assert_almost_equal(lengths, lengths[:1])


def test_spectrogram_model_train__batch_size_sensitivity():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 7
    padding_len = 2

    # NOTE: The random generator for dropout varies based on the tensor size; therefore, it's
    # dependent on the `BatchSize` and we need to disable it. For example:
    # >>> import torch
    # >>> torch.manual_seed(123)
    # >>> batch_dropout = torch.nn.functional.dropout(torch.ones(5, 5))
    # >>> torch.manual_seed(123)
    # >>> dropout = torch.nn.functional.dropout(torch.ones(5))
    # >>> batch_dropout[0] != dropout
    add_config({
        'src.spectrogram_model': {
            'attention.LocationSensitiveAttention.__init__': HParams(dropout=0.0),
            'decoder.AutoregressiveDecoder.__init__': HParams(stop_net_dropout=0.0),
            'encoder.Encoder.__init__': HParams(dropout=0.0),
            'model.SpectrogramModel.__init__': HParams(speaker_embed_dropout=0.0),
            'pre_net.PreNet.__init__': HParams(dropout=0.0),
        },
    })

    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # Ensure `LayerNorm` perturbs the input instead of being just an identity.
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.uniform_(module.weight)
            torch.nn.init.uniform_(module.bias)

    # NOTE: 1-index in `random_` to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    input_[-padding_len:, 0] = 0  # Add padding on the end

    speaker = torch.randint(1, num_speakers, (1, batch_size))

    target_frames = torch.randn(num_frames, batch_size, frame_channels)
    target_frames[-padding_len:] = 0  # Add padding on the end

    batched_num_tokens = torch.randint(1, num_tokens, (batch_size,))
    batched_num_tokens[1] = num_tokens  # NOTE: One of the lengths must be at max length
    batched_num_tokens[0] = num_tokens - padding_len

    batched_num_frames = torch.randint(1, num_frames, (batch_size,))
    batched_num_frames[1] = num_frames  # NOTE: One of the lengths must be at max length
    batched_num_frames[0] = num_frames - padding_len

    # frames [num_frames, batch_size, frame_channels]
    # frames_with_residual [num_frames, batch_size, frame_channels]
    # stop_token [num_frames, batch_size]
    # alignment [num_frames, batch_size, num_tokens]
    with fork_rng(seed=123):
        (batched_frames, batched_frames_with_residual, batched_stop_token,
         batched_alignment) = model(
             input_,
             speaker,
             num_tokens=batched_num_tokens,
             target_lengths=batched_num_frames,
             target_frames=target_frames)
        (batched_frames_with_residual[:, :1].sum() + batched_stop_token[:, :1].sum()).backward()
        batched_gradient_sum = sum([p.grad.sum() for p in model.parameters() if p.grad is not None])
        model.zero_grad()

    with fork_rng(seed=123):
        frames, frames_with_residual, stop_token, alignment = model(
            input_[:-padding_len, :1],
            speaker[:, :1],
            target_frames=target_frames[:-padding_len, :1],
            num_tokens=batched_num_tokens[:1],
            target_lengths=batched_num_frames[:1])
        (frames_with_residual.sum() + stop_token.sum()).backward()
        gradient_sum = sum([p.grad.sum() for p in model.parameters() if p.grad is not None])
        model.zero_grad()

    assert_almost_equal = lambda a, b: numpy.testing.assert_almost_equal(
        a.detach().numpy(), b.detach().numpy(), decimal=5)

    assert_almost_equal(frames, batched_frames[:batched_num_frames[0], :1])
    assert_almost_equal(frames_with_residual,
                        batched_frames_with_residual[:batched_num_frames[0], :1])
    assert_almost_equal(stop_token, batched_stop_token[:batched_num_frames[0], :1])
    assert_almost_equal(alignment,
                        batched_alignment[:batched_num_frames[0], :1, :int(batched_num_tokens[0])])
    numpy.isclose(gradient_sum, batched_gradient_sum)


class MockSigmoid(torch.nn.Module):

    def forward(self, tensor):
        return torch.rand(*tensor.shape, device=tensor.device)


def test_spectrogram_model__filter_reached_max():
    batch_size = 64
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1

    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens)
    model.stop_sigmoid = MockSigmoid()

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    batched_num_tokens = torch.full((batch_size,), num_tokens,
                                    dtype=torch.long).random_(1, num_tokens)
    batched_num_tokens[0] = num_tokens

    frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
        input_, speaker, num_tokens=batched_num_tokens, filter_reached_max=True)

    assert reached_max.type() == 'torch.BoolTensor'
    assert reached_max.sum().item() >= 0

    num_reached_max = reached_max.sum().item()
    max_length = lengths.max().item()

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (max_length, batch_size - num_reached_max, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (max_length, batch_size - num_reached_max, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (max_length, batch_size - num_reached_max)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (max_length, batch_size - num_reached_max, num_tokens)

    assert lengths.shape == (1, batch_size - num_reached_max)

    for length in lengths[0].tolist():
        assert length > 0
        assert length <= max_length


def test_spectrogram_model__filter_all():
    batch_size = 64
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1

    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    batched_num_tokens = torch.full((batch_size,), num_tokens,
                                    dtype=torch.long).random_(1, num_tokens)
    batched_num_tokens[0] = num_tokens

    frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
        input_, speaker, num_tokens=batched_num_tokens, filter_reached_max=True)

    assert reached_max.type() == 'torch.BoolTensor'
    assert reached_max.sum().item() == batch_size

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, 0, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, 0, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, 0)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, 0, num_tokens)

    assert lengths.shape == (1, 0)


def test_spectrogram_model__random_sigmoid():
    batch_size = 64
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    batched_num_tokens = torch.full((batch_size,), num_tokens,
                                    dtype=torch.long).random_(1, num_tokens)
    batched_num_tokens[0] = num_tokens

    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens)

    model.stop_sigmoid = MockSigmoid()

    frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
        input_, speaker, num_tokens=batched_num_tokens)

    max_length = lengths.max().item()

    assert reached_max.type() == 'torch.BoolTensor'
    assert reached_max.sum().item() >= 0

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (max_length, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (max_length, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (max_length, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (max_length, batch_size, num_tokens)

    assert lengths.shape == (1, batch_size)
    for length in lengths[0].tolist():
        assert length > 0
        assert length <= max_length


def test_spectrogram_model():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    batched_num_tokens = torch.full((batch_size,), num_tokens,
                                    dtype=torch.long).random_(1, num_tokens)
    batched_num_tokens[0] = num_tokens

    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].bias, -math.inf)

    frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
        input_, speaker, num_tokens=batched_num_tokens)

    assert reached_max.type() == 'torch.BoolTensor'
    assert reached_max.sum().item() == batch_size

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    assert lengths.shape == (1, batch_size)
    for length in lengths[0].tolist():
        assert length > 0
        assert length <= num_frames


def test_spectrogram_model_unbatched():
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 3
    model = SpectrogramModel(
        vocab_size,
        num_speakers,
        frame_channels=frame_channels,
        max_frames_per_token=num_frames / num_tokens).eval()

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token[-1].bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens).random_(1, vocab_size)
    speaker = torch.LongTensor(1, 1).fill_(0)

    frames, frames_with_residual, stop_token, alignment, lengths, reached_max = model(
        input_, speaker)

    assert reached_max.type() == 'torch.BoolTensor'
    assert reached_max

    assert torch.equal(lengths, torch.tensor([num_frames]))

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames,)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, num_tokens)


def test_spectrogram_model_target():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_speakers = 2
    num_frames = 5
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.zeros(1, batch_size, dtype=torch.long)
    target_frames = torch.rand(num_frames, batch_size, frame_channels)
    target_lengths = torch.full((batch_size,), num_frames, dtype=torch.long)
    batched_num_tokens = torch.full((batch_size,), num_tokens, dtype=torch.long)
    frames, frames_with_residual, stop_token, alignment = model(
        input_,
        speaker,
        num_tokens=batched_num_tokens,
        target_frames=target_frames,
        target_lengths=target_lengths)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    frames_with_residual.sum().backward()


def test_spectrogram_model_target_unbatched():
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_speakers = 1
    num_frames = 5
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens).random_(1, vocab_size)
    speaker = torch.zeros(1, 1, dtype=torch.long)
    target_frames = torch.rand(num_frames, frame_channels)
    target_lengths = torch.tensor(num_frames, dtype=torch.long)
    frames, frames_with_residual, stop_token, alignment = model(
        input_, speaker, target_frames=target_frames, target_lengths=target_lengths)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames,)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, num_tokens)

    frames_with_residual.sum().backward()
