import math

from hparams import add_config
from hparams import HParams
from torchnlp.random import fork_rng

import torch
import numpy

from src.spectrogram_model import SpectrogramModel


def test_spectrogram_model_inference__batch_size_sensativity():
    """
    NOTE: This batch size sensativity does not test variable padding associated typically different
    batch sizes.
    TODO: Consider testing the training batch size sensativity.
    """
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 7

    # Ensure that the model computes the full length
    add_config({'src.spectrogram_model.model.SpectrogramModel._infer': HParams(stop_threshold=1.0)})

    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels).eval()

    # NOTE: Dropout result change in response to `BatchSize` because the dropout is computed
    # all together. For example:
    # >>> import torch
    # >>> torch.manual_seed(123)
    # >>> batch_dropout = torch.nn.functional.dropout(torch.ones(5, 5))
    # >>> torch.manual_seed(123)
    # >>> dropout = torch.nn.functional.dropout(torch.ones(5))
    # >>> batch_dropout[0] != dropout
    model.decoder.pre_net.layers[0][2].p = 0
    model.decoder.pre_net.layers[1][2].p = 0

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.randint(1, num_speakers, (1, batch_size))
    batched_num_tokens = torch.full((batch_size,), num_tokens, dtype=torch.long)

    # frames [num_frames, batch_size, frame_channels]
    # frames_with_residual [num_frames, batch_size, frame_channels]
    # stop_token [num_frames, batch_size]
    # alignment [num_frames, batch_size, num_tokens]
    with fork_rng(seed=123):
        (batched_frames, batched_frames_with_residual, batched_stop_token, batched_alignment,
         batched_lengths) = model(
             input_,
             speaker,
             num_tokens=batched_num_tokens,
             max_frames_per_token=num_frames / num_tokens)

    with fork_rng(seed=123):
        frames, frames_with_residual, stop_token, alignment, lengths = model(
            input_[:, :1], speaker[:, :1], max_frames_per_token=num_frames / num_tokens)

    numpy.testing.assert_almost_equal(
        frames.detach().numpy(), batched_frames[:, :1].detach().numpy(), decimal=6)
    numpy.testing.assert_almost_equal(
        frames_with_residual.detach().numpy(),
        batched_frames_with_residual[:, :1].detach().numpy(),
        decimal=6)
    numpy.testing.assert_almost_equal(
        stop_token.detach().numpy(), batched_stop_token[:, :1].detach().numpy(), decimal=6)
    numpy.testing.assert_almost_equal(
        alignment.detach().numpy(), batched_alignment[:, :1].detach().numpy(), decimal=6)
    numpy.testing.assert_almost_equal(lengths.numpy(), lengths[:1].numpy(), decimal=6)


def test_spectrogram_model_train__batch_size_sensativity():
    """
    NOTE: This batch size sensativity does not test variable padding associated typically
    different batch sizes.
    """
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 7

    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels).eval()

    # NOTE: Dropout result change in response to `BatchSize` because the dropout is computed
    # all together. For example:
    # >>> import torch
    # >>> torch.manual_seed(123)
    # >>> batch_dropout = torch.nn.functional.dropout(torch.ones(5, 5))
    # >>> torch.manual_seed(123)
    # >>> dropout = torch.nn.functional.dropout(torch.ones(5))
    # >>> batch_dropout[0] != dropout
    model.decoder.pre_net.layers[0][2].p = 0
    model.decoder.pre_net.layers[1][2].p = 0

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.randint(1, num_speakers, (1, batch_size))
    target_frames = torch.randn(num_frames, batch_size, frame_channels)
    batched_num_tokens = torch.randint(1, num_tokens, (batch_size,))
    batched_num_tokens[0] = num_tokens  # NOTE: One of the lengths must be at max length
    batched_num_frames = torch.randint(1, num_frames, (batch_size,))
    batched_num_frames[0] = num_frames  # NOTE: One of the lengths must be at max length

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

    with fork_rng(seed=123):
        frames, frames_with_residual, stop_token, alignment = model(
            input_[:, :1],
            speaker[:, :1],
            target_frames=target_frames[:, :1],
            num_tokens=batched_num_tokens[:1],
            target_lengths=batched_num_frames[:1])

    numpy.testing.assert_almost_equal(
        frames.detach().numpy(), batched_frames[:, :1].detach().numpy(), decimal=6)
    numpy.testing.assert_almost_equal(
        frames_with_residual.detach().numpy(),
        batched_frames_with_residual[:, :1].detach().numpy(),
        decimal=6)
    numpy.testing.assert_almost_equal(
        stop_token.detach().numpy(), batched_stop_token[:, :1].detach().numpy(), decimal=6)
    numpy.testing.assert_almost_equal(
        alignment.detach().numpy(), batched_alignment[:, :1].detach().numpy(), decimal=6)


def test_spectrogram_model():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token.bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    batched_num_tokens = torch.full((batch_size,), num_tokens, dtype=torch.long)

    frames, frames_with_residual, stop_token, alignment, lengths = model(
        input_,
        speaker,
        num_tokens=batched_num_tokens,
        max_frames_per_token=num_frames / num_tokens)

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
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels).eval()

    # Make sure that stop-token is not predicted; therefore, reaching ``max_frames_per_token``
    torch.nn.init.constant_(model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token.bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens).random_(1, vocab_size)
    speaker = torch.LongTensor(1, 1).fill_(0)

    frames, frames_with_residual, stop_token, alignment, _ = model(
        input_, speaker, max_frames_per_token=num_frames / num_tokens)

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
    speaker = torch.LongTensor(1, batch_size).fill_(0)
    target_frames = torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1)
    target_lengths = torch.full((batch_size,), num_frames)
    batched_num_tokens = torch.full((batch_size,), num_tokens).long()
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
    speaker = torch.LongTensor(1, 1).fill_(0)
    target_frames = torch.FloatTensor(num_frames, frame_channels).uniform_(0, 1)
    target_lengths = torch.tensor(num_frames)
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
