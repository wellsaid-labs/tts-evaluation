import math

import torch
import numpy

from src.spectrogram_model import SpectrogramModel


def test_spectrogram_model_batch_size_sensativity():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 7
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels).eval()
    model.decoder.pre_net.layers[0][2].p = 0  # Remove random dropout from AlwaysDropout
    model.decoder.pre_net.layers[1][2].p = 0  # Remove random dropout from AlwaysDropout

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)

    # frames [num_frames, batch_size, frame_channels]
    # frames_with_residual [num_frames, batch_size, frame_channels]
    # stop_token [num_frames, batch_size]
    # alignment [num_frames, batch_size, num_tokens]
    (batched_frames, batched_frames_with_residual, batched_stop_token, batched_alignment,
     batched_lengths) = model(
         input_, speaker, max_recursion=num_frames)

    frames, frames_with_residual, stop_token, alignment, lengths = model(
        input_[:, :1], speaker[:, :1], max_recursion=num_frames)

    numpy.testing.assert_almost_equal(frames.detach().numpy(),
                                      batched_frames[:, :1].detach().numpy())
    numpy.testing.assert_almost_equal(frames_with_residual.detach().numpy(),
                                      batched_frames_with_residual[:, :1].detach().numpy())
    numpy.testing.assert_almost_equal(stop_token.detach().numpy(),
                                      batched_stop_token[:, :1].detach().numpy())
    numpy.testing.assert_almost_equal(alignment.detach().numpy(),
                                      batched_alignment[:, :1].detach().numpy())
    numpy.testing.assert_almost_equal(lengths, lengths[:1])


def test_spectrogram_model():
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 1
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token.bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    speaker = torch.LongTensor(1, batch_size).fill_(0)

    frames, frames_with_residual, stop_token, alignment, lengths = model(
        input_, speaker, max_recursion=num_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    assert len(lengths) == batch_size
    for length in lengths:
        assert length > 0
        assert length <= num_frames


def test_spectrogram_model_unbatched():
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    num_speakers = 3
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels).eval()

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(model.decoder.linear_stop_token.weight, -math.inf)
    torch.nn.init.constant_(model.decoder.linear_stop_token.bias, -math.inf)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens).random_(1, vocab_size)
    speaker = torch.LongTensor(1, 1).fill_(0)

    frames, frames_with_residual, stop_token, alignment, _ = model(
        input_, speaker, max_recursion=num_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames,)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, num_tokens)


def test_spectrogram_model_ground_truth():
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
    ground_truth_frames = torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1)
    frames, frames_with_residual, stop_token, alignment = model(
        input_, speaker, ground_truth_frames=ground_truth_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    frames_with_residual.sum().backward()


def test_spectrogram_model_ground_truth_unbatched():
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_speakers = 1
    num_frames = 5
    model = SpectrogramModel(vocab_size, num_speakers, frame_channels=frame_channels)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens).random_(1, vocab_size)
    speaker = torch.LongTensor(1, 1).fill_(0)
    ground_truth_frames = torch.FloatTensor(num_frames, frame_channels).uniform_(0, 1)
    frames, frames_with_residual, stop_token, alignment = model(
        input_, speaker, ground_truth_frames=ground_truth_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames,)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, num_tokens)

    frames_with_residual.sum().backward()
