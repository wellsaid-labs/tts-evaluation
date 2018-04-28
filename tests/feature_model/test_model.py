import torch

from src.feature_model import FeatureModel


def test_feature_model():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    model = FeatureModel(
        vocab_size,
        encoder_hidden_size=encoder_hidden_size,
        frame_channels=frame_channels,
        max_recursion=10)
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.autograd.Variable(
        torch.LongTensor(batch_size, num_tokens).random_(1, vocab_size))
    frames, frames_with_residual, stop_token = model(input_)

    assert frames.data.type() == 'torch.FloatTensor'
    assert frames.shape == (1, batch_size, frame_channels)

    assert frames_with_residual.data.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (1, batch_size, frame_channels)

    assert stop_token.data.type() == 'torch.FloatTensor'
    assert stop_token.shape == (1, batch_size)


def test_feature_model_ground_truth():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 5
    model = FeatureModel(
        vocab_size,
        encoder_hidden_size=encoder_hidden_size,
        frame_channels=frame_channels,
        max_recursion=10)
    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.autograd.Variable(
        torch.LongTensor(batch_size, num_tokens).random_(1, vocab_size))
    ground_truth_frames = torch.autograd.Variable(
        torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1))
    frames, frames_with_residual, stop_token = model(
        input_, ground_truth_frames=ground_truth_frames)

    assert frames.data.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.data.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.data.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)
