import torch

from src.feature_model import FeatureModel


def test_feature_model():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 3
    model = FeatureModel(
        vocab_size, encoder_hidden_size=encoder_hidden_size, frame_channels=frame_channels)

    # Make sure that stop-token is not predicted; therefore, reaching ``max_recursion``
    torch.nn.init.constant_(model.decoder.linear_stop_token[0].weight, -1)
    torch.nn.init.constant_(model.decoder.linear_stop_token[0].bias, -1)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)

    frames, frames_with_residual, stop_token, alignment = model(input_, max_recursion=num_frames)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)

    # Smoke test backward
    frames_with_residual.sum().backward()


def test_feature_model_ground_truth():
    encoder_hidden_size = 32
    batch_size = 5
    num_tokens = 6
    frame_channels = 20
    vocab_size = 20
    num_frames = 5
    model = FeatureModel(
        vocab_size, encoder_hidden_size=encoder_hidden_size, frame_channels=frame_channels)

    # NOTE: 1-index to avoid using 0 typically associated with padding
    input_ = torch.LongTensor(num_tokens, batch_size).random_(1, vocab_size)
    ground_truth_frames = torch.FloatTensor(num_frames, batch_size, frame_channels).uniform_(0, 1)
    frames, frames_with_residual, stop_token, alignment = model(
        input_, ground_truth_frames=ground_truth_frames, max_recursion=10)

    assert frames.type() == 'torch.FloatTensor'
    assert frames.shape == (num_frames, batch_size, frame_channels)

    assert frames_with_residual.type() == 'torch.FloatTensor'
    assert frames_with_residual.shape == (num_frames, batch_size, frame_channels)

    assert stop_token.type() == 'torch.FloatTensor'
    assert stop_token.shape == (num_frames, batch_size)

    assert alignment.type() == 'torch.FloatTensor'
    assert alignment.shape == (num_frames, batch_size, num_tokens)
