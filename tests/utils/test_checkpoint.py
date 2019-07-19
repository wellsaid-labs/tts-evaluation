import random

from torch import nn

import torch

from src.optimizers import Optimizer
from src.utils.checkpoint import Checkpoint
from src.environment import set_random_generator_state


def test_load_most_recent_checkpoint():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.pt')
    assert isinstance(checkpoint, Checkpoint)
    assert 'tests/_test_data/step_100.pt' in str(checkpoint.path)

    set_random_generator_state(checkpoint.random_generator_state)
    assert 666765114 == random.randint(1, 2**31)  # Checkpoint set random generator state


def test_load_most_recent_checkpoint_none():
    checkpoint = Checkpoint.most_recent('tests/_test_data/**/*.abc')
    assert checkpoint is None


def test_load_save_checkpoint():
    model = nn.LSTM(10, 10)
    optimizer = Optimizer(
        torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
    checkpoint = Checkpoint(
        directory='tests/_test_data/', model=model, step=1000, optimizer=optimizer)
    filename = checkpoint.save()
    assert filename.is_file()

    # Smoke test
    Checkpoint.from_path(filename)
