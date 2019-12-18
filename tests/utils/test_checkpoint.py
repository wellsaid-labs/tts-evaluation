import random

from torch import nn
from torchnlp.random import set_random_generator_state

import torch
import pytest

from src.optimizers import Optimizer
from src.utils.checkpoint import Checkpoint
from src.environment import TEST_DATA_PATH

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'utils'


def test_load_most_recent_checkpoint():
    checkpoint = Checkpoint.most_recent(str(TEST_DATA_PATH_LOCAL / '**/*.pt'))
    assert isinstance(checkpoint, Checkpoint)
    assert str(TEST_DATA_PATH_LOCAL / 'step_100.pt') in str(checkpoint.path)

    set_random_generator_state(checkpoint.random_generator_state)
    assert 819796839 == random.randint(1, 2**31)  # Checkpoint set random generator state


def test_load_most_recent_checkpoint_none():
    checkpoint = Checkpoint.most_recent(str(TEST_DATA_PATH_LOCAL / '**/*.abc'))
    assert checkpoint is None


def test_load_save_checkpoint():
    model = nn.LSTM(10, 10)
    optimizer = Optimizer(
        torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters())))
    checkpoint = Checkpoint(
        directory=TEST_DATA_PATH_LOCAL, model=model, step=1000, optimizer=optimizer)
    filename = checkpoint.save()

    with pytest.raises(ValueError):  # Detects overwrite
        checkpoint.save()

    assert filename.is_file()

    # Smoke test
    Checkpoint.from_path(filename)
