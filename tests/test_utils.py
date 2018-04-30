import os

from torch import nn
from torch.nn import functional
from torchnlp.text_encoders import PADDING_INDEX

import torch
import pytest

from src.utils import split_dataset
from src.utils import get_root_path
from src.utils import get_total_parameters
from src.utils import pad_batch
from src.utils import pad_tensor


class MockModel(nn.Module):
    # REFERENCE: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        super(MockModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(functional.relu(self.conv1(x)))
        x = self.pool(functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_get_total_parameters():
    model = MockModel()
    assert 62006 == get_total_parameters(model)


def test_get_root_path():
    root_path = get_root_path()
    assert os.path.isfile(os.path.join(root_path, 'requirements.txt'))


def test_split_dataset():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits) == [[1, 2, 3], [4], [5]]


def test_pad_tensor():
    padded = pad_tensor(torch.LongTensor([1, 2, 3]), 5, PADDING_INDEX)
    assert padded.tolist() == [1, 2, 3, PADDING_INDEX, PADDING_INDEX]


def test_pad_tensor_multiple_dim():
    padded = pad_tensor(torch.LongTensor(1, 2, 3), 5, PADDING_INDEX)
    assert padded.size() == (5, 2, 3)
    assert padded[1].sum().item() == pytest.approx(0)


def test_pad_tensor_multiple_dim_float_tensor():
    padded = pad_tensor(torch.FloatTensor(778, 80), 804, PADDING_INDEX)
    assert padded.size() == (804, 80)
    assert padded[-1].sum().item() == pytest.approx(0)
    assert padded.type() == 'torch.FloatTensor'


def test_pad_batch():
    batch = [torch.LongTensor([1, 2, 3]), torch.LongTensor([1, 2]), torch.LongTensor([1])]
    padded, lengths = pad_batch(batch, PADDING_INDEX)
    padded = [r.tolist() for r in padded]
    assert padded == [[1, 2, 3], [1, 2, PADDING_INDEX], [1, PADDING_INDEX, PADDING_INDEX]]
    assert lengths == [3, 2, 1]
