import os

from src.utils import batch
from src.utils import split_dataset
from src.utils import get_root_path
from src.utils import get_total_parameters

import torch.nn as nn
import torch.nn.functional as F


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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_get_root_path():
    root_path = get_root_path()
    assert os.path.isfile(os.path.join(root_path, 'requirements.txt'))


def test_split_dataset():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_dataset(dataset, splits) == [[1, 2, 3], [4], [5]]


def test_batch_generator():

    def generator():
        for i in range(11):
            yield i

    assert len(list(batch(generator(), n=2))) == 6


def test_batch():
    assert len(list(batch([i for i in range(11)], n=2))) == 6


def test_get_total_parameters():
    model = MockModel()
    assert 62006 == get_total_parameters(model)
