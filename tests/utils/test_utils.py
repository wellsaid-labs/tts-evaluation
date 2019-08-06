from collections import Counter
from copy import deepcopy

import random
import time

from torch import nn

import pytest
import torch

from src.utils.utils import balance_list
from src.utils.utils import dict_collapse
from src.utils.utils import evaluate
from src.utils.utils import flatten
from src.utils.utils import flatten_parameters
from src.utils.utils import get_average_norm
from src.utils.utils import get_total_parameters
from src.utils.utils import get_weighted_stdev
from src.utils.utils import identity
from src.utils.utils import log_runtime
from src.utils.utils import ResetableTimer
from src.utils.utils import slice_by_cumulative_sum
from src.utils.utils import sort_together
from src.utils.utils import split_list
from src.utils.utils import get_chunks
from src.utils.utils import bash_time_label


def test_bash_time_label():
    """ Test to ensure that no bash special characters appear in the label, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    """
    label = bash_time_label()
    # NOTE (michael p): `:` wasn't mentioned explicitly; however, in my shell it required an escape.
    for character in ([
            '`', '~', '!', '#', '$', '&', '*', '(', ')', ' ', '\t', '\n', '{', '}', '[', ']', '|',
            ';', '\'', '"', '<', '>', '?'
    ] + [':']):
        assert character not in label


def test_get_chunks():
    assert list(get_chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_resetable_timer():
    called = 0

    def _helper():
        nonlocal called
        called += 1

    timer = ResetableTimer(0.5, _helper)
    timer.start()
    time.sleep(0.25)
    assert called == 0

    timer.reset()
    time.sleep(0.25)
    assert called == 0

    time.sleep(0.3)
    assert called == 1


def test_flatten():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_slice_by_cumulative_sum():
    assert [1, 2, 3] == slice_by_cumulative_sum([1, 2, 3, 4], max_total_value=6)
    assert [(1, 1), (1, 2)] == slice_by_cumulative_sum([(1, 1), (1, 2), (1, 3), (1, 4)],
                                                       max_total_value=4,
                                                       get_value=lambda x: x[1])


def test_sort_together():
    assert ['c', 'a', 'b'] == sort_together(['a', 'b', 'c'], [2, 3, 1])


def test_log_runtime__smoke_test():
    func = lambda x: x + 1
    func = log_runtime(func)
    func(1)


def test_dict_collapse():
    assert {
        'a': 1,
        'b.c': 2,
        'b.d.e': 3,
        'g': []
    } == dict_collapse({
        'a': 1,
        'b': {
            'c': 2,
            'd': {
                'e': 3
            }
        },
        'f': {},
        'g': []
    })


def test_evaluate():
    model = nn.LSTM(10, 10).train(mode=True)
    model_two = nn.LSTM(10, 10).train(mode=True)
    with evaluate(model, model_two, device=torch.device('cpu')):
        # NOTE: Unable to test ``no_grad`` without a PyTorch helper function
        # NOTE: Unable to test device change without a secondary device available.
        assert not model.training
        assert not model_two.training
    assert model.training
    assert model_two.training


def test_identity():
    assert identity(2) == 2


def test_get_average_norm__shape_invariant():
    tensor = torch.Tensor([0.5, 0.2, 0.3])
    expanded = tensor.unsqueeze(1).expand(3, 4)
    assert get_average_norm(tensor) == get_average_norm(expanded)


def test_get_average_norm__mask_invariant():
    tensor = torch.randn(3, 4, 5)
    mask = torch.ones(3, 5).byte()
    assert get_average_norm(tensor, dim=1) == get_average_norm(tensor, mask=mask, dim=1)


def test_get_weighted_stdev__constant():
    tensor = torch.Tensor([0, 1, 0])
    standard_deviation = get_weighted_stdev(tensor, dim=0)
    assert standard_deviation == 0.0  # 0.0 is the standard deviation for a constant


def test_get_weighted_stdev__bias():
    tensor = torch.Tensor([.25, .25, .25, .25])
    standard_deviation = get_weighted_stdev(tensor, dim=0)
    # Population standard deviation for 1, 2, 3, 4
    assert standard_deviation == pytest.approx(1.1180339887499)


def test_get_weighted_stdev():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    standard_deviation = get_weighted_stdev(tensor, dim=2)
    assert standard_deviation == pytest.approx(0.5791246294975281)


def test_get_weighted_stdev__mask():
    tensor = torch.Tensor([[[0.33333, 0.33333, 0.33334], [0, 0.5, 0.5]],
                           [[0, 0.5, 0.5], [0, 0.5, 0.5]]])
    mask = torch.ByteTensor([[1, 0], [0, 0]])
    standard_deviation = get_weighted_stdev(tensor, dim=2, mask=mask)
    # Population standard deviation for 1,2,3
    assert standard_deviation == pytest.approx(0.81649658093, rel=1.0e-04)


class MockModel(nn.Module):
    # REFERENCE: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


def test_get_total_parameters():
    assert 62006 == get_total_parameters(MockModel())


def test_flatten_parameters__smoke_test():
    flatten_parameters(MockModel())
    flatten_parameters(nn.LSTM(10, 10))


def test_balance_list():
    balanced = balance_list(['a', 'a', 'b', 'b', 'c'])
    assert len(balanced) == 3
    assert len(set(balanced)) == 3


def test_balance_list__determinism():
    """ Test to ensure that `balance_list` is deterministic when `random_seed` is provided. """
    random_ = random.Random(123)
    list_ = [(i, random_.choice('abcde')) for i in range(99)]
    balanced = balance_list(list_, get_class=lambda i: i[1], random_seed=123)
    assert len(balanced) == 70
    count = Counter([e[1] for e in balanced])
    assert len(set(count.values())) == 1  # Ensure that the list is balanced.
    assert [e[0] for e in balanced[:10]] == [7, 33, 62, 51, 14, 50, 19, 73, 56, 21]


def test_balance_list__non_determinism():
    """ Test to ensure that `balance_list` is not deterministic when `random_seed` is not provided.
    """
    random_ = random.Random(123)
    list_ = [(i, random_.choice('abcde')) for i in range(10000)]
    balanced = balance_list(list_, get_class=lambda i: i[1])

    list_other = deepcopy(list_)
    balanced_other = balance_list(list_other, get_class=lambda i: i[1])
    # NOTE: This test should fail one in 10000 times.
    assert balanced_other[0][0] != balanced[0][0]


def test_balance_list__get_weight():
    list_ = [(1, 'a'), (1, 'a'), (2, 'b')]
    balanced = balance_list(list_, get_class=lambda i: i[1], get_weight=lambda i: i[0])
    assert len(balanced) == 3


def test_split_list():
    dataset = [1, 2, 3, 4, 5]
    splits = (.6, .2, .2)
    assert split_list(dataset, splits) == [[1, 2, 3], [4], [5]]


def test_split_list_rounding():
    dataset = [1]
    splits = (.33, .33, .34)
    assert split_list(dataset, splits) == [[], [], [1]]
