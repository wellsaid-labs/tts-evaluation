import math
import pathlib
import time

from torch import nn
from torchnlp.random import fork_rng

import pytest
import torch

from src.utils.utils import assert_no_overwritten_files
from src.utils.utils import bash_time_label
from src.utils.utils import dict_collapse
from src.utils.utils import evaluate
from src.utils.utils import flatten
from src.utils.utils import flatten_parameters
from src.utils.utils import get_average_norm
from src.utils.utils import get_chunks
from src.utils.utils import get_weighted_stdev
from src.utils.utils import identity
from src.utils.utils import log_runtime
from src.utils.utils import mean
from src.utils.utils import random_sample
from src.utils.utils import RepeatTimer
from src.utils.utils import ResetableTimer
from src.utils.utils import slice_by_cumulative_sum
from src.utils.utils import sort_together


def test_assert_no_overwritten_files():

    @assert_no_overwritten_files
    def _helper(path):
        pass

    path_ = pathlib.Path('tests/_test_data/test_assert_no_overwritten_files.txt')
    path_.write_text('blah')
    _helper(path_)
    path_.write_text('blah')

    with pytest.raises(AssertionError):
        _helper(path_)


def test_mean():
    assert mean(range(3)) == 1
    assert mean([]) is math.nan


def test_random_sample():
    assert random_sample([], 5) == []
    with fork_rng(seed=1234):
        assert random_sample([1, 2, 3], 5) == [2, 1, 3]


def test_repeat_timer():
    """ Ensure the repeat timer continues to execute `func`. """
    calls = 0

    def func():
        nonlocal calls
        calls += 1

    timer = RepeatTimer(0.1, func)
    timer.start()
    time.sleep(0.15)

    assert calls == 1

    time.sleep(0.5)

    assert calls == 6

    timer.cancel()

    time.sleep(0.15)

    assert calls == 6  # No more calls after `cancel`


def test_bash_time_label():
    """ Test to ensure that no bash special characters appear in the label, learn more:
    https://unix.stackexchange.com/questions/270977/what-characters-are-required-to-be-escaped-in-command-line-arguments
    """
    label = bash_time_label()
    # NOTE (michael p): `:` and `=` wasn't mentioned explicitly; however, in my shell it required
    # an escape.
    for character in ([
            '`', '~', '!', '#', '$', '&', '*', '(', ')', ' ', '\t', '\n', '{', '}', '[', ']', '|',
            ';', '\'', '"', '<', '>', '?', '='
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


def test_resetable_timer__race_condition():
    """ Test to ensure that subsequent calls for `reset` do not trigger a race condition. """
    called = 0

    def _helper():
        nonlocal called
        called += 1

    timer = ResetableTimer(0.1, _helper)
    timer.start()
    for i in range(10000):
        timer.reset()

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
    mask = torch.ones(3, 5).bool()
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
    mask = torch.BoolTensor([[1, 0], [0, 0]])
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


def test_flatten_parameters__smoke_test():
    flatten_parameters(MockModel())
    flatten_parameters(nn.LSTM(10, 10))
