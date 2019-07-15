from pathlib import Path

import logging
import os
import sys

from torch import nn

import pytest
import torch

from src.utils.utils import assert_enough_disk_space
from src.utils.utils import balance_list
from src.utils.utils import dict_collapse
from src.utils.utils import duplicate_stream
from src.utils.utils import evaluate
from src.utils.utils import flatten
from src.utils.utils import flatten_parameters
from src.utils.utils import get_average_norm
from src.utils.utils import get_tensors_dim_length
from src.utils.utils import get_total_parameters
from src.utils.utils import get_weighted_stdev
from src.utils.utils import identity
from src.utils.utils import log_runtime
from src.utils.utils import parse_hparam_args
from src.utils.utils import ROOT_PATH
from src.utils.utils import set_basic_logging_config
from src.utils.utils import slice_by_cumulative_sum
from src.utils.utils import sort_together
from src.utils.utils import split_list


def test_flatten():
    assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]


def test_slice_by_cumulative_sum():
    assert [1, 2, 3] == slice_by_cumulative_sum([1, 2, 3, 4], max_total_value=6)
    assert [(1, 1), (1, 2)] == slice_by_cumulative_sum([(1, 1), (1, 2), (1, 3), (1, 4)],
                                                       max_total_value=4,
                                                       get_value=lambda x: x[1])


def test_assert_enough_disk_space__smoke_test():
    assert_enough_disk_space(0)


def test_sort_together():
    assert ['c', 'a', 'b'] == sort_together(['a', 'b', 'c'], [2, 3, 1])


def test_get_tensors_dim_length():
    assert [5, 5] == get_tensors_dim_length([torch.randn(5, 5) for _ in range(2)],
                                            dim=1,
                                            use_tqdm=True)


def test_log_runtime__smoke_test():
    func = lambda x: x + 1
    func = log_runtime(func)
    func(1)


def test_set_basic_logging_config__smoke_test():
    set_basic_logging_config()
    set_basic_logging_config(torch.device('cpu'))


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


def test_duplicate_stream(capsys):
    stdout_log = Path('tests/_test_data/stdout.log')
    with capsys.disabled():  # Disable capsys because it messes with sys.stdout
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)
        stop = duplicate_stream(sys.stdout, stdout_log)

        print('1')
        logger.info('2')
        os.system('echo 3')

        # Flush and close
        stop()
        logger.removeHandler(handler)

    assert stdout_log.is_file()
    output = stdout_log.read_text()
    assert set(output.split()) == set(['1', '2', '3'])


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


def test_get_root_path():
    assert (ROOT_PATH / 'requirements.txt').is_file()


def test_parse_hparam_args():
    hparam_args = ['--foo 0.01', '--bar WaveNet', '--moo.foo=1']
    assert parse_hparam_args(hparam_args) == {'foo': 0.01, 'bar': 'WaveNet', 'moo.foo': 1}


def test_balance_list():
    balanced = balance_list(['a', 'a', 'b', 'b', 'c'])
    assert len(balanced) == 3
    assert len(set(balanced)) == 3


def test_balance_list_determinism():
    list_ = [(i, 'a' if i % 2 == 0 else 'b') for i in range(99)]
    balanced = balance_list(list_, get_class=lambda i: i[1], random_seed=123)
    assert len(balanced) == 98
    assert balanced[0][0] == 57


def test_balance_list_get_weight():
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
