from unittest import mock

import random

import pytest
import torch
import numpy

from src.environment import assert_enough_disk_space
from src.environment import check_module_versions
from src.environment import fork_rng
from src.environment import get_random_generator_state
from src.environment import ROOT_PATH
from src.environment import set_basic_logging_config
from src.environment import set_random_generator_state
from src.environment import set_seed


def test_fork_rng():
    set_seed(123)
    pre_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    with fork_rng(seed=123):
        random.randint(1, 2**31)

    post_randint = [random.randint(1, 2**31), random.randint(1, 2**31)]

    set_seed(123)
    assert pre_randint != post_randint
    assert pre_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]
    assert post_randint == [random.randint(1, 2**31), random.randint(1, 2**31)]


def test_set_seed__smoke_test():
    set_seed(123)


def test_set_basic_logging_config__smoke_test():
    set_basic_logging_config()


def test_assert_enough_disk_space__smoke_test():
    assert_enough_disk_space(min_space=0)


def test_random_generator_state():
    # TODO: Test `torch.cuda` random as well.
    state = get_random_generator_state()
    randint = random.randint(1, 2**31)
    numpy_randint = numpy.random.randint(1, 2**31)
    torch_randint = int(torch.randint(1, 2**31, (1,)))

    set_random_generator_state(state)
    post_randint = random.randint(1, 2**31)
    post_numpy_randint = numpy.random.randint(1, 2**31)
    post_torch_randint = int(torch.randint(1, 2**31, (1,)))

    assert randint == post_randint
    assert numpy_randint == post_numpy_randint
    assert torch_randint == post_torch_randint


def test_get_root_path():
    assert (ROOT_PATH / 'requirements.txt').is_file()


@mock.patch('src.environment.subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('src.environment.Path.read_text', return_value='torch==0.4.1\n')
def test_check_module_versions(_, __):
    check_module_versions()


@mock.patch('subprocess.check_output', return_value='torch==0.4.1'.encode())
@mock.patch('src.environment.Path.read_text', return_value='torch==0.4.0\n')
def test_check_module_versions__wrong_version(_, __):
    with pytest.raises(RuntimeError):
        check_module_versions()


@mock.patch('subprocess.check_output', return_value='tensorflow==0.4.0'.encode())
@mock.patch('src.environment.Path.read_text', return_value='torch==0.4.0\n')
def test_check_module_versions__missing_install(_, __):
    with pytest.raises(RuntimeError):
        check_module_versions()
