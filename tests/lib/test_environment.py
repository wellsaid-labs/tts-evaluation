from unittest import mock

import pytest

from src.environment import assert_enough_disk_space
from src.environment import check_module_versions
from src.environment import ROOT_PATH
from src.environment import set_basic_logging_config


def test_set_basic_logging_config__smoke_test():
    set_basic_logging_config()


def test_assert_enough_disk_space__smoke_test():
    assert_enough_disk_space(min_space=0)


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
