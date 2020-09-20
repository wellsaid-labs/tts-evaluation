import urllib.request

import numpy
import torch

import lib

TEST_DATA_PATH = lib.environment.ROOT_PATH / 'tests' / '_test_data'


def assert_almost_equal(a: torch.Tensor, b: torch.Tensor, **kwargs):
    numpy.testing.assert_almost_equal(a.cpu().detach().numpy(), b.cpu().detach().numpy(), **kwargs)


def first_parameter_url_side_effect(url: str, *args, **kwargs):
    """ `unittest.mock.side_effect` for functions with a first parameter url.  """
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


# `unittest.mock.side_effect` for functions with a second parameter url.
url_second_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(*args, **kwargs)
