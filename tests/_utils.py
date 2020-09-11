import shutil
import urllib.request
import pathlib

import numpy
import pytest
import torch


def assert_almost_equal(a: torch.Tensor, b: torch.Tensor, **kwargs):
    numpy.testing.assert_almost_equal(a.cpu().detach().numpy(), b.cpu().detach().numpy(), **kwargs)


def create_disk_garbage_collection_fixture(root_directory: pathlib.Path, **kwargs):
    """ Create a fixture that deletes files and directories created during testing. """

    @pytest.fixture(**kwargs)
    def fixture():
        all_paths = lambda: set(root_directory.rglob('*')) if root_directory.exists() else set()

        before = all_paths()
        yield root_directory
        after = all_paths()

        for path in after.difference(before):
            if not path.exists():
                continue

            # NOTE: These `print`s will help debug a test if it fails; otherwise, they are ignored.
            if path.is_dir():
                print('Deleting directory: ', path)
                shutil.rmtree(str(path))
            elif path.is_file():
                print('Deleting file: ', path)
                path.unlink()

        assert before == all_paths()

    return fixture


def first_parameter_url_side_effect(url: str, *args, **kwargs):
    """ `unittest.mock.side_effect` for functions with a first parameter url.  """
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


# `unittest.mock.side_effect` for functions with a second parameter url.
url_second_side_effect = lambda _, *args, **kwargs: url_first_side_effect(*args, **kwargs)
