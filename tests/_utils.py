import pathlib
import subprocess
import typing
import urllib.request

import numpy
import pytest
import torch

import lib

TEST_DATA_PATH = lib.environment.ROOT_PATH / "tests" / "_test_data"


def assert_almost_equal(a: torch.Tensor, b: torch.Tensor, **kwargs):
    # TODO: Instead of this, should we use `torch.allclose`?
    numpy.testing.assert_almost_equal(a.cpu().detach().numpy(), b.cpu().detach().numpy(), **kwargs)


def first_parameter_url_side_effect(url: str, *_, **__):
    """`unittest.mock.side_effect` for functions with a first parameter url."""
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


def assert_uniform_distribution(counter: typing.Counter, **kwargs):
    """Assert that the counted distribution is uniform."""
    total = sum(counter.values())
    for value in counter.values():
        assert value / total == pytest.approx(1 / len(counter), **kwargs)


def subprocess_run_side_effect(command, *args, _command: str = "", _func=subprocess.run, **kwargs):
    """`subprocess.run.side_effect` that only returns if `_command` not in `command`."""
    if _command not in command:
        return _func(command, *args, **kwargs)


def make_metadata(
    path: pathlib.Path = pathlib.Path("."),
    sample_rate=1,
    num_channels=0,
    encoding=lib.audio.AudioEncoding.PCM_INT_8_BIT,
    bit_rate="",
    precision="",
    num_samples=0,
) -> lib.audio.AudioMetadata:
    """Make a `AudioMetadata` for testing."""
    return lib.audio.AudioMetadata(
        path=path,
        sample_rate=sample_rate,
        num_channels=num_channels,
        encoding=encoding,
        bit_rate=bit_rate,
        precision=precision,
        num_samples=num_samples,
    )


# NOTE: `unittest.mock.side_effect` for functions with a second parameter url.
second_parameter_url_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(
    *args, **kwargs
)


def print_params(label: str, params: typing.Iterable[typing.Tuple[str, torch.Tensor]]):
    """Print `params` in a copy pastable format for testing the model version."""
    print(label + " = {")
    for name, parameter in params:
        print(f'    "{name}": torch.tensor({parameter.sum().item():.6f}),')
    print("}")
