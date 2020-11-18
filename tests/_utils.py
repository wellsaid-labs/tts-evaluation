import functools
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
    numpy.testing.assert_almost_equal(a.cpu().detach().numpy(), b.cpu().detach().numpy(), **kwargs)


def first_parameter_url_side_effect(url: str, *_, **__):
    """ `unittest.mock.side_effect` for functions with a first parameter url.  """
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


def assert_uniform_distribution(counter: typing.Counter, **kwargs):
    """ Assert that the counted distribution is uniform. """
    total = sum(counter.values())
    for value in counter.values():
        assert value / total == pytest.approx(1 / len(counter), **kwargs)


def subprocess_run_side_effect(command, *args, _command: str = "", _func=subprocess.run, **kwargs):
    """`subprocess.run.side_effect` that only returns if `_command` not in `command`. """
    if _command not in command:
        return _func(command, *args, **kwargs)


def make_metadata(
    path: pathlib.Path = pathlib.Path("."),
    sample_rate=0,
    num_channels=0,
    encoding="",
    length=0,
    bit_rate="",
    precision="",
) -> lib.audio.AudioFileMetadata:
    """ Make a `AudioFileMetadata` for testing. """
    return lib.audio.AudioFileMetadata(
        path, sample_rate, num_channels, encoding, length, bit_rate, precision
    )


def get_audio_metadata_side_effect(
    paths: typing.List[pathlib.Path],
    _func=lib.audio.get_audio_metadata,
    **kwargs,
) -> typing.List[lib.audio.AudioFileMetadata]:
    """`get_audio_metadata.side_effect` that returns placeholder metadata if the path doesn't
    exist."""
    partial = functools.partial(
        lib.audio.AudioFileMetadata,
        sample_rate=0,
        num_channels=0,
        encoding="",
        length=0,
        bit_rate="",
        precision="",
    )
    existing = [p for p in paths if p.exists() and p.is_file()]
    metadatas = _func(existing, **kwargs)
    # TODO: Handle `get_audio_metadata` non-list return type.
    return [metadatas.pop(0) if p.exists() else partial(p) for p in paths]


# NOTE: `unittest.mock.side_effect` for functions with a second parameter url.
second_parameter_url_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(
    *args, **kwargs
)
