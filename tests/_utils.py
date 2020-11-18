import math
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
    length=math.inf,
    bit_rate="",
    precision="",
) -> lib.audio.AudioFileMetadata:
    """ Make a `AudioFileMetadata` for testing. """
    return lib.audio.AudioFileMetadata(
        path, sample_rate, num_channels, encoding, length, bit_rate, precision
    )


def make_passage(
    alignments: typing.Tuple[lib.datasets.Alignment, ...],
    speaker: lib.datasets.Speaker = lib.datasets.Speaker(""),
    audio_file: lib.audio.AudioFileMetadata = make_metadata(),
    script: typing.Optional[str] = None,
    transcript: typing.Optional[str] = None,
    **kwargs,
) -> lib.datasets.Passage:
    """ Make a `Passage` for testing. """
    max_ = lambda attr: max([getattr(a, attr)[1] for a in alignments])
    make_str = lambda attr: "." * max_(attr) if len(alignments) else ""
    script = make_str("script") if script is None else script
    transcript = make_str("transcript") if transcript is None else transcript
    return lib.datasets.Passage(audio_file, speaker, script, transcript, alignments, **kwargs)


def get_audio_metadata_side_effect(
    paths: typing.Union[pathlib.Path, typing.List[pathlib.Path]],
    _func=lib.audio.get_audio_metadata,
    **kwargs,
) -> typing.Union[lib.audio.AudioFileMetadata, typing.List[lib.audio.AudioFileMetadata]]:
    """`get_audio_metadata.side_effect` that returns placeholder metadata if the path doesn't
    exist."""
    is_list = isinstance(paths, list)
    paths = paths if isinstance(paths, list) else [paths]
    existing = [p for p in paths if p.exists() and p.is_file()]
    metadatas: typing.List[lib.audio.AudioFileMetadata] = _func(existing, **kwargs)
    metadatas = [metadatas.pop(0) if p.exists() else make_metadata(path=p) for p in paths]
    return metadatas if is_list else metadatas[0]


# NOTE: `unittest.mock.side_effect` for functions with a second parameter url.
second_parameter_url_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(
    *args, **kwargs
)
