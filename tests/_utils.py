import pathlib
import subprocess
import typing
import urllib.request

import numpy
import pytest
import torch

import lib
import run.data._loader
from lib.utils import Tuple
from run.data._loader import Alignment

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
    sample_rate=1,
    num_channels=0,
    encoding=lib.audio.AudioEncoding.PCM_INT_8_BIT,
    bit_rate="",
    precision="",
    num_samples=0,
) -> lib.audio.AudioMetadata:
    """ Make a `AudioMetadata` for testing. """
    return lib.audio.AudioMetadata(
        path=path,
        sample_rate=sample_rate,
        num_channels=num_channels,
        encoding=encoding,
        bit_rate=bit_rate,
        precision=precision,
        num_samples=num_samples,
    )


def make_passage(
    alignments: Tuple[Alignment] = Alignment.stow([]),
    nonalignments: Tuple[Alignment] = Alignment.stow([]),
    speaker: run.data._loader.Speaker = run.data._loader.Speaker(""),
    audio_file: lib.audio.AudioMetadata = make_metadata(),
    script: typing.Optional[str] = None,
    transcript: typing.Optional[str] = None,
    speech_segments: typing.Optional[typing.Tuple[slice, ...]] = None,
    **kwargs,
) -> run.data._loader.Passage:
    """Make a `Passage` for testing.

    TODO: Move this from `tests/_utils.py` to `tests/run/_utils.py`, for better locality.
    """
    max_ = lambda attr: max([getattr(a, attr)[1] for a in alignments])
    make_str: typing.Callable[[str], str]
    make_str = lambda attr: "." * max_(attr) if len(alignments) else ""
    script_ = make_str("script") if script is None else script
    transcript_ = make_str("transcript") if transcript is None else transcript
    passage = run.data._loader.Passage(
        audio_file, speaker, script_, transcript_, alignments, **kwargs
    )
    object.__setattr__(passage, "nonalignments", nonalignments)
    default_speech_segments = tuple(passage[i] for i in range(len(alignments)))
    speech_segments = default_speech_segments if speech_segments is None else speech_segments
    object.__setattr__(passage, "speech_segments", speech_segments)
    return passage


def make_unprocessed_passage(
    audio_path=pathlib.Path("."),
    speaker=run.data._loader.Speaker(""),
    script="",
    transcript="",
    alignments=None,
) -> run.data._loader.utils.UnprocessedPassage:
    """Make a `UnprocessedPassage` for testing.

    TODO: Move this from `tests/_utils.py` to `tests/run/_utils.py`, for better locality.
    """
    return run.data._loader.utils.UnprocessedPassage(
        audio_path, speaker, script, transcript, alignments
    )


def get_audio_metadata_side_effect(
    paths: typing.Union[pathlib.Path, typing.List[pathlib.Path]],
    _func=lib.audio.get_audio_metadata,
    **kwargs,
) -> typing.Union[lib.audio.AudioMetadata, typing.List[lib.audio.AudioMetadata]]:
    """`get_audio_metadata.side_effect` that returns placeholder metadata if the path doesn't
    exist."""
    is_list = isinstance(paths, list)
    paths = paths if isinstance(paths, list) else [paths]
    existing = [p for p in paths if p.exists() and p.is_file()]
    metadatas: typing.List[lib.audio.AudioMetadata] = _func(existing, **kwargs)
    metadatas = [metadatas.pop(0) if p.exists() else make_metadata(path=p) for p in paths]
    return metadatas if is_list else metadatas[0]


# NOTE: `unittest.mock.side_effect` for functions with a second parameter url.
second_parameter_url_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(
    *args, **kwargs
)
