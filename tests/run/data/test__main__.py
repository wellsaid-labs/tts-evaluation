import contextlib
import functools
import random
import tempfile
import typing
from pathlib import Path
from unittest import mock

import pytest

from run.data.__main__ import logger, numberings, rename


def _rename_side_effect(path: Path, _expected=Path):
    """ Side-effect for `Path.rename`. """
    assert path == _expected


def _assert_rename(name: str, renamed: str, **kwargs):
    """ Helper function for `__main__.rename`. """
    with tempfile.TemporaryDirectory() as directory_:
        directory = Path(directory_)
        path = directory / name
        path.touch()
        with mock.patch("pathlib.Path.rename") as mock_rename:
            partial = functools.partial(_rename_side_effect, _expected=directory / renamed)
            mock_rename.side_effect = partial
            rename(path.parent, **kwargs)


@contextlib.contextmanager
def _mock_directory() -> typing.Iterator[
    typing.Tuple[Path, Path, Path, typing.List[Path], typing.List[Path]]
]:
    """ Create a temporary directory with empty files for testing. """
    with tempfile.TemporaryDirectory() as directory_:
        directory = Path(directory_)
        recordings_path = directory / "recordings"
        recordings_path.mkdir()
        recordings = []
        for name in [
            "Copy of WSL - Adrienne WalkerScript10.wav",
            "WSL_EliseRandall_DIPHONE_Script-1.wav",
            "WSL_EliseRandall_ENTHUSIASTIC_Script-1.wav",
            "Copy of WSL - Adrienne WalkerScript16-21.wav",
        ]:
            recordings.append(recordings_path / name)
            recordings[-1].touch()

        scripts_path = directory / "scripts"
        scripts_path.mkdir()
        scripts = []
        for name in [
            "Script10.csv",
            "DIPHONE_Script-1.csv",
            "ENTHUSIASTIC_Script1.csv",
            "Script16-21.csv",
        ]:
            scripts.append(scripts_path / name)
            scripts[-1].touch()

        yield (directory, recordings_path, scripts_path, recordings, scripts)


def test_rename():
    """ Test `__main__.rename` against a couple of basic cases. """
    _assert_rename(
        "Copy of WSL_SMurphyScript16-21.wav",
        "copy_of_wsl_smurphyscript16-21.wav",
        only_numbers=False,
    )
    _assert_rename("Copy of WSL_SMurphyScript16-21.wav", "16-21.wav", only_numbers=True)
    with mock.patch.object(logger, "warning") as warning_mock:
        _assert_rename(
            "Copy of WSL_SMurphyScript.wav", "copy_of_wsl_smurphyscript.wav", only_numbers=True
        )
        assert warning_mock.called


def test_rename__duplicate():
    """ Test `__main__.rename` against duplicate numberings. """
    with _mock_directory() as (directory, *_):
        with pytest.raises(AssertionError) as info:
            rename(directory, only_numbers=True)
        assert "duplicate file names" in str(info.value)


def test_numberings():
    """ Test `__main__.numberings` against a basic case. """
    with _mock_directory() as (_, recordings_path, scripts_path, *_):
        numberings(recordings_path, scripts_path)


def test_numberings__duplicate_numbering():
    """ Test `__main__.numberings` handles duplicate numberings. """
    with _mock_directory() as (_, recordings_path, scripts_path, _, scripts):
        scripts[1].unlink()
        with pytest.raises(AssertionError):
            numberings(recordings_path, scripts_path)


def test_numberings__missing():
    """ Test `__main__.numberings` handles a missing file. """
    with _mock_directory() as (_, recordings_path, scripts_path, recordings, scripts):
        random.choice(recordings + scripts).unlink()
        with pytest.raises(AssertionError):
            numberings(recordings_path, scripts_path)
