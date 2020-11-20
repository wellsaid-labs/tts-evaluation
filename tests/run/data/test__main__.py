import contextlib
import functools
import random
import tempfile
import typing
from pathlib import Path
from unittest import mock

import pytest

from run.data.__main__ import _normalize_file_name, logger, numberings, rename


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


def test__normalize_file_name():
    """ Test `__main__._normalize_file_name` handles all kinds of casing and numbering. """
    norm = _normalize_file_name
    assert norm("ENTHUSIASTIC_Script1") == "enthusiastic_script_1"
    assert norm("Script16-21") == "script_16-21"
    assert norm("Copy of WSL_SMurphyScript16-21") == "copy_of_wsl_s_murphy_script_16-21"
    assert norm("A1BondedTermite_111315") == "a1_bonded_termite_111315"
    assert norm("Applewood(PSL Group)_022317") == "applewood_(psl_group)_022317"
    assert norm("Applewood[PSL Group]_022317") == "applewood_[psl_group]_022317"
    assert norm("Applewood||PSL Group_022317") == "applewood_||_psl_group_022317"
    assert norm("applewood|psl Group_022317") == "applewood_|_psl_group_022317"
    assert norm("WhiteCanyonMotors_IVRPrompts_020116") == "white_canyon_motors_ivr_prompts_020116"
    assert norm("VisionaryCentreForWomen_061218") == "visionary_centre_for_women_061218"
    assert norm("SmileStudio87_010419") == "smile_studio_87_010419"


def test_rename():
    """ Test `__main__.rename` against a couple of basic cases. """
    _assert_rename(
        "Copy of WSL_SMurphyScript16-21.wav",
        "copy_of_wsl_s_murphy_script_16-21.wav",
        only_numbers=False,
    )
    _assert_rename("Copy of WSL_SMurphyScript16-21.wav", "16-21.wav", only_numbers=True)
    with mock.patch.object(logger, "error") as error_mock:
        _assert_rename(
            "Copy of WSL_SMurphyScript.mp3",
            "copy_of_wsl_s_murphy_script.mp3",
            only_numbers=True,
        )
        assert error_mock.called


def test_rename__duplicate():
    """ Test `__main__.rename` against duplicate numberings. """
    with _mock_directory() as (directory, *_):
        with pytest.raises(AssertionError) as info:
            rename(directory, only_numbers=True)
        assert "duplicate file name" in str(info.value)


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
