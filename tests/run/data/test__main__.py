import functools
import tempfile
from pathlib import Path
from unittest import mock

from run.data.__main__ import logger, rename


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
