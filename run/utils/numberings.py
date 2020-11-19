import pathlib
import re
import typing

import typer


def files(directory: pathlib.Path) -> typing.Iterator[str]:
    for path in directory.iterdir():
        if path.is_file():
            yield "-".join(re.findall(r"\d+", path.stem))


def main(directory: pathlib.Path, other_directory: pathlib.Path):
    """ Check that DIRECTORY and OTHER_DIRECTORY have files with similar numberings. """
    difference = set(files(directory)).symmetric_difference(set(files(other_directory)))
    message = (
        "Directories did not have equal numberings. "
        f"File(s) numbered {difference} were only found in one of the two directories."
    )
    assert len(difference) == 0, message


if __name__ == "__main__":  # pragma: no cover
    typer.run(main)
