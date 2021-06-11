""" Tool for checking licenses of packages managed by `pip`.

Usage:

    $ python -m run.utils.check_licenses
"""

import subprocess
import typing
from io import StringIO

import pandas
import typer


class Row(typing.TypedDict):
    Name: str
    Version: str
    License: str


ACCEPTABLE = {
    "3-Clause BSD License",
    "Apache 2.0 License",
    "Apache 2",
    "Apache Software License",
    "BSD 3-Clause",
    "BSD License",
    "BSD",
    "GNU General Public License (GPL)",
    "GNU General Public License v2 or later (GPLv2+)",
    "GNU Lesser General Public License v2 or later (LGPLv2+)",
    "GNU Library or Lesser General Public License (LGPL), BSD License",
    "GNU Library or Lesser General Public License (LGPL)",
    "GPL",
    "Historical Permission Notice and Disclaimer (HPND)",
    "ISC License (ISCL)",
    "MIT License",
    "MIT",
    "Mozilla Public License 2.0 (MPL 2.0)",
    "new BSD",
    "Public Domain",
    "Python Software Foundation License",
}


def main():
    output = subprocess.check_output(["pip-licenses", "--format=csv"]).decode().strip()
    df = pandas.read_csv(StringIO(output))
    for index, row in df.iterrows():
        row = typing.cast(Row, row)
        for license_ in row["License"].split(","):
            license_ = license_.strip()
            if license_ not in ACCEPTABLE:
                full_name = f"{row['Name']}@{row['Version']}"
                print(f"WARNING: `{full_name}` has not support license '{license_}'!")


if __name__ == "__main__":
    typer.run(main)
