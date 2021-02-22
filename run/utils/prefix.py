"""Add a prefix before each argument.

Learn more about how to accomplish this in the terminal:
https://unix.stackexchange.com/questions/445430/expand-glob-with-flag-inserted-before-each-filename

Usage:
    $ python -m run.utils.prefix --prefix-flag abc.txt def.txt ghi.txt
    --prefix-flag abc.txt --prefix-flag def.txt --prefix-flag ghi.txt
"""
import sys
import typing

import lib


def main(flag: str, args: typing.List[str]):
    print(" ".join([f"{flag} {a}" for a in args]), end="")


if __name__ == "__main__":
    main(sys.argv[1], lib.utils.flatten_2d([a.split() for a in sys.argv[2:]]))
