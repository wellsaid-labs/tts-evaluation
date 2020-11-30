""" Sort `sys.stdin` using numbers first and `lib.text.natural_keys` second. """

import re
import sys

import lib

lines = [l for l in sys.stdin]
numbers = lambda l: [int(i) for i in re.findall(r"\d+", l)]
key = lambda l: tuple(numbers(l) + lib.text.natural_keys(l))  # type: ignore
lines = sorted(lines, key=key)
for line in lines:
    print(line.strip())
