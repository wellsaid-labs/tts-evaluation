""" Sort `sys.stdin` using `lib.text.numbers_then_natural_keys`. """

import sys

import lib

lines = [l for l in sys.stdin]
lines = sorted(lines, key=lib.text.numbers_then_natural_keys)
for line in lines:
    print(line.strip())
