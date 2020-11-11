""" Sort `sys.stdin` with `lib.text.natural_keys`. """

import sys

import lib

lines = [l for l in sys.stdin]
lines = sorted(lines, key=lib.text.natural_keys)
for line in lines:
    print(line)
