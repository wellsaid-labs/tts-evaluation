import collections
import dataclasses
import gc
import pickle
import sys
import timeit
import typing

import pyximport
from attrs import define

from run.data._loader import Speaker
from run.data._loader.english.lj_speech import LINDA_JOHNSON

pyximport.install(language_level=sys.version_info[0])

import bam  # type: ignore


@dataclasses.dataclass(frozen=True)
class data:
    a: int
    b: int


class tupl(tuple):
    @property
    def a(self):
        return self[0]

    @property
    def b(self):
        return self[1]


class clss:
    __slots__ = "args"

    def __init__(self, args) -> None:
        self.args = args

    @property
    def a(self):
        return self.args[0]

    @property
    def b(self):
        return self.args[1]

    def __getstate__(self):
        return self.args

    def __setstate__(self, args):
        self.args = args


@define
class attr_:
    x: int
    y: int


class typed(typing.NamedTuple):
    a: int
    b: Speaker


print(dataclasses.astuple(LINDA_JOHNSON))

spk = ("linda_johnson", "LibriVox", "English", "English (United States)", None, None, False, None)

nt = collections.namedtuple("nt", "a b")
my10k = [bam.MyClass(i, i) for i in range(2000)]
data10k = [data(i, i) for i in range(2000)]
tupl10k = [tupl((i, i)) for i in range(2000)]
a10k = [attr_(x=1, y=2) for i in range(2000)]
cl10k = [clss((1, 2)) for i in range(2000)]
nt10k = {nt(i, LINDA_JOHNSON): 1 for i in range(2000)}
t10k = {(i, spk): 1 for i in range(2000)}
ntt10k = [typed(1, LINDA_JOHNSON) for i in range(2000)]
dict10k = [{"a": 1, "b": 2} for i in range(2000)]
print(timeit.timeit("pickle.loads(pickle.dumps(my10k))", globals=globals(), number=100))
print(timeit.timeit("pickle.loads(pickle.dumps(data10k))", globals=globals(), number=100))
print(timeit.timeit("pickle.loads(pickle.dumps(tupl10k))", globals=globals(), number=100))
print(timeit.timeit("pickle.loads(pickle.dumps(a10k))", globals=globals(), number=100))
print("nt10k", timeit.timeit("pickle.loads(pickle.dumps(nt10k))", globals=globals(), number=100))
print(
    "dict10k", timeit.timeit("pickle.loads(pickle.dumps(dict10k))", globals=globals(), number=100)
)
print("cl10k", timeit.timeit("pickle.loads(pickle.dumps(cl10k))", globals=globals(), number=100))
print("t10k", timeit.timeit("pickle.loads(pickle.dumps(t10k))", globals=globals(), number=100))
print("ntt10k", timeit.timeit("pickle.loads(pickle.dumps(ntt10k))", globals=globals(), number=100))
[nt._make(t) for t in pickle.loads(pickle.dumps([tuple(t) for t in nt10k]))]
print(
    timeit.timeit(
        "[nt._make(t) for t in pickle.loads(pickle.dumps([tuple(t) for t in nt10k]))]",
        globals=globals(),
        number=100,
    )
)
