# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import dataclasses
import pickle
import typing


@dataclasses.dataclass(frozen=True)
class LinkedList:
    speaker: str
    passages: typing.Tuple[LinkedList] = dataclasses.field(init=False, repr=False, compare=False)
    index: int = dataclasses.field(init=False, repr=False, compare=False)

    @property
    def prev(self):
        return None if self.index == 0 else self.passages[self.index - 1]

    @property
    def next(self):
        return None if self.index == len(self.passages) - 1 else self.passages[self.index + 1]

    def __getstate__(self):
        if self.index != 0:
            copy = self.__dict__.copy()
            del copy["passages"]
            copy["__first"] = self.passages[0]
            return copy
        return self.__dict__

    def __setstate__(self, state):
        if "__first" in state:
            del state["__first"]
            if hasattr(self, "passages"):
                state["passages"] = self.passages
            object.__setattr__(self, "__dict__", state)
        else:
            object.__setattr__(self, "__dict__", state)
            for i, link in enumerate(self.passages):
                object.__setattr__(link, "index", i)
                object.__setattr__(link, "passages", self.passages)


links = []
for i in range(10):
    links.append(LinkedList(str(i)))

links = tuple(links)
for i, link in enumerate(links):
    object.__setattr__(link, "passages", links)
    object.__setattr__(link, "index", i)


links = pickle.loads(pickle.dumps(links))
for link in links:
    for link in link.passages:
        assert link.prev is None or link == link.prev.next
        assert link.next is None or link == link.next.prev
assert len(set(l.passages for l in links)) == 1

four = pickle.loads(pickle.dumps(links[4]))
for link in four.passages:
    assert link.prev is None or link == link.prev.next
    assert link.next is None or link == link.next.prev
