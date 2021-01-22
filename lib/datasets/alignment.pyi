# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import typing

class Alignment:
    _fields: typing.Tuple[str]
    def __init__(
        self,
        script: typing.Tuple[int, int],
        audio: typing.Tuple[float, float],
        transcript: typing.Tuple[int, int],
    ): ...
    @property
    def script(self) -> typing.Tuple[int, int]: ...
    @property
    def audio(self) -> typing.Tuple[float, float]: ...
    @property
    def transcript(self) -> typing.Tuple[int, int]: ...
    def _replace(self, **kwargs) -> Alignment: ...
    def __iter__(
        self,
    ) -> typing.Iterator[typing.Union[typing.Tuple[int, int], typing.Tuple[float, float]]]: ...
