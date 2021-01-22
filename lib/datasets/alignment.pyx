"""
`Alignment` is a memory optimized version of the below `NamedTuple`. The `NamedTuple` is around
232 bytes while the memory optimized version is 40 bytes.

```
class Alignment(typing.NamedTuple):
    script: typing.Tuple[int, int]
    audio: typing.Tuple[float, float]
    transcript: typing.Tuple[int, int]
```

Learn more:
https://habr.com/en/post/458518/

TODO: Store all alignments together in a `numpy.ndarray` to further reduce their size.
"""

_attrs = [
    "_script_start",
    "_script_end",
    "_audio_start",
    "_audio_end",
    "_transcript_start",
    "_transcript_end",
]

cdef class Alignment:
    _fields = ("script", "audio", "transcript")
    cdef public int _script_start, _script_end
    cdef public float _audio_start, _audio_end
    cdef public int _transcript_start, _transcript_end

    def __init__(self, script, audio, transcript):
        self._script_start = script[0]
        self._script_end = script[1]
        self._audio_start = audio[0]
        self._audio_end = audio[1]
        self._transcript_start = transcript[0]
        self._transcript_end = transcript[1]

    @property
    def script(self):
        return (self._script_start, self._script_end)

    @property
    def audio(self):
        return (self._audio_start, self._audio_end)

    @property
    def transcript(self):
        return (self._transcript_start, self._transcript_end)

    def _replace(self, **kwargs):
        _kwargs = {"script": self.script, "audio": self.audio, "transcript": self.transcript}
        _kwargs.update(kwargs)
        return Alignment(**_kwargs)

    def __iter__(self):
        return iter((self.script, self.audio, self.transcript))

    def __eq__(self, other):
        if not isinstance(other, Alignment):
            return False
        return all(getattr(self, a) == getattr(other, a) for a in _attrs)

    def __hash__(self):
        return hash((getattr(self, a) for a in _attrs))
