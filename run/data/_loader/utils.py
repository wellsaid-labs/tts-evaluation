# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import csv
import dataclasses
import functools
import itertools
import json
import logging
import math
import os
import pathlib
import random
import subprocess
import typing
from concurrent import futures
from dataclasses import field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from hparams import HParam, configurable
from third_party import LazyLoader
from tqdm import tqdm

import lib
from lib.audio import AudioDataType, AudioEncoding, AudioFormat, AudioMetadata, get_audio_metadata
from lib.utils import Interval, Timeline, Tuple, flatten_2d

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")


logger = logging.getLogger(__name__)

FloatFloat = typing.Tuple[float, float]
IntInt = typing.Tuple[int, int]
Slice = slice  # NOTE: `pylance` is buggy if we use `slice` directly for typing.


class NonalignmentSpans(typing.NamedTuple):
    """Nonalignments for a `Passage` or `Span`.

    Args:
        prev: The last nonalignment from the previous passage.
        next: The first nonalignment from the next passage.
        spans: Nonalignments from the current passage, or span.
    """

    prev: typing.Optional[Span]
    next: typing.Optional[Span]
    spans: typing.List[Span]


def read_audio(audio_file: AudioMetadata, *args, **kwargs) -> np.ndarray:
    """Read `audio_file` into a `np.float32` array."""
    try:
        assert audio_file.encoding == AudioEncoding.PCM_FLOAT_32_BIT
        audio = lib.audio.read_wave_audio(audio_file, *args, **kwargs)
    except AssertionError:
        audio = lib.audio.read_audio(audio_file.path, *args, **kwargs)
    assert audio.dtype == np.float32, "Invariant failed. Audio `dtype` must be `np.float32`."
    return audio


@configurable
def normalize_audio_suffix(path: Path, suffix: str = HParam()) -> Path:
    """ Normalize the last suffix to `suffix` in `path`. """
    assert len(path.suffixes) == 1, "`path` has multiple suffixes."
    return path.parent / (path.stem + suffix)


@configurable
def normalize_audio(
    source: Path,
    destination: Path,
    suffix: str = HParam(),
    data_type: AudioDataType = HParam(),
    bits: int = HParam(),
    sample_rate: int = HParam(),
    num_channels: int = HParam(),
):
    return lib.audio.normalize_audio(
        source, destination, suffix, data_type, bits, sample_rate, num_channels
    )


@configurable
def is_normalized_audio_file(
    audio_file: AudioMetadata, audio_format: AudioFormat = HParam(), suffix: str = HParam()
):
    """Check if `audio_file` is normalized to `audio_format`."""
    attrs = [f.name for f in dataclasses.fields(AudioFormat) if f.name != "encoding"]
    bool_ = audio_file.path.suffix == suffix
    # NOTE: Use `.name` because of this bug:
    # https://github.com/streamlit/streamlit/issues/2379
    bool_ = bool_ and audio_file.encoding.name == audio_format.encoding.name
    return bool_ and all(getattr(audio_format, a) == getattr(audio_file, a) for a in attrs)


@configurable
def _cache_path(
    original: pathlib.Path, prefix: str, suffix: str, cache_dir=HParam(), **kwargs
) -> pathlib.Path:
    """Make `Path` for caching results given the `original` file.

    Args:
        ...
        suffix: Cache path suffix, starting with a dot.
        cache_dir: Relative or absolute path to a directory dedicated for caching.
        ...
    """
    assert suffix[0] == "."
    params = ",".join(f"{k}={v}" for k, v in kwargs.items())
    parent = original.parent if original.parent.stem == cache_dir else original.parent / cache_dir
    cache_path = parent / f"{prefix}({original.stem},{params}){suffix}"
    cache_path.parent.mkdir(exist_ok=True)
    return cache_path


@configurable
def maybe_normalize_audio_and_cache(
    audio_file: AudioMetadata,
    suffix: str = HParam(),
    data_type: AudioDataType = HParam(),
    bits: int = HParam(),
    sample_rate: int = HParam(),
    num_channels: int = HParam(),
    **kwargs,
) -> pathlib.Path:
    """Normalize `audio_file`, if it's not already normalized, and cache the results."""
    if is_normalized_audio_file(audio_file):
        return audio_file.path
    kwargs_ = dict(
        suffix=suffix,
        bits=bits,
        sample_rate=sample_rate,
        num_channels=num_channels,
        data_type=data_type,
    )
    name = maybe_normalize_audio_and_cache.__wrapped__.__name__
    cache = _cache_path(audio_file.path, name, **kwargs_, **kwargs)
    if not cache.exists():
        lib.audio.normalize_audio(audio_file.path, cache, **kwargs_)
    return cache


@configurable
def get_non_speech_segments_and_cache(
    audio_file: AudioMetadata,
    low_cut: int = HParam(),
    frame_length: float = HParam(),
    hop_length: float = HParam(),
    threshold: float = HParam(),
    **kwargs,
) -> Timeline[FloatFloat]:
    """Get non-speech segments in `audio_file` and cache."""
    format: typing.Callable[[typing.Iterable[FloatFloat]], Timeline[FloatFloat]]
    format = lambda l: Timeline([Interval(s, s) for s in l])
    kwargs_ = dict(
        low_cut=low_cut, frame_length=frame_length, hop_length=hop_length, threshold=threshold
    )
    name = get_non_speech_segments_and_cache.__wrapped__.__name__
    cache_path = _cache_path(audio_file.path, name, ".npy", **kwargs_, **kwargs)
    if cache_path.exists():
        return format(tuple(t) for t in np.load(cache_path, allow_pickle=False))

    audio = read_audio(audio_file, memmap=True)
    vad: typing.List[FloatFloat] = lib.audio.get_non_speech_segments(audio, audio_file, **kwargs_)
    np.save(cache_path, np.array(vad), allow_pickle=False)
    return format(vad)


def voiced_nonalignment_spans(
    span: typing.Union[Passage, Span]
) -> typing.Tuple[NonalignmentSpans, typing.List[bool]]:
    """In addition to `NonalignmentSpans`, this returns if a nonalignment should be voiced,
    probably.

    NOTE: This assumes that the `transcript` and `script` from the previous and next passage
    affect mistranscriptions in the current passage.
    """
    spans = span.nonalignment_spans()
    text = [s.script + s.transcript for s in spans.spans]
    text[0] += "" if spans.prev is None else spans.prev.script + spans.prev.transcript
    text[-1] += "" if spans.next is None else spans.next.script + spans.next.transcript
    return spans, [lib.text.is_voiced(t) for t in text]


def has_a_mistranscription(span: typing.Union[Passage, Span]) -> bool:
    """Return `True` if `span` contains a mistranscription, probably."""
    _, is_voiced = voiced_nonalignment_spans(span)
    return any(is_voiced)


_alignment_dtype = [
    ("script", np.dtype([("start", np.uint32), ("stop", np.uint32)])),
    ("audio", np.dtype([("start", np.float32), ("stop", np.float32)])),
    ("transcript", np.dtype([("start", np.uint32), ("stop", np.uint32)])),
]
alignment_dtype = np.dtype(_alignment_dtype)


class Alignment(typing.NamedTuple):
    """An aligned `script`, `audio` and `transcript` slice.

    Args:
        script: The start and end of a script slice in characters.
        audio: The start and end of a audio recording slice in seconds.
        transcript: The start and end of a trasnscript slice in characters.
    """

    script: IntInt
    audio: FloatFloat
    transcript: IntInt

    def to_json(self):
        return [list(self.script), list(self.audio), list(self.transcript)]

    @classmethod
    def from_json(cls, args: typing.List[typing.List[float]]):
        return cls(
            script=tuple(args[0]),
            audio=tuple(args[1]),
            transcript=tuple(args[2]),
        )

    @staticmethod
    def stow(alignments: typing.Sequence[Alignment]) -> Tuple[Alignment]:
        return lib.utils.stow(alignments, dtype=alignment_dtype)


class Speaker(typing.NamedTuple):
    label: str
    name: typing.Optional[str] = None
    gender: typing.Optional[str] = None


@dataclasses.dataclass(frozen=True)
class UnprocessedPassage:
    """Raw data for a voiced passage.

    Args:
        audio_path: Audio file corresponding to a voice-over of the `script`.
        speaker: An identifier of the voice.
        script: The `script` the `speaker` was reading from.
        transcript: The `transcript` of the `audio`.
        alignments: Alignments (sorted) that align the `script`, `transcript` and `audio`.
        other_metadata: Additional metadata associated with this passage.
    """

    audio_path: pathlib.Path
    speaker: Speaker
    script: str
    transcript: str
    alignments: typing.Optional[typing.Tuple[Alignment, ...]] = None
    other_metadata: typing.Dict = field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Passage:
    """A voiced passage.

    TODO: Create a `ConventionalPassage` or `ConventionalSpan` for storing tens of thousands
    of single alignment pre-cut spans. We could more efficiently and accurately handle script
    and audio updates. We could more efficiently create a `ConventionalSpan`.

    Args:
        audio_file: A voice-over of the `script`.
        speaker: An identifier of the voice.
        script: The `script` the `speaker` was reading from.
        transcript: The `transcript` of the `audio`.
        alignments: Alignments (sorted) that align the `script`, `transcript` and `audio`.
        nonalignments: Nonalignments are alignments, in between, alignments. For example,
            in between two valid alignments, there may be a misalignment between the script
            and transcript. Also, a nonalignment may strech beyond the edges of the `Passage`.
        speech_segments: Speech segments represented by `Span`s with pauses on either
            end. In normal speech, one typically finds many consecutive words being said with no
            pauses between them. Learn more:
            https://en.wikipedia.org/wiki/Speech_segmentation
            https://english.stackexchange.com/questions/365470/do-you-take-a-break-between-words-when-pronouncing
            https://en.wikipedia.org/wiki/detect_voice_activity
        other_metadata: Additional metadata associated with this passage.
    """

    audio_file: AudioMetadata
    speaker: Speaker
    script: str
    transcript: str
    alignments: Tuple[Alignment]
    other_metadata: typing.Dict = field(default_factory=dict, compare=False, hash=False)
    nonalignments: Tuple[Alignment] = field(init=False, repr=False, compare=False)
    speech_segments: typing.Tuple[Span, ...] = field(
        init=False, repr=False, compare=False, hash=False
    )
    first: Alignment = field(init=False, repr=False, compare=False)
    last: Alignment = field(init=False, repr=False, compare=False)
    prev: typing.Optional[Passage] = field(default=None, init=False, repr=False, compare=False)
    next: typing.Optional[Passage] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.alignments) > 0:  # NOTE: Cache `first` and `last`, if they exist.
            object.__setattr__(self, "first", self.alignments[0])
            object.__setattr__(self, "last", self.alignments[-1])

    def audio(self):
        return read_audio(self.audio_file)

    def aligned_audio_length(self) -> float:
        return self.last.audio[-1] - self.first.audio[0]

    def span(self, *args, **kwargs) -> Span:
        return Span(self, self.alignments, *args, **kwargs)

    def nonalignment_span(self, *args, **kwargs) -> Span:
        return Span(self, self.nonalignments, *args, **kwargs)

    def nonalignment_spans(self) -> NonalignmentSpans:
        """Get a `Span` for every `nonalignment`.

        NOTE: Additionally, this returns two `Span`s from the neighboring `Passage`s. For example,
        the last `nonalignment` of the previous passage overlaps with the first `nonalignment`
        in this passage.
        """
        cut = lambda x: slice(x, x + 1)
        prev = self.prev
        return NonalignmentSpans(
            None if prev is None else prev.nonalignment_span(cut(len(prev.nonalignments) - 1)),
            None if self.next is None else self.next.nonalignment_span(cut(0)),
            [self.nonalignment_span(cut(i)) for i in range(len(self.nonalignments))],
        )

    def __getitem__(self, key) -> Span:
        if isinstance(key, int):
            key = len(self.alignments) + key if key < 0 else key
            key = slice(key, key + 1)
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self.alignments))
            assert step == 1, f"Step size {step} is not supported."
            return self.span(slice(start, stop))
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))

    def _get(self, field: str) -> typing.List[float]:
        """Get the values for `field` in `self.alignments`."""
        return [typing.cast(float, v) for a in self.alignments for v in getattr(a, field)]

    @staticmethod
    def _no_white_space(s: str) -> bool:
        return s.strip() == s

    def check_invariants(self, eps: float = 1e-6):
        """Check datastructure invariants."""
        assert hasattr(self, "nonalignments")
        assert hasattr(self, "speech_segments")
        assert hasattr(self, "prev")
        assert hasattr(self, "next")

        assert self.prev is None or self == self.prev.next
        assert self.next is None or self == self.next.prev
        assert len(self.nonalignments) == len(self.alignments) + 1
        assert len(self.alignments) == 0 or len(self.speech_segments) > 0

        assert all(a.script[0] <= a.script[1] for a in self.alignments)
        assert all(a.audio[0] <= a.audio[1] for a in self.alignments)
        assert all(a.transcript[0] <= a.transcript[1] for a in self.alignments)

        slices = (self.script[a.script[0] : a.script[1]] for a in self.alignments)
        assert all(self._no_white_space(s) for s in slices)
        slices = (self.transcript[a.transcript[0] : a.transcript[1]] for a in self.alignments)
        assert all(self._no_white_space(s) for s in slices)

        pairs = zip(self.alignments, self.alignments[1:])
        assert all(a.script[1] <= b.script[0] for a, b in pairs)
        assert all(a.transcript[1] <= b.transcript[0] for a, b in pairs)
        # NOTE: The `audio` alignments may overlap by a little bit, at the edges.
        assert all(a.audio[0] < b.audio[1] for a, b in pairs)

        if len(self.alignments) != 0:
            # NOTE: `eps` allows for minor rounding errors.
            assert max(self._get("audio")) <= self.audio_file.length + eps
            assert max(self._get("script")) <= len(self.script)
            assert max(self._get("transcript")) <= len(self.transcript)
            assert min(self._get("audio")) >= 0
            assert min(self._get("script")) >= 0
            assert min(self._get("transcript")) >= 0

        return self


SpanType = typing.TypeVar("SpanType", bound="Span")


@dataclasses.dataclass(frozen=True)
class Span:
    """A span of the voiced passage.

    NOTE: The first and last `Alignment`s are cached for performance. The goal is to avoid
    accessing `self.passage_alignments` due to it's mediocre performance.
    TODO: Instead of storing `passage_alignments` and `slice`, consolidate into a `ListView` class,
    similar to: https://stackoverflow.com/questions/3485475/can-i-create-a-view-on-a-python-list

    Args:
        passage: The original passage, for context.
        passage_alignments: The original passage alignments, for context.
        slice: A `slice` of `passage.alignments`.
        audio_slice_: By default, the `audio_slice` is based on `passage_alignments[slice]`. This
            allows `audio_slice_` to be customized.
    """

    passage: Passage = field(repr=False)
    passage_alignments: Tuple[Alignment] = field(repr=False)
    slice: Slice
    audio_slice_: typing.Optional[Slice] = None
    _first_cache: Alignment = field(init=False, repr=False, compare=False)
    _last_cache: Alignment = field(init=False, repr=False, compare=False)

    @property
    def _first(self) -> Alignment:
        """NOTE: This property is private because it's not offset correctly similar to
        `self.alignments`.
        """
        if not hasattr(self, "_first_cache"):
            object.__setattr__(self, "_first_cache", self.passage_alignments[self.slice.start])
        return self._first_cache

    @property
    def _last(self) -> Alignment:
        if self.slice.stop - 1 == self.slice.start:
            return self._first
        if not hasattr(self, "_last_cache"):
            object.__setattr__(self, "_last_cache", self.passage_alignments[self.slice.stop - 1])
        return self._last_cache

    @property
    def audio_start(self):
        """Start of audio span in `self.audio_file`."""
        return self._first.audio[0] if self.audio_slice_ is None else self.audio_slice_.start

    @property
    def audio_stop(self):
        """End of audio span in `self.audio_file`."""
        return self._last.audio[-1] if self.audio_slice_ is None else self.audio_slice_.stop

    @property
    def speaker(self):
        return self.passage.speaker

    @property
    def audio_file(self):
        return self.passage.audio_file

    @property
    def alignment(self):
        """Get `Span` as an `Alignment` for `self.passage`."""
        return Alignment(
            script=(self._first.script[0], self._last.script[-1]),
            audio=(self.audio_start, self.audio_stop),
            transcript=(self._first.transcript[0], self._last.transcript[-1]),
        )

    @property
    def other_metadata(self):
        return self.passage.other_metadata

    @property
    def script_slice(self):
        """Slice of script in `self.passage.script`."""
        return slice(self._first.script[0], self._last.script[-1])

    @property
    def audio_slice(self):
        """Slice of script in `self.audio_file`."""
        return slice(self.audio_start, self.audio_stop)

    @property
    def transcript_slice(self):
        """Slice of script in `self.passage.transcript`."""
        return slice(self._first.transcript[0], self._last.transcript[-1])

    @property
    def script(self):
        return self.passage.script[self.script_slice]

    @property
    def transcript(self):
        return self.passage.transcript[self.transcript_slice]

    @staticmethod
    def _offset_helper(a: FloatFloat, b: float):
        return (a[0] - b, a[1] - b)

    def _offset(self, alignment: Alignment):
        return alignment._replace(
            script=self._offset_helper(alignment.script, self._first.script[0]),
            transcript=self._offset_helper(alignment.transcript, self._first.transcript[0]),
            audio=self._offset_helper(alignment.audio, self.audio_start),
        )

    @property
    def alignments(self):
        return Alignment.stow([self._offset(a) for a in self.passage_alignments[self.slice]])

    @property
    def audio_length(self):
        return self.audio_stop - self.audio_start

    def audio(self) -> np.ndarray:
        return read_audio(self.passage.audio_file, self.audio_start, self.audio_length)

    def nonalignment_spans(self) -> NonalignmentSpans:
        """See `self.passage.nonalignment_spans()` docs."""
        assert self.passage_alignments == self.passage.alignments
        spans = self.passage.nonalignment_spans()
        return NonalignmentSpans(
            prev=spans.prev if self.slice.start == 0 else None,
            next=spans.next if self.slice.stop == len(self.passage_alignments) else None,
            spans=spans.spans[self.slice.start : self.slice.stop + 1],
        )

    def __len__(self) -> int:
        return self.slice.stop - self.slice.start

    def __getitem__(self: SpanType, key) -> SpanType:
        if isinstance(key, int):
            key = len(self) + key if key < 0 else key
            key = slice(key, key + 1)
        if isinstance(key, slice):
            offset = self.slice.start
            start, stop, step = key.indices(len(self))
            assert step == 1, f"Step size {step} is not supported."
            slice_ = slice(offset + start, offset + stop)
            return self.__class__(self.passage, self.passage_alignments, slice_)
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))

    def __iter__(self: SpanType) -> typing.Iterator[SpanType]:
        return (self.__getitem__(i) for i in range(self.slice.stop - self.slice.start))

    def check_invariants(self):
        """ Check datastructure invariants. """
        self.passage.check_invariants()
        assert self.slice.stop > self.slice.start, "`Span` must have `Alignments`."
        assert self.slice.stop <= len(self.passage_alignments) and self.slice.stop >= 0
        assert self.slice.start < len(self.passage_alignments) and self.slice.start >= 0
        assert self.audio_slice_ is None or (
            self.audio_slice_.start <= self._first.audio[0]
            and self.audio_slice_.stop >= self._last.audio[-1]
        )
        return self


def _merge(
    a: Alignment, b: Alignment, script: bool = False, transcript: bool = False, audio: bool = False
) -> Alignment:
    """Merge alignments `a` and `b` iff they are connected."""
    if not script and not transcript and not audio:
        return a

    return a._replace(
        script=(b if script else a).script,
        transcript=(b if transcript else a).transcript,
        audio=(b if audio else a).audio,
    )


def _make_nonalignments(doc: typing.List[Passage], index: int, **kwargs) -> Tuple[Alignment]:
    """Get nonalignments in between `data.alignments`, and in between
    `[prev.alignments[-1], data.alignments, next.alignments[0]]`.

    Args:
        curr: The current passage.
        audio_metadata: Audio metadata for the current passage.
        prev: The last passage.
        next: The next passage.
        **kwargs: Keyword arguments passed to `_merge`. They determine the connection between
            `prev`, `curr`, and `next`.
    """
    prev = next((p for p in reversed(doc[:index]) if len(p.alignments) > 0), None)
    next_ = next((p for p in doc[index + 1 :] if len(p.alignments) > 0), None)
    curr = doc[index]

    prev_alignment = Alignment(script=(0, 0), audio=(0.0, 0.0), transcript=(0, 0))
    if prev is not None:
        prev_alignment = _merge(prev_alignment, prev.alignments[-1], **kwargs)

    next_alignment = Alignment(
        script=(len(curr.script), len(curr.script)),
        audio=(curr.audio_file.length, curr.audio_file.length),
        transcript=(len(curr.transcript), len(curr.transcript)),
    )
    if next_ is not None:
        next_alignment = _merge(next_alignment, next_.alignments[0], **kwargs)

    alignments = [prev_alignment] + list(curr.alignments) + [next_alignment]
    nonalignments = []
    for prev_, next_ in zip(alignments, alignments[1:]):
        assert prev_.script[-1] <= next_.script[0], "Alignments shouldn't overlap."
        assert prev_.transcript[-1] <= next_.transcript[0], "Alignments shouldn't overlap."
        nonalignment = Alignment(
            script=(prev_.script[-1], next_.script[0]),
            transcript=(prev_.transcript[-1], next_.transcript[0]),
            audio=(prev_.audio[-1], next_.audio[0]),
        )
        nonalignments.append(nonalignment)
    return Alignment.stow(nonalignments)


def _exists(path: Path) -> bool:
    """ Helper function for `make_passages` that can be easily mocked. """
    return path.exists() and path.is_file()


def _filter_non_speech_segments(
    timeline: Timeline[typing.Tuple[int, Alignment]], non_speech_segments: typing.Iterable[slice]
) -> typing.Iterable[slice]:
    """Filter out `non_speech_segments` which are not in between alignments.

    NOTE: To better understand the various cases, take a look at the unit tests for this function.
    """
    for slice_ in non_speech_segments:
        vals = list(timeline[slice_])

        # NOTE: Check if any interval is inside any other interval.
        intervals = [o[1].audio for o in vals]
        permutations = itertools.permutations(intervals + [(slice_.start, slice_.stop)], 2)
        if any(a[0] <= b[0] and b[1] <= a[1] for a, b in permutations):
            continue

        # NOTE: Check if any alignment intervals overlap.
        if any(sum(max(a[0], b[0]) < min(a[1], b[1]) for b in intervals) > 1 for a in intervals):
            continue

        assert len(vals) < 3, "It should be impossible to overlap three alignments."
        message = "Alignments should be back-to-back."
        assert len(vals) != 2 or abs(vals[0][0] - vals[1][0]) == 1, message

        yield slice_


@configurable
def _make_speech_segments(
    passage: Passage, nss_timeline: typing.Optional[Timeline[FloatFloat]], padding: float = HParam()
) -> typing.Tuple[Span, ...]:
    """Make a list of `Span`s that start and end with silence.

    TODO: Include `non_speech_segments` in `Passage` metadata. It'd be useful to have this info
    down stream. For example:
    - We could use it to filter out bad alignments
    - We could use it for a better calculation of the speakers speed.
    TODO: Instead of including the start and end of the `passage` by default, we should consider
    looking for pauses slightly before and after the `passage`. This could help ensure that
    speech segments on the boundaries, start and end with a silence, as well.

    Args:
        ...
        nss_timeline: Timeline for looking up non-speech segments (NSS) an interval of audio.
        padding: Seconds to add to either side of the speech segment.
    """
    if len(passage.alignments) == 0:
        return tuple()
    if len(passage.alignments) == 1:
        return (passage.span(slice(0, 1)),)

    start, stop = passage.alignments[0].audio[0], passage.alignments[-1].audio[-1]

    assert nss_timeline is not None
    non_speech_segments = [slice(*s) for s in nss_timeline[start:stop]]
    timeline = Timeline([Interval(a.audio, (i, a)) for i, a in enumerate(passage.alignments)])
    non_speech_segments = list(_filter_non_speech_segments(timeline, non_speech_segments))
    if len(non_speech_segments) == 0:
        return (passage.span(slice(0, len(passage.alignments))),)

    non_speech_segments = [slice(start, start)] + non_speech_segments + [slice(stop, stop)]
    speech_segments: typing.List[Span] = []
    for prev, next_ in zip(non_speech_segments, non_speech_segments[1:]):
        span = timeline[prev.stop : next_.start]
        if len(span) != 0:
            max_length = passage.audio_file.length
            span = passage.span(
                slice(min(i for i, _ in span), max(i for i, _ in span) + 1),
                slice(max(prev.stop - padding, 0.0), min(next_.start + padding, max_length)),
            )
            speech_segments.append(span)

    assert all(a.slice.stop <= b.slice.start for a, b in zip(speech_segments, speech_segments[1:]))
    return tuple(speech_segments)


def _maybe_normalize_vo_script(script: str) -> str:
    """Normalize a script if it's not normalized."""
    if not lib.text.is_normalized_vo_script(script):
        return lib.text.normalize_vo_script(script)
    return script


def _check_updated_script(
    name: str, passage: UnprocessedPassage, updated_script: str, updated_transcript: str
):
    """Check if updated script and transcript is compatible with `passage.alignments`."""
    updates = (
        ("script", passage.script, updated_script),
        ("transcript", passage.transcript, updated_transcript),
    )
    for label, original, updated in updates:
        if passage.alignments is not None and original != updated:
            diff = [o for o, u in zip(original, updated) if o != u]
            lib.utils.call_once(logger.error, f"[{name}] `{label}` was not normalized: {diff}")
            assert len(original) == len(updated), "Alignments and script are out-of-sync."


UnprocessedDataset = typing.List[typing.List[UnprocessedPassage]]


@lib.utils.log_runtime
def make_passages(
    name: str, dataset: UnprocessedDataset, add_tqdm: bool = False, **kwargs
) -> typing.List[Passage]:
    """Process `UnprocessedPassage` and return a list of `Passage`s.

    NOTE: This function processes passages in a batch; therefore, it'd be ideal to pass as many
    items at once as possible.
    TODO: In order to encourage parallelism, the longest files should be run through
    `_maybe_normalize_audio_and_cache` first.

    Args:
        dataset: Dataset with a list of documents each with a list of passsages.
        **kwargs: Keyword arguments passed to `_make_nonalignments`.
    """
    executor = futures.ThreadPoolExecutor()
    tqdm_ = partial(tqdm, disable=not add_tqdm)
    get_audio_metadata_ = partial(get_audio_metadata, add_tqdm=add_tqdm)
    audio_paths = list(set(p.audio_path for l in dataset for p in l))
    audio_paths = [a for a, e in zip(audio_paths, executor.map(_exists, audio_paths)) if e]
    audio_files = get_audio_metadata_(audio_paths)

    logger.info(f"[{name}] Normalizing audio files...")
    iterator = executor.map(maybe_normalize_audio_and_cache, audio_files)
    normal_audio_paths = list(tqdm_(iterator, total=len(audio_files)))
    normal_audio_files = {
        a: n for a, n in zip(audio_paths, get_audio_metadata_(normal_audio_paths))
    }

    logger.info(f"[{name}] Normalizing scripts...")
    scripts = set(s for l in dataset for p in l for s in (p.script, p.transcript))
    normal_scripts = {s: _maybe_normalize_vo_script(s) for s in tqdm_(scripts)}

    logger.info(f"[{name}] Making passages...")
    passages: typing.List[typing.List[Passage]] = [[] for _ in range(len(dataset))]
    iter_: typing.Iterable[typing.Tuple[int, UnprocessedPassage]]
    iter_ = iter(tqdm_([(i, p) for i, d in enumerate(dataset) for p in d]))
    for i, item in iter_:
        if item.audio_path not in normal_audio_files:
            logger.warning(f"[{name}] Skipping, audio path (%s) isn't a file.", item.audio_path)
            continue

        audio_file = normal_audio_files[item.audio_path]
        script = normal_scripts[item.script]
        transcript = normal_scripts[item.transcript]
        _check_updated_script(name, item, script, transcript)
        args = ((0, len(script)), (0.0, audio_file.length), (0, len(transcript)))
        alignments = (Alignment(*args),) if item.alignments is None else item.alignments
        alignments = Alignment.stow(alignments)
        speaker, other_metadata = item.speaker, item.other_metadata
        passage = Passage(audio_file, speaker, script, transcript, alignments, other_metadata)
        passages[i].append(passage)

    logger.info(f"[{name}] Getting non-speech segments...")
    audio_files = list(set(p.audio_file for l in passages for p in l if len(p.alignments) > 1))
    nss_timelines = {a: get_non_speech_segments_and_cache(a) for a in tqdm_(audio_files)}

    logger.info(f"[{name}] Updating passages...")
    for doc in typing.cast(typing.List[typing.List[Passage]], tqdm_(passages)):
        doc_ = typing.cast(typing.List[typing.Optional[Passage]], doc)
        buffer = typing.cast(typing.List[typing.Optional[Passage]], [None])
        for i, (prev, curr, next_) in enumerate(zip(buffer + doc_, doc, doc_[1:] + buffer)):
            object.__setattr__(curr, "prev", prev)
            object.__setattr__(curr, "next", next_)
            object.__setattr__(curr, "nonalignments", _make_nonalignments(doc, i, **kwargs))
            nss_timeline = nss_timelines.get(curr.audio_file, None)
            object.__setattr__(curr, "speech_segments", _make_speech_segments(curr, nss_timeline))
        [passage.check_invariants() for passage in doc]

    logger.info(f"[{name}] Done! {lib.utils.mazel_tov()}")

    return flatten_2d(passages)


class SpanGenerator(typing.Iterator[Span]):
    """Randomly generate `Span`(s) that are at most `max_seconds` long.

    NOTE:
    - Every `Alignment` has an equal chance of getting sampled, assuming there are no overlaps.
    - Larger groups of `Alignment` are less likely to be sampled. Learn more here:
      https://stats.stackexchange.com/questions/484329/how-do-you-uniformly-sample-spans-from-a-bounded-line/484332#484332
      https://www.reddit.com/r/MachineLearning/comments/if8icr/d_how_do_you_sample_spans_uniformly_from_a_time/
      https://pytorch.slack.com/archives/C205U7WAF/p1598227661001700
      https://www.reddit.com/r/math/comments/iftev6/uniformly_sampling_from_timeseries_data/
    - A considerable amount of effort has been put into profiling, and optimizing the performance of
      `SpanGenerator`.

    TODO: A sufficiently large pause would slow this algorithm to a hault because it'd be
    sampled continuously and ignored. We could handle that better by excluding large samples
    from the sampling distribution.

    TODO: Use `lib.utils.corrected_random_choice` in order to ensure a uniform length distribution.

    Args:
        passages
        max_seconds: The maximum interval length.
        **kwargs: Additional key-word arguments passed `Timeline`.
    """

    def __init__(self, passages: typing.List[Passage], max_seconds: float, **kwargs):
        assert max_seconds > 0, "The maximum interval length must be a positive number."
        self.passages = passages
        self.max_seconds = max_seconds
        self._weights = torch.tensor([p.last.audio[1] - p.first.audio[0] for p in self.passages])
        get_interval: typing.Callable[[Span], FloatFloat]
        get_interval = lambda s: (s.audio_start, s.audio_stop)
        make_timeline: typing.Callable[[Passage], Timeline[int]]
        make_timeline = lambda p: Timeline(
            [Interval(get_interval(s), i) for i, s in enumerate(p.speech_segments)], **kwargs
        )
        self._timelines = None if max_seconds == math.inf else [make_timeline(p) for p in passages]

    @staticmethod
    def _overlap(x1: float, x2: float, y1: float, y2: float) -> float:
        """ Get the percentage overlap between x and the y slice. """
        if x2 == x1:
            return 1.0 if x1 >= y1 and x2 <= y2 else 0.0
        return (min(x2, y2) - max(x1, y1)) / (x2 - x1)

    @functools.lru_cache(maxsize=None)
    def _is_include(self, x1: float, x2: float, y1: float, y2: float):
        return self._overlap(x1, x2, y1, y2) >= random.random()

    def __iter__(self) -> typing.Iterator[Span]:
        return self

    def __next__(self) -> Span:
        if len(self.passages) == 0:
            raise StopIteration()

        # NOTE: For a sufficiently large `max_seconds`, the span length tends to be larger than the
        # passage length; therefore, the entire passage tends to be selected every time.
        if self.max_seconds == float("inf"):
            return random.choice(self.passages)[:]

        while True:
            length = random.uniform(0, self.max_seconds)

            # NOTE: The `weight` is based on `start` (i.e. the number of spans)
            # NOTE: For some reason, `torch.multinomial(replacement=True)` is faster by a lot.
            index = int(torch.multinomial(self._weights + length, 1, replacement=True).item())
            passage, timeline = self.passages[index], self._timelines[index]

            # NOTE: Uniformly sample a span of audio.
            start = random.uniform(passage.first.audio[0] - length, passage.last.audio[1])
            stop = min(start + length, passage.last.audio[1])
            start = max(start, passage.first.audio[0])

            # NOTE: Based on the overlap, decide which alignments to include in the span.
            overlapping = timeline.get(slice(start, stop))
            self._is_include.cache_clear()
            _is_include = partial(self._is_include, y1=start, y2=stop)
            begin: typing.Optional[Interval[int]]
            begin = next((i for i in overlapping if _is_include(i.start, i.stop)), None)
            end: typing.Optional[Interval[int]]
            end = next((i for i in reversed(overlapping) if _is_include(i.start, i.stop)), None)
            if (
                (begin is not None and end is not None)
                and end.stop - begin.start > 0
                and end.stop - begin.start <= self.max_seconds
            ):
                segments = passage.speech_segments[begin.val : end.val + 1]
                slice_ = slice(segments[0].slice.start, segments[-1].slice.stop)
                audio_slice = slice(segments[0].audio_start, segments[-1].audio_stop)
                return passage.span(slice_, audio_slice)


"""
Using `guppy3`, the Jan 30th, 2021 dataset looks like...
```
Partition of a set of 2335137 objects. Total size = 408087380 bytes.
 Index  Count   %     Size   % Cumulative  % Kind (class / dict of class)
     0 202667   9 143100604  35 143100604  35 numpy.ndarray
     1 721114  31 116036028  28 259136632  64 str
     2 119866   5 30258136   7 289394768  71 dict (no owner)
     3 101242   4 14578848   4 303973616  74 dict of lib.datasets.utils.Passage
     4  75937   3 14269800   3 318243416  78 list
     5 154112   7 11615920   3 329859336  81 tuple
     6 202484   9  9719232   2 339578568  83 lib.utils.Tuples
     7  43190   2  7721103   2 347299671  85 types.CodeType
     8  66526   3  6918704   2 354218375  87 pathlib.PosixPath
     9  83951   4  6775874   2 360994249  88 bytes
<2006 more rows. Type e.g. '_.more' to view.>
```
NOTE: Without changing the actual data stored, >63% of this datastructure is optimally stored:
`str` and `numpy.ndarray`.
TODO: Explore adding `__slots__` to `Passage` to save space.
TODO: Explore using a `NamedTuple` instead of `dict` for `other_metadata`. That'll help reduce
"dict (no owner)".
TODO: Instead of storing `start`, and `stop` in `numpy.ndarray` we could store `start` and `length`.
And, in order to save space, we could use `np.uint16` for `length` since the `length` should
be smaller than 65535.
TODO: Ensure the `str` is normalized to ASCII, in order to save space, see:
https://rushter.com/blog/python-strings-and-memory/
TODO: We could use `float16` for representing seconds. The number of seconds for a 20-hour audio
file is around 72,000 and the precision of the alignments is 0.1, so it should be okay.
"""


def _temporary_fix_for_transcript_offset(
    transcript: str, alignments: typing.List[typing.List[typing.List[float]]]
) -> typing.List[typing.List[typing.List[float]]]:
    """Temporary fix for a bug in `sync_script_with_audio.py`.

    TODO: Remove after datasets are reprocessed.
    """
    return_ = []
    for alignment_ in alignments:
        alignment = Alignment.from_json(alignment_)
        word = transcript[alignment.transcript[0] : alignment.transcript[1]]
        if word.strip() != word:
            update = (alignment.transcript[0] - 1, alignment.transcript[1] - 1)
            alignment = alignment._replace(transcript=update)
            corrected = transcript[alignment.transcript[0] : alignment.transcript[1]]
            logger.warning("Corrected '%s' to '%s'.", word, corrected)
        return_.append(alignment.to_json())
    return return_


def dataset_loader(
    directory: Path,
    root_directory_name: str,
    gcs_path: str,
    speaker: Speaker,
    alignments_directory_name: str = "alignments",
    alignments_suffix: str = ".json",
    recordings_directory_name: str = "recordings",
    recordings_suffix: str = ".wav",
    scripts_directory_name: str = "scripts",
    scripts_suffix: str = ".csv",
    text_column: str = "Content",
    strict: bool = False,
    add_tqdm: bool = False,
) -> typing.List[Passage]:
    """Load an alignment text-to-speech (TTS) dataset from GCS.

    TODO: Add `-m` when this issue is resolved:
    https://github.com/GoogleCloudPlatform/gsutil/pull/1107
    TODO: It's faster to run `gsutil` without `-m` if the data already exists on disk;
    therefore, if the directory already exists, we should skip multiprocessing.
    TODO: For performance, support batch loading multiple datasets at the same time.

    The structure of the dataset should be:
        - The file structure is similar to:
            {gcs_path}/
            ├── {alignments_directory_name}/  # Alignments between recordings and scripts
            │   ├── audio1.json
            │   └── ...
            ├── {recordings_directory_name}/  # Voice overs
            │   ├── audio1.wav                # NOTE: Most audio file formats are accepted.
            │   └── ...
            └── {scripts_directory_name}/     # Voice over scripts with related metadata
                ├── audio1-script.csv
                └── ...
        - The alignments, recordings, and scripts directory should contain the same number of
          similarly named files.
        - The dataset contain data representing only one speaker.
        - The scripts only contain ASCII characters.

    Args:
        directory: Directory to cache the dataset.
        root_directory_name: Name of the directory inside `directory` to store data.
        gcs_path: The base GCS path storing the data.
        speaker: The speaker represented by this dataset.
        alignments_gcs_path: The name of the alignments directory on GCS.
        alignments_suffix
        recordings_gcs_path: The name of the voice over directory on GCS.
        recordings_suffix
        scripts_gcs_path: The name of the voice over script directory on GCS.
        scripts_suffix
        text_column: The voice over script column in the CSV script files.
        strict: Use `gsutil` to validate the source files.
        add_tqdm
    """
    logger.info("Loading `%s` speech dataset", root_directory_name)

    root = (Path(directory) / root_directory_name).absolute()
    root.mkdir(exist_ok=True)
    names = [alignments_directory_name, recordings_directory_name, scripts_directory_name]
    suffixes = (alignments_suffix, recordings_suffix, scripts_suffix)
    directories = [root / d for d in names]

    files: typing.List[typing.List[Path]] = []
    for directory, suffix in zip(directories, suffixes):
        if strict or not directory.exists():
            directory.mkdir(exist_ok=True)
            command = f"gsutil cp -n {gcs_path}/{directory.name}/*{suffix} {directory}/"
            subprocess.run(command.split(), check=True)
        files_ = [p for p in directory.iterdir() if p.suffix == suffix]
        message = "Expecting an equal number of recording, alignment, and script files."
        assert len(files) == 0 or len(files_) == len(files[-1]), message
        files.append(sorted(files_, key=lambda p: lib.text.numbers_then_natural_keys(p.name)))

    dataset: UnprocessedDataset = []
    iterator = typing.cast(typing.Iterator[typing.Tuple[Path, Path, Path]], zip(*tuple(files)))
    for alignment_path, recording_path, script_path in iterator:
        scripts = pandas.read_csv(str(script_path.absolute()))
        json_ = json.loads(alignment_path.read_text())
        error = f"Each script ({script_path}) must have an alignment ({alignment_path})."
        assert len(scripts) == len(json_["alignments"]), error
        document = []
        for (_, script), alignments in zip(scripts.iterrows(), json_["alignments"]):
            alignments = _temporary_fix_for_transcript_offset(json_["transcript"], alignments)
            passage = UnprocessedPassage(
                audio_path=recording_path,
                speaker=speaker,
                script=typing.cast(str, script[text_column]),
                transcript=json_["transcript"],
                alignments=tuple(Alignment.from_json(a) for a in alignments),
                other_metadata={k: v for k, v in script.items() if k not in (text_column,)},
            )
            document.append(passage)
        dataset.append(document)
    return make_passages(root_directory_name, dataset, add_tqdm, transcript=True, audio=True)


def conventional_dataset_loader(
    directory: Path,
    speaker: Speaker,
    metadata_path_template: str = "{directory}/metadata.csv",
    metadata_audio_column: typing.Union[str, int] = 0,
    metadata_text_column: typing.Union[str, int] = 2,
    metadata_kwargs={"quoting": csv.QUOTE_NONE, "header": None, "delimiter": "|"},
    audio_path_template: str = "{directory}/wavs/{file_name}.wav",
    additional_metadata: typing.Dict = {},
) -> typing.List[UnprocessedPassage]:
    """Load a conventional speech dataset.

    A conventional speech dataset has these invariants:
        - The dataset has already been segmented, and the segments have been audited.
        - The file structure is similar to:
            {directory}/
                metadata.csv
                wavs/
                    audio1.wav
                    audio2.wav
        - The metadata CSV file contains a mapping of audio transcriptions to audio filenames.
        - The dataset contains one speaker.
        - The dataset is stored in a `tar` or `zip` at some url.

    Args:
        directory: Directory the dataset is stored in.
        speaker: The dataset speaker.
        metadata_path_template: A template specifying the location of the metadata file.
        metadata_audio_column: The column name or index with the audio filename.
        metadata_text_column: The column name or index with the audio transcript.
        metadata_kwargs: Keyword arguments passed to `pandas.read_csv` for reading the metadata.
        audio_path_template: A template specifying the location of an audio file.
        additional_metadata: Additional metadata to include along with the returned passages.
    """
    metadata_path = Path(metadata_path_template.format(directory=directory))
    if os.stat(str(metadata_path)).st_size == 0:
        return []
    df = pandas.read_csv(metadata_path, **metadata_kwargs)
    get_audio_path = lambda n: Path(audio_path_template.format(directory=directory, file_name=n))
    handled_columns = [metadata_text_column, metadata_audio_column]
    get_other_metadata = lambda r: {k: v for k, v in r.items() if k not in handled_columns}
    return [
        UnprocessedPassage(
            audio_path=get_audio_path(row[metadata_audio_column]),
            speaker=speaker,
            script=row[metadata_text_column].strip(),
            transcript=row[metadata_text_column].strip(),
            alignments=None,
            other_metadata={**get_other_metadata(row), **additional_metadata},
        )
        for _, row in df.iterrows()
    ]
