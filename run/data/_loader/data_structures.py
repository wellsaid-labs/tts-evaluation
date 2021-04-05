# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import dataclasses
import itertools
import logging
import pathlib
import typing
from concurrent import futures
from dataclasses import field
from functools import partial
from pathlib import Path

import numpy as np
from hparams import HParam, configurable
from tqdm import tqdm

import lib
from lib.audio import AudioMetadata, get_audio_metadata
from lib.utils import Timeline, Tuple, flatten_2d
from run.data import _loader

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
            script=typing.cast(IntInt, tuple(args[0])),
            audio=typing.cast(FloatFloat, tuple(args[1])),
            transcript=typing.cast(IntInt, tuple(args[2])),
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
        return _loader.utils.read_audio(self.audio_file)

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

    @staticmethod
    def _get(alignments: Tuple[Alignment], field: str) -> typing.List[float]:
        """Get the values for `field` in `self.alignments`."""
        return [typing.cast(float, v) for a in alignments for v in getattr(a, field)]

    @staticmethod
    def _no_white_space(s: str) -> bool:
        return s.strip() == s

    def check_invariants(self):
        """Check datastructure invariants."""
        assert hasattr(self, "nonalignments")
        assert hasattr(self, "speech_segments")
        assert hasattr(self, "prev")
        assert hasattr(self, "next")

        assert self.prev is None or self == self.prev.next
        assert self.next is None or self == self.next.prev
        assert len(self.nonalignments) == len(self.alignments) + 1
        assert len(self.alignments) == 0 or len(self.speech_segments) > 0

        # NOTE: `self` must align something.
        fields = Alignment._fields
        if len(self.alignments) == 0:
            get_ = lambda f: getattr(self.nonalignments[0], f)
            assert any(get_(f)[0] != get_(f)[1] for f in fields)
        else:
            get_ = lambda i, f: getattr(self.alignments[i], f)
            assert any(get_(0, f)[0] != get_(-1, f)[1] for f in fields)

        assert all(a.script[0] <= a.script[1] for a in self.alignments)
        assert all(a.audio[0] <= a.audio[1] for a in self.alignments)
        assert all(a.transcript[0] <= a.transcript[1] for a in self.alignments)

        # NOTE: `self.alignments` must not have extra whitespaces on it's edges.
        slices = (self.script[a.script[0] : a.script[1]] for a in self.alignments)
        assert all(self._no_white_space(s) for s in slices)
        slices = (self.transcript[a.transcript[0] : a.transcript[1]] for a in self.alignments)
        assert all(self._no_white_space(s) for s in slices)

        # NOTE: `self.speech_segments` must be sorted.
        pairs = zip(self.speech_segments, self.speech_segments[1:])
        assert all(a.slice.start <= b.slice.start for a, b in pairs)

        for alignments in (self.nonalignments, self.alignments):
            # NOTE: `self.alignments`, and `self.nonalignments` must be sorted.
            pairs = zip(alignments, alignments[1:])
            assert all(a.script[1] <= b.script[0] for a, b in pairs)
            assert all(a.transcript[1] <= b.transcript[0] for a, b in pairs)
            # NOTE: The `audio` alignments may overlap by a little bit, at the edges.
            assert all(a.audio[0] < b.audio[1] for a, b in pairs)

            if len(alignments) != 0:
                dtype = alignment_dtype["audio"]["stop"].type
                max_length = max(dtype(self.audio_file.length), self.audio_file.length)
                assert max(self._get(alignments, "audio")) <= max_length
                assert max(self._get(alignments, "script")) <= len(self.script)
                assert max(self._get(alignments, "transcript")) <= len(self.transcript)
                assert min(self._get(alignments, "audio")) >= 0
                assert min(self._get(alignments, "script")) >= 0
                assert min(self._get(alignments, "transcript")) >= 0

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
        return _loader.utils.read_audio(
            self.passage.audio_file, self.audio_start, self.audio_length
        )

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
        # NOTE: `self.audio_slice_` must partially contain all alignments.
        assert self.audio_slice_ is None or (
            self.audio_slice_.start <= self._first.audio[1]
            and self.audio_slice_.stop >= self._last.audio[0]
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
    return typing.cast(Tuple[Alignment], nonalignments)


def _exists(path: Path) -> bool:
    """ Helper function for `make_passages` that can be easily mocked. """
    return path.exists() and path.is_file()


def _filter_non_speech_segments(
    passage: Passage, timeline: Timeline, non_speech_segments: typing.Iterable[slice]
) -> typing.Iterable[slice]:
    """Filter out `non_speech_segments` which are not in between alignments.

    NOTE: To better understand the various cases, take a look at the unit tests for this function.
    """
    for slice_ in non_speech_segments:
        indicies = list(timeline.indicies(slice_))

        # NOTE: Check if any interval is inside any other interval.
        intervals = [passage.alignments[i].audio for i in indicies]
        permutations = itertools.permutations(intervals + [(slice_.start, slice_.stop)], 2)
        if any(a[0] <= b[0] and b[1] <= a[1] for a, b in permutations):
            continue

        # NOTE: Check if any alignment intervals overlap.
        if any(sum(max(a[0], b[0]) < min(a[1], b[1]) for b in intervals) > 1 for a in intervals):
            continue

        assert len(indicies) < 3, "It should be impossible to overlap three alignments."
        message = "Alignments should be back-to-back."
        assert len(indicies) != 2 or abs(indicies[0] - indicies[1]) == 1, message

        yield slice_


@configurable
def _make_speech_segments(
    passage: Passage, nss_timeline: typing.Optional[Timeline], pad: float = HParam()
) -> typing.Tuple[Span, ...]:
    """Make a list of `Span`s that start and end with silence.

    TODO: Include `non_speech_segments` in `Passage` metadata. It'd be useful to have this info
    down stream. For example:
    - We could use it to filter out bad alignments
    - We could use it for a better calculation of the speakers speed.
    TODO: Instead of including the start and end of the `passage` by default, we should consider
    looking for pauses slightly before and after the `passage`. This could help ensure that
    speech segments on the boundaries, start and end with a silence, as well.
    - Instead of including start and end, use `passage.prev` and `passage.next` to the get
    the previous and next alignments. We can use those alignments as boundaries for finding
    pauses.

    Args:
        ...
        nss_timeline: Timeline for looking up non-speech segments (NSS) an interval of audio.
        pad: Seconds to add to either side of the speech segment.
    """
    if len(passage.alignments) == 0:
        return tuple()
    if len(passage.alignments) == 1:
        return (passage.span(slice(0, 1)),)

    start, stop = passage.alignments[0].audio[0], passage.alignments[-1].audio[-1]

    assert nss_timeline is not None
    non_speech_segments = [slice(s[0], s[1]) for s in nss_timeline[start:stop]]
    timeline = Timeline([a.audio for a in passage.alignments])
    non_speech_segments = list(_filter_non_speech_segments(passage, timeline, non_speech_segments))
    if len(non_speech_segments) == 0:
        return (passage.span(slice(0, len(passage.alignments))),)

    clamp = partial(lib.utils.clamp, min_=start, max_=stop)
    non_speech_segments = [slice(clamp(s.start), clamp(s.stop)) for s in non_speech_segments]
    non_speech_segments = [slice(start, start)] + non_speech_segments + [slice(stop, stop)]
    speech_segments: typing.List[Span] = []
    for prev, next_ in zip(non_speech_segments, non_speech_segments[1:]):
        if prev.stop > next_.start:  # NOTE: In rare cases, the pauses may overlap.
            continue

        indicies = list(timeline.indicies(slice(prev.stop, next_.start)))
        if (
            len(indicies) != 0
            # NOTE: The pauses must contain all the alignments fully, not just partially.
            and prev.start <= passage.alignments[indicies[0]].audio[0]
            and next_.stop >= passage.alignments[indicies[-1]].audio[1]
        ):
            max_length = passage.audio_file.length
            audio_slice = slice(max(prev.stop - pad, 0.0), min(next_.start + pad, max_length))
            speech_segments.append(passage.span(slice(indicies[0], indicies[-1] + 1), audio_slice))

    assert all(a.slice.stop <= b.slice.start for a, b in zip(speech_segments, speech_segments[1:]))
    return tuple(speech_segments)


def _maybe_normalize_vo_script(script: str) -> str:
    """Normalize a script if it's not normalized."""
    if not lib.text.is_normalized_vo_script(script):
        return lib.text.normalize_vo_script(script)
    return script


def _check_updated_script_helper(name: str, label: str, original: str, updated: str):
    diff = [o for o, u in zip(original, updated) if o != u]
    lib.utils.call_once(logger.error, f"[{name}] `{label}` was not normalized: {diff}")
    assert len(original) == len(updated), "Alignments and script are out-of-sync."


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
            lib.utils.call_once(_check_updated_script_helper, name, label, original, updated)


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
    iterator = executor.map(_loader.utils.maybe_normalize_audio_and_cache, audio_files)
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
        alignments = typing.cast(Tuple[Alignment], alignments)
        speaker, other_metadata = item.speaker, item.other_metadata
        passage = Passage(audio_file, speaker, script, transcript, alignments, other_metadata)
        passages[i].append(passage)

    logger.info(f"[{name}] Getting non-speech segments...")
    audio_files = list(set(p.audio_file for l in passages for p in l if len(p.alignments) > 1))
    nss_timelines = {
        a: _loader.utils.get_non_speech_segments_and_cache(a) for a in tqdm_(audio_files)
    }

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

    logger.info(f"[{name}] Checking invariants and packing...")
    for doc in tqdm_(passages):
        for passage in doc:
            passage.check_invariants()
            object.__setattr__(passage, "alignments", Alignment.stow(passage.alignments))
            object.__setattr__(passage, "nonalignments", Alignment.stow(passage.nonalignments))

    logger.info(f"[{name}] Done! {lib.utils.mazel_tov()}")

    return flatten_2d(passages)
