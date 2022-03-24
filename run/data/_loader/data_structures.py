# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import dataclasses
import itertools
import logging
import typing
from concurrent import futures
from dataclasses import field
from enum import Enum
from functools import partial
from pathlib import Path

import config as cf
import numpy as np
import spacy.tokens.doc

import lib
import run
from lib.audio import AudioMetadata, get_audio_metadata
from lib.utils import Timeline, Tuple, flatten_2d, tqdm_
from run.data import _loader

logger = logging.getLogger(__name__)

FloatFloat = typing.Tuple[float, float]
IntInt = typing.Tuple[int, int]
Slice = slice  # NOTE: `pylance` is buggy if we use `slice` directly for typing.


def get_nlp():
    return lib.text.load_en_core_web_md(disable=("parser", "ner", "tagger"))


class NonalignmentSpans(typing.NamedTuple):
    """Nonalignments for a `Passage` or `Span`.

    Args:
        passage: The passage for `spans`.
        spans: Nonalignments from the current passage, or span.
        prev: The last nonalignment from the previous passage.
        next: The first nonalignment from the next passage.
    """

    passage: Passage
    spans: typing.List[Span]
    prev: typing.Optional[Span]
    next: typing.Optional[Span]


def voiced_nonalignment_spans(
    span: typing.Union[Passage, Span]
) -> typing.Tuple[NonalignmentSpans, typing.List[bool]]:
    """In addition to `NonalignmentSpans`, this returns if a nonalignment should be voiced,
    probably.

    NOTE: This assumes that the `transcript` and `script` from the previous and next passage
    affect mistranscriptions in the current passage if their `transcript`s are linked.
    """
    spans = span.nonalignment_spans()
    text = [s.script + s.transcript for s in spans.spans]
    if spans.passage.is_linked.transcript:
        text[0] += "" if spans.prev is None else spans.prev.script + spans.prev.transcript
        text[-1] += "" if spans.next is None else spans.next.script + spans.next.transcript
    return spans, [run._config.is_voiced(t, span.speaker.language) for t in text]


def _is_alignment_voiced(passage: Passage, alignment: Alignment) -> bool:
    """Return `True` if a `passage` voiced at `alignment`."""
    for attr in ("script", "transcript"):
        interval = getattr(alignment, attr)
        if interval[0] < interval[-1] and run._config.is_voiced(
            getattr(passage, attr)[interval[0] : interval[-1]], passage.speaker.language
        ):
            return True
    return False


def has_a_mistranscription(span: typing.Union[Passage, Span]) -> bool:
    """Return `True` if `span` contains a mistranscription, probably.

    NOTE: This is equivalent and ~3x faster than: `any(voiced_nonalignment_spans(span)[1])`
    """
    # NOTE: Use duck typing because of this issue:
    # https://github.com/streamlit/streamlit/issues/2379
    is_span = hasattr(span, "passage")
    if is_span:
        span = typing.cast(Span, span)
        slice_ = span.nonalignments_slice
        span = span.passage
    else:
        span = typing.cast(Passage, span)
        slice_ = slice(0, len(span.nonalignments))

    slices = [(span, slice_)]
    is_linked = span.is_linked.transcript
    if is_linked and span.prev is not None and slice_.start == 0:
        slices.append((span.prev, slice(-1, None)))
    if is_linked and span.next is not None and slice_.stop == len(span.nonalignments):
        slices.append((span.next, slice(0, 1)))

    return any(_is_alignment_voiced(p, a) for p, s in slices for a in p.nonalignments[s])


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


class Language(Enum):
    ENGLISH: typing.Final = "English"
    GERMAN: typing.Final = "German"
    PORTUGUESE_BR: typing.Final = "Portuguese"
    SPANISH_CO: typing.Final = "Spanish"


class Speaker(typing.NamedTuple):
    # TODO: Move style, language, post processing attributes to `Passage` because the same speaker
    # could have multiple styles, languages, and filters applied.
    label: str
    language: Language
    name: typing.Optional[str] = None
    gender: typing.Optional[str] = None


make_en_speaker = lambda label, *a, **kw: Speaker(label, Language.ENGLISH, *a, **kw)
make_de_speaker = lambda label, *a, **kw: Speaker(label, Language.GERMAN, *a, **kw)
make_es_speaker = lambda label, *a, **kw: Speaker(label, Language.SPANISH_CO, *a, **kw)
make_pt_speaker = lambda label, *a, **kw: Speaker(label, Language.PORTUGUESE_BR, *a, **kw)

# TODO: Implement `__str__` so that we have a more succinct string representation for logging
Session = typing.NewType("Session", typing.Tuple[Speaker, str])


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

    audio_path: Path
    speaker: Speaker
    script: str
    transcript: str
    alignments: typing.Optional[typing.Tuple[Alignment, ...]] = None
    other_metadata: typing.Dict = field(default_factory=dict)


class IsLinked(typing.NamedTuple):
    script: bool = False
    transcript: bool = False
    audio: bool = False


@dataclasses.dataclass(frozen=True)
class Passage:
    """A voiced passage.

    TODO: Create a `ConventionalPassage` or `ConventionalSpan` for storing tens of thousands
    of single alignment pre-cut spans. We could more efficiently and accurately handle script
    and audio updates. We could more efficiently create a `ConventionalSpan`.

    Args:
        audio_file: A voice-over of the `script`.
        session: A label used to group passages recorded together.
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
    session: Session
    script: str
    transcript: str
    alignments: Tuple[Alignment]
    other_metadata: typing.Dict = field(default_factory=dict, compare=False)
    is_linked: IsLinked = field(default_factory=IsLinked)
    nonalignments: Tuple[Alignment] = field(init=False, repr=False, compare=False)
    speech_segments: typing.Tuple[Span, ...] = field(init=False, repr=False, compare=False)
    non_speech_segments: Timeline = field(init=False, repr=False, compare=False)
    first: Alignment = field(init=False, repr=False, compare=False)
    last: Alignment = field(init=False, repr=False, compare=False)
    passages: typing.List[Passage] = field(init=False, repr=False, compare=False)
    index: int = field(init=False, repr=False, compare=False)
    _doc: spacy.tokens.doc.Doc = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.alignments) > 0:  # NOTE: Cache `first` and `last`, if they exist.
            object.__setattr__(self, "first", self.alignments[0])
            object.__setattr__(self, "last", self.alignments[-1])

        # NOTE: Error if `dataclasses.replace` is run on a linked `Passage`.
        if hasattr(self, "passages"):
            assert self is self.passages[self.index]

        assert not hasattr(self, "_doc")

    @property
    def prev(self):
        # NOTE: Our linked list uses a underlying `list` because recursive data structures are hard
        # to pickle, learn more:
        # https://stackoverflow.com/questions/2912841/python-pickling-highly-recursive-objects-without-using-setrecursionlimit
        return None if self.index == 0 else self.passages[self.index - 1]

    @property
    def next(self):
        return None if self.index == len(self.passages) - 1 else self.passages[self.index + 1]

    @property
    def audio_start(self) -> float:
        return self.speech_segments[0].audio_start

    @property
    def audio_stop(self) -> float:
        return self.speech_segments[-1].audio_stop

    @property
    def speaker(self):
        return self.session[0]

    @property
    def doc(self) -> spacy.tokens.doc.Doc:
        if hasattr(self, "_doc"):
            return self._doc
        # NOTE: For performance, process all `self.passages` together, and cache the results.
        docs = get_nlp().pipe(s.script for s in self.passages)
        for passage, doc in zip(self.passages, docs):
            object.__setattr__(passage, "_doc", doc)
        return self._doc

    def __getstate__(self):
        # NOTE: Delete temporary `_doc`, learn more here:  https://spacy.io/usage/saving-loading
        state = self.__dict__.copy()
        if hasattr(state, "_doc"):
            del state["_doc"]
        return state

    def audio(self):
        return _loader.utils.read_audio(self.audio_file)

    def segmented_audio_length(self) -> float:
        return self.audio_stop - self.audio_start

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
            self,
            [self.nonalignment_span(cut(i)) for i in range(len(self.nonalignments))],
            None if prev is None else prev.nonalignment_span(cut(len(prev.nonalignments) - 1)),
            None if self.next is None else self.next.nonalignment_span(cut(0)),
        )

    def _first_alignment(self) -> Alignment:
        """Get the zero length first alignment iff this is the first `Passage`."""
        return Alignment(script=(0, 0), audio=(0.0, 0.0), transcript=(0, 0))

    def _last_alignment(self) -> Alignment:
        """Get the zero length last alignment iff this is the last `Passage`."""
        return Alignment(
            script=(len(self.script), len(self.script)),
            audio=(self.audio_file.length, self.audio_file.length),
            transcript=(len(self.transcript), len(self.transcript)),
        )

    def _merge(self, a: Alignment, b: Alignment) -> Alignment:
        """Merge alignments `a` and `b` iff they are linked."""
        if all(not getattr(self.is_linked, f) for f in IsLinked._fields):
            return a

        return a._replace(
            script=(b if self.is_linked.script else a).script,
            transcript=(b if self.is_linked.transcript else a).transcript,
            audio=(b if self.is_linked.audio else a).audio,
        )

    def _prev_alignment(self) -> Alignment:
        """Get an alignment representing a left boundary.

        NOTE: The aligned `transcript`, `script`, and `audio` may not match due to the constraints
        of `is_linked`.
        """
        if self.prev is None:
            return self._first_alignment()
        prev = self.prev
        alignment = prev.alignments[-1] if len(prev.alignments) > 0 else prev._prev_alignment()
        return self._merge(self._first_alignment(), alignment)

    def _next_alignment(self) -> Alignment:
        """Get an alignment representing a right boundary.

        NOTE: The aligned `transcript`, `script`, and `audio` may not match due to the constraints
        of `is_linked`.
        """
        if self.next is None:
            return self._last_alignment()
        next = self.next
        alignment = next.alignments[0] if len(next.alignments) > 0 else next._next_alignment()
        return self._merge(self._last_alignment(), alignment)

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
        assert hasattr(self, "passages")
        assert hasattr(self, "index")

        assert self is self.passages[self.index]
        assert self.prev is None or self == self.prev.next
        assert self.next is None or self == self.next.prev
        assert self.prev is None or self.is_linked == self.prev.is_linked
        assert self.next is None or self.is_linked == self.next.is_linked
        assert len(self.nonalignments) == len(self.alignments) + 1

        # NOTE: `self` must align some `transcript` or `script`.
        fields = ("transcript", "script")
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
    def audio_start(self) -> float:
        """Start of audio span in `self.audio_file`."""
        return self._first.audio[0] if self.audio_slice_ is None else self.audio_slice_.start

    @property
    def audio_stop(self) -> float:
        """End of audio span in `self.audio_file`."""
        return self._last.audio[-1] if self.audio_slice_ is None else self.audio_slice_.stop

    @property
    def speaker(self):
        return self.passage.session[0]

    @property
    def session(self) -> Session:
        return self.passage.session

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
        """Slice of audio in `self.audio_file`."""
        return slice(self.audio_start, self.audio_stop)

    @property
    def transcript_slice(self):
        """Slice of transcript in `self.passage.transcript`."""
        return slice(self._first.transcript[0], self._last.transcript[-1])

    @property
    def nonalignments_slice(self):
        """Slice of `Alignment`s in `self.passage.nonalignments`."""
        assert self.passage_alignments is self.passage.alignments
        return slice(self.slice.start, self.slice.stop + 1)

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

    @property
    def spacy(self) -> spacy.tokens.span.Span:
        span = self.passage.doc.char_span(self.script_slice.start, self.script_slice.stop)
        assert span is not None, "Invalid `spacy.tokens.Span` selected."
        return span

    def spacy_with_context(self, max_words: int) -> spacy.tokens.span.Span:
        """Get a `spacy.tokens.span.Span` with the required context for voicing `self`."""
        doc = self.passage.doc
        end = min(self.spacy.end + max_words, len(doc))
        return doc[max(self.spacy.start - max_words, 0) : end]

    def audio(self) -> np.ndarray:
        return _loader.utils.read_audio(
            self.passage.audio_file, self.audio_start, self.audio_length
        )

    def nonalignment_spans(self) -> NonalignmentSpans:
        """See `self.passage.nonalignment_spans()` docs."""
        assert self.passage_alignments is self.passage.alignments
        spans = self.passage.nonalignment_spans()
        return NonalignmentSpans(
            passage=self.passage,
            spans=spans.spans[self.nonalignments_slice],
            prev=spans.prev if self.slice.start == 0 else None,
            next=spans.next if self.slice.stop == len(self.passage_alignments) else None,
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
        """Check datastructure invariants."""
        self.passage.check_invariants()
        assert (
            self.passage_alignments is self.passage.nonalignments
            or self.passage_alignments is self.passage.alignments
        )
        assert self.slice.stop > self.slice.start, "`Span` must have `Alignments`."
        assert self.slice.stop <= len(self.passage_alignments) and self.slice.stop >= 0
        assert self.slice.start < len(self.passage_alignments) and self.slice.start >= 0
        # NOTE: `self.audio_slice_` must partially contain all alignments.
        assert self.audio_slice_ is None or (
            self.audio_slice_.stop > self.audio_slice_.start
            and self.audio_slice_.start <= self._first.audio[1]
            and self.audio_slice_.stop >= self._last.audio[0]
        )
        return self


def _make_nonalignments(passage: Passage) -> Tuple[Alignment]:
    """Get nonalignments in between `data.alignments`, and in between
    `[prev.alignments[-1], data.alignments, next.alignments[0]]`.
    """
    alignments = [passage._prev_alignment()] + list(passage.alignments)
    alignments += [passage._next_alignment()]
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
    """Helper function for `make_passages` that can be easily mocked."""
    return path.exists() and path.is_file()


def _filter_non_speech_segments(
    alignments: typing.List[FloatFloat],
    alignments_timeline: Timeline,
    non_speech_segments: typing.Iterable[slice],
) -> typing.Iterable[slice]:
    """Filter out `non_speech_segments` which are not in between alignments.

    NOTE: To better understand the various cases, take a look at the unit tests for this function.
    """
    for slice_ in non_speech_segments:
        indicies = list(alignments_timeline.indicies(slice_))

        if len(indicies) >= 3:
            continue

        # NOTE: Check if any interval is inside any other interval.
        intervals = [alignments[i] for i in indicies]
        permutations = itertools.permutations(intervals + [(slice_.start, slice_.stop)], 2)
        if any(
            (a[0] < b[0] and b[1] < a[1]) or (a[0] == b[0] and b[1] == a[1])
            for a, b in permutations
        ):
            continue

        # NOTE: Check if any alignment intervals overlap.
        if any(sum(max(a[0], b[0]) < min(a[1], b[1]) for b in intervals) > 1 for a in intervals):
            continue

        message = "Alignments should be back-to-back."
        assert len(indicies) != 2 or abs(indicies[0] - indicies[1]) == 1, message

        yield slice_


def _make_speech_segments_helper(
    alignments: typing.List[FloatFloat],
    prev_alignment: FloatFloat,
    next_alignment: FloatFloat,
    max_length: float,
    nss_timeline: Timeline,
    pad: float,
) -> typing.Tuple[typing.Tuple[slice, ...], ...]:
    """Make a list of `Span`s that start and end with silence.

    NOTE: Long alignments (more than 1s) can include a pause. Due to this, there may be a pause
    inside a speech segment.

    Args:
        ...
        nss_timeline: Timeline for looking up non-speech segments (NSS) an interval of audio.
        pad: Seconds to add to either side of the speech segment.
    """
    if len(alignments) == 0:
        return tuple()

    alignments_ = [prev_alignment] + alignments + [next_alignment]
    start, stop = alignments_[0][0], alignments_[-1][-1]
    assert stop <= max_length
    alignments_timeline = Timeline(alignments_)

    nss = [slice(s[0], s[1]) for s in nss_timeline[start:stop]]
    nss = list(_filter_non_speech_segments(alignments_, alignments_timeline, nss))
    if len(nss) == 0:
        return tuple()

    speech_segments: typing.List[typing.Tuple[slice, ...]] = []
    pairs = [i for i in zip(nss, nss[1:]) if i[0].stop <= i[1].start]  # NOTE: Pauses may overlap.
    for a, b in pairs:
        idx = list(alignments_timeline.indicies(slice(a.stop, b.start)))
        if (
            len(idx) != 0
            # NOTE: The pauses must contain all the alignments fully, not just partially.
            and (a.start <= alignments_[idx[0]][0] or b.stop >= alignments_[idx[-1]][1])
            # NOTE: The speech segment must only contain `alignments`.
            and (0 not in idx and len(alignments_) - 1 not in idx)
        ):
            idx = [i - 1 for i in idx]
            audio_slice = slice(max(a.stop - pad, 0.0), min(b.start + pad, max_length))
            speech_segments.append((slice(idx[0], idx[-1] + 1), audio_slice))

    return tuple(speech_segments)


def _normalize_non_standard_characters(text: str) -> str:
    """Google STT transcripts occasionally utilizes unexpected symbols that haven't been normalized.
    If the transcripts are normalized during training, the alignments get out of sync due to the
    normalization (coming from `unidecode`) using more characters than the nonstandard character.
        Examples: ° would be normalized to deg
                  € would be normalized to eur

    TODO: Normalize Google STT transcripts before syncing and aligning.
    TODO: Remove this function once all datasets have been preprocessed on latest code.
    """
    text = text.replace("¹", "'")
    text = text.replace("°", "d")
    text = text.replace("€", "e")
    return text


def _maybe_normalize_vo_script(script: str, language: Language) -> str:
    """Normalize a script if it's not normalized."""
    script = _normalize_non_standard_characters(script)
    if not run._config.is_normalized_vo_script(script, language):
        return run._config.normalize_vo_script(script, language)
    return script


def _check_updated_script_helper(label: str, attr: str, original: str, updated: str):
    """Helper function for `_check_updated_script`."""
    diff = [o for o, u in zip(original, updated) if o != u]
    lib.utils.call_once(logger.error, f"[{label}] `{attr}` was not normalized: {diff}")
    assert len(original) == len(updated), "Alignments and script are out-of-sync."


def _check_updated_script(
    label: str, passage: UnprocessedPassage, updated_script: str, updated_transcript: str
):
    """Check if updated script and transcript is compatible with `passage.alignments`."""
    updates = (
        ("script", passage.script, updated_script),
        ("transcript", passage.transcript, updated_transcript),
    )
    for attr, original, updated in updates:
        if passage.alignments is not None and original != updated:
            lib.utils.call_once(_check_updated_script_helper, label, attr, original, updated)


UnprocessedDataset = typing.List[typing.List[UnprocessedPassage]]


def _normalize_audio_files(
    dataset: UnprocessedDataset, no_tqdm: bool
) -> typing.Dict[Path, AudioMetadata]:
    """Map every audio file to a normalized audio file.

    TODO: In order to encourage parallelism, the longest files should be run through
    `maybe_normalize_audio_and_cache` first.
    """
    executor = futures.ThreadPoolExecutor()
    get_audio_metadata_ = partial(get_audio_metadata, add_tqdm=not no_tqdm)
    audio_paths = list(set(p.audio_path for l in dataset for p in l))
    audio_paths = [a for a, e in zip(audio_paths, executor.map(_exists, audio_paths)) if e]
    audio_files = get_audio_metadata_(audio_paths)
    maybe_norm = cf.partial(_loader.utils.maybe_normalize_audio_and_cache)
    generator = executor.map(maybe_norm, audio_files)
    normal_audio_paths = list(tqdm_(generator, total=len(audio_files), disable=no_tqdm))
    return {a: n for a, n in zip(audio_paths, get_audio_metadata_(normal_audio_paths))}


def _get_non_speech_segments(
    dataset: UnprocessedDataset,
    normalized_audio_files: typing.Dict[Path, AudioMetadata],
    no_tqdm: bool,
    threshold: int = 5 * 60,
) -> typing.Dict[AudioMetadata, Timeline]:
    """Map every audio file to a non speech segments `Timeline`.

    Args:
        ...
        threshold: If all `audio_file`s are smaller than `threshold`, use multithreading.
            This is a naive method for controlling memory usage.
    """
    audio_paths = list(set(p.audio_path for l in dataset for p in l))
    audio_files = [normalized_audio_files[p] for p in audio_paths if p in normalized_audio_files]
    get_nss_and_cache = cf.partial(_loader.utils.get_non_speech_segments_and_cache)
    if max([f.length for f in audio_files]) > threshold:
        iterator = tqdm_(audio_files, disable=no_tqdm)
        return {a: get_nss_and_cache(a) for a in iterator}

    executor = futures.ThreadPoolExecutor()
    generator = executor.map(get_nss_and_cache, audio_files)
    non_speech_segments = list(tqdm_(generator, total=len(audio_files), disable=no_tqdm))
    return {a: n for a, n in zip(audio_files, non_speech_segments)}


def _normalize_scripts(
    label: str, dataset: UnprocessedDataset, no_tqdm: bool
) -> UnprocessedDataset:
    """Normalize `transcript` and `script` in `dataset`.

    TODO: Support `Passage`s with no content. At the moment `Passage` requires some content.
    """
    scripts = set(
        (script, passage.speaker.language)
        for document in dataset
        for passage in document
        for script in (passage.script, passage.transcript)
    )
    new_scripts = {
        (script, language): _maybe_normalize_vo_script(script, language)
        for script, language in tqdm_(scripts, disable=no_tqdm)
    }
    new_dataset: UnprocessedDataset = [[] for _ in range(len(dataset))]
    iterator = tqdm_([(p, n) for d, n in zip(dataset, new_dataset) for p in d], disable=no_tqdm)
    for passage, new_document in iterator:
        if len(passage.script) == 0 and len(passage.transcript) == 0:
            message = f"[{label}] Skipping, passage ({passage.audio_path.name}) has no content."
            logger.error(message)
            continue
        script = new_scripts[(passage.script, passage.speaker.language)]
        transcript = new_scripts[(passage.transcript, passage.speaker.language)]
        _check_updated_script(label, passage, script, transcript)
        new_document.append(dataclasses.replace(passage, script=script, transcript=transcript))
    return new_dataset


def _normalize_alignments(
    passage: UnprocessedPassage, audio_file: AudioMetadata
) -> Tuple[Alignment]:
    """Normalize `passage.alignments`."""
    alignments = passage.alignments
    args = ((0, len(passage.script)), (0.0, audio_file.length), (0, len(passage.transcript)))
    default_alignments = (Alignment(*args),)
    return typing.cast(Tuple[Alignment], default_alignments if alignments is None else alignments)


def _make_speech_segments(passage: Passage) -> typing.List[Span]:
    """Make `speech_segments` for `passage`."""
    length = passage.audio_file.length
    if len(passage.alignments) == 1 and passage.alignments[0].audio == (0, length):
        return [passage[:]]
    speech_segments = _make_speech_segments_helper(
        [a.audio for a in passage.alignments],
        passage._prev_alignment().audio,
        passage._next_alignment().audio,
        passage.audio_file.length,
        passage.non_speech_segments,
        **cf.get(),
    )
    return [passage.span(*s) for s in speech_segments]


def _default_session(passage: UnprocessedPassage) -> Session:
    """By default, this assumes that each audio file was recorded, individually."""
    return Session((passage.speaker, passage.audio_path.stem))


@lib.utils.log_runtime
def make_passages(
    label: str,
    dataset: UnprocessedDataset,
    add_tqdm: bool = False,
    get_session: typing.Callable[[UnprocessedPassage], Session] = _default_session,
    **kwargs,
) -> typing.List[Passage]:
    """Process `UnprocessedPassage` and return a list of `Passage`s.

    NOTE: This function processes passages in a batch; therefore, it'd be ideal to pass as many
    items at once as possible.
    TODO: Add `check_invariants` to `UnprocessedPassage`, so that, we can enforce invariants
    that this code relies on.
    TODO: Load the correct spaCy model based on language.

    Args:
        dataset: Dataset with a list of documents each with a list of passsages.
        **kwargs: Keyword arguments passed to `Passage`.
    """
    no_tqdm = not add_tqdm
    tqdm = partial(tqdm_, disable=no_tqdm)

    logger.info(f"[{label}] Normalizing audio files...")
    normalized_audio_files = _normalize_audio_files(dataset, no_tqdm)

    logger.info(f"[{label}] Getting non-speech segments...")
    non_speech_segments = _get_non_speech_segments(dataset, normalized_audio_files, no_tqdm)

    logger.info(f"[{label}] Normalizing scripts...")
    dataset = _normalize_scripts(label, dataset, no_tqdm)

    logger.info(f"[{label}] Making passages...")
    documents: typing.List[typing.List[Passage]] = [[] for _ in range(len(dataset))]
    iterator = tqdm([(i, p) for i, d in enumerate(dataset) for p in d])
    for i, item in iterator:
        if item.audio_path not in normalized_audio_files:
            logger.warning(f"[{label}] Skipping, audio path ({item.audio_path.name}) isn't a file.")
            continue
        kwargs = {**kwargs, "transcript": item.transcript, "other_metadata": item.other_metadata}
        kwargs = {**kwargs, "session": get_session(item), "script": item.script}
        audio_file = normalized_audio_files[item.audio_path]
        alignments = _normalize_alignments(item, audio_file)
        documents[i].append(Passage(audio_file=audio_file, alignments=alignments, **kwargs))

    logger.info(f"[{label}] Linking passages...")
    for doc in tqdm(documents):
        for i, passage in enumerate(doc):
            object.__setattr__(passage, "passages", doc)
            object.__setattr__(passage, "index", i)

    flat = flatten_2d(documents)
    logger.info(f"[{label}] Making nonalignments and speech segments...")
    for passage in tqdm(flat):
        object.__setattr__(passage, "nonalignments", _make_nonalignments(passage))
        object.__setattr__(passage, "non_speech_segments", non_speech_segments[passage.audio_file])
        object.__setattr__(passage, "speech_segments", _make_speech_segments(passage))

    logger.info(f"[{label}] Checking invariants and packing...")
    for passage in tqdm(flat):
        passage.check_invariants()
        object.__setattr__(passage, "alignments", Alignment.stow(passage.alignments))
        object.__setattr__(passage, "nonalignments", Alignment.stow(passage.nonalignments))
        for span in passage.speech_segments:
            object.__setattr__(span, "passage_alignments", passage.alignments)

    logger.info(f"[{label}] Done! {lib.utils.mazel_tov()}")
    return flat
