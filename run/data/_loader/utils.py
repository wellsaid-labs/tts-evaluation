# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import csv
import dataclasses
import functools
import json
import logging
import math
import multiprocessing
import os
import pathlib
import random
import subprocess
import typing
from functools import partial
from pathlib import Path

import numpy as np
import torch
from third_party import LazyLoader

import lib
from lib.audio import AudioMetadata, get_audio_metadata
from lib.utils import Interval, Timeline, Tuple, flatten_2d

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")


logger = logging.getLogger(__name__)


def read_audio(audio_file: AudioMetadata, *args, **kwargs) -> np.ndarray:
    """Read `audio_file` into a `np.float32` array."""
    try:
        assert audio_file.encoding == AudioEncoding.PCM_FLOAT_32_BIT
        audio = lib.audio.read_wave_audio(audio_file, *args, **kwargs)
    except AssertionError:
        audio = lib.audio.read_audio(audio_file.path, *args, **kwargs)
    assert audio.dtype == np.float32, "Invariant failed. Audio `dtype` must be `np.float32`."
    return audio




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

    script: typing.Tuple[int, int]
    audio: typing.Tuple[float, float]
    transcript: typing.Tuple[int, int]

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
    other_metadata: typing.Dict = dataclasses.field(default_factory=dict)


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
        other_metadata: Additional metadata associated with this passage.
    """

    audio_file: AudioMetadata
    speaker: Speaker
    script: str
    transcript: str
    alignments: Tuple[Alignment]
    nonalignments: Tuple[Alignment] = dataclasses.field(compare=False, hash=False)
    other_metadata: typing.Dict = dataclasses.field(default_factory=dict, compare=False, hash=False)
    first: Alignment = dataclasses.field(init=False, repr=False, compare=False)
    last: Alignment = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.alignments) > 0:  # NOTE: Cache `first` and `last`, if they exist.
            object.__setattr__(self, "first", self.alignments[0])
            object.__setattr__(self, "last", self.alignments[-1])

    def audio(self):
        return read_audio(self.audio_file)

    def aligned_audio_length(self) -> float:
        return self.last.audio[-1] - self.first.audio[0]

    def spans(self, alignments: Tuple[Alignment]) -> typing.Iterable[Span]:
        """ Get `Span`s for every `Alignment` in `alignments`. """
        return (Span(self, alignments, slice(i + 1)) for i in range(len(alignments)))

    def nonalignment_spans(self) -> typing.Iterable[Span]:
        """ Get a `Span` for every `nonalignment`. """
        return self.spans(self.nonalignments)

    def __getitem__(self, key) -> Span:
        if isinstance(key, int):
            key = len(self.alignments) + key if key < 0 else key
            key = slice(key, key + 1)
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self.alignments))
            assert step == 1, f"Step size {step} is not supported."
            return Span(self, self.alignments, slice(start, stop))
        else:
            raise TypeError("Invalid argument type: {}".format(type(key)))

    def _get(self, field: str) -> typing.List[float]:
        """Get the values for `field` in `self.alignments`."""
        return [typing.cast(float, v) for a in self.alignments for v in getattr(a, field)]

    @staticmethod
    def _no_white_space(s: str) -> bool:
        return s.strip() == s

    def check_invariants(self, eps=1e-6):
        """Check datastructure invariants."""
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
        assert all(a.audio[0] <= b.audio[1] for a, b in pairs)

        if len(self.alignments) != 0:
            # NOTE: `eps` allows for minor rounding errors.
            assert max(self._get("audio")) <= self.audio_file.length + eps
            assert max(self._get("script")) <= len(self.script)
            assert max(self._get("transcript")) <= len(self.transcript)
            assert min(self._get("audio")) >= 0
            assert min(self._get("script")) >= 0
            assert min(self._get("transcript")) >= 0


SpanType = typing.TypeVar("SpanType", bound="Span")
Slice = slice


@dataclasses.dataclass(frozen=True)
class Span:
    """A span of the voiced passage.

    NOTE: The first and last `Alignment`s are cached for performance. The goal is to avoid
    accessing `self.passage_alignments` due to it's mediocre performance.

    Args:
        passage: The original passage, for context.
        passage_alignments: The original passage alignments, for context.
        slice: A `slice` of `passage.alignments`.
    """

    passage: Passage
    passage_alignments: Tuple[Alignment]
    slice: Slice
    _first_cache: Alignment = dataclasses.field(init=False, repr=False, compare=False)
    _last_cache: Alignment = dataclasses.field(init=False, repr=False, compare=False)

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
    def speaker(self):
        return self.passage.speaker

    @property
    def audio_file(self):
        return self.passage.audio_file

    @property
    def other_metadata(self):
        return self.passage.other_metadata

    @property
    def script_slice(self):
        return slice(self._first.script[0], self._last.script[-1])

    @property
    def audio_slice(self):
        return slice(self._first.audio[0], self._last.audio[-1])

    @property
    def transcript_slice(self):
        return slice(self._first.transcript[0], self._last.transcript[-1])

    @property
    def script(self):
        return self.passage.script[self.script_slice]

    @property
    def transcript(self):
        return self.passage.transcript[self.transcript_slice]

    @staticmethod
    def _offset_helper(a: typing.Tuple[float, float], b: float):
        return (a[0] - b, a[1] - b)

    def _offset(self, alignment: Alignment):
        return alignment._replace(
            script=self._offset_helper(alignment.script, self._first.script[0]),
            transcript=self._offset_helper(alignment.transcript, self._first.transcript[0]),
            audio=self._offset_helper(alignment.audio, self._first.audio[0]),
        )

    @property
    def alignments(self):
        return Alignment.stow([self._offset(a) for a in self.passage_alignments[self.slice]])

    @property
    def audio_length(self):
        return self._last.audio[-1] - self._first.audio[0]

    def audio(self) -> np.ndarray:
        return read_audio(self.passage.audio_file, self._first.audio[0], self.audio_length)

    def nonalignment_spans(self) -> typing.Iterable[Span]:
        nonalignments = self.passage.nonalignments[self.slice.start : self.slice.stop + 1]
        return self.passage.spans(nonalignments)

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


def _merge(
    a: Alignment, b: Alignment, script: bool = False, transcript: bool = False, audio: bool = False
) -> Alignment:
    """Merge alignments `a` and `b` iff they are connected."""
    return a._replace(
        script=(b if script else a).script,
        transcript=(b if transcript else a).transcript,
        audio=(b if audio else a).audio,
    )


def make_nonalignments(
    curr: UnprocessedPassage,
    audio_metadata: AudioMetadata,
    prev: UnprocessedPassage = None,
    next: UnprocessedPassage = None,
    **kwargs,
) -> Tuple[Alignment]:
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
    assert next is None or (next.alignments is not None and len(next.alignments) > 0)
    assert prev is None or (prev.alignments is not None and len(prev.alignments) > 0)
    assert curr.alignments is not None

    prev_alignment = Alignment(script=(0, 0), audio=(0, 0), transcript=(0, 0))
    if prev is not None:
        prev_alignment = _merge(prev_alignment, prev.alignments[-1], **kwargs)

    next_alignment = Alignment(
        script=(len(curr.script), len(curr.script)),
        audio=(audio_metadata.length, audio_metadata.length),
        transcript=(len(curr.transcript), len(curr.transcript)),
    )
    if next is not None:
        next_alignment = _merge(next_alignment, next.alignments[0], **kwargs)

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


def make_passages(
    dataset: typing.List[typing.List[UnprocessedPassage]], **kwargs
) -> typing.Iterator[Passage]:
    """Process `UnprocessedPassage` and return a list of `Passage`s.

    NOTE: This function processes passages in a batch; therefore, it'd be ideal to pass as many
    items at once as possible.

    Args:
        dataset: Dataset with a list of documents each with a list of passsages.
        **kwargs: Keyword arguments passed to `make_nonalignments`.
    """
    audio_paths = list(set(d.audio_path for d in flatten_2d(dataset)))
    with multiprocessing.pool.ThreadPool() as pool:
        audio_paths = [p for e, p in zip(pool.map(_exists, audio_paths), audio_paths) if e]
    metadatas = get_audio_metadata(typing.cast(typing.List[pathlib.Path], list(audio_paths)))
    lookup = {m.path: m for m in metadatas}
    for document in dataset:
        for i, curr in enumerate(document):
            if curr.audio_path not in lookup:
                logger.warning("Skipping, audio path (%s) isn't a file.", curr.audio_path)
                continue

            audio_file = lookup[curr.audio_path]

            if curr.alignments is None:
                alignment = Alignment(
                    (0, len(curr.script)), (0.0, audio_file.length), (0, len(curr.transcript))
                )
                curr = dataclasses.replace(curr, alignments=(alignment,))
                assert curr.alignments is not None
                prev, next_ = None, None
            else:
                # NOTE: While not explicit, this function will error if there is a mix of
                # undefined, and defined alignments.
                prev = next((p for p in reversed(document[:i]) if len(p.alignments) > 0), None)
                next_ = next((p for p in document[i + 1 :] if len(p.alignments) > 0), None)

            passage = Passage(
                audio_file=audio_file,
                speaker=curr.speaker,
                script=curr.script,
                transcript=curr.transcript,
                alignments=Alignment.stow(curr.alignments),
                nonalignments=make_nonalignments(curr, audio_file, prev, next_, **kwargs),
                other_metadata=curr.other_metadata,
            )
            passage.check_invariants()
            yield passage


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
        make_timeline: typing.Callable[[Passage], Timeline[int]]
        make_timeline = lambda p: Timeline(
            [Interval(a.audio, i) for i, a in enumerate(p.alignments)], **kwargs
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
                return passage[begin.val : end.val + 1]


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

    dataset = []
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
    return list(make_passages(dataset, script=False, transcript=True, audio=True))


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
        other_metadata: Additional metadata to include along with the returned passages.
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


def update_conventional_passage_script(passage: Passage, script: str) -> Passage:
    """Update `passage.script` with `passage.transcript` and `passage.alignments` accordingly
    in a conventional passage."""
    assert len(passage.alignments) == 1
    slice = (0, len(script))
    alignments = [passage.alignments[0]._replace(script=slice, transcript=slice)]
    alignments_ = stow(alignments, dtype=alignment_dtype)
    return dataclasses.replace(passage, script=script, transcript=script, alignments=alignments_)


def _clamp(alignment: Alignment, audio_file: AudioMetadata) -> Alignment:
    """ Helped function for `update_passage_audio`. """
    if alignment.audio[-1] <= audio_file.length:
        return alignment
    new = (min(alignment.audio[0], audio_file.length), min(alignment.audio[-1], audio_file.length))
    return alignment._replace(audio=new)


def update_passage_audio(passage: Passage, audio_file: AudioMetadata, eps: float = 1e-4) -> Passage:
    """Update `passage.audio_file` with a new `audio_file` that has a similar length to the old
    audio file."""
    message = "The audio files must have similar length."
    assert abs(passage.audio_file.length - audio_file.length) < eps, message
    if len(passage.alignments) == 0 or passage.alignments[-1].audio[-1] <= audio_file.length:
        return dataclasses.replace(passage, audio_file=audio_file)
    updated = stow([_clamp(a, audio_file) for a in passage.alignments], alignment_dtype)
    return dataclasses.replace(passage, alignments=updated, audio_file=audio_file)
