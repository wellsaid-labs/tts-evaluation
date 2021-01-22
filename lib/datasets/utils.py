# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import csv
import dataclasses
import functools
import json
import logging
import os
import random
import subprocess
import sys
import typing
from dataclasses import fields
from math import ceil, floor
from pathlib import Path

import numpy
import pyximport
import torch
from third_party import LazyLoader

import lib
from lib.audio import AudioFileMetadata, get_audio_metadata
from lib.utils import flatten, list_to_tuple

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")

logger = logging.getLogger(__name__)
UnalignedType = typing.Tuple[str, str, typing.Tuple[float, float]]
pyximport.install(language_level=sys.version_info[0])

from lib.datasets.alignment import Alignment  # type: ignore


def _handle_get_item_key(length: int, key: typing.Any) -> slice:
    """ Normalize `__getitem__` key. """
    if length == 0 and isinstance(key, slice) and key.start is None and key.stop is None:
        return slice(0, 0)
    items = list(range(length))[key]
    return slice(items[0], items[-1] + 1) if isinstance(items, list) else slice(items, items + 1)


def _to_string(self: typing.Union[Span, Passage], *fields: str) -> str:
    """ Create a string representation of a `dataclass` with a limited number of `fields`. """
    values = ", ".join(f"{f}={getattr(self, f)}" for f in fields)
    return f"{self.__class__.__name__}({values})"


class Speaker(typing.NamedTuple):
    # NOTE: `gender` is not a required property for a `Speaker`.
    label: str
    name: typing.Optional[str] = None
    gender: typing.Optional[str] = None


class IsConnected(typing.NamedTuple):
    script: bool
    audio: bool
    transcript: bool


@dataclasses.dataclass
class Passage:
    """A voiced passage.

    Args:
        audio_file: A voice-over of the `script`.
        speaker: An identifier of the voice.
        script: The `script` the `speaker` was reading from.
        transcript: The `transcript` of the `audio`.
        alignments: Alignments (sorted) that align the `script`, `transcript` and `audio`.
        index: The index of `self` in `self.passages`.
        passages: Other neighboring passages (sorted), for context.
        is_connected: Iff `True`, the `script`, `audio` or `transcript` alignments are on a
            continuous timeline from passage to passage.
        other_metadata: Additional metadata associated with this passage.
    """

    audio_file: AudioFileMetadata
    speaker: Speaker
    script: str
    transcript: str
    alignments: typing.Tuple[Alignment, ...]
    index: int = 0
    passages: typing.List[Passage] = None  # type: ignore
    is_connected: IsConnected = IsConnected(False, False, False)
    other_metadata: typing.Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.passages is None:
            self.passages = [self]
        self._check_invariants()

    def audio(self):
        return lib.audio.read_audio(self.audio_file.path)

    def aligned_audio_length(self) -> float:
        return self.alignments[-1].audio[-1] - self.alignments[0].audio[0]

    def to_string(self, *fields):
        return _to_string(self, *fields)

    def __getitem__(self, key) -> Span:
        return Span(self, _handle_get_item_key(len(self.alignments), key))

    @property
    def key(self):
        not_included = ("passages", "other_metadata")
        return tuple(getattr(self, f.name) for f in fields(self) if f.name not in not_included)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other: typing.Any):
        return self.key == other.key if isinstance(other, Passage) else NotImplemented

    @property
    def prev(self) -> typing.Optional[Passage]:
        assert self.passages[self.index] == self
        return self.passages[self.index - 1] if self.index != 0 else None

    @property
    def next(self) -> typing.Optional[Passage]:
        assert self.passages[self.index] == self
        return self.passages[self.index + 1] if self.index != len(self.passages) - 1 else None

    def _merge(self, a: Alignment, other: Passage, b: Alignment) -> Alignment:
        """Merge alignments `a` and `b` iff they are part of a larger continuous timeline."""
        assert b in other.alignments
        a = a._replace(script=b.script) if self.is_connected.script else a
        a = a._replace(transcript=b.transcript) if self.is_connected.transcript else a
        return a._replace(audio=b.audio) if self.is_connected.audio else a

    def _prev_alignment(self) -> Alignment:
        """ Helper function for `_unaligned`. """
        alignment = Alignment(script=(0, 0), audio=(0, 0), transcript=(0, 0))
        prev = self.prev
        while prev is not None and len(prev.alignments) == 0:
            prev = prev.prev
        return alignment if prev is None else self._merge(alignment, prev, prev.alignments[-1])

    def _next_alignment(self) -> Alignment:
        """ Helper function for `_unaligned`. """
        script = (len(self.script), len(self.script))
        transcript = (len(self.transcript), len(self.transcript))
        audio = (self.audio_file.length, self.audio_file.length)
        alignment = Alignment(script=script, audio=audio, transcript=transcript)
        next = self.next
        while next is not None and len(next.alignments) == 0:
            next = next.next
        return alignment if next is None else self._merge(alignment, next, next.alignments[0])

    def _unaligned(self) -> typing.Iterator[UnalignedType]:
        alignments = [self._prev_alignment()] + list(self.alignments) + [self._next_alignment()]
        message = "Alignments in `passages` that share a `transcript`, `audio_file`, or "
        message += "`script` must be continuous."
        for prev, next in zip(alignments, alignments[1:]):
            assert prev.script[-1] <= next.script[0], message
            assert prev.transcript[-1] <= next.transcript[0], message
            script = self.script[prev.script[-1] : next.script[0]]
            transcript = self.transcript[prev.transcript[-1] : next.transcript[0]]
            yield (script, transcript, (prev.audio[-1], next.audio[0]))

    @property
    def unaligned(self) -> typing.List[UnalignedType]:
        """List of unaligned `script` text, `transcript` text, and `audio` spans between each
        alignment.

        If the `passages` are part of a continuous `transcript`, `script` or `audio_file`, then the
        `unaligned` text or audio may bleed into other passages at the edges.

        NOTE: This property may change if `self.passages` is mutated.
        """
        return list(self._unaligned())

    def _check_invariants(self):
        """Check datastructure invariants.

        NOTE: The `audio` alignments may overlap by a little bit, at the edges.
        NOTE: `self.index` may not be in `self.passages`.
        """
        pairs = zip(self.alignments, self.alignments[1:])
        assert all(a.script[0] <= a.script[1] for a in self.alignments)
        assert all(a.audio[0] <= a.audio[1] for a in self.alignments)
        assert all(a.transcript[0] <= a.transcript[1] for a in self.alignments)
        assert all(a.script[1] <= b.script[0] for a, b in pairs)
        assert all(a.transcript[1] <= b.transcript[0] for a, b in pairs)
        assert all(a.audio[0] <= b.audio[1] for a, b in pairs)

        if len(self.alignments) != 0:
            get_ = lambda f: flatten([list(getattr(a, f)) for a in self.alignments])
            assert max(get_("audio")) <= self.audio_file.length
            assert max(get_("script")) <= len(self.script)
            assert max(get_("transcript")) <= len(self.transcript)
            assert min(get_("audio")) >= 0
            assert min(get_("script")) >= 0
            assert min(get_("transcript")) >= 0

        if self.index < len(self.passages):
            assert self.passages[self.index] is self
        for passage in self.passages:
            assert passage.passages is self.passages


SpanType = typing.TypeVar("SpanType", bound="Span")


@dataclasses.dataclass(frozen=True)
class Span:
    """A span of the voiced passage.

    Args:
        passage: The original passage, for context.
        span: A `slice` of `passage.alignments`.
        script: A span of text within `script`.
        transcript: A span of text within `transcript`.
        alignments: A span of alignments that align the `script`, `transcript` and `audio`.
        ...
    """

    passage: Passage
    span: slice
    script_span: slice = dataclasses.field(init=False)
    audio_span: slice = dataclasses.field(init=False)
    transcript_span: slice = dataclasses.field(init=False)
    script: str = dataclasses.field(init=False)
    transcript: str = dataclasses.field(init=False)
    alignments: typing.Tuple[Alignment, ...] = dataclasses.field(init=False)
    audio_file: AudioFileMetadata = dataclasses.field(init=False)
    audio_length: float = dataclasses.field(init=False)
    speaker: Speaker = dataclasses.field(init=False)
    other_metadata: typing.Dict = dataclasses.field(init=False)
    unaligned: typing.List[UnalignedType] = dataclasses.field(init=False)

    @staticmethod
    def _subtract(a: typing.Tuple[float, float], b: typing.Tuple[float, float]):
        return tuple([a[0] - b[0], a[1] - b[0]])

    def __post_init__(self):
        self._check_invariants()

        # Learn more about using `__setattr__`:
        # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        set = object.__setattr__
        set(self, "speaker", self.passage.speaker)
        set(self, "audio_file", self.passage.audio_file)
        set(self, "other_metadata", self.passage.other_metadata)
        set(self, "unaligned", self.passage.unaligned[slice(self.span.start, self.span.stop + 1)])

        span = self.passage.alignments[self.span]
        script = self.passage.script[span[0].script[0] : span[-1].script[-1]]
        audio_length = span[-1].audio[-1] - span[0].audio[0]
        transcript = self.passage.transcript[span[0].transcript[0] : span[-1].transcript[-1]]
        alignments = [tuple([self._subtract(a, b) for a, b in zip(a, span[0])]) for a in span]

        set(self, "script_span", slice(span[0].script[0], span[-1].script[-1]))
        set(self, "audio_span", slice(span[0].audio[0], span[-1].audio[-1]))
        set(self, "transcript_span", slice(span[0].transcript[0], span[-1].transcript[-1]))
        set(self, "script", script)
        set(self, "audio_length", audio_length)
        set(self, "transcript", transcript)
        set(self, "alignments", tuple(Alignment(*a) for a in alignments))  # type: ignore

    def audio(self) -> numpy.ndarray:
        start = self.passage.alignments[self.span][0].audio[0]
        return lib.audio.read_audio_slice(self.passage.audio_file.path, start, self.audio_length)

    def to_string(self, *fields) -> str:
        return _to_string(self, *fields)

    def __getitem__(self: SpanType, key) -> SpanType:
        slice_ = _handle_get_item_key(len(self.alignments), key)
        slice_ = slice(slice_.start + self.span.start, slice_.stop + self.span.start)
        return self.__class__(self.passage, slice_)

    def _check_invariants(self):
        """ Check datastructure invariants. """
        self.passage._check_invariants()
        assert self.span.stop - self.span.start, "`Span` must have `Alignments`."
        assert self.span.stop <= len(self.passage.alignments) and self.span.stop >= 0
        assert self.span.start < len(self.passage.alignments) and self.span.start >= 0


def _overlap(slice: typing.Tuple[float, float], other: typing.Tuple[float, float]) -> float:
    """ Get the percentage overlap between `slice` and the `other` slice. """
    if other[-1] == other[0]:
        return 1.0 if other[0] >= slice[0] and other[-1] <= slice[1] else 0.0
    return (min(slice[1], other[-1]) - max(slice[0], other[0])) / (other[-1] - other[0])


def span_generator(
    passages: typing.List[Passage], max_seconds: float, step: float = 10.0, eps=1e-8
) -> typing.Iterator[Span]:
    """Randomly generate `Span`(s) that are at most `max_seconds` long.

    NOTE:
    - Every `Alignment` has an equal chance of getting sampled, assuming there are no overlaps.
    - Larger groups of `Alignment` are less likely to be sampled. Learn more here:
      https://stats.stackexchange.com/questions/484329/how-do-you-uniformly-sample-spans-from-a-bounded-line/484332#484332
      https://www.reddit.com/r/MachineLearning/comments/if8icr/d_how_do_you_sample_spans_uniformly_from_a_time/
      https://pytorch.slack.com/archives/C205U7WAF/p1598227661001700
      https://www.reddit.com/r/math/comments/iftev6/uniformly_sampling_from_timeseries_data/

    TODO: A sufficiently large pause would slow this algorithm to a hault because it'd be
    sampled continuously and ignored. We could handle that better by excluding large samples
    from the sampling distribution.
    TODO: Add a parameter for step size to trade-off between memory and performance.

    Args:
        passages
        max_seconds: The maximum interval length.
        step: A lower step size is more performant but uses more memory, and vice versa.
        eps: Add small number so that the end point is also included within the range.
    """
    assert max_seconds > 0, "The maximum interval length must be a positive number."
    assert step > 0, "Step must be a positive number."
    if len(passages) == 0:
        return

    # NOTE: For a sufficiently large `max_seconds`, the span length tends to be larger than the
    # passage length; therefore, the entire passage tends to be selected every time.
    if max_seconds == float("inf"):
        while True:
            yield random.choice(passages)[:]

    min_ = lambda passage: passage.alignments[0].audio[0]
    max_ = lambda passage: passage.alignments[-1].audio[1]
    map_ = lambda i, passage: (i - min_(passage)) / step
    slice_ = lambda i, j, p: (int(floor(map_(i, p))), int(ceil((map_(j, p) + eps))))

    # NOTE: `lookup` allows fast lookups of alignments for a point in time.
    lookup_: typing.List[typing.List[typing.List[int]]]
    lookup_ = [[[] for _ in range(ceil(map_(max_(p), p) + eps))] for p in passages]
    for i, passage in enumerate(passages):
        for j, alignment in enumerate(passage.alignments):
            for k in range(*slice_(alignment.audio[0], alignment.audio[1], passage)):
                lookup_[i][k].append(j)
    lookup = typing.cast(typing.Tuple[typing.Tuple[typing.Tuple[int]]], list_to_tuple(lookup_))

    weights = torch.tensor([float(max_(p) - min_(p)) for p in passages])
    random_ = functools.lru_cache(maxsize=None)(lambda _: random.random())
    get_alignment = lambda p, i: p.alignments[i].audio
    while True:
        length = random.uniform(0, max_seconds)
        # NOTE: The `weight` is based on `start` (i.e. the number of spans)
        index = int(torch.multinomial(weights + length, 1).item())
        passage = passages[index]

        # NOTE: Uniformly sample a span of audio.
        start = random.uniform(min_(passage) - length, max_(passage))
        end = min(start + length, max_(passage))
        start = max(start, min_(passage))

        # NOTE: Based on the overlap, decide which alignments to include in the span.
        part = flatten(lib.utils.tuple_to_list(lookup[index][slice(*slice_(start, end, passage))]))
        random_.cache_clear()
        get = functools.partial(get_alignment, passage)
        bounds = (
            next((i for i in part if _overlap((start, end), get(i)) >= random_(i)), None),
            next((i for i in reversed(part) if _overlap((start, end), get(i)) >= random_(i)), None),
        )
        if (
            bounds[0] is not None
            and bounds[1] is not None
            and bounds[0] <= bounds[1]
            and get(bounds[1])[1] - get(bounds[0])[0] > 0
            and get(bounds[1])[1] - get(bounds[0])[0] <= max_seconds
        ):
            yield passage[bounds[0] : bounds[1] + 1]


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
) -> typing.List[Passage]:
    """Load an alignment text-to-speech (TTS) dataset from GCS.

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
    """
    logger.info("Loading `%s` speech dataset", root_directory_name)

    root = (Path(directory) / root_directory_name).absolute()
    root.mkdir(exist_ok=True)
    names = [alignments_directory_name, recordings_directory_name, scripts_directory_name]
    suffixes = (alignments_suffix, recordings_suffix, scripts_suffix)
    directories = [root / d for d in names]

    files: typing.List[typing.List[Path]] = []
    for directory, suffix in zip(directories, suffixes):
        directory.mkdir(exist_ok=True)
        # TODO: Add `-m` when this issue is resolved:
        # https://github.com/GoogleCloudPlatform/gsutil/pull/1107
        # TODO: It's faster to run `gsutil` without `-m` if the data already exists on disk;
        # therefore, if the directory already exists, we should skip multiprocessing.
        command = "gsutil cp -n "
        command += f"{gcs_path}/{directory.name}/*{suffix} {directory}/"
        subprocess.run(command.split(), check=True)
        files_ = [p for p in directory.iterdir() if p.suffix == suffix]
        files.append(sorted(files_, key=lambda p: lib.text.numbers_then_natural_keys(p.name)))

    return_ = []
    audio_file_metadatas = get_audio_metadata(files[1])
    Iterator = typing.Iterator[typing.Tuple[Path, Path, Path, AudioFileMetadata]]
    iterator = typing.cast(Iterator, zip(*tuple(files), audio_file_metadatas))
    is_connected = IsConnected(script=False, transcript=True, audio=True)
    for alignment_path, _, script_path, recording_file_metadata in iterator:
        scripts = pandas.read_csv(str(script_path.absolute()))
        json_ = json.loads(alignment_path.read_text())

        error = f"Each script ({script_path}) must have an alignment ({alignment_path})."
        assert len(scripts) == len(json_["alignments"]), error

        passages = []
        for (_, script), alignments in zip(scripts.iterrows(), json_["alignments"]):
            passage = Passage(
                audio_file=recording_file_metadata,
                speaker=speaker,
                script=script[text_column],
                transcript=json_["transcript"],
                alignments=tuple(Alignment(*a) for a in list_to_tuple(alignments)),  # type: ignore
                other_metadata={k: v for k, v in script.items() if k not in (text_column,)},
                index=len(passages),
                passages=passages,
                is_connected=is_connected,
            )
            passages.append(passage)
        return_.extend(passages)
    return return_


def _exists(path: Path) -> bool:
    """ Helper function for `conventional_dataset_loader` that can be easily mocked. """
    return path.exists() and path.is_file()


def conventional_dataset_loader(
    directory: Path,
    speaker: Speaker,
    metadata_path_template: str = "{directory}/metadata.csv",
    metadata_audio_column: typing.Union[str, int] = 0,
    metadata_text_column: typing.Union[str, int] = 2,
    metadata_quoting: int = csv.QUOTE_NONE,
    metadata_delimiter: str = "|",
    metadata_header: typing.Optional[bool] = None,
    audio_path_template: str = "{directory}/wavs/{file_name}.wav",
    additional_metadata: typing.Dict = {},
) -> typing.List[Passage]:
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

    TODO: These passages could be linked together with `Passage.passages`, consider that.

    Args:
        directory: Directory the dataset is stored in.
        speaker: The dataset speaker.
        metadata_path_template: A template specifying the location of the metadata file.
        metadata_audio_column: The column name or index with the audio filename.
        metadata_text_column: The column name or index with the audio transcript.
        metadata_quoting
        metadata_delimiter: The metadata file column delimiter.
        metadata_header
        audio_path_template: A template specifying the location of an audio file.
        other_metadata: Additional metadata to include along with the returned passages.
    """
    metadata_path = Path(metadata_path_template.format(directory=directory))
    if os.stat(str(metadata_path)).st_size == 0:
        return []
    df = pandas.read_csv(  # type: ignore
        metadata_path,
        delimiter=metadata_delimiter,
        header=metadata_header,
        quoting=metadata_quoting,
    )
    audio_paths = [
        Path(audio_path_template.format(directory=directory, file_name=r[metadata_audio_column]))
        for _, r in df.iterrows()
    ]
    handled_columns = [metadata_text_column, metadata_audio_column]
    _get_other_metadata = lambda r: {k: v for k, v in r.items() if k not in handled_columns}
    _get_alignments = lambda s, l: (Alignment((0, len(s)), (0.0, l), (0, len(s))),)
    audio_paths_, rows = [], []
    for audio_path, (_, row) in zip(audio_paths, df.iterrows()):
        if not _exists(audio_path):
            logger.warning("Skipping, audio path (%s) isn't a file.", audio_path)
            continue
        audio_paths_.append(audio_path)
        rows.append(row)
    return [
        Passage(
            audio_file=audio_metadata,
            speaker=speaker,
            script=row[metadata_text_column].strip(),
            transcript=row[metadata_text_column].strip(),
            alignments=_get_alignments(row[metadata_text_column].strip(), audio_metadata.length),
            other_metadata={**_get_other_metadata(row), **additional_metadata},
        )
        for audio_metadata, row in zip(get_audio_metadata(audio_paths_), rows)
    ]


def update_conventional_passage_script(passage: Passage, script: str) -> Passage:
    """Update `passage.script` with `passage.transcript` and `passage.alignments` accordingly
    in a conventional passage."""
    passage.script = script
    passage.transcript = script
    assert len(passage.alignments) == 1
    slice = (0, len(script))
    passage.alignments = (passage.alignments[0]._replace(script=slice, transcript=slice),)
    return passage


def update_passage_audio(passage: Passage, audio_file: AudioFileMetadata, tolerance: float = 0.001):
    """Update `passage.audio_file` with a new `audio_file` that has a similar length to the old
    audio file.

    TODO: Should this be included in the `Passage` object?
    """
    message = "The audio files must have similar length."
    assert abs(passage.audio_file.length - audio_file.length) < tolerance, message
    clamp_ = lambda a: (min(a[0], audio_file.length), min(a[1], audio_file.length))
    updated = tuple(a._replace(audio=clamp_(a.audio)) for a in passage.alignments)
    passage.alignments = updated
    passage.audio_file = audio_file
    return passage
