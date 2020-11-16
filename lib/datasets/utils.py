# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import csv
import dataclasses
import functools
import json
import logging
import os
import pathlib
import random
import subprocess
import typing
from dataclasses import fields
from functools import lru_cache
from math import ceil, floor
from pathlib import Path

import torch
from third_party import LazyLoader

import lib
from lib.utils import list_to_tuple

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")

logger = logging.getLogger(__name__)


read_audio_slice = functools.lru_cache(maxsize=None)(lib.audio.read_audio_slice)
read_audio = functools.lru_cache(maxsize=None)(lib.audio.read_audio)


def _handle_get_item_key(length: int, key: typing.Any) -> slice:
    """ Normalize `__getitem__` key. """
    if length == 0 and isinstance(key, slice) and key.start is None and key.stop is None:
        return slice(0, 0)
    items = list(range(length))[key]
    return slice(items[0], items[-1] + 1) if isinstance(items, list) else slice(items, items + 1)


class Alignment(typing.NamedTuple):
    """An aligned `text` and `audio` slice.

    Args:
        script: The start and end of a script slice in characters.
        audio: The start and end of a audio recording slice in seconds.
        transcript: The start and end of a trasnscript slice in characters.
    """

    script: typing.Tuple[int, int]
    audio: typing.Tuple[float, float]
    transcript: typing.Tuple[int, int]


class Speaker(typing.NamedTuple):
    # NOTE: `gender` is not a required property for a `Speaker`.
    name: str
    gender: typing.Optional[str] = None


@dataclasses.dataclass
class Passage:
    """A voiced passage.

    Args:
        audio_path: A voice-over of the `script`.
        speaker: An identifier of the voice.
        script: The `script` the `speaker` was reading from.
        transcript: The `transcript` of the `audio`.
        alignments: Alignments that align the `script`, `transcript` and `audio`.
        other_metadata: Additional metadata associated with this passage.
        index: The index of `self` in `self.passages`.
        passages: Other neighboring passages, for context.
    """

    audio_path: Path
    speaker: Speaker
    script: str
    transcript: str
    alignments: typing.Tuple[Alignment, ...]
    other_metadata: typing.Dict
    index: int = 0
    passages: typing.List[Passage] = None  # type: ignore

    def __post_init__(self):
        if self.index == 0 and self.passages is None:
            self.passages = [self]

    @property
    def audio(self):
        return read_audio(self.audio_path)

    def to_string(self, *fields):
        values = ", ".join(f"{f}={getattr(self, f)}" for f in fields)
        return f"{self.__class__.__name__}({values})"

    def __getitem__(self, key) -> Span:
        return Span(self, _handle_get_item_key(len(self.alignments), key))

    def key(self):
        return tuple(getattr(self, f.name) for f in fields(self) if f.name != "passages")

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other: typing.Any):
        return self.key() == other.key() if isinstance(other, Passage) else NotImplemented


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
    script: str = dataclasses.field(init=False)
    transcript: str = dataclasses.field(init=False)
    alignments: typing.Tuple[Alignment, ...] = dataclasses.field(init=False)
    audio_path: Path = dataclasses.field(init=False)
    audio_length: float = dataclasses.field(init=False)
    speaker: Speaker = dataclasses.field(init=False)
    other_metadata: typing.Dict = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.span.stop - self.span.start, "Cannot create `Span` without any `Alignments`."

        # Learn more about using `__setattr__`:
        # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        set = object.__setattr__
        set(self, "speaker", self.passage.speaker)
        set(self, "audio_path", self.passage.audio_path)
        set(self, "other_metadata", self.passage.other_metadata)

        span = self.passage.alignments[self.span]
        script = self.passage.script[span[0].script[0] : span[-1].script[-1]]
        audio_length = span[-1].audio[-1] - span[0].audio[0]
        transcript = self.passage.transcript[span[0].transcript[0] : span[-1].transcript[-1]]
        subtract = lambda a, b: tuple([a[0] - b[0], a[1] - b[0]])
        alignments = [tuple([subtract(a, b) for a, b in zip(a, span[0])]) for a in span]

        set(self, "script", script)
        set(self, "audio_length", audio_length)
        set(self, "transcript", transcript)
        set(self, "alignments", tuple(Alignment(*a) for a in alignments))  # type: ignore

    @property
    def audio(self):
        start = self.passage.alignments[self.span][0].audio[0]
        return read_audio_slice(self.passage.audio_path, start, self.audio_length)

    def to_string(self, *fields):
        values = ", ".join(f"{f}={getattr(self, f)}" for f in fields)
        return f"{self.__class__.__name__}({values})"

    def __getitem__(self, key) -> Span:
        return Span(self.passage, _handle_get_item_key(len(self.alignments), key))


def _overlap(slice: typing.Tuple[float, float], other: typing.Tuple[float, float]) -> float:
    """ Get the percentage overlap between `slice` and the `other` slice. """
    if other[-1] == other[0]:
        return 1.0 if other[0] >= slice[0] and other[-1] <= slice[1] else 0.0
    return (min(slice[1], other[-1]) - max(slice[0], other[0])) / (other[-1] - other[0])


def span_generator(passages: typing.List[Passage], max_seconds: float) -> typing.Iterator[Span]:
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

    Args:
        passages
        max_seconds: The maximum interval length.
    """
    assert max_seconds > 0, "The maximum interval length must be a positive number."
    if len(passages) == 0:
        return

    # NOTE: For a sufficiently large `max_seconds`, the span length tends to be larger than the
    # passage length; therefore, the entire passage tends to be selected every time.
    if max_seconds == float("inf"):
        while True:
            yield random.choice(passages)[:]

    min_ = lambda passage: passage.alignments[0].audio[0]
    max_ = lambda passage: passage.alignments[-1].audio[1]
    offset = lambda passage: floor(min_(passage))

    # NOTE: `lookup` allows fast lookups of alignments for a point in time.
    lookup: typing.List[typing.List[typing.List[int]]]
    lookup = [[[] for _ in range(ceil(max_(p)) - offset(p) + 1)] for p in passages]
    for i, passage in enumerate(passages):
        for j, alignment in enumerate(passage.alignments):
            for k in range(int(floor(alignment.audio[0])), int(ceil(alignment.audio[1])) + 1):
                lookup[i][k - offset(passage)].append(j)

    weights = torch.tensor([float(max_(p) - min_(p)) for p in passages])
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
        part_ = lookup[index][int(start) - offset(passage) : int(end) - offset(passage) + 1]
        part = lib.utils.flatten(part_)
        get = lambda i: passage.alignments[i].audio
        random_ = lru_cache(maxsize=None)(lambda i: random.random())
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

    files: typing.List[typing.List[pathlib.Path]] = []
    for directory, suffix in zip(directories, suffixes):
        directory.mkdir(exist_ok=True)
        command = f"gsutil -m cp -n {gcs_path}/{directory.name}/*{suffix} {directory}/"
        subprocess.run(command.split(), check=True)
        files_ = [p for p in directory.iterdir() if p.suffix == suffix]
        files.append(sorted(files_, key=lambda p: lib.text.natural_keys(p.name)))

    return_ = []
    iterator = typing.cast(typing.Iterator[typing.Tuple[pathlib.Path, ...]], zip(*tuple(files)))
    for alignment_file_path, recording_file_path, script_file_path in iterator:
        scripts = pandas.read_csv(str(script_file_path.absolute()))
        json_ = json.loads(alignment_file_path.read_text())

        error = f"Each script ({script_file_path}) must have an alignment ({alignment_file_path})."
        assert len(scripts) == len(json_["alignments"]), error

        passages = []
        for (_, script), alignments in zip(scripts.iterrows(), json_["alignments"]):
            passage = Passage(
                audio_path=recording_file_path,
                speaker=speaker,
                script=script[text_column],
                transcript=json_["transcript"],
                alignments=tuple(Alignment(*a) for a in list_to_tuple(alignments)),  # type: ignore
                other_metadata={k: v for k, v in script.items() if k not in (text_column,)},
                index=len(passages),
                passages=passages,
            )
            passages.append(passage)
        return_.extend(passages)
    return return_


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
    iterator = zip(audio_paths, lib.audio.get_audio_metadata(audio_paths), df.iterrows())
    # TODO: These passages could be linked together, with `passages`, consider that.
    return [
        Passage(
            audio_path=audio_path,
            speaker=speaker,
            script=row[metadata_text_column].strip(),
            transcript=row[metadata_text_column].strip(),
            alignments=_get_alignments(row[metadata_text_column].strip(), audio_metadata.length),
            other_metadata={**_get_other_metadata(row), **additional_metadata},
        )
        for audio_path, audio_metadata, (_, row) in iterator
    ]
