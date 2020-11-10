import dataclasses
import functools
import json
import logging
import pathlib
import pprint
import random
import subprocess
import typing
from functools import lru_cache
from math import ceil, floor
from pathlib import Path

import torch
from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")

logger = logging.getLogger(__name__)
pprinter = pprint.PrettyPrinter(indent=4)


class Alignment(typing.NamedTuple):
    """An aligned `text` and `audio` slice.

    Args:
        text: The start and end of a slice of text in characters.
        audio: The start and end of a slice of audio in seconds.
    """

    text: typing.Tuple[int, int]
    audio: typing.Tuple[float, float]


class Speaker(typing.NamedTuple):
    # NOTE: `gender` is not a required property for a `Speaker`.
    name: str
    gender: typing.Optional[str] = None


read_audio_slice = functools.lru_cache(maxsize=None)(lib.audio.read_audio_slice)
read_audio = functools.lru_cache(maxsize=None)(lib.audio.read_audio)


@dataclasses.dataclass(frozen=True)
class Example:
    """Given a script (`text`), this is an `Example` voice-over stored at the `audio_path`.

    Args:
        alignments: List of alignments between `text` and `audio_path`.
        text: The text read in `audio_path`.
        audio_path: A voice over of the `text.`
        speaker: The voice.
        metadata: Additional metadata associated with this example.
    """

    audio_path: Path
    speaker: Speaker
    alignments: typing.Tuple[Alignment, ...]
    text: str
    metadata: typing.Dict

    @property
    def audio(self):
        return read_audio(self.audio_path)


@dataclasses.dataclass(frozen=True)
class Span:
    """Span of `text` and voice-over sliced from `example`.

    Args:
        example: Example voice-over of a script.
        slice: The start and end of an `alignments` span.
        ...
    """

    example: Example
    slice: typing.Tuple[int, int]
    alignments_slice: typing.Tuple[Alignment, ...] = dataclasses.field(init=False)
    speaker: Speaker = dataclasses.field(init=False)
    audio_path: Path = dataclasses.field(init=False)
    metadata: typing.Dict = dataclasses.field(init=False)
    text: str = dataclasses.field(init=False)
    audio_length: float = dataclasses.field(init=False)
    alignments: typing.Tuple[Alignment, ...] = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.slice[1] - self.slice[0], "Cannot create `Span` without any `Alignments`."
        # Learn more about using `__setattr__`:
        # https://stackoverflow.com/questions/53756788/how-to-set-the-value-of-dataclass-field-in-post-init-when-frozen-true
        set = object.__setattr__
        set(self, "speaker", self.example.speaker)
        set(self, "audio_path", self.example.audio_path)
        set(self, "metadata", self.example.metadata)
        set(self, "alignments_slice", self.example.alignments[slice(*self.slice)])
        text_slice = slice(self.alignments_slice[0].text[0], self.alignments_slice[-1].text[-1])
        set(self, "text", self.example.text[text_slice])
        audio_length = self.alignments_slice[-1].audio[-1] - self.alignments_slice[0].audio[0]
        set(self, "audio_length", audio_length)
        set(self, "alignments", self._get_alignments())

    def _get_alignments(self) -> typing.Tuple[Alignment, ...]:
        subtract = lambda t, o: (t[0] - o, t[1] - o)
        first = self.alignments_slice[0]
        alignments = [
            Alignment(subtract(a.text, first.text[0]), subtract(a.audio, first.audio[0]))
            for a in self.alignments_slice
        ]
        return tuple(alignments)

    @property
    def audio(self):
        start = self.alignments_slice[0].audio[0]
        return read_audio_slice(self.example.audio_path, start, self.audio_length)


def _overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    """ Get the percentage overlap. """
    if other_end == other_start:
        return 1.0 if other_start >= start and other_end <= end else 0.0
    return (min(end, other_end) - max(start, other_start)) / (other_end - other_start)


def span_generator(data: typing.List[Example], max_seconds: float) -> typing.Iterator[Span]:
    """Randomly generate `Example`(s) that are at most `max_seconds` long.

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
        data: List of examples to sample from.
        max_seconds: The maximum interval length.
    """
    assert max_seconds > 0, "The maximum interval length must be a positive number."
    if len(data) == 0:
        return

    # NOTE: For a sufficiently large `max_seconds`, the span length tends to be larger than the
    # example length; therefore, the entire example tends to be selected every time.
    if max_seconds == float("inf"):
        while True:
            example = random.choice(data)
            yield Span(example, slice=(0, len(example.alignments)))

    min_ = lambda e: e.alignments[0].audio[0]
    max_ = lambda e: e.alignments[-1].audio[1]
    offset = lambda e: floor(min_(e))

    # NOTE: `lookup` allows fast lookups of alignments for a point in time.
    lookup: typing.List[typing.List[typing.List[int]]]
    lookup = [[[] for _ in range(ceil(max_(e)) - offset(e) + 1)] for e in data]
    for i, example in enumerate(data):
        for j, alignment in enumerate(example.alignments):
            for k in range(int(floor(alignment.audio[0])), int(ceil(alignment.audio[1])) + 1):
                lookup[i][k - offset(example)].append(j)

    weights = torch.tensor([float(max_(e) - min_(e)) for e in data])
    while True:
        length = random.uniform(0, max_seconds)
        # NOTE: The `weight` is based on `start` (i.e. the number of spans)
        index = int(torch.multinomial(weights + length, 1).item())
        example = data[index]
        # NOTE: Uniformly sample a span of audio.
        start = random.uniform(min_(example) - length, max_(example))
        end = min(start + length, max_(example))
        start = max(start, min_(example))

        # NOTE: Based on the overlap, decide which alignments to include in the span.
        part = lib.utils.flatten(
            lookup[index][int(start) - offset(example) : int(end) - offset(example) + 1]
        )
        get = lambda i: example.alignments[i].audio
        random_ = lru_cache(maxsize=None)(lambda i: random.random())
        bounds = (
            next((i for i in part if _overlap(start, end, *get(i)) >= random_(i)), None),
            next((i for i in reversed(part) if _overlap(start, end, *get(i)) >= random_(i)), None),
        )
        if (
            bounds[0] is not None
            and bounds[1] is not None
            and bounds[0] <= bounds[1]
            and get(bounds[1])[1] - get(bounds[0])[0] > 0
            and get(bounds[1])[1] - get(bounds[0])[0] <= max_seconds
        ):
            slice_ = typing.cast(typing.Tuple[int, int], (bounds[0], bounds[1] + 1))
            yield Span(example, slice=slice_)


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
) -> typing.List[Example]:
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

    Returns: List of voice-over examples in the dataset.
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

    examples = []
    iterator = typing.cast(typing.Iterator[typing.Tuple[pathlib.Path, ...]], zip(*tuple(files)))
    for alignment_file_path, recording_file_path, script_file_path in iterator:
        scripts = pandas.read_csv(str(script_file_path.absolute()))
        script_alignments: typing.List[typing.List[typing.List[typing.List[float]]]]
        script_alignments = json.loads(alignment_file_path.read_text())
        error = f"Each script ({script_file_path}) must have an alignment ({alignment_file_path})."
        assert len(scripts) == len(script_alignments), error
        for (_, script), alignments in zip(scripts.iterrows(), script_alignments):
            alignments_ = [
                Alignment((int(a[0][0]), int(a[0][1])), (a[1][0], a[1][1])) for a in alignments
            ]
            example = Example(
                audio_path=recording_file_path,
                speaker=speaker,
                alignments=tuple(alignments_),
                text=script[text_column],
                metadata={k: v for k, v in script.items() if k not in (text_column,)},
            )
            examples.append(example)

    return examples


def precut_dataset_loader(
    directory: Path,
    root_directory_name: str,
    url: str,
    speaker: Speaker,
    url_filename: typing.Optional[str] = None,
    check_files: typing.List[str] = ["{metadata_filename}"],
    metadata_filename: str = "{directory}/{root_directory_name}/metadata.csv",
    metadata_text_column: typing.Union[str, int] = "Content",
    metadata_audio_column: typing.Union[str, int] = "WAV Filename",
    metadata_audio_path: str = (
        "{directory}/{root_directory_name}/wavs/{metadata_audio_column_value}"
    ),
    **kwargs,
) -> typing.List[Example]:
    """Load a precut speech dataset.

    A precut speech dataset has these invariants:
        - The dataset has already been segmented, and the segments have been audited.
        - The file structure is similar to:
            {root_directory_name}/
                metadata.csv
                wavs/
                    audio1.wav
                    audio2.wav
        - The metadata CSV file contains a mapping of audio transcriptions to audio filenames.
        - The dataset contains one speaker.
        - The dataset is stored in a `tar` or `zip` at some url.

    Args:
        directory: Directory to cache the dataset.
        root_directory_name: Name of the directory inside `directory` to store data.
        url: URL of the dataset file.
        speaker: The dataset speaker.
        url_filename: Name of the file downloaded; Otherwise, a filename is extracted from the url.
        check_files: The download is considered successful, if these files exist.
        metadata_filename: The filename for the metadata file.
        metadata_text_column: Column name or index with the audio transcript.
        metadata_audio_column: Column name or index with the audio filename.
        metadata_audio_path: String template for the audio path given the `metadata_audio_column`
            value.
        **kwargs: Key word arguments passed to `pandas.read_csv`.

    Returns: List of voice-over examples in the dataset.
    """
    logger.info("Loading `%s` speech dataset", root_directory_name)

    directory = Path(directory)
    metadata_filename = metadata_filename.format(
        directory=directory, root_directory_name=root_directory_name
    )
    check_files = [f.format(metadata_filename=metadata_filename) for f in check_files]
    check_files = [str(Path(f).absolute()) for f in check_files]
    download_file_maybe_extract(
        url=url,
        directory=str(directory.absolute()),
        check_files=check_files,
        filename=url_filename,
    )

    data_frame = pandas.read_csv(Path(metadata_filename), **kwargs)
    _get_audio_path = lambda r: Path(
        metadata_audio_path.format(
            directory=directory,
            root_directory_name=root_directory_name,
            metadata_audio_column_value=r[metadata_audio_column],
        )
    )
    audio_paths = [_get_audio_path(r) for _, r in data_frame.iterrows()]
    audio_metadatas = lib.audio.get_audio_metadata(audio_paths)
    texts = [r[metadata_text_column] for _, r in data_frame.iterrows()]
    _get_metadata = lambda r: {
        k: v for k, v in r.items() if k not in [metadata_text_column, metadata_audio_column]
    }
    iterator = zip(audio_paths, audio_metadatas, texts, data_frame.iterrows())
    return [
        Example(
            audio_path=audio_path,
            speaker=speaker,
            alignments=(Alignment((0, len(text)), (0.0, audio_metadata.length)),),
            text=text,
            metadata=_get_metadata(row),
        )
        for audio_path, audio_metadata, text, (_, row) in iterator
    ]
