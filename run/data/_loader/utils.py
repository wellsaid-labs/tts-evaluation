import csv
import dataclasses
import functools
import json
import logging
import math
import os
import pathlib
import random
import subprocess
import typing
from pathlib import Path

import config as cf
import numpy as np
import torch
from third_party import LazyLoader

import lib
from lib.audio import AudioDataType, AudioEncoding, AudioFormat, AudioMetadata
from lib.utils import Timeline
from run.data._loader import structures as struc

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")


logger = logging.getLogger(__name__)

FloatFloat = typing.Tuple[float, float]
IntInt = typing.Tuple[int, int]
Slice = slice  # NOTE: `pylance` is buggy if we use `slice` directly for typing.


def read_audio(audio_file: AudioMetadata, *args, **kwargs) -> np.ndarray:
    """Read `audio_file` into a `np.float32` array."""
    try:
        assert audio_file.encoding == AudioEncoding.PCM_FLOAT_32_BIT
        audio = lib.audio.read_wave_audio(audio_file, *args, **kwargs)
    except AssertionError:
        audio = lib.audio.read_audio(audio_file.path, *args, **kwargs)
    assert audio.dtype == np.float32, "Invariant failed. Audio `dtype` must be `np.float32`."
    return audio


def normalize_audio_suffix(path: Path, suffix: str) -> Path:
    """Normalize the last suffix to `suffix` in `path`."""
    assert len(path.suffixes) == 1, "`path` has multiple suffixes."
    return path.parent / (path.stem + suffix)


def normalize_audio(
    source: Path,
    destination: Path,
    suffix: str,
    data_type: AudioDataType,
    bits: int,
    sample_rate: int,
    num_channels: int,
):
    return lib.audio.normalize_audio(
        source, destination, suffix, data_type, bits, sample_rate, num_channels
    )


def is_normalized_audio_file(audio_file: AudioMetadata, audio_format: AudioFormat, suffix: str):
    """Check if `audio_file` is normalized to `audio_format`."""
    attrs = [f.name for f in dataclasses.fields(AudioFormat) if f.name != "encoding"]
    bool_ = audio_file.path.suffix == suffix
    # NOTE: Use `.name` because of this bug:
    # https://github.com/streamlit/streamlit/issues/2379
    bool_ = bool_ and audio_file.encoding.name == audio_format.encoding.name
    return bool_ and all(getattr(audio_format, a) == getattr(audio_file, a) for a in attrs)


def _cache_path(
    original: pathlib.Path, prefix: str, suffix: str, cache_dir, **kwargs
) -> pathlib.Path:
    """Make `Path` for caching results given the `original` file.

    TODO: Factor this out into `_config` so we have guarentees that there is no collisions in the
    disk.

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


def maybe_normalize_audio_and_cache(
    audio_file: AudioMetadata,
    suffix: str,
    data_type: AudioDataType,
    bits: int,
    format_: AudioFormat,
    **kwargs,
) -> pathlib.Path:
    """Normalize `audio_file`, if it's not already normalized, and cache the results.

    TODO: Remove redundancy in parameters.
    """
    if is_normalized_audio_file(audio_file, format_, suffix):
        return audio_file.path

    kwargs_: typing.Dict[str, typing.Any] = dict(
        suffix=suffix,
        bits=bits,
        sample_rate=format_.sample_rate,
        num_channels=format_.num_channels,
        data_type=data_type,
    )
    name = maybe_normalize_audio_and_cache.__name__
    cache = cf.partial(_cache_path)(audio_file.path, name, **kwargs_, **kwargs)
    if not cache.exists():
        lib.audio.normalize_audio(audio_file.path, cache, **kwargs_)
    return cache


def get_non_speech_segments_and_cache(
    audio_file: AudioMetadata,
    low_cut: int,
    frame_length: float,
    hop_length: float,
    threshold: float,
    **kwargs,
) -> Timeline:
    """Get non-speech segments in `audio_file` and cache."""
    kwargs_: typing.Dict[str, typing.Any] = dict(
        low_cut=low_cut, frame_length=frame_length, hop_length=hop_length, threshold=threshold
    )
    name = get_non_speech_segments_and_cache.__name__
    cache_path = cf.partial(_cache_path)(audio_file.path, name, ".npy", **kwargs_, **kwargs)
    if cache_path.exists():
        try:
            loaded = np.load(cache_path, allow_pickle=False)
            return Timeline(list(typing.cast(FloatFloat, tuple(t)) for t in loaded))
        except ValueError:
            cache_path.unlink()

    audio = read_audio(audio_file, memmap=True)
    vad: typing.List[FloatFloat] = lib.audio.get_non_speech_segments(audio, audio_file, **kwargs_)
    np.save(cache_path, np.array(vad), allow_pickle=False)
    return Timeline(vad)


_Float = typing.Union[np.floating, float]


class SpanGenerator(typing.Iterator[struc.Span]):
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
        max_pause: The maximum pause length between speech segments
        **kwargs: Additional key-word arguments passed `Timeline`.
    """

    def __init__(
        self,
        passages: typing.List[struc.Passage],
        max_seconds: float,
        max_pause: float,
        **kwargs,
    ):
        assert max_seconds > 0, "The maximum interval length must be a positive number."
        self.passages = [p for p in passages if len(p.speech_segments) > 0]
        if len(self.passages) != len(passages):
            message = "Filtered out %d of %d passages without speech segments."
            logger.warning(message, len(passages) - len(self.passages), len(passages))
        self.max_seconds = max_seconds
        self.max_pause = max_pause
        lengths = [p.segmented_audio_length() for p in self.passages]
        self._weights = torch.tensor(lengths)
        make_timeline: typing.Callable[[struc.Passage], Timeline]
        make_timeline = lambda p: Timeline(
            [(s.audio_start, s.audio_stop) for s in p.speech_segments], **kwargs
        )
        self._timelines = (
            None if max_seconds == math.inf else [make_timeline(p) for p in self.passages]
        )

    @staticmethod
    def _overlap(x1: _Float, x2: _Float, y1: _Float, y2: _Float) -> _Float:
        """Get the percentage overlap between x and the y slice."""
        if x2 == x1:
            return 1.0 if x1 >= y1 and x2 <= y2 else 0.0
        min_ = typing.cast(_Float, min(x2, y2))  # type: ignore
        max_ = typing.cast(_Float, max(x1, y1))  # type: ignore
        return (min_ - max_) / (x2 - x1)

    @functools.lru_cache(maxsize=None)
    def _is_include(self, x1: _Float, x2: _Float, y1: _Float, y2: _Float):
        return self._overlap(x1, x2, y1, y2) >= random.random()

    def next(self, length: float) -> typing.Optional[struc.Span]:
        if len(self.passages) == 0:
            raise StopIteration()

        # NOTE: For a sufficiently large `max_seconds`, the span length tends to be larger than the
        # passage length; therefore, the entire passage tends to be selected every time.
        if self.max_seconds == float("inf"):
            return random.choice(self.passages)[:]

        # NOTE: The `weight` is based on `start` (i.e. the number of spans)
        # NOTE: For some reason, `torch.multinomial(replacement=True)` is faster by a lot.
        index = int(torch.multinomial(self._weights + length, 1, replacement=True).item())
        assert self._timelines is not None
        passage, timeline = self.passages[index], self._timelines[index]

        # NOTE: Uniformly sample a span of audio.
        start = random.uniform(passage.audio_start - length, passage.audio_stop)
        stop = min(start + length, passage.audio_stop)
        start = max(start, passage.audio_start)

        # NOTE: Based on the overlap, decide which alignments to include in the span.
        indicies = list(timeline.indicies(slice(start, stop)))
        self._is_include.cache_clear()
        _filter = lambda i: self._is_include(timeline.start(i), timeline.stop(i), start, stop)
        begin = next((i for i in iter(indicies) if _filter(i)), None)
        end = next((i for i in reversed(indicies) if _filter(i)), None)
        if begin is None or end is None:
            return

        segments = passage.speech_segments[begin : end + 1]
        if len(segments) > 1:
            pairs = zip(segments, segments[1:])
            if any(b.audio_start - a.audio_stop > self.max_pause for a, b in pairs):
                return

        length_ = timeline.stop(end) - timeline.start(begin)
        if length_ > 0 and length_ <= self.max_seconds:
            slice_ = slice(segments[0].slice.start, segments[-1].slice.stop)
            audio_slice = slice(segments[0].audio_start, segments[-1].audio_stop)
            return passage.span(slice_, audio_slice)

    def __next__(self) -> struc.Span:
        while True:
            span = self.next(random.uniform(0, self.max_seconds))
            if span is not None:
                return span

    def __iter__(self) -> typing.Iterator[struc.Span]:
        return self


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
    transcript: str, alignments: typing.Tuple[struc.Alignment, ...]
) -> typing.Tuple[struc.Alignment, ...]:
    """Temporary fix for a bug in `sync_script_with_audio.py`.

    TODO: Remove after datasets are reprocessed.
    """
    return_: typing.List[struc.Alignment] = []
    for alignment in alignments:
        word = transcript[alignment.transcript[0] : alignment.transcript[1]]
        if word.strip() != word:
            update = (alignment.transcript[0] - 1, alignment.transcript[1] - 1)
            alignment = alignment._replace(transcript=update)
            corrected = transcript[alignment.transcript[0] : alignment.transcript[1]]
            logger.warning("Corrected '%s' to '%s'.", word, corrected)
        return_.append(alignment)
    return tuple(return_)


DataLoader = typing.Callable[[Path], typing.List[struc.Passage]]
DataLoaders = typing.Dict[struc.Speaker, DataLoader]


def dataset_loader(
    directory: Path,
    root_directory_name: str,
    gcs_path: str,
    speaker: struc.Speaker,
    alignments_directory_name: str = "alignments",
    alignments_suffix: str = ".json",
    recordings_directory_name: str = "recordings",
    recordings_suffix: str = ".wav",
    scripts_directory_name: str = "scripts",
    scripts_suffix: str = ".csv",
    text_column: str = "Content",
    strict: bool = False,
    add_tqdm: bool = False,
) -> typing.List[struc.Passage]:
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

    dataset: struc.UnprocessedDataset = []
    iterator = typing.cast(typing.Iterator[typing.Tuple[Path, Path, Path]], zip(*tuple(files)))
    for alignment_path, recording_path, script_path in iterator:
        scripts = pandas.read_csv(str(script_path.absolute()))
        json_ = json.loads(alignment_path.read_text())
        error = f"Each script ({script_path}) must have an alignment ({alignment_path})."
        assert len(scripts) == len(json_["alignments"]), error
        document = []
        for (_, script), alignments in zip(scripts.iterrows(), json_["alignments"]):
            alignments_ = tuple(struc.Alignment.from_json(a) for a in alignments)
            alignments_ = _temporary_fix_for_transcript_offset(json_["transcript"], alignments_)
            passage = struc.UnprocessedPassage(
                audio_path=recording_path,
                speaker=speaker,
                script=typing.cast(str, script[text_column]),
                transcript=json_["transcript"],
                alignments=alignments_,
                other_metadata={k: v for k, v in script.items() if k not in (text_column,)},
            )
            document.append(passage)
        dataset.append(document)
    is_linked = struc.IsLinked(transcript=True, audio=True)
    return struc.make_passages(root_directory_name, dataset, is_linked=is_linked, add_tqdm=add_tqdm)


def conventional_dataset_loader(
    directory: Path,
    speaker: struc.Speaker,
    metadata_path_template: str = "{directory}/metadata.csv",
    metadata_audio_column: typing.Union[str, int] = 0,
    metadata_text_column: typing.Union[str, int] = 2,
    metadata_kwargs={"quoting": csv.QUOTE_NONE, "header": None, "delimiter": "|"},
    audio_path_template: str = "{directory}/wavs/{file_name}.wav",
    additional_metadata: typing.Dict = {},
) -> typing.List[struc.UnprocessedPassage]:
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
    df = typing.cast(
        pandas.DataFrame, pandas.read_csv(metadata_path, **metadata_kwargs, keep_default_na=False)
    )
    get_audio_path = lambda n: Path(audio_path_template.format(directory=directory, file_name=n))
    handled_columns = [metadata_text_column, metadata_audio_column]
    get_other_metadata = lambda r: {k: v for k, v in r.items() if k not in handled_columns}
    return [
        struc.UnprocessedPassage(
            audio_path=get_audio_path(row[metadata_audio_column]),
            speaker=speaker,
            script=typing.cast(str, row[metadata_text_column]).strip(),
            transcript=typing.cast(str, row[metadata_text_column]).strip(),
            alignments=None,
            other_metadata={**get_other_metadata(row), **additional_metadata},
        )
        for _, row in df.iterrows()
    ]


def wsl_gcs_dataset_loader(
    directory: Path,
    speaker: struc.Speaker,
    gcs_path: str = "gs://wellsaid_labs_datasets",
    prefix: str = "",
    post_suffix="__manual_post",
    recordings_directory_name: str = "recordings",
    data_directory: str = "processed",
    **kwargs,
) -> typing.List[struc.Passage]:
    """
    Load WellSaid Labs dataset from Google Cloud Storage (GCS).

    The file structure is similar to:
        {bucket}/
        └── {speaker_label}/
            └── {data_directory}/
                ...

        {data_directory}/
                ├── {alignments_directory_name}/  # Alignments between recordings and scripts
                │   ├── audio1.json
                ├── {recordings_directory_name}/  # Voice overs
                │   ├── audio1.wav                # NOTE: Most audio file formats are accepted.
                │   └── ...
                ├── {recordings_directory_name}{post_suffix}/  # Optional post processed voice over
                │   ├── audio1.wav                # NOTE: Most audio file formats are accepted.
                │   └── ...
                └── {scripts_directory_name}/     # Voice over scripts with related metadata
                    ├── audio1-script.csv
                    └── ...

    Args:
        directory: Directory to cache the dataset.
        speaker: The speaker represented by this dataset.
        gcs_path: The base GCS path storing the data.
        prefix: The prefix path after `gcs_path` and before the speaker label.
        post_processing_suffix: An optional suffix added at the end of the speaker label.
        recordings_directory_name
        data_directory
        **kwargs
    """
    suffix = post_suffix if speaker.post else ""
    assert speaker.gcs_dir is not None
    label = speaker.gcs_dir.replace(post_suffix, "")
    gcs_path = "/".join([s for s in [gcs_path, prefix, label, data_directory] if len(s) > 0])
    kwargs = dict(recordings_directory_name=recordings_directory_name + suffix, **kwargs)
    return dataset_loader(directory, label, gcs_path, speaker, **kwargs)
