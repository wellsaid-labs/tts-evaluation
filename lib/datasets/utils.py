import json
import logging
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
    import librosa
    import pandas
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
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


class Example(typing.NamedTuple):
    """Given the `text`, this is an `Example` voice-over stored at the `audio_path`.

    Args:
        alignments: List of alignments between `text` and `audio_path`.
        text: The text read in `audio_path`.
        audio_path: A voice over of the `text.`
        speaker: The voice.
        metadata: Additional metadata associated with this example.
    """

    audio_path: Path
    speaker: Speaker
    alignments: typing.Optional[typing.Tuple[Alignment, ...]] = None
    text: str = ""
    metadata: typing.Dict[str, typing.Any] = {}


def dataset_generator(
    data: typing.List[Example], max_seconds: float
) -> typing.Generator[Example, None, None]:
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
            yield random.choice(data)

    min_ = lambda e: e.alignments[0].audio[0]
    max_ = lambda e: e.alignments[-1].audio[1]
    offset = lambda e: floor(min_(e))

    # NOTE: `lookup` allows fast lookups of alignments for a point in time.
    lookup: typing.List[typing.List[typing.List[int]]]
    lookup = [[[] for _ in range(ceil(max_(e)) - offset(e) + 1)] for e in data]
    for i, example in enumerate(data):
        assert example.alignments is not None, "`alignments` must be defined."
        for j, alignment in enumerate(example.alignments):
            for k in range(int(floor(alignment.audio[0])), int(ceil(alignment.audio[1])) + 1):
                lookup[i][k - offset(example)].append(j)

    weights = torch.tensor([float(max_(e) - min_(e)) for e in data])
    while True:
        length = random.uniform(0, max_seconds)
        # NOTE: The `weight` is based on `start` (i.e. the number of spans)
        index = int(torch.multinomial(weights + length, 1).item())
        example = data[index]
        assert example.alignments is not None, "`alignments` must be defined."
        # NOTE: Uniformly sample a span of audio.
        start = random.uniform(min_(example) - length, max_(example))
        end = min(start + length, max_(example))
        start = max(start, min_(example))

        # NOTE: Based on the overlap, decide which alignments to include in the span.
        part = lib.utils.flatten(
            lookup[index][int(start) - offset(example) : int(end) - offset(example) + 1]
        )
        get = lambda i: example.alignments[i].audio
        overlap = lambda i: (min(end, get(i)[1]) - max(start, get(i)[0])) / (get(i)[1] - get(i)[0])
        random_ = lru_cache(maxsize=None)(lambda i: random.random())
        bounds = (
            next((i for i in part if overlap(i) >= random_(i)), None),
            next((i for i in reversed(part) if overlap(i) >= random_(i)), None),
        )
        if (
            bounds[0] is not None
            and bounds[1] is not None
            and bounds[0] <= bounds[1]
            and get(bounds[1])[1] - get(bounds[0])[0] > 0
            and get(bounds[1])[1] - get(bounds[0])[0] <= max_seconds
        ):
            yield example._replace(alignments=tuple(example.alignments[bounds[0] : bounds[1] + 1]))


def dataset_loader(
    directory: Path,
    root_directory_name: str,
    gcs_path: str,
    speaker: Speaker,
    alignments_directory_name: str = "alignments",
    recordings_directory_name: str = "recordings",
    scripts_directory_name: str = "scripts",
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
        recordings_gcs_path: The name of the voice over directory on GCS.
        scripts_gcs_path: The name of the voice over script directory on GCS.
        text_column: The voice over script column in the CSV script files.

    Returns: List of voice-over examples in the dataset.
    """
    logger.info("Loading `%s` speech dataset", root_directory_name)

    root = (Path(directory) / root_directory_name).absolute()
    root.mkdir(exist_ok=True)
    names = [
        alignments_directory_name,
        recordings_directory_name,
        scripts_directory_name,
    ]
    directories = [root / d for d in names]
    for directory, suffix in zip(directories, (".json", "", ".csv")):
        directory.mkdir(exist_ok=True)
        command = f"gsutil -m cp -n {gcs_path}/{directory.name}/*{suffix} {directory}/"
        subprocess.run(command.split(), check=True)

    files = (sorted(d.iterdir(), key=lambda p: lib.text.natural_keys(p.name)) for d in directories)
    examples = []
    for alignment_file_path, recording_file_path, script_file_path in zip(*tuple(files)):
        scripts = pandas.read_csv(str(script_file_path.absolute()))
        script_alignments: typing.List[typing.List[typing.List[typing.List[float]]]]
        script_alignments = json.loads(alignment_file_path.read_text())
        assert len(scripts) == len(script_alignments)
        for (_, script), alignments in zip(scripts.iterrows(), script_alignments):
            assert lib.text.is_normalized_vo_script(
                script[text_column]
            ), "The script must be normalized."
            example_alignments = [
                Alignment((int(a[0][0]), int(a[0][1])), (a[1][0], a[1][1])) for a in alignments
            ]
            example = Example(
                audio_path=recording_file_path,
                speaker=speaker,
                alignments=tuple(example_alignments),
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
    return [
        Example(
            text=row[metadata_text_column],
            audio_path=Path(
                metadata_audio_path.format(
                    directory=directory,
                    root_directory_name=root_directory_name,
                    metadata_audio_column_value=row[metadata_audio_column],
                )
            ),
            speaker=speaker,
            metadata={
                k: v
                for k, v in row.items()
                if k not in [metadata_text_column, metadata_audio_column]
            },
        )
        for _, row in pandas.read_csv(Path(metadata_filename), **kwargs).iterrows()
    ]
