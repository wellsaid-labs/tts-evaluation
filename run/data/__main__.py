""" Command-line interface (CLI) for processing data files. """
import collections
import functools
import logging
import math
import multiprocessing
import pathlib
import re
import subprocess
import tempfile
import time
import typing

import hparams
import numpy as np
import tabulate
import tqdm
import typer
from third_party import LazyLoader

import lib
import run
from run._utils import gcs_uri_to_blob

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
    from spacy.lang import en as spacy_en
else:
    pandas = LazyLoader("pandas", globals(), "pandas")
    spacy_en = LazyLoader("spacy_en", globals(), "spacy.lang.en")


lib.environment.set_basic_logging_config()
logger = logging.getLogger(__name__)
app = typer.Typer(context_settings=dict(max_content_width=math.inf))
audio_app = typer.Typer()
app.add_typer(audio_app, name="audio")
csv_app = typer.Typer()
app.add_typer(csv_app, name="csv")
run._config.configure()


def _get_length(path: pathlib.Path) -> float:
    """ Helper for `_get_total_length`."""
    return lib.audio.get_audio_metadata([path])[0].length


def _get_total_length(paths: typing.List[pathlib.Path], max_parallel: int = 16) -> float:
    """ Get the sum of the lengths of each audio file in `paths`. """
    with multiprocessing.pool.ThreadPool(max_parallel) as pool:
        return sum(tqdm.tqdm(pool.imap_unordered(_get_length, paths), total=len(paths)))


@app.command()
def download():
    """Download the dataset."""
    run._config.get_dataset()


@app.command()
def rename(directory: pathlib.Path):
    """ Normalize the name of every directory and file in DIRECTORY."""
    for path in directory.glob("**/*"):
        normalized = path.name.replace(" ", "_").lower()
        if normalized != path.name:
            logger.info('Renaming file name "%s" to "%s"', path.name, normalized)
            path.rename(path.parent / normalized)


@app.command("diff")
def diff(gcs_uri: str, other_gcs_uri: str):
    """ View diff between GCS-URI and OTHER-GCS-URI. """
    blob = gcs_uri_to_blob(gcs_uri)
    file_ = tempfile.NamedTemporaryFile()
    path = pathlib.Path(file_.name)
    blob.download_to_filename(str(path))

    other_blob = gcs_uri_to_blob(other_gcs_uri)
    other_file = tempfile.NamedTemporaryFile()
    other_path = pathlib.Path(other_file.name)
    other_blob.download_to_filename(str(other_path))

    subprocess.run(f"code --diff {path.absolute()} {other_path.absolute()}", shell=True)

    while True:
        time.sleep(1)  # NOTE: Keep the temporary files around, until the process is exited.


@audio_app.command()
def loudness(paths: typing.List[pathlib.Path]):
    """ Print the loudness for each file in PATHS. """
    # TODO: Get loudness faster by...
    # - Adding parallel processing with chunking for large audio files
    # - Find a faster loudness implementation
    results = []
    progress_bar = tqdm.tqdm(total=round(_get_total_length(paths)))
    for path in paths:
        metadata = lib.audio.get_audio_metadata([path])[0]
        meter = lib.audio.get_pyloudnorm_meter(metadata.sample_rate, "DeMan")
        audio = lib.audio.read_audio(path)
        lufs = meter.integrated_loudness(audio)
        progress_bar.update(round(metadata.length))
        results.append((lufs, metadata.path))
    typer.echo(tabulate.tabulate(sorted(results), headers=["LUFS", "Path"]))


class _SharedAudioFileMetadata(typing.NamedTuple):
    sample_rate: int
    num_channels: int
    encoding: str
    bit_rate: str
    precision: str


def _metadata(path: pathlib.Path) -> typing.Tuple[pathlib.Path, _SharedAudioFileMetadata]:
    """ Helper for the `metadata` command."""
    metadata = lib.audio.get_audio_metadata([path])[0]
    return path, _SharedAudioFileMetadata(
        sample_rate=metadata.sample_rate,
        num_channels=metadata.num_channels,
        encoding=metadata.encoding,
        bit_rate=metadata.bit_rate,
        precision=metadata.precision,
    )


@audio_app.command()
def metadata(paths: typing.List[pathlib.Path], max_parallel: int = typer.Option(16)):
    """ Print the metadata for each file in PATHS. """
    num_parallel = lib.utils.clamp(len(paths), max_=max_parallel)
    with multiprocessing.pool.ThreadPool(num_parallel) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(_metadata, paths), total=len(paths)))

    groups = collections.defaultdict(list)
    for path, metadata in results:
        groups[metadata].append(path)
    for metadata, paths in groups.items():
        logger.info("Found %d files with `%s` metadata:\n%s", len(paths), metadata, paths)


@audio_app.command("normalize")
def audio_normalize(
    paths: typing.List[pathlib.Path],
    dest: pathlib.Path,
    encoding: typing.Optional[str] = typer.Option(None),
):
    """ Normalize audio file format(s) in PATHS and save to DEST. """
    params = hparams.HParams(audio_filters=lib.audio.AudioFilters(""))
    if encoding is not None:
        params.update(encoding=encoding)
    hparams.add_config({lib.audio.normalize_audio: params})

    progress_bar = tqdm.tqdm(total=round(_get_total_length(paths)))
    for path in paths:
        dest_path = dest / path.name
        if dest_path.exists():
            logger.error("Skipping, file already exists: %s", dest_path)
        else:
            lib.audio.normalize_audio(path, dest_path)
            progress_bar.update(round(lib.audio.get_audio_metadata([path])[0].length))


def _csv_normalize(text: str, nlp: spacy_en.English) -> str:
    """Helper for the `csv_normalize` command.

    TODO: Consider adding:
    - Non-standard word verbalizer
    - Grammar and spell check, for example:
      https://github.com/explosion/spaCy/issues/315
      https://pypi.org/project/pyspellchecker/
      https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
    - Visualize any text changes for quality assurance
    - Remove any stray HTML
    - Visualize any strange words that may need to be normalized
    """
    text = lib.text.normalize_vo_script(text)
    text = text.replace("®", "")
    text = text.replace("™", "")
    # NOTE: Remove HTML tags
    text = re.sub("<.*?>", "", text)
    # NOTE: Fix for a missing space between end and beginning of a sentence. Example match is
    # deliminated with angle brackets, see here:
    #   the cold w<ar.T>he term 'business ethics'
    text = lib.text.add_spaces_between_sentences(nlp(text))
    return text


@csv_app.command("normalize")
def csv_normalize(paths: typing.List[pathlib.Path], dest: pathlib.Path):
    """Normalize csv file(s) in PATHS and save to DEST."""
    nlp = lib.text.load_en_core_web_md(disable=("tagger", "ner"))
    partial = functools.partial(_csv_normalize, nlp=nlp)
    for path in tqdm.tqdm(paths):
        dest_path = dest / path.name
        if dest_path.exists():
            logger.error("Skipping, file already exists: %s", dest_path)
        else:
            data_frame = typing.cast(pandas.DataFrame, pandas.read_csv(path))
            data_frame = data_frame.applymap(partial)
            data_frame.to_csv(dest_path, index=False)


@csv_app.command()
def combine(
    csvs: typing.List[pathlib.Path] = typer.Argument(..., help="List of CSVs to combine."),
    csv: pathlib.Path = typer.Argument(..., help="Combined CSV filename."),
):
    """Combine a list of CSVS into one CSV.

    Also, this adds an additional "__csv" column with the original filename.
    """
    df = pandas.read_csv(csvs[0])
    df["__csv"] = csvs[0]
    for csv in csvs[1:]:
        df_csv = pandas.read_csv(csv)
        df_csv["__csv"] = csv
        df = df.append(df_csv, ignore_index=True)
    df.to_csv(csv, index=False)


@csv_app.command()
def shuffle(
    source: pathlib.Path = typer.Option(...), destination: pathlib.Path = typer.Option(...)
):
    """ Shuffle SOURCE csv and save it to DESTINATION. """
    df = pandas.read_csv(source)
    df = typing.cast(pandas.DataFrame, df.iloc[np.random.permutation(len(df))])  # type: ignore
    df.to_csv(destination, index=False)


@csv_app.command()
def prefix(
    source: pathlib.Path = typer.Option(...),
    destination: pathlib.Path = typer.Option(...),
    column: str = typer.Option(...),
    prefix: str = typer.Option(...),
):
    """ Add a PREFIX to every value in the SOURCE csv under COLUMN and save to DESTINATION. """
    df = pandas.read_csv(source)
    df[column] = prefix + df[column].astype(str)
    df.to_csv(destination, index=False)


if __name__ == "__main__":
    app()
