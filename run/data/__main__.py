""" Command-line interface (CLI) for processing data files.

TODO: Should we add a tool to check for noise floors? We could report back the lowest decibel
seen in a similar way to `lib.audio.get_non_speech_segments`.
TODO: Ensure that CSV column names are consistent in the various Python data processing modules.
TODO: Automatically detect if a file is of TSV or CSV format in `csv_normalize`.
"""
import collections
import dataclasses
import functools
import logging
import math
import pathlib
import re
import shlex
import subprocess
import sys
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
from run.data import _loader

if typing.TYPE_CHECKING:  # pragma: no cover
    import Levenshtein
    import pandas
    from spacy.lang import en as spacy_en
else:
    Levenshtein = LazyLoader("Levenshtein", globals(), "Levenshtein")
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


def _get_total_length(paths: typing.List[pathlib.Path]) -> float:
    """Get the sum of the lengths of each audio file in `paths`."""
    return sum([m.length for m in lib.audio.get_audio_metadata(paths)])


def _is_include(
    speaker_name: str,
    include: typing.List[str],
    exclude: typing.List[str],
    normalize: typing.Callable[[str], str] = lambda s: s.lower(),
) -> bool:
    """Include `speaker_name` if it matches any substring in `include` and it doesn't match
    any substring in `exclude`."""
    is_exclude = any(normalize(e) in normalize(speaker_name) for e in exclude)
    return any(normalize(i) in normalize(speaker_name) for i in include) and not is_exclude


@app.command()
def download(
    include: typing.List[str] = typer.Option(
        [""], help="Only download speakers whose name includes the substring(s) --include."
    ),
    exclude: typing.List[str] = typer.Option(
        [], help="Only download speakers whose name excludes the substring(s) --exclude."
    ),
):
    """Download dataset(s)."""
    is_include = functools.partial(_is_include, include=include, exclude=exclude)
    datasets = {k: v for k, v in run.data._loader.DATASETS.items() if is_include(k.label)}
    [loader(run._config.DATA_PATH) for loader in datasets.values()]


def _file_numberings(directory: pathlib.Path) -> typing.List[str]:
    """Get every file numbering in `directory`."""
    numbers = lambda n: "-".join([str(int(i)) for i in re.findall(r"\d+", n)])
    return sorted([numbers(p.stem) for p in directory.iterdir() if p.is_file()])


@app.command()
def numberings(
    directory: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
    other_directory: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
):
    """Check that DIRECTORY and OTHER_DIRECTORY have files with similar numberings."""
    numberings = _file_numberings(directory)
    other_numberings = _file_numberings(other_directory)
    message = f"Directories did not have equal numberings:\n{numberings}\n{other_numberings}"
    assert numberings == other_numberings, message
    logger.info(f"The file numberings match up! {lib.utils.mazel_tov()}")


@app.command()
def pair(
    recording: typing.List[pathlib.Path] = typer.Option(..., exists=True, dir_okay=False),
    script: typing.List[pathlib.Path] = typer.Option(..., exists=True, dir_okay=False),
    encoding: str = typer.Option("utf-8"),
):
    """Sort and analyze pairs consisting of a RECORING and a SCRIPT."""
    assert len(recording) == len(script)
    rows = []
    iterator = tqdm.tqdm(zip(lib.audio.get_audio_metadata(recording), script), total=len(script))
    for metadata, script_ in iterator:
        num_characters = len(script_.read_text(encoding=encoding))
        row = {
            "Script Name": script_.name,
            "Recording Name": metadata.path.name,
            "Number of Characters": num_characters,
            "Number of Seconds": metadata.length,
            "Seconds per Character": metadata.length / float(num_characters),
        }
        rows.append(row)
    rows = sorted(rows, key=lambda r: r["Seconds per Character"])
    typer.echo("\n")
    typer.echo(tabulate.tabulate(rows, headers="keys"))


def _normalize_file_name(name):
    """Learn more about this normalization:
    https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
    https://stackoverflow.com/questions/7593969/regex-to-split-camelcase-or-titlecase-advanced/7599674#7599674
    """
    name = name.replace("_", " ")
    # NOTE: Put a space before the start of a non-lower case sequence, non-closing bracket, non
    # numeric sequence.
    name = re.sub(r"([^a-z0-9\)\]\-]+)", r" \1", name)
    # NOTE: Put a spaces surrounding a "|" character.
    name = re.sub(r"([\|]+)", r" \1 ", name)
    name = "_".join(re.sub(r"([A-Z][a-z]+)", r" \1", name).split()).lower()
    return name.replace("_-", "-").replace("-_", "-")


_ONLY_NUMBERS_HELP = "Include only dashes and numbers in the normalized file name."


@app.command()
def rename(
    directory: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
    only_numbers: bool = typer.Option(False, help=_ONLY_NUMBERS_HELP),
):
    """Normalize the name of every directory and file in DIRECTORY."""
    paths = list(directory.glob("**/*"))
    updates = []
    for path in paths:
        stem = _normalize_file_name(path.stem)
        if only_numbers and any(c.isdigit() for c in stem):
            stem = "-".join(re.findall(r"\d+", stem))
        elif only_numbers:
            logger.error("Skipping, path has no numbers: %s", path)
        updates.append((path, path.parent / (stem + path.suffix)))

    unique = set()
    for _, updated in updates:
        message = f"Found duplicate file name ({updated.name}) after normalization."
        assert updated not in unique, message
        unique.add(updated)

    for path, updated in updates:
        if updated != path:
            logger.info('Renaming file name "%s" to "%s"', path.name, updated.name)
            path.rename(updated)
        else:
            logger.info("Skipping, file already exists: %s", updated)


def _download(gcs_uri: str) -> typing.Tuple[typing.IO[bytes], str]:
    """Helper function for `diff`."""
    blob = gcs_uri_to_blob(gcs_uri)
    file_ = tempfile.NamedTemporaryFile(prefix=blob.name.split(".")[0].split("/")[-1])
    path = pathlib.Path(file_.name)
    blob.download_to_filename(str(path))
    return file_, shlex.quote(str(path.absolute()))


@app.command("diff")
def diff(gcs_uri: str, other_gcs_uri: str):
    """View diff between GCS-URI and OTHER-GCS-URI."""
    file_, path = _download(gcs_uri)
    other_file, other_path = _download(other_gcs_uri)
    subprocess.run(f"code --diff {path} {other_path}", shell=True)
    while True:
        time.sleep(1)  # NOTE: Keep the temporary files around until the process is exited.


@audio_app.command()
def loudness(paths: typing.List[pathlib.Path] = typer.Argument(..., exists=True, dir_okay=False)):
    """Print the loudness for each file in PATHS."""
    # TODO: Get loudness faster by...
    # - Adding parallel processing with chunking for large audio files
    # - Find a faster loudness implementation
    results = []
    progress_bar = tqdm.tqdm(total=round(_get_total_length(paths)))
    for path in paths:
        metadata = lib.audio.get_audio_metadata(path)
        meter = lib.audio.get_pyloudnorm_meter(metadata.sample_rate, "DeMan")
        audio = lib.audio.read_audio(path)
        lufs = meter.integrated_loudness(audio)
        progress_bar.update(round(metadata.length))
        results.append((lufs, metadata.path))
    progress_bar.close()
    typer.echo("\n")
    typer.echo(tabulate.tabulate(sorted(results), headers=["LUFS", "Path"]))


@audio_app.command()
def print_format(
    paths: typing.List[pathlib.Path] = typer.Argument(..., exists=True, dir_okay=False)
):
    """Print the format for each file in PATHS."""
    metadatas = lib.audio.get_audio_metadata(paths)
    groups = collections.defaultdict(list)
    for metadata in metadatas:
        fields = dataclasses.fields(lib.audio.AudioFormat)
        format_ = lib.audio.AudioFormat(**{f.name: getattr(metadata, f.name) for f in fields})
        groups[format_].append(metadata.path)
    for format_, paths in groups.items():
        list_ = "\n".join([str(p.relative_to(p.parent.parent)) for p in paths])
        logger.info("Found %d file(s) with `%s` audio file format:\n%s", len(paths), format_, list_)


@audio_app.command("normalize")
def audio_normalize(
    paths: typing.List[pathlib.Path] = typer.Argument(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
    data_type: typing.Optional[lib.audio.AudioDataType] = typer.Option(None),
    bits: typing.Optional[int] = typer.Option(None),
):
    """Normalize audio file format(s) in PATHS and save to directory DEST."""
    if bits is not None:
        hparams.add_config({_loader.normalize_audio: hparams.HParams(bits=bits)})
    if data_type is not None:
        hparams.add_config({_loader.normalize_audio: hparams.HParams(data_type=data_type)})

    progress_bar = tqdm.tqdm(total=round(_get_total_length(paths)))
    for path in paths:
        dest_path = _loader.normalize_audio_suffix(dest / path.name)
        if dest_path.exists():
            logger.error("Skipping, file already exists: %s", dest_path)
        else:
            _loader.normalize_audio(path, dest_path)
            progress_bar.update(round(lib.audio.get_audio_metadata(path).length))


@csv_app.command()
def text(
    paths: typing.List[pathlib.Path] = typer.Argument(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
    column: str = "Content",
    encoding: str = typer.Option("utf-8"),
):
    """Convert text file(s) in PATHS to CSV file(s), and save to directory DEST with one row and one
    column."""
    for path in tqdm.tqdm(paths):
        dest_path = dest / (path.stem + ".csv")
        if dest_path.exists():
            logger.error("Skipping, file already exists: %s", dest_path)
            continue
        text = path.read_text(encoding=encoding)
        pandas.DataFrame({column: [text]}).to_csv(dest_path, index=False)


def _csv_normalize(text: str, nlp: typing.Optional[spacy_en.English], decode: bool = True) -> str:
    """Helper for the `csv_normalize` command.

    TODO: Consider adding:
    - Non-standard word verbalizer
    - Grammar and spell check, for example:
      https://github.com/explosion/spaCy/issues/315
      https://pypi.org/project/pyspellchecker/
      https://textblob.readthedocs.io/en/dev/quickstart.html#spelling-correction
    - Visualize any text changes for quality assurance
    - Visualize any strange words that may need to be normalized
    """
    text = lib.text.normalize_vo_script(text, decode)
    text = text.replace("®", "")
    text = text.replace("™", "")
    # NOTE: Remove HTML tags
    text = re.sub("<.*?>", "", text)
    # TODO: For non-English, could use nlp if available. Didn't seem critical for German case.
    # NOTE: Fix for a missing space between end and beginning of a sentence. For example:
    #   the cold war.The term 'business ethics'
    if nlp:
        text = lib.text.add_space_between_sentences(nlp(text))
    return text


def _read_csv(
    path: pathlib.Path,
    required_column: str,
    optional_columns: typing.List[str] = [],
    encoding: str = "utf-8",
) -> pandas.DataFrame:
    """Read a CSV or TSV file with required and optional column(s).

    TODO: Improve the TSV vs CSV logic so that it is more precise.
    """
    # NOTE: There is a bug with `setprofile` and `read_csv`:
    # https://github.com/pandas-dev/pandas/issues/41069
    sys.setprofile(None)

    read_csv = lambda *a, **k: typing.cast(pandas.DataFrame, pandas.read_csv(*a, **k))
    all_columns = list(optional_columns) + [required_column]

    text = path.read_text(encoding=encoding)
    separator = ","
    if text.count("\t") > len(text.split("\n")) // 2:
        message = "There are a lot of tabs (%d) so this (%s) will be parsed as a TSV file."
        logger.warning(message, text.count("\t"), path)
        separator = "\t"

    read_csv_kwgs = dict(sep=separator, keep_default_na=False, index_col=False)
    data_frame = read_csv(path, **read_csv_kwgs)
    if not any(c in data_frame.columns for c in all_columns):
        logger.warning("None of the optional or required column(s) were found...")
        if len(data_frame.columns) == 1:
            message = "There is only 1 column so this will assume that column is the "
            message += f"reqiured '{required_column}' column."
            logger.warning(message)
            data_frame = read_csv(path, header=None, names=[required_column], **read_csv_kwgs)
    if required_column not in data_frame.columns:
        message = f"The required '{required_column}' column couldn't be found or inferred."
        logger.error(message)
        raise typer.Exit(code=1)

    dropped = list(set(data_frame.columns) - set(all_columns))
    if len(dropped) > 0:
        logger.warning("[%s] Dropping extra columns: %s", path.name, dropped)
    data_frame = data_frame.drop(columns=dropped)
    for column in optional_columns:
        if column not in data_frame.columns:
            logger.warning("[%s] Adding missing column: '%s'", path.name, column)
            data_frame[column] = ""
    return data_frame[all_columns]


@csv_app.command("normalize")
def csv_normalize(
    paths: typing.List[pathlib.Path] = typer.Argument(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Argument(..., exists=True, file_okay=False),
    no_spacy: bool = typer.Option(False, "--no_spacy"),
    no_decode: bool = typer.Option(False, "--no_decode"),
    required_column: str = typer.Option("Content"),
    optional_columns: typing.List[str] = typer.Option(["Source", "Title"]),
    encoding: str = typer.Option("utf-8"),
):
    """Normalize csv file(s) in PATHS and save to directory DEST."""
    nlp = None if no_spacy else lib.text.load_en_core_web_md(disable=("tagger", "ner"))

    results = []
    for path in tqdm.tqdm(paths):
        dest_path = dest / path.name
        if dest_path.exists():
            logger.error("Skipping, file already exists: %s", dest_path)
            continue

        text = path.read_text(encoding=encoding)
        data_frame = _read_csv(path, required_column, optional_columns, encoding)
        data_frame = data_frame.applymap(
            functools.partial(_csv_normalize, nlp=nlp, decode=not no_decode)
        )
        data_frame.to_csv(dest_path, index=False)

        # TODO: Count the number of alphanumeric edits instead of punctuation mark edits.
        num_edits = Levenshtein.distance(text, dest_path.read_text())  # type: ignore
        results.append(((num_edits / len(text)) * 100, num_edits, len(text), path.name))

    headers = ["% Edited", "# Edits", "# Characters", "File Name"]
    typer.echo("\n" + tabulate.tabulate(sorted(results), headers=headers))


@csv_app.command()
def combine(
    csvs: typing.List[pathlib.Path] = typer.Option(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Argument(...),
):
    """Combine a list of CSVS into DEST csv.

    Also, this adds an additional "__csv" column with the original filename.
    """
    assert dest.parent.exists(), "DEST parent directory must exist."
    assert not dest.exists(), "DEST must not exist."
    df = pandas.read_csv(csvs[0])
    df["__csv"] = csvs[0]
    for csv in csvs[1:]:
        df_csv = pandas.read_csv(csv)
        df_csv["__csv"] = csv
        df = df.append(df_csv, ignore_index=True)
    df.to_csv(dest, index=False)


@csv_app.command()
def shuffle(
    source: pathlib.Path = typer.Option(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Option(...),
):
    """Shuffle SOURCE csv and save it to DEST csv."""
    assert dest.parent.exists(), "DEST parent directory must exist."
    assert not dest.exists(), "DEST must not exist."
    df = pandas.read_csv(source)
    df = typing.cast(pandas.DataFrame, df.iloc[np.random.permutation(len(df))])  # type: ignore
    df.to_csv(dest, index=False)


@csv_app.command()
def prefix(
    source: pathlib.Path = typer.Option(..., exists=True, dir_okay=False),
    dest: pathlib.Path = typer.Option(...),
    column: str = typer.Option(...),
    prefix: str = typer.Option(...),
):
    """Add a PREFIX to every value in the SOURCE csv under COLUMN and save to DEST csv."""
    assert dest.parent.exists(), "DEST parent directory must exist."
    assert not dest.exists(), "DEST must not exist."
    df = pandas.read_csv(source)
    df[column] = prefix + df[column].astype(str)
    df.to_csv(dest, index=False)


if __name__ == "__main__":
    app()
