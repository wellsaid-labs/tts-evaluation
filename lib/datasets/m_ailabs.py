"""
Module to download and process the M-AILABS speech dataset
http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/.

Most of the data is based on LibriVox and Project Gutenberg. The training data consist of nearly
thousand hours of audio and the text-files in prepared format.

A transcription is provided for each clip. Clips vary in length from 1 to 20 seconds and have a
total length of approximately shown in the list (and in the respective info.txt-files) below.

The texts were published between 1884 and 1964, and are in the public domain. The audio was
recorded by the LibriVox project and is also in the public domain – except for Ukrainian.

M-AILABS directory structure follows this format:

    en_US/by_book/[Speaker Gender]/[Speaker Name]/[Book Title]

Within each book directory, there's metadata.csv file and a wavs directory following the
convention of the LJSpeech dataset.

The current implementation uses types for books, genders, speakers to allows robust error checking.
"""
import csv
import logging
import os
import typing
from pathlib import Path

from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract

import lib
from lib.datasets.utils import Example, Speaker

if typing.TYPE_CHECKING:  # pragma: no cover
    import pandas
else:
    pandas = LazyLoader("pandas", globals(), "pandas")

logger = logging.getLogger(__name__)


class Book(typing.NamedTuple):
    speaker: Speaker
    title: str


JUDY_BIEBER = Speaker("Judy Bieber", "female")
MARY_ANN = Speaker("Mary Ann", "female")
ELLIOT_MILLER = Speaker("Elliot Miller", "male")
ELIZABETH_KLETT = Speaker("Elizabeth Klett", "female")

THE_SEA_FAIRIES = Book(JUDY_BIEBER, "the_sea_fairies")
THE_MASTER_KEY = Book(JUDY_BIEBER, "the_master_key")
RINKITINK_IN_OZ = Book(JUDY_BIEBER, "rinkitink_in_oz")
DOROTHY_AND_WIZARD_OZ = Book(JUDY_BIEBER, "dorothy_and_wizard_oz")
SKY_ISLAND = Book(JUDY_BIEBER, "sky_island")
OZMA_OF_OZ = Book(JUDY_BIEBER, "ozma_of_oz")
EMERALD_CITY_OF_OZ = Book(JUDY_BIEBER, "emerald_city_of_oz")

MIDNIGHT_PASSENGER = Book(MARY_ANN, "midnight_passenger")
NORTH_AND_SOUTH = Book(MARY_ANN, "northandsouth")

PIRATES_OF_ERSATZ = Book(ELLIOT_MILLER, "pirates_of_ersatz")
POISONED_PEN = Book(ELLIOT_MILLER, "poisoned_pen")
SILENT_BULLET = Book(ELLIOT_MILLER, "silent_bullet")
HUNTERS_SPACE = Book(ELLIOT_MILLER, "hunters_space")
PINK_FAIRY_BOOK = Book(ELLIOT_MILLER, "pink_fairy_book")

JANE_EYRE = Book(ELIZABETH_KLETT, "jane_eyre")
WIVES_AND_DAUGHTERS = Book(ELIZABETH_KLETT, "wives_and_daughters")


def m_ailabs_en_us_judy_bieber_speech_dataset(
    *args,
    books=[
        THE_SEA_FAIRIES,
        THE_MASTER_KEY,
        RINKITINK_IN_OZ,
        DOROTHY_AND_WIZARD_OZ,
        SKY_ISLAND,
        OZMA_OF_OZ,
        EMERALD_CITY_OF_OZ,
    ],
):
    return m_ailabs_en_us_speech_dataset(*args, books=books)  # type: ignore


def m_ailabs_en_us_mary_ann_speech_dataset(*args, books=[MIDNIGHT_PASSENGER, NORTH_AND_SOUTH]):
    return m_ailabs_en_us_speech_dataset(*args, books=books)  # type: ignore


def m_ailabs_en_us_elliot_miller_speech_dataset(
    *args,
    books=[PIRATES_OF_ERSATZ, POISONED_PEN, SILENT_BULLET, HUNTERS_SPACE, PINK_FAIRY_BOOK],
):
    return m_ailabs_en_us_speech_dataset(*args, books=books)  # type: ignore


def m_ailabs_en_uk_elizabeth_klett_speech_dataset(*args, books=[JANE_EYRE, JANE_EYRE]):
    return m_ailabs_en_us_speech_dataset(*args, books=books)  # type: ignore


def m_ailabs_en_us_speech_dataset(
    directory,
    url="http://www.caito.de/data/Training/stt_tts/en_US.tgz",
    check_files=["en_US/by_book/info.txt"],
    extracted_name="en_US",
    books=[
        THE_SEA_FAIRIES,
        THE_MASTER_KEY,
        RINKITINK_IN_OZ,
        DOROTHY_AND_WIZARD_OZ,
        SKY_ISLAND,
        OZMA_OF_OZ,
        EMERALD_CITY_OF_OZ,
        MIDNIGHT_PASSENGER,
        NORTH_AND_SOUTH,
        PIRATES_OF_ERSATZ,
        POISONED_PEN,
        SILENT_BULLET,
        HUNTERS_SPACE,
        PINK_FAIRY_BOOK,
    ],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `en_US` dataset.

    The dataset is 8GB compressed. The file to be downloaded is called `en_US.tgz`. It contains 102
    hours of audio. When extracted, it creates a list of 14 books.

    NOTE: Based on 100 clips from the M-AILABS dataset, around 10% of the clips would end too early.
    Furthermore, it seemed like the text was verbalized accuractely.
    """
    return _m_ailabs_speech_dataset(directory, extracted_name, url, check_files, books, **kwargs)


def m_ailabs_en_uk_speech_dataset(
    directory,
    url="http://www.caito.de/data/Training/stt_tts/en_UK.tgz",
    check_files=["en_UK/by_book/info.txt"],
    extracted_name="en_UK",
    books=[JANE_EYRE, WIVES_AND_DAUGHTERS],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `en_UK` dataset.

    The dataset is 4GB compressed. The file to be downloaded is called `en_US.tgz`. It contains
    45 hours of audio. When extracted, it creates a list of 2 books.
    """
    return _m_ailabs_speech_dataset(directory, extracted_name, url, check_files, books, **kwargs)


def _m_ailabs_speech_dataset(
    directory: typing.Union[str, Path],
    extracted_name: str,
    url: str,
    check_files: typing.List[str],
    books: typing.List[Book],
    root_directory_name: str = "M-AILABS",
    metadata_pattern: str = "**/metadata.csv",
    metadata_path_column: str = "metadata_path",
    metadata_audio_column: typing.Union[str, int] = 0,
    metadata_audio_path: str = "wavs/{}.wav",
    metadata_text_column: typing.Union[str, int] = 2,
    metadata_quoting: int = csv.QUOTE_NONE,
    metadata_delimiter: str = "|",
    metadata_header: typing.Optional[bool] = None,
) -> typing.List[Example]:
    """Download, extract, and process a M-AILABS dataset.

    NOTE: The original URL is `http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/`. Use
    `curl -I <URL>` to find the redirected URL.

    Args:
        directory: Directory to cache the dataset.
        extracted_name: Name of the extracted dataset directory.
        url: URL of the dataset `tar.gz` file.
        check_files: These file(s) should exist if the download was successful.
        books: List of books to load.
        root_directory_name: Name of the dataset directory.
        metadata_pattern: Pattern for all `metadata.csv` files containing (filename, text)
            information.
        metadata_path_column: Column name to store the metadata path.
        metadata_audio_column: Column name or index with the audio filename.
        metadata_audio_path: Given the audio column, this template determines the filename.
        metadata_text_column: Column name or index with the audio transcript.
        metadata_quoting: Control field quoting behavior per `csv.QUOTE_*` constants for the
            metadata file.
        metadata_delimiter: Delimiter for the metadata file.
        metadata_header: If `True`, `metadata_file` has a header to parse.

    Returns: List of voice-over examples in the dataset.
    """
    logger.info("Loading `M-AILABS %s` speech dataset", extracted_name)
    directory = Path(directory) / root_directory_name
    directory.mkdir(exist_ok=True)
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)

    # NOTE: This makes sure that the download succeeds by checking against defined books in `books`.
    metadata_paths = list((directory / extracted_name).glob(metadata_pattern))
    downloaded_books = set([_path_to_book(path, directory=directory) for path in metadata_paths])
    assert len(set(books) - downloaded_books) == 0

    generator = _read_m_ailabs_data(
        books,
        directory,
        extracted_name,
        metadata_path_column,
        quoting=metadata_quoting,
        header=metadata_header,
        delimiter=metadata_delimiter,
    )
    data = list(generator)
    _get_audio_path = lambda r: Path(
        r[metadata_path_column].parent,
        metadata_audio_path.format(r[metadata_audio_column]),  # type: ignore
    )
    audio_paths = [_get_audio_path(r) for r in data]
    audio_metadatas = lib.audio.get_audio_metadata(audio_paths)
    texts = [r[metadata_text_column].strip() for r in data]  # type: ignore
    books = [_path_to_book(r[metadata_path_column], directory=directory) for r in data]
    iterator = zip(audio_paths, audio_metadatas, texts, books)
    return [
        Example(
            audio_path=audio_path,
            speaker=book.speaker,
            alignments=(lib.datasets.Alignment((0, len(text)), (0.0, audio_metadata.length)),),
            text=text,
            metadata={"book": book},
        )
        for audio_path, audio_metadata, text, book in iterator
    ]


def _book_to_path(book: Book, directory: Path, extracted_name: str) -> Path:
    """Given a book of `Book` type, returns the relative path to its metadata.csv file."""
    name = book.speaker.name.lower().replace(" ", "_")
    assert book.speaker.gender is not None
    gender = book.speaker.gender.lower()
    return directory / extracted_name / "by_book" / gender / name / book.title / "metadata.csv"


def _path_to_book(metadata_path: Path, directory: Path) -> Book:
    """Given a path to a book's metadata.csv, returns the corresponding `Book` object."""
    # EXAMPLE: "en_US/by_book/female/judy_bieber/dorothy_and_wizard_oz/metadata.csv"
    metadata_path = metadata_path.relative_to(directory)
    speaker_gender, speaker_name, book_title = metadata_path.parts[2:5]
    speaker = Speaker(speaker_name.replace("_", " ").title(), speaker_gender.lower())
    return Book(speaker, book_title)


def _book_to_data(
    book: Book, directory: Path, extracted_name: str, metadata_path_column: str, **kwargs
) -> typing.Iterator[typing.Dict]:
    """Given a book, yield pairs of (text, audio_path) for that book.

    Args:
        book
        directory: Directory that M-AILABS was downloaded.
        extracted_name: Name of the extracted dataset directory.
        metadata_path_column: Column name to store the metadata_path.
    """
    metadata_path = _book_to_path(book, directory=directory, extracted_name=extracted_name)
    if os.stat(str(metadata_path)).st_size == 0:
        logger.warning("%s is empty, skipping for now", str(metadata_path))
        yield from []
    else:
        data_frame = typing.cast(pandas.DataFrame, pandas.read_csv(metadata_path, **kwargs))
        data_frame[metadata_path_column] = metadata_path
        yield from (row.to_dict() for _, row in data_frame.iterrows())


def _read_m_ailabs_data(books: typing.List[Book], *args, **kwargs) -> typing.Iterator[typing.Dict]:
    for book in books:
        yield from _book_to_data(book, *args, **kwargs)
