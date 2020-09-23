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
from pathlib import Path

import csv
import logging
import os
import typing

from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract

pandas = LazyLoader('pandas', globals(), 'pandas')

from lib.datasets.utils import Example
from lib.datasets.utils import Speaker

logger = logging.getLogger(__name__)


class Book(typing.NamedTuple):
    speaker: Speaker
    title: str


JUDY_BIEBER = Speaker('Judy Bieber', 'female')
MARY_ANN = Speaker('Mary Ann', 'female')
ELLIOT_MILLER = Speaker('Elliot Miller', 'male')
ELIZABETH_KLETT = Speaker('Elizabeth Klett', 'female')

THE_SEA_FAIRIES = Book(JUDY_BIEBER, 'the_sea_fairies')
THE_MASTER_KEY = Book(JUDY_BIEBER, 'the_master_key')
RINKITINK_IN_OZ = Book(JUDY_BIEBER, 'rinkitink_in_oz')
DOROTHY_AND_WIZARD_OZ = Book(JUDY_BIEBER, 'dorothy_and_wizard_oz')
SKY_ISLAND = Book(JUDY_BIEBER, 'sky_island')
OZMA_OF_OZ = Book(JUDY_BIEBER, 'ozma_of_oz')
EMERALD_CITY_OF_OZ = Book(JUDY_BIEBER, 'emerald_city_of_oz')

MIDNIGHT_PASSENGER = Book(MARY_ANN, 'midnight_passenger')
NORTH_AND_SOUTH = Book(MARY_ANN, 'northandsouth')

PIRATES_OF_ERSATZ = Book(ELLIOT_MILLER, 'pirates_of_ersatz')
POISONED_PEN = Book(ELLIOT_MILLER, 'poisoned_pen')
SILENT_BULLET = Book(ELLIOT_MILLER, 'silent_bullet')
HUNTERS_SPACE = Book(ELLIOT_MILLER, 'hunters_space')
PINK_FAIRY_BOOK = Book(ELLIOT_MILLER, 'pink_fairy_book')

JANE_EYRE = Book(ELIZABETH_KLETT, 'jane_eyre')
WIVES_AND_DAUGHTERS = Book(ELIZABETH_KLETT, 'wives_and_daughters')


def m_ailabs_en_us_judy_bieber_speech_dataset(books=[
    THE_SEA_FAIRIES, THE_MASTER_KEY, RINKITINK_IN_OZ, DOROTHY_AND_WIZARD_OZ, SKY_ISLAND, OZMA_OF_OZ,
    EMERALD_CITY_OF_OZ
]):
    return m_ailabs_en_us_speech_dataset(books=books)


def m_ailabs_en_us_mary_ann_speech_dataset(books=[MIDNIGHT_PASSENGER, NORTH_AND_SOUTH]):
    return m_ailabs_en_us_speech_dataset(books=books)


def m_ailabs_en_us_elliot_miller_speech_dataset(
        books=[PIRATES_OF_ERSATZ, POISONED_PEN, SILENT_BULLET, HUNTERS_SPACE, PINK_FAIRY_BOOK]):
    return m_ailabs_en_us_speech_dataset(books=books)


def m_ailabs_en_uk_elizabeth_klett_speech_dataset(books=[JANE_EYRE, JANE_EYRE]):
    return m_ailabs_en_us_speech_dataset(books=books)


def m_ailabs_en_us_speech_dataset(directory,
                                  url='http://www.caito.de/data/Training/stt_tts/en_US.tgz',
                                  check_files=['en_US/by_book/info.txt'],
                                  extracted_name='en_US',
                                  books=[
                                      THE_SEA_FAIRIES, THE_MASTER_KEY, RINKITINK_IN_OZ,
                                      DOROTHY_AND_WIZARD_OZ, SKY_ISLAND, OZMA_OF_OZ,
                                      EMERALD_CITY_OF_OZ, MIDNIGHT_PASSENGER, NORTH_AND_SOUTH,
                                      PIRATES_OF_ERSATZ, POISONED_PEN, SILENT_BULLET, HUNTERS_SPACE,
                                      PINK_FAIRY_BOOK
                                  ],
                                  **kwargs):
    """ Download, extract, and process the M-AILABS `en_US` dataset.

    The dataset is 8GB compressed. The file to be downloaded is called `en_US.tgz`. It contains 102
    hours of audio. When extracted, it creates a list of 14 books.

    NOTE: Based on 100 clips from the M-AILABS dataset, around 10% of the clips would end too early.
    Furthermore, it seemed like the text was verbalized accuractely.
    """
    return _m_ailabs_speech_dataset(
        directory=directory,
        extracted_name=extracted_name,
        url=url,
        check_files=check_files,
        books=books,
        **kwargs)


def m_ailabs_en_uk_speech_dataset(directory,
                                  url='http://www.caito.de/data/Training/stt_tts/en_UK.tgz',
                                  check_files=['en_UK/by_book/info.txt'],
                                  extracted_name='en_UK',
                                  books=[JANE_EYRE, WIVES_AND_DAUGHTERS],
                                  **kwargs):
    """ Download, extract, and process the M-AILABS `en_UK` dataset.

    The dataset is 4GB compressed. The file to be downloaded is called `en_US.tgz`. It contains
    45 hours of audio. When extracted, it creates a list of 2 books.
    """
    return _m_ailabs_speech_dataset(
        directory=directory,
        extracted_name=extracted_name,
        url=url,
        check_files=check_files,
        books=books,
        **kwargs)


def _m_ailabs_speech_dataset(
        directory: typing.Union[str, Path],
        extracted_name: str,
        url: str,
        check_files: typing.List[str],
        books: typing.List[Book],
        metadata_pattern: str = '**/metadata.csv',
        metadata_path_column: str = 'metadata_path',
        metadata_audio_column: typing.Union[str, int] = 0,
        metadata_audio_path_template: str = 'wavs/{}.wav',
        metadata_text_column: typing.Union[str, int] = 2) -> typing.List[Example]:
    """ Download, extract, and process a M-AILABS dataset.

    NOTE: The original URL is `http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/`. Use
    `curl -I <URL>` to find the redirected URL.

    Args:
        directory: Directory to cache the dataset.
        extracted_name: Name of the extracted dataset directory.
        url: URL of the dataset `tar.gz` file.
        check_files: These file(s) should exist if the download was successful.
        books: List of books to load.
        metadata_pattern: Pattern for all `metadata.csv` files containing (filename, text)
            information.
        metadata_path_column: Column name to store the metadata path.
        metadata_audio_column: Column name or index with the audio filename.
        metadata_audio_path_template: Given the audio column, this template determines the filename.
        metadata_text_column: Column name or index with the audio transcript.

    Returns: List of voice-over examples in the dataset.

    Example:
        >>> from lib.hparams import set_hparams # doctest: +SKIP
        >>> from lib.datasets import m_ailabs_speech_dataset # doctest: +SKIP
        >>> train, dev = m_ailabs_speech_dataset() # doctest: +SKIP
    """
    logger.info('Loading `M-AILABS %s` speech dataset', extracted_name)
    directory = Path(directory)
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)

    # NOTE: This makes sure that the download succeeds by checking against defined books in
    # `books`.
    metadata_paths = list((directory / extracted_name).glob('**/metadata.csv'))
    downloaded_books = set([_path_to_book(path, directory=directory) for path in metadata_paths])
    assert len(set(books) - downloaded_books) == 0

    data = _read_m_ailabs_data(
        books,
        directory=directory,
        extracted_name=extracted_name,
        metadata_path_column=metadata_path_column)
    return [
        Example(
            text=row[metadata_text_column].strip(),
            audio_path=Path(row[metadata_path_column].parent,
                            metadata_audio_path_template.format(row[metadata_audio_column])),
            speaker=_path_to_book(row[metadata_path_column], directory=directory).speaker,
            metadata={'book': _path_to_book(row[metadata_path_column], directory=directory)})
        for row in data
    ]


def _book_to_path(book: Book, directory: Path, extracted_name: str) -> Path:
    """ Given a book of `Book` type, returns the relative path to its metadata.csv file.

    Examples:
        >>> from lib.environment import DATA_PATH
        >>> _book_to_path(SKY_ISLAND, DATA_PATH / 'M-AILABS', 'en_US').relative_to(DATA_PATH)
        PosixPath('M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')
    """
    name = book.speaker.name.lower().replace(' ', '_')
    assert book.speaker.gender is not None
    gender = book.speaker.gender.lower()
    return directory / extracted_name / 'by_book' / gender / name / book.title / 'metadata.csv'


def _path_to_book(metadata_path: Path, directory: Path) -> Book:
    """ Given a path to a book's metadata.csv, returns the corresponding `Book` object.

    Examples:
        >>> from lib.environment import DATA_PATH
        >>> path = DATA_PATH / 'M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv'
        >>> _path_to_book(metadata_path=path, directory=DATA_PATH / 'M-AILABS') # noqa: E501
        Book(speaker=Speaker(name='Judy Bieber', gender=FEMALE), title='sky_island')
    """
    metadata_path = metadata_path.relative_to(directory)
    # EXAMPLE: metadata_path=en_US/by_book/female/judy_bieber/dorothy_and_wizard_oz/metadata.csv
    speaker_gender, speaker_name, book_title = metadata_path.parts[2:5]
    speaker = Speaker(speaker_name.replace('_', ' ').title(), speaker_gender.lower())
    return Book(speaker, book_title)


def _book_to_data(book: Book,
                  directory: Path,
                  extracted_name: str,
                  quoting: int = csv.QUOTE_NONE,
                  delimiter: str = '|',
                  header: typing.Optional[bool] = None,
                  metadata_path_column: str = 'metadata_path',
                  index: bool = False):
    """ Given a book, yield pairs of (text, audio_path) for that book.

    Args:
        book
        directory: Directory that M-AILABS was downloaded.
        extracted_name :  Name of the extracted dataset directory.
        quoting: Control field quoting behavior per csv.QUOTE_* constants for the metadata file.
        delimiter: Delimiter for the metadata file.
        header: If `True`, `metadata_file` has a header to parse.
        metadata_path_column: Column name to store the metadata_path.
        index: If `True`, return the index as the first element of the tuple in
            `pandas.DataFrame.itertuples`.

    Returns:
        iterable
    """
    metadata_path = _book_to_path(book, directory=directory, extracted_name=extracted_name)
    if os.stat(str(metadata_path)).st_size == 0:
        logger.warning('%s is empty, skipping for now', str(metadata_path))
        yield from []
    else:
        data_frame = pandas.read_csv(
            metadata_path, delimiter=delimiter, header=header, quoting=quoting)
        data_frame[metadata_path_column] = metadata_path
        yield from (row.to_dict() for _, row in data_frame.iterrows())


def _read_m_ailabs_data(books: typing.List[Book], **kwargs):
    for book in books:
        yield from _book_to_data(book, **kwargs)
