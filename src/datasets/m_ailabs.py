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
from collections import namedtuple
from pathlib import Path

import csv
import logging
import os

from torchnlp.download import download_file_maybe_extract

import pandas

from src.datasets.constants import Gender
from src.datasets.constants import Speaker
from src.datasets.constants import TextSpeechRow
from src.environment import DATA_PATH

logger = logging.getLogger(__name__)
Book = namedtuple('Book', 'speaker title')

JUDY_BIEBER = Speaker('Judy Bieber', Gender.FEMALE)
MARY_ANN = Speaker('Mary Ann', Gender.FEMALE)
ELLIOT_MILLER = Speaker('Elliot Miller', Gender.MALE)
ELIZABETH_KLETT = Speaker('Elizabeth Klett', Gender.FEMALE)

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

DOWNLOAD_DIRECTORY = DATA_PATH / 'M-AILABS'


def m_ailabs_en_us_speech_dataset(directory=DOWNLOAD_DIRECTORY,
                                  url='http://www.caito.de/data/Training/stt_tts/en_US.tgz',
                                  check_files=['en_US/by_book/info.txt'],
                                  extracted_name='en_US',
                                  all_books=[
                                      THE_SEA_FAIRIES, THE_MASTER_KEY, RINKITINK_IN_OZ,
                                      DOROTHY_AND_WIZARD_OZ, SKY_ISLAND, OZMA_OF_OZ,
                                      EMERALD_CITY_OF_OZ, MIDNIGHT_PASSENGER, NORTH_AND_SOUTH,
                                      PIRATES_OF_ERSATZ, POISONED_PEN, SILENT_BULLET, HUNTERS_SPACE,
                                      PINK_FAIRY_BOOK
                                  ],
                                  **kwargs):
    """ Load the M-AILABS ``en_US`` dataset.

    Download, extract, and process the M-AILABS ``en_US`` dataset, which is 8GB compressed. The file
    to be downloaded is called ``en_US.tgz``. It contains 102 hours of audio. When extracted, it
    creates a list of 14 books.

    NOTE: A cursory analysis 100 clips from the M-AILABS dataset was that it was 10% of the clips
    would end to early. The text was verbalized accurately, during the analysis.

    """
    return _m_ailabs_speech_dataset(
        directory=directory,
        extracted_name=extracted_name,
        url=url,
        check_files=check_files,
        all_books=all_books,
        **kwargs)


def m_ailabs_en_uk_speech_dataset(directory=DOWNLOAD_DIRECTORY,
                                  url='http://www.caito.de/data/Training/stt_tts/en_UK.tgz',
                                  check_files=['en_UK/by_book/info.txt'],
                                  extracted_name='en_UK',
                                  all_books=[JANE_EYRE, WIVES_AND_DAUGHTERS],
                                  **kwargs):
    """ Load the M-AILABS en_UK dataset.

    Download, extract, and process the M-AILABS ``en_UK`` dataset, which is 4GB compressed. The file
    to be downloaded is called ``en_US.tgz``. It contains 45 hours of audio. When extracted, it
    creates a list of 2 books.
    """
    return _m_ailabs_speech_dataset(
        directory=directory,
        extracted_name=extracted_name,
        url=url,
        check_files=check_files,
        all_books=all_books,
        **kwargs)


def _m_ailabs_speech_dataset(directory,
                             extracted_name,
                             url,
                             check_files,
                             all_books,
                             metadata_pattern='**/metadata.csv',
                             metadata_path_column='metadata_path',
                             metadata_audio_column=0,
                             metadata_audio_path_template='wavs/{}.wav',
                             metadata_text_column=2):
    """ Load a M-AILABS dataset.

    Download, extract, and process a M-AILABS dataset.
    The original URL is ``http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/``.
    Use ``curl -I <URL>`` to find the redirected URL.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        extracted_name (str, optional): Name of the extracted dataset directory.
        url (str, optional): URL of the dataset ``tar.gz`` file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        all_books (list of Book): List of books to load.
        metadata_pattern (str, optional): Pattern for all ``metadata.csv`` files containing
            (filename, text) information.
        metadata_path_column (str, optional): Column name to store the metadata path.
        metadata_audio_column (int, optional): Column name or index with the audio filename.
        metadata_audio_path_template (str, optional): Given the audio column, this template
            determines the filename.
        metadata_text_column (int, optional): Column name or index with the audio transcript.

     Returns:
          list: A M-AILABS dataset with audio filenames and text annotations.

    Example:
        >>> from src.hparams import set_hparams # doctest: +SKIP
        >>> from src.datasets import m_ailabs_speech_dataset # doctest: +SKIP
        >>> set_hparams() # doctest: +SKIP
        >>> train, dev = m_ailabs_speech_dataset() # doctest: +SKIP
    """
    logger.info('Loading `M-AILABS %s` speech dataset', extracted_name)
    directory = Path(directory)
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)

    # Making sure that the download succeeds by checking against defined books in _ALL_BOOKS
    metadata_paths = list((directory / extracted_name).glob('**/metadata.csv'))
    actual_books = [_path2book(path, directory=directory) for path in metadata_paths]
    assert sorted(actual_books, key=lambda x: x.title) == sorted(all_books, key=lambda x: x.title)

    data = _read_m_ailabs_data(
        all_books,
        directory=directory,
        extracted_name=extracted_name,
        metadata_path_column=metadata_path_column)
    return [
        TextSpeechRow(
            text=row[metadata_text_column].strip(),
            audio_path=Path(row[metadata_path_column].parent,
                            metadata_audio_path_template.format(row[metadata_audio_column])),
            speaker=_path2book(row[metadata_path_column], directory=directory).speaker,
            metadata=None) for row in data
    ]


def _book2path(book, directory, extracted_name):
    """ Given a book of :class:`Book` type, returns the relative path to its metadata.csv file.

    Examples:
        >>> from src.environment import DATA_PATH
        >>> _book2path(SKY_ISLAND, DATA_PATH / 'M-AILABS', 'en_US').relative_to(DATA_PATH)
        PosixPath('M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')
    """
    name = book.speaker.name.lower().replace(' ', '_')
    gender = book.speaker.gender.name.lower()
    return directory / extracted_name / 'by_book' / gender / name / book.title / 'metadata.csv'


def _path2book(metadata_path, directory):
    """ Given a path to a book's metadata.csv, returns the corresponding :class:`Book` object.

    Examples:
        >>> from src.environment import DATA_PATH
        >>> path = DATA_PATH / 'M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv'
        >>> _path2book(metadata_path=path, directory=DATA_PATH / 'M-AILABS') # noqa: E501
        Book(speaker=Speaker(name='Judy Bieber', gender=FEMALE), title='sky_island')
    """
    metadata_path = metadata_path.relative_to(directory)
    # EXAMPLE: metadata_path=en_US/by_book/female/judy_bieber/dorothy_and_wizard_oz/metadata.csv
    speaker_gender, speaker_name, book_title = metadata_path.parts[2:5]
    speaker_gender = getattr(Gender, speaker_gender.upper())
    speaker = Speaker(speaker_name.replace('_', ' ').title(), speaker_gender)
    return Book(speaker, book_title)


def _book2speech_data(book,
                      directory,
                      extracted_name,
                      quoting=csv.QUOTE_NONE,
                      delimiter='|',
                      header=None,
                      metadata_path_column='metadata_path',
                      index=False):
    """ Given a book, yield pairs of (text, audio_path) for that book.

    Args:
        book (Book)
        directory (Path or str): Directory that M-AILABS was downloaded.
        extracted_name (str):  Name of the extracted dataset directory.
        quoting (int, optional): Control field quoting behavior per csv.QUOTE_* constants for
            the metadata file.
        delimiter (str, optional): Delimiter for the metadata file.
        header (bool, optional): If ``True``, ``metadata_file`` has a header to parse.
        metadata_path_column (str, optional): Column name to store the metadata_path.
        index (bool, optional): If ``True``, return the index as the first element of the tuple in
            ``pandas.DataFrame.itertuples``.

    Returns:
        iterable
    """
    metadata_path = _book2path(book, directory=directory, extracted_name=extracted_name)
    if os.stat(str(metadata_path)).st_size == 0:
        logger.warning('%s is empty, skipping for now', str(metadata_path))
        yield from []
    else:
        data_frame = pandas.read_csv(
            metadata_path, delimiter=delimiter, header=header, quoting=quoting)
        data_frame['metadata_path'] = metadata_path
        yield from (row.to_dict() for _, row in data_frame.iterrows())


def _read_m_ailabs_data(all_books, **kwargs):
    for book in all_books:
        yield from _book2speech_data(book, **kwargs)
