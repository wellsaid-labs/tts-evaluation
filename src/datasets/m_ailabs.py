"""
Module to download and process the M-AILABS speech dataset
http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/.

Most of the data is based on LibriVox and Project Gutenberg. The training data consist of nearly
thousand hours of audio and the text-files in prepared format.

A transcription is provided for each clip. Clips vary in length from 1 to 20 seconds and have a
total length of approximately shown in the list (and in the respective info.txt-files) below.

The texts were published between 1884 and 1964, and are in the public domain. The audio was
recorded by the LibriVox project and is also in the public domain – except for Ukrainian.

This module targets the en_US dataset only. The file to be downloaded is called en_US.tgz.
It contains 102 hours of audio. When extracted, it creates a list of 14 books with directory
structure of the format:

    en_US/by_book/[Speaker Gender]/[Speaker Name]/[Book Title]

Within each book directory, there's metadata.csv file and a wavs directory following the
convention of the LJSpeech dataset.

The current implementation uses types for books, genders, speakers to allows robust error checking
and flexibility to select data for a specific gender, speaker, or book.

There are several missing audio files. During processing warnings are given for these files.
"""
from collections import namedtuple
from functools import partial
from pathlib import Path

import csv
import logging
import os

import pandas

from src.datasets.constants import Gender
from src.datasets.constants import Speaker
from src.datasets.constants import TextSpeechRow
from src.datasets.process import _download_file_maybe_extract
from src.datasets.process import _normalize_audio_and_cache
from src.datasets.process import _process_in_parallel
from src.datasets.process import _split_dataset
from src.hparams import configurable

logger = logging.getLogger(__name__)
Book = namedtuple('Book', 'speaker title')

THE_SEA_FAIRIES = Book(Speaker.JUDY_BIEBER, 'the_sea_fairies')
THE_MASTER_KEY = Book(Speaker.JUDY_BIEBER, 'the_master_key')
RINKITINK_IN_OZ = Book(Speaker.JUDY_BIEBER, 'rinkitink_in_oz')
DOROTHY_AND_WIZARD_OZ = Book(Speaker.JUDY_BIEBER, 'dorothy_and_wizard_oz')
SKY_ISLAND = Book(Speaker.JUDY_BIEBER, 'sky_island')
OZMA_OF_OZ = Book(Speaker.JUDY_BIEBER, 'ozma_of_oz')
EMERALD_CITY_OF_OZ = Book(Speaker.JUDY_BIEBER, 'emerald_city_of_oz')

MIDNIGHT_PASSENGER = Book(Speaker.MARY_ANN, 'midnight_passenger')
NORTH_AND_SOUTH = Book(Speaker.MARY_ANN, 'northandsouth')

PIRATES_OF_ERSATZ = Book(Speaker.ELLIOT_MILLER, 'pirates_of_ersatz')
POISONED_PEN = Book(Speaker.ELLIOT_MILLER, 'poisoned_pen')
SILENT_BULLET = Book(Speaker.ELLIOT_MILLER, 'silent_bullet')
HUNTERS_SPACE = Book(Speaker.ELLIOT_MILLER, 'hunters_space')
PINK_FAIRY_BOOK = Book(Speaker.ELLIOT_MILLER, 'pink_fairy_book')

_allbooks = [
    THE_SEA_FAIRIES, THE_MASTER_KEY, RINKITINK_IN_OZ, DOROTHY_AND_WIZARD_OZ, SKY_ISLAND, OZMA_OF_OZ,
    EMERALD_CITY_OF_OZ, MIDNIGHT_PASSENGER, NORTH_AND_SOUTH, PIRATES_OF_ERSATZ, POISONED_PEN,
    SILENT_BULLET, HUNTERS_SPACE, PINK_FAIRY_BOOK
]

DOWNLOAD_DIRECTORY = Path('data/M-AILABS')


def _processing_row(row,
                    directory,
                    kwargs,
                    metadata_path_column,
                    metadata_audio_column=0,
                    metadata_audio_path_template='wavs/{}.wav',
                    metadata_text_column=2):
    """
    Note:
        - ``# pragma: no cover`` is used because this functionality is run with multiprocessing
          that is not compatible with the coverage module.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        metadata_path_column (str, optional): Column name to store the metadata path.
        kwargs: Arguments passed to process dataset audio.
        metadata_audio_column (int, optional): Column name or index with the audio filename.
        metadata_audio_path_template (str, optional): Given the audio column, this template
            determines the filename.
        metadata_text_column (int, optional): Column name or index with the audio transcript.

    Returns:
        (TextSpeechRow) Processed row.
    """
    text = row[metadata_text_column].strip()
    metadata_path = row[metadata_path_column]
    book = _path2book(metadata_path, directory=directory)
    audio_path = Path(metadata_path.parent,
                      metadata_audio_path_template.format(row[metadata_audio_column]))

    if not audio_path.is_file():
        logger.warning('Not found audio file, skipping: %s', audio_path)
        return None

    if len(text) == 0:
        logger.warning('Text is absent, skipping: %s', audio_path)
        return None

    audio_path = _normalize_audio_and_cache(audio_path, **kwargs)
    return TextSpeechRow(text=text, audio_path=audio_path, speaker=book.speaker, metadata=None)


@configurable
def m_ailabs_speech_dataset(directory=DOWNLOAD_DIRECTORY,
                            url='http://www.caito.de/data/Training/stt_tts/en_US.tgz',
                            check_files=['en_US/by_book/info.txt'],
                            metadata_pattern='**/metadata.csv',
                            metadata_path_column='metadata_path',
                            picker=None,
                            splits=(.8, .2),
                            **kwargs):
    """ Load the M-AILABS en_US dataset.

    Download, extract, and process the M-AILABS en_US dataset, which is 8GB compressed.
    The original URL is ``http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/``.
    Use ``curl -I <URL>`` to find the redirected URL.

    NOTE: A cursory analysis 100 clips from the M-AILABS dataset was that it was 10% of the clips
    would end to early. The text was verbalized accurately, during the analysis.

    Args:
        directory (str or Path, optional): Directory to cache the dataset.
        url (str, optional): URL of the dataset ``tar.gz`` file.
        check_files (list of str, optional): Check this file exists if the download was successful.
        metadata_pattern (str, optional): Pattern for all ``metadata.csv`` files containing
            (filename, text) information.
        metadata_path_column (str, optional): Column name to store the metadata path.
        picker (None or Book or Speaker or Gender): Argument that dictates which dataset subset to
            pick.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.

     Returns:
          list: M-AILABS en_us dataset with audio filenames and text annotations.

    Example:
        >>> from src.hparams import set_hparams # doctest: +SKIP
        >>> from src.datasets import m_ailabs_speech_dataset # doctest: +SKIP
        >>> set_hparams() # doctest: +SKIP
        >>> train, dev = m_ailabs_speech_dataset() # doctest: +SKIP
    """
    logger.info('Loading M-AILABS speech dataset')
    directory = Path(directory)
    _download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)

    # Making sure that the download succeeds by checking against defined books in _allbooks
    metadata_paths = list(directory.glob('**/metadata.csv'))
    actual_books = [_path2book(path, directory=directory) for path in metadata_paths]
    assert sorted(actual_books, key=lambda x: x.title) == sorted(_allbooks, key=lambda x: x.title)

    # Process data
    data = _read_mailabs_data(
        picker=picker, directory=directory, metadata_path_column=metadata_path_column)

    logger.info('Normalizing text and audio...')
    data = _process_in_parallel(
        data,
        partial(
            _processing_row,
            directory=directory,
            metadata_path_column=metadata_path_column,
            kwargs=kwargs))
    data = list(filter(None.__ne__, data))
    return _split_dataset(data, splits=splits)


def _book2path(book, directory=DOWNLOAD_DIRECTORY):
    """ Given a book of :class:`Book` type, returns the relative path to its metadata.csv file.

    Examples:
        >>> _book2path(SKY_ISLAND)
        PosixPath('data/M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')
    """
    name = book.speaker.name.lower().replace(' ', '_')
    gender = book.speaker.gender.name.lower()
    return directory / 'en_US/by_book' / gender / name / book.title / 'metadata.csv'


def _path2book(metadata_path, directory=DOWNLOAD_DIRECTORY):
    """ Given a path to a book's metadata.csv, returns the corresponding :class:`Book` object.

    Examples:
        >>> _path2book(Path('data/M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')) # noqa: E501
        Book(speaker=Speaker(name='Judy Bieber', gender=FEMALE, id=0), title='sky_island')
    """
    metadata_path = metadata_path.relative_to(directory)
    # EXAMPLE: metadata_path=en_US/by_book/female/judy_bieber/dorothy_and_wizard_oz/metadata.csv
    speaker_gender, speaker_name, book_title = metadata_path.parts[2:5]
    speaker = getattr(Speaker, speaker_name.upper())
    return Book(speaker, book_title)


def _book2speech_data(book,
                      directory=DOWNLOAD_DIRECTORY,
                      quoting=csv.QUOTE_NONE,
                      delimiter='|',
                      header=None,
                      metadata_path_column='metadata_path',
                      index=False):
    """ Given a book, yield pairs of (text, audio_path) for that book.

    Args:
        book (Book)
        directory (Path or str): Directory that M-AILABS was downloaded.
        quoting (int, optional): Control field quoting behavior per csv.QUOTE_* constants for
            the metadata file.
        delimiter (str, optional): Delimiter for the metadata file.
        header (bool, optional): If True, ``metadata_file`` has a header to parse.
        metadata_path_column (str, optional): Column name to store the metadata_path.
        index (bool, optional): If True, return the index as the first element of the tuple in
            ``pandas.DataFrame.itertuples``.
    """
    metadata_path = _book2path(book, directory=directory)
    if os.stat(str(metadata_path)).st_size == 0:
        logger.warning('%s is empty, skipping for now', str(metadata_path))
        yield from []
    else:
        data_frame = pandas.read_csv(
            metadata_path, delimiter=delimiter, header=header, quoting=quoting)
        data_frame['metadata_path'] = metadata_path
        yield from (row.to_dict() for _, row in data_frame.iterrows())


def _speaker2speech_data(speaker, **kwargs):
    """
    Given an speaker, yield pairs of (text, audio_path) for that speaker.
    """
    for book in _allbooks:
        if book.speaker == speaker:
            yield from _book2speech_data(book, **kwargs)


def _gender2speech_data(gender, **kwargs):
    """
    Given a gender (male or female), yield pairs of (text, audio_path) for all books read
    by speakers of that gender.
    """
    for book in _allbooks:
        if book.speaker.gender == gender:
            yield from _book2speech_data(book, **kwargs)


def _read_mailabs_data(picker=None, **kwargs):
    """
    Yield pairs of (text, audio_path) from the MAILABS dataset.
    Args:
        picker (None or Book or Speaker or Gender): Argument that dictates which dataset subset to
            pick.

    Examples:
        _read_mailabs_data()
        _read_mailabs_data(SKY_ISLAND)
        _read_mailabs_data(JUDY_BIEBER)
        _read_mailabs_data(Gender.MALE)
    """
    if picker is None:
        for book in _allbooks:
            yield from _book2speech_data(book, **kwargs)
    elif isinstance(picker, Book):
        yield from _book2speech_data(picker, **kwargs)
    elif isinstance(picker, Speaker):
        yield from _speaker2speech_data(picker, **kwargs)
    elif isinstance(picker, Gender):
        yield from _gender2speech_data(picker, **kwargs)
    else:
        raise ValueError(
            "Input {}, if not None, should be of type Book, Speaker, or Gender".format(picker))
