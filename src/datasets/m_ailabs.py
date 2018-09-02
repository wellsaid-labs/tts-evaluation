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

    en_US/by_book/[Author Gender]/[Author Name]/[Book Title]

Within each book directory, there's metadata.csv file and a wavs directory following the
convention of the LJSpeech dataset.

The current implementation uses types for books, genders, authors to allows robust error checking
and flexibility to select data for a specific gender, author, or book.

There are several missing audio files. During processing warnings are given for these files.
"""

from collections import namedtuple
from enum import Enum
from pathlib import Path

from torchnlp.download import download_file_maybe_extract

from src.datasets.process import read_speech_data, process_audio, process_all
from src.utils.configurable import configurable

Book = namedtuple('Book', 'gender author_name book_title')


class Gender(Enum):
    FEMALE = 'female'
    MALE = 'male'

    def __str__(self):
        return self.value


class Author(Enum):
    JUDY_BIEBER = 'judy_bieber'
    MARY_ANN = 'mary_ann'
    ELLIOT_MILLER = 'elliot_miller'

    def __str__(self):
        return self.value


THE_SEA_FAIRIES = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'the_sea_fairies')
THE_MASTER_KEY = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'the_master_key')
RINKITINK_IN_OZ = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'rinkitink_in_oz')
DOROTHY_AND_WIZARD_OZ = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'dorothy_and_wizard_oz')
SKY_ISLAND = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'sky_island')
OZMA_OF_OZ = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'ozma_of_oz')
EMERALD_CITY_OF_OZ = Book(Gender.FEMALE, Author.JUDY_BIEBER, 'emerald_city_of_oz')

MIDNIGHT_PASSENGER = Book(Gender.FEMALE, Author.MARY_ANN, 'midnight_passenger')
NORTH_AND_SOUTH = Book(Gender.FEMALE, Author.MARY_ANN, 'northandsouth')

PIRATES_OF_ERSATZ = Book(Gender.MALE, Author.ELLIOT_MILLER, 'pirates_of_ersatz')
POISONED_PEN = Book(Gender.MALE, Author.ELLIOT_MILLER, 'poisoned_pen')
SILENT_BULLET = Book(Gender.MALE, Author.ELLIOT_MILLER, 'silent_bullet')
HUNTERS_SPACE = Book(Gender.MALE, Author.ELLIOT_MILLER, 'hunters_space')
PINK_FAIRY_BOOK = Book(Gender.MALE, Author.ELLIOT_MILLER, 'pink_fairy_book')

_allbooks = [
    THE_SEA_FAIRIES, THE_MASTER_KEY, RINKITINK_IN_OZ, DOROTHY_AND_WIZARD_OZ, SKY_ISLAND, OZMA_OF_OZ,
    EMERALD_CITY_OF_OZ, MIDNIGHT_PASSENGER, NORTH_AND_SOUTH, PIRATES_OF_ERSATZ, POISONED_PEN,
    SILENT_BULLET, HUNTERS_SPACE, PINK_FAIRY_BOOK
]

DOWNLOAD_DIRECTORY = Path('data/M-AILABS')


@configurable
def mailabs_speech_dataset(directory=DOWNLOAD_DIRECTORY,
                           url='http://data.m-ailabs.bayern/data/Training/stt_tts/en_US.tgz',
                           check_files=['en_US/by_book/info.txt'],
                           picker=None,
                           resample=24000,
                           norm=False,
                           guard=False,
                           lower_hertz=0,
                           upper_hertz=12000,
                           loudness=False,
                           random_seed=123,
                           splits=(.8, .2)):
    """
    Download, extract, and process the M-AILABS en_US dataset, which is 7.5GB compressed.
    The original URL is `http://www.m-ailabs.bayern/?ddownload=411`.
    Use ``curl -I <URL>`` to find the redirected URL.

    Args:
        directory (str, optional): Directory to cache the dataset.
        url (str, optional): URL of the dataset `tar.gz` file.
        resample (int or None, optional): If integer is provided, uses SoX to create resampled
            files.
        norm (bool, optional): Automatically invoke the gain effect to guard against clipping and to
            normalise the audio.
        guard (bool, optional): Automatically invoke the gain effect to guard against clipping.
        lower_hertz (int, optional): Apply a sinc kaiser-windowed high-pass.
        upper_hertz (int, optional): Apply a sinc kaiser-windowed low-pass.
        loudness (bool, optioanl): Normalize the subjective perception of loudness level based on
            ISO 226.
        random_seed (int, optional): Random seed used to determine the splits.
        splits (tuple, optional): The number of splits and cardinality of dataset splits.

     Returns:
          :class:`torchnlp.datasets.Dataset`: M-AILABS en_us dataset with audio filenames and text
          annotations.
    """
    download_file_maybe_extract(url=url, directory=str(directory), check_files=check_files)

    # Making sure that the download succeeds by checking against defined books in _allbooks
    metadata_paths = list(DOWNLOAD_DIRECTORY.glob('**/metadata.csv'))
    actual_books = [_path2book(path) for path in metadata_paths]
    title_key = lambda x: x.book_title
    assert sorted(actual_books, key=title_key) == sorted(_allbooks, key=title_key)

    def extract_fun(args):
        text, wav_filename = args
        processed_wav_filename = process_audio(
            str(wav_filename),
            resample=resample,
            norm=norm,
            guard=guard,
            lower_hertz=lower_hertz,
            upper_hertz=upper_hertz,
            loudness=loudness)
        return {'text': text, 'wav_filename': processed_wav_filename}

    data = _read_mailabs_data(picker)
    return process_all(extract_fun, data, splits, random_seed)


def _book2path(book):
    """
    Given a book of :class:`Book` type, returns the relative path to its metadata.csv file.

    Examples:
        >>> _book2path(SKY_ISLAND)
        PosixPath('data/M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')
    """
    return DOWNLOAD_DIRECTORY / 'en_US/by_book' / '/'.join(map(str, book)) / 'metadata.csv'


def _path2book(metadata_path):
    """
    Given a path to a book's metadata.csv, returns the corresponding :class:`Book` object.

    Examples:
        >>> _path2book(Path('data/M-AILABS/en_US/by_book/female/judy_bieber/sky_island/metadata.csv')) # noqa: E501
        Book(gender='female', author_name='judy_bieber', book_title='sky_island')
    """
    gender, author_name, book_title = metadata_path.parts[4:7]
    return Book(Gender(gender), Author(author_name), book_title)


def _book2speech_data(book):
    """
    Given a book, yield pairs of (text, wav_filename) for that book.
    For now, use the cleaned text (text_column=2).
    """
    yield from read_speech_data(_book2path(book), text_column=2)


def _author2speech_data(author):
    """
    Given an author, yield pairs of (text, wav_filename) for that author.
    """
    for book in _allbooks:
        if book.author_name == author:
            yield from _book2speech_data(book)


def _gender2speech_data(gender):
    """
    Given a gender (male or female), yield pairs of (text, wav_filename) for all books read
    by authors of that gender.
    """
    for book in _allbooks:
        if book.gender == gender:
            yield from _book2speech_data(book)


def _read_mailabs_data(picker=None):
    """
    Yield pairs of (text, wav_filename) from the MAILABS dataset.
    Args:
        picker: argument that dictates which subset to pick.

    Examples:
        _read_mailabs_data()
        _read_mailabs_data(SKY_ISLAND)
        _read_mailabs_data(Author.JUDY_BIEBER)
        _read_mailabs_data(Gender.MALE)
    """
    if picker is None:
        for book in _allbooks:
            yield from _book2speech_data(book)
    elif isinstance(picker, Book):
        yield from _book2speech_data(picker)
    elif isinstance(picker, Author):
        yield from _author2speech_data(picker)
    elif isinstance(picker, Gender):
        yield from _gender2speech_data(picker)
    else:
        raise ValueError(
            "Input {}, if not None, should be of type Book, Author, or Gender".format(picker))
