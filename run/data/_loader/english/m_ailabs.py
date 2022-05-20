"""
Sub-module of M-AILABS module for downloading and processing ENGLISH datasets.
"""
from functools import partial

from run.data._loader import structures as struc
from run.data._loader.m_ailabs import Book, m_ailabs_speech_dataset, make_speaker

JUDY_BIEBER = make_speaker("judy_bieber", struc.Dialect.EN_US, "female")
MARY_ANN = make_speaker("mary_ann", struc.Dialect.EN_US, "female")
ELLIOT_MILLER = make_speaker("elliot_miller", struc.Dialect.EN_US, "male")
ELIZABETH_KLETT = make_speaker("elizabeth_klett", struc.Dialect.EN_UK, "male")

THE_SEA_FAIRIES = Book(struc.Dialect.EN_US, JUDY_BIEBER, "the_sea_fairies")
THE_MASTER_KEY = Book(struc.Dialect.EN_US, JUDY_BIEBER, "the_master_key")
RINKITINK_IN_OZ = Book(struc.Dialect.EN_US, JUDY_BIEBER, "rinkitink_in_oz")
DOROTHY_AND_WIZARD_OZ = Book(struc.Dialect.EN_US, JUDY_BIEBER, "dorothy_and_wizard_oz")
SKY_ISLAND = Book(struc.Dialect.EN_US, JUDY_BIEBER, "sky_island")
OZMA_OF_OZ = Book(struc.Dialect.EN_US, JUDY_BIEBER, "ozma_of_oz")
EMERALD_CITY_OF_OZ = Book(struc.Dialect.EN_US, JUDY_BIEBER, "emerald_city_of_oz")

MIDNIGHT_PASSENGER = Book(struc.Dialect.EN_US, MARY_ANN, "midnight_passenger")
NORTH_AND_SOUTH = Book(struc.Dialect.EN_US, MARY_ANN, "northandsouth")

PIRATES_OF_ERSATZ = Book(struc.Dialect.EN_US, ELLIOT_MILLER, "pirates_of_ersatz")
POISONED_PEN = Book(struc.Dialect.EN_US, ELLIOT_MILLER, "poisoned_pen")
SILENT_BULLET = Book(struc.Dialect.EN_US, ELLIOT_MILLER, "silent_bullet")
HUNTERS_SPACE = Book(struc.Dialect.EN_US, ELLIOT_MILLER, "hunters_space")
PINK_FAIRY_BOOK = Book(struc.Dialect.EN_US, ELLIOT_MILLER, "pink_fairy_book")

JANE_EYRE = Book(struc.Dialect.EN_UK, ELIZABETH_KLETT, "jane_eyre")
WIVES_AND_DAUGHTERS = Book(struc.Dialect.EN_UK, ELIZABETH_KLETT, "wives_and_daughters")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]

UK_BOOKS = [b for b in BOOKS if b.dialect == struc.Dialect.EN_UK]
US_BOOKS = [b for b in BOOKS if b.dialect == struc.Dialect.EN_US]


def m_ailabs_en_us_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/en_US.tgz",
    extracted_name="en_US",
    books=US_BOOKS,
    dialect=struc.Dialect.EN_US,
    check_files=["en_US/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `en_US` dataset.

    The dataset is 8GB compressed. The file to be downloaded is called `en_US.tgz`. It contains 102
    hours of audio. When extracted, it creates a list of 14 books.

    NOTE: Based on 100 clips from the M-AILABS dataset, around 10% of the clips would end too early.
    Furthermore, it seemed like the text was verbalized accuractely.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, dialect, check_files, **kwargs
    )


def m_ailabs_en_uk_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/en_UK.tgz",
    extracted_name="en_UK",
    books=UK_BOOKS,
    dialect=struc.Dialect.EN_UK,
    check_files=["en_UK/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `en_UK` dataset.

    The dataset is 4GB compressed. The file to be downloaded is called `en_UK.tgz`. It contains
    45 hours of audio. When extracted, it creates a list of 2 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, dialect, check_files, **kwargs
    )


_loaders = {
    struc.Dialect.EN_UK: m_ailabs_en_uk_speech_dataset,
    struc.Dialect.EN_US: m_ailabs_en_us_speech_dataset,
}
_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
M_AILABS_DATASETS = {
    s: partial(_loaders[s.dialect], books=[b for b in BOOKS if b.speaker == s]) for s in _speakers
}
