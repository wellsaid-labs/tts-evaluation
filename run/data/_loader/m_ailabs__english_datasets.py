"""
Sub-module of M-AILABS module for downloading and processing ENGLISH datasets.
"""
from run.data._loader.data_structures import Speaker, WSL_Languages
from run.data._loader.m_ailabs import Book, Dataset, m_ailabs_speech_dataset

UK_DATASET = Dataset("en_UK")
US_DATASET = Dataset("en_US")

JUDY_BIEBER = Speaker("judy_bieber", gender="female")
MARY_ANN = Speaker("mary_ann", gender="female")
ELLIOT_MILLER = Speaker("elliot_miller", gender="male")
ELIZABETH_KLETT = Speaker("elizabeth_klett", gender="female")

THE_SEA_FAIRIES = Book(US_DATASET, JUDY_BIEBER, "the_sea_fairies")
THE_MASTER_KEY = Book(US_DATASET, JUDY_BIEBER, "the_master_key")
RINKITINK_IN_OZ = Book(US_DATASET, JUDY_BIEBER, "rinkitink_in_oz")
DOROTHY_AND_WIZARD_OZ = Book(US_DATASET, JUDY_BIEBER, "dorothy_and_wizard_oz")
SKY_ISLAND = Book(US_DATASET, JUDY_BIEBER, "sky_island")
OZMA_OF_OZ = Book(US_DATASET, JUDY_BIEBER, "ozma_of_oz")
EMERALD_CITY_OF_OZ = Book(US_DATASET, JUDY_BIEBER, "emerald_city_of_oz")

MIDNIGHT_PASSENGER = Book(US_DATASET, MARY_ANN, "midnight_passenger")
NORTH_AND_SOUTH = Book(US_DATASET, MARY_ANN, "northandsouth")

PIRATES_OF_ERSATZ = Book(US_DATASET, ELLIOT_MILLER, "pirates_of_ersatz")
POISONED_PEN = Book(US_DATASET, ELLIOT_MILLER, "poisoned_pen")
SILENT_BULLET = Book(US_DATASET, ELLIOT_MILLER, "silent_bullet")
HUNTERS_SPACE = Book(US_DATASET, ELLIOT_MILLER, "hunters_space")
PINK_FAIRY_BOOK = Book(US_DATASET, ELLIOT_MILLER, "pink_fairy_book")

JANE_EYRE = Book(UK_DATASET, ELIZABETH_KLETT, "jane_eyre")
WIVES_AND_DAUGHTERS = Book(UK_DATASET, ELIZABETH_KLETT, "wives_and_daughters")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]
JUDY_BIEBER_BOOKS = [b for b in BOOKS if b.speaker == JUDY_BIEBER]
MARY_ANN_BOOKS = [b for b in BOOKS if b.speaker == MARY_ANN]
ELLIOT_MILLER_BOOKS = [b for b in BOOKS if b.speaker == ELLIOT_MILLER]
ELIZABETH_KLETT_BOOKS = [b for b in BOOKS if b.speaker == ELIZABETH_KLETT]
UK_BOOKS = [b for b in BOOKS if b.dataset == UK_DATASET]
US_BOOKS = [b for b in BOOKS if b.dataset == US_DATASET]


def m_ailabs_en_us_judy_bieber_speech_dataset(*args, books=JUDY_BIEBER_BOOKS, **kwargs):
    return m_ailabs_en_us_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_en_us_mary_ann_speech_dataset(*args, books=MARY_ANN_BOOKS, **kwargs):
    return m_ailabs_en_us_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_en_us_elliot_miller_speech_dataset(*args, books=ELLIOT_MILLER_BOOKS, **kwargs):
    return m_ailabs_en_us_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_en_uk_elizabeth_klett_speech_dataset(*args, books=ELIZABETH_KLETT_BOOKS, **kwargs):
    return m_ailabs_en_uk_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_en_us_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/en_US.tgz",
    extracted_name=str(US_DATASET),
    books=US_BOOKS,
    language=WSL_Languages.ENGLISH,
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
        directory, extracted_name, url, books, language, check_files, **kwargs
    )


def m_ailabs_en_uk_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/en_UK.tgz",
    extracted_name=str(UK_DATASET),
    books=UK_BOOKS,
    language=WSL_Languages.ENGLISH,
    check_files=["en_UK/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `en_UK` dataset.

    The dataset is 4GB compressed. The file to be downloaded is called `en_UK.tgz`. It contains
    45 hours of audio. When extracted, it creates a list of 2 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, language, check_files, **kwargs
    )
