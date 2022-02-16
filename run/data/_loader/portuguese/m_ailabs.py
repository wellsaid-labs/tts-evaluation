"""
Sub-module of M-AILABS module for downloading and processing SPANISH datasets.
"""
from run.data._loader.data_structures import Language, make_es_speaker
from run.data._loader.m_ailabs import Book, Dataset, m_ailabs_speech_dataset

ES_ES_DATASET = Dataset("es_ES")

KAREN_SAVAGE = make_es_speaker("karen_savage", gender="female")
VICTOR_VILLARRAZA = make_es_speaker("victor_villarraza", gender="male")
TUX = make_es_speaker("tux", gender="male")

ANGELINA = Book(ES_ES_DATASET, KAREN_SAVAGE, "angelina")

CUENTOS_CLASICOS_DEL_NORTE = Book(ES_ES_DATASET, VICTOR_VILLARRAZA, "cuentos_clasicos_del_norte")
LA_DAMA_DE_LAS_CAMELIAS = Book(ES_ES_DATASET, VICTOR_VILLARRAZA, "la_dama_de_las_camelias")

BAILEN = Book(ES_ES_DATASET, TUX, "bailen")
NAPOLEON_EN_CHAMARTIN = Book(ES_ES_DATASET, TUX, "napoleon_en_chamartin")
EL_19_DE_MARZO = Book(ES_ES_DATASET, TUX, "el_19_de_marzo_y_el_2_de_nayo")
LA_BATALLA_DE_LOS_ARAPILES = Book(ES_ES_DATASET, TUX, "la_batalla_de_los_arapiles")
TRAFALGAR = Book(ES_ES_DATASET, TUX, "trafalgar")
ENEIDA = Book(ES_ES_DATASET, TUX, "eneida")
LA_CORTE_DE_CARLOS_IV = Book(ES_ES_DATASET, TUX, "la_corte_de_carlos_iv")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]
KAREN_SAVAGE_BOOKS = [b for b in BOOKS if b.speaker == KAREN_SAVAGE]
VICTOR_VILLARRAZA_BOOKS = [b for b in BOOKS if b.speaker == VICTOR_VILLARRAZA]
TUX_BOOKS = [b for b in BOOKS if b.speaker == TUX]
ES_ES_BOOKS = [b for b in BOOKS if b.dataset == ES_ES_DATASET]


def m_ailabs_es_es_karen_savage_speech_dataset(*args, books=KAREN_SAVAGE_BOOKS, **kwargs):
    return m_ailabs_es_es_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_es_es_victor_v_speech_dataset(*args, books=VICTOR_VILLARRAZA_BOOKS, **kwargs):
    return m_ailabs_es_es_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_es_es_tux_speech_dataset(*args, books=TUX_BOOKS, **kwargs):
    return m_ailabs_es_es_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_es_es_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/es_ES.tgz",
    extracted_name=str(ES_ES_DATASET),
    books=ES_ES_BOOKS,
    language=Language.SPANISH_CO,
    check_files=["es_ES/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `es_ES` dataset.

    The dataset is is 15GB compressed. The file to be downloaded is called ``es_ES.tgz``. It
    contains ???? hours of audio. When extracted, it creates a list of 9 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, language, check_files, **kwargs
    )
