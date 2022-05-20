"""
Sub-module of M-AILABS module for downloading and processing SPANISH datasets.
"""
from functools import partial

from run.data._loader import structures as struc
from run.data._loader.m_ailabs import Book, m_ailabs_speech_dataset, make_speaker

ES_ES = struc.Dialect.ES_ES

KAREN_SAVAGE = make_speaker("karen_savage", ES_ES, "female")
VICTOR_VILLARRAZA = make_speaker("victor_villarraza", ES_ES, "male")
TUX = make_speaker("tux", ES_ES, "male")

ANGELINA = Book(ES_ES, KAREN_SAVAGE, "angelina")

CUENTOS_CLASICOS_DEL_NORTE = Book(ES_ES, VICTOR_VILLARRAZA, "cuentos_clasicos_del_norte")
LA_DAMA_DE_LAS_CAMELIAS = Book(ES_ES, VICTOR_VILLARRAZA, "la_dama_de_las_camelias")

BAILEN = Book(ES_ES, TUX, "bailen")
NAPOLEON_EN_CHAMARTIN = Book(ES_ES, TUX, "napoleon_en_chamartin")
EL_19_DE_MARZO = Book(ES_ES, TUX, "el_19_de_marzo_y_el_2_de_nayo")
LA_BATALLA_DE_LOS_ARAPILES = Book(ES_ES, TUX, "la_batalla_de_los_arapiles")
TRAFALGAR = Book(ES_ES, TUX, "trafalgar")
ENEIDA = Book(ES_ES, TUX, "eneida")
LA_CORTE_DE_CARLOS_IV = Book(ES_ES, TUX, "la_corte_de_carlos_iv")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]
ES_ES_BOOKS = [b for b in BOOKS if b.dialect == ES_ES]


def m_ailabs_es_es_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/es_ES.tgz",
    extracted_name=str("es_ES"),
    books=ES_ES_BOOKS,
    dialect=ES_ES,
    check_files=["es_ES/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `es_ES` dataset.

    The dataset is is 15GB compressed. The file to be downloaded is called ``es_ES.tgz``. It
    contains 83 hours of audio. When extracted, it creates a list of 9 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, dialect, check_files, **kwargs
    )


_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
M_AILABS_DATASETS = {
    s: partial(m_ailabs_es_es_speech_dataset, books=[b for b in BOOKS if b.speaker == s])
    for s in _speakers
}
