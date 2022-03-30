"""
Sub-module of M-AILABS module for downloading and processing GERMAN datasets.
"""
from functools import partial

from run.data._loader import structures as struc
from run.data._loader.m_ailabs import Book, m_ailabs_speech_dataset, make_speaker

DE_DE = struc.Dialect.DE_DE

ANGELA_MERKEL = make_speaker("angela_merkel", DE_DE, "female")
EVA_K = make_speaker("eva_k", DE_DE, "female")
RAMONA_DEININGER = make_speaker("ramona_deininger", DE_DE, "female")
REBECCA_BRAUNERT_PLUNKETT = make_speaker("rebecca_braunert_plunkett", DE_DE, "female")
KARLSSON = make_speaker("karlsson", DE_DE, "female")

MERKEL_ALONE = Book(DE_DE, ANGELA_MERKEL, "merkel_alone")

GRUNE_HAUSE = Book(DE_DE, EVA_K, "grune_haus")
KLEINE_LORD = Book(DE_DE, EVA_K, "kleine_lord")
TOTEN_SEELEN = Book(DE_DE, EVA_K, "toten_seelen")
WERDE_DIE_DU_BIST = Book(DE_DE, EVA_K, "werde_die_du_bist")

ALTER_AFRIKANER = Book(DE_DE, RAMONA_DEININGER, "alter_afrikaner")
CASPAR = Book(DE_DE, RAMONA_DEININGER, "caspar")
FRANKENSTEIN = Book(DE_DE, RAMONA_DEININGER, "frankenstein")
GRUNE_GESICHT = Book(DE_DE, RAMONA_DEININGER, "grune_gesicht")
MENSCHENHASSER = Book(DE_DE, RAMONA_DEININGER, "menschenhasser")
STERBEN = Book(DE_DE, RAMONA_DEININGER, "sterben")
TOM_SAWYER = Book(DE_DE, RAMONA_DEININGER, "tom_sawyer")
TSCHUN = Book(DE_DE, RAMONA_DEININGER, "tschun")
WEIHNACHTSABEND = Book(DE_DE, RAMONA_DEININGER, "weihnachtsabend")

DAS_LETZTE_MARCHEN = Book(DE_DE, REBECCA_BRAUNERT_PLUNKETT, "das_letzte_marchen")
FERIEN_VOM_ICH = Book(DE_DE, REBECCA_BRAUNERT_PLUNKETT, "ferien_vom_ich")
MAERCHEN = Book(DE_DE, REBECCA_BRAUNERT_PLUNKETT, "maerchen")
MEIN_WEG_ALS_DEUTSCHER = Book(DE_DE, REBECCA_BRAUNERT_PLUNKETT, "mein_weg_als_deutscher_und_jude")

ALTEHOUS = Book(DE_DE, KARLSSON, "altehous")
HERRNARNESSCHATZ = Book(DE_DE, KARLSSON, "herrnarnesschatz")
KAMMACHER = Book(DE_DE, KARLSSON, "kammmacher")
KLEINZACHES = Book(DE_DE, KARLSSON, "kleinzaches")
KOENIGSGAUKLER = Book(DE_DE, KARLSSON, "koenigsgaukler")
LIEBESBRIEFE = Book(DE_DE, KARLSSON, "liebesbriefe")
MAEDCHEN_VON_MOORHOF = Book(DE_DE, KARLSSON, "maedchen_von_moorhof")
ODYSSEUS = Book(DE_DE, KARLSSON, "odysseus")
REISE_TILSIT = Book(DE_DE, KARLSSON, "reise_tilsit")
SANDMANN = Book(DE_DE, KARLSSON, "sandmann")
SCHMIED_SEINES_GLUECKES = Book(DE_DE, KARLSSON, "schmied_seines_glueckes")
SPIEGEL_KAETZCHEN = Book(DE_DE, KARLSSON, "spiegel_kaetzchen")
UNDINE = Book(DE_DE, KARLSSON, "undine")
UNTERM_BIRNBAUM = Book(DE_DE, KARLSSON, "unterm_birnbaum")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]
DE_BOOKS = [b for b in BOOKS if b.dialect == DE_DE]


def m_ailabs_de_de_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/de_DE.tgz",
    extracted_name="de_DE",
    books=DE_BOOKS,
    dialect=DE_DE,
    check_files=["de_DE/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `de_DE` dataset.

    The dataset is 21GB compressed. The file to be downloaded is called `de_DE.tgz`. It contains
    237 hours of audio. When extracted, it creates a list of 32 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, DE_DE, check_files, **kwargs
    )


_speakers = [s for s in locals().values() if isinstance(s, struc.Speaker)]
M_AILABS_DATASETS = {
    s: partial(m_ailabs_de_de_speech_dataset, books=[b for b in BOOKS if b.speaker == s])
    for s in _speakers
}
