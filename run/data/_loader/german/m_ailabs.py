"""
Sub-module of M-AILABS module for downloading and processing GERMAN datasets.
"""
from run.data._loader.data_structures import Language, make_german_speaker
from run.data._loader.m_ailabs import Book, Dataset, m_ailabs_speech_dataset

DE_DATASET = Dataset("de_DE")

ANGELA_MERKEL = make_german_speaker(label="angela_merkel", gender="female")
EVA_K = make_german_speaker(label="eva_k", gender="female")
RAMONA_DEININGER = make_german_speaker(label="ramona_deininger", gender="female")
REBECCA_BRAUNERT_PLUNKETT = make_german_speaker(label="rebecca_braunert_plunkett", gender="female")

KARLSSON = make_german_speaker("karlsson", gender="male")

MERKEL_ALONE = Book(DE_DATASET, ANGELA_MERKEL, "merkel_alone")

GRUNE_HAUSE = Book(DE_DATASET, EVA_K, "grune_haus")
KLEINE_LORD = Book(DE_DATASET, EVA_K, "kleine_lord")
TOTEN_SEELEN = Book(DE_DATASET, EVA_K, "toten_seelen")
WERDE_DIE_DU_BIST = Book(DE_DATASET, EVA_K, "werde_die_du_bist")

ALTER_AFRIKANER = Book(DE_DATASET, RAMONA_DEININGER, "alter_afrikaner")
CASPAR = Book(DE_DATASET, RAMONA_DEININGER, "caspar")
FRANKENSTEIN = Book(DE_DATASET, RAMONA_DEININGER, "frankenstein")
GRUNE_GESICHT = Book(DE_DATASET, RAMONA_DEININGER, "grune_gesicht")
MENSCHENHASSER = Book(DE_DATASET, RAMONA_DEININGER, "menschenhasser")
STERBEN = Book(DE_DATASET, RAMONA_DEININGER, "sterben")
TOM_SAWYER = Book(DE_DATASET, RAMONA_DEININGER, "tom_sawyer")
TSCHUN = Book(DE_DATASET, RAMONA_DEININGER, "tschun")
WEIHNACHTSABEND = Book(DE_DATASET, RAMONA_DEININGER, "weihnachtsabend")

DAS_LETZTE_MARCHEN = Book(DE_DATASET, REBECCA_BRAUNERT_PLUNKETT, "das_letzte_marchen")
FERIEN_VOM_ICH = Book(DE_DATASET, REBECCA_BRAUNERT_PLUNKETT, "ferien_vom_ich")
MAERCHEN = Book(DE_DATASET, REBECCA_BRAUNERT_PLUNKETT, "maerchen")
MEIN_WEG_ALS_DEUTSCHER = Book(
    DE_DATASET, REBECCA_BRAUNERT_PLUNKETT, "mein_weg_als_deutscher_und_jude"
)

ALTEHOUS = Book(DE_DATASET, KARLSSON, "altehous")
HERRNARNESSCHATZ = Book(DE_DATASET, KARLSSON, "herrnarnesschatz")
KAMMACHER = Book(DE_DATASET, KARLSSON, "kammmacher")
KLEINZACHES = Book(DE_DATASET, KARLSSON, "kleinzaches")
KOENIGSGAUKLER = Book(DE_DATASET, KARLSSON, "koenigsgaukler")
LIEBESBRIEFE = Book(DE_DATASET, KARLSSON, "liebesbriefe")
MAEDCHEN_VON_MOORHOF = Book(DE_DATASET, KARLSSON, "maedchen_von_moorhof")
ODYSSEUS = Book(DE_DATASET, KARLSSON, "odysseus")
REISE_TILSIT = Book(DE_DATASET, KARLSSON, "reise_tilsit")
SANDMANN = Book(DE_DATASET, KARLSSON, "sandmann")
SCHMIED_SEINES_GLUECKES = Book(DE_DATASET, KARLSSON, "schmied_seines_glueckes")
SPIEGEL_KAETZCHEN = Book(DE_DATASET, KARLSSON, "spiegel_kaetzchen")
UNDINE = Book(DE_DATASET, KARLSSON, "undine")
UNTERM_BIRNBAUM = Book(DE_DATASET, KARLSSON, "unterm_birnbaum")

BOOKS = [v for v in locals().values() if isinstance(v, Book)]
ANGELA_MERKEL_BOOKS = [b for b in BOOKS if b.speaker == ANGELA_MERKEL]
EVA_K_BOOKS = [b for b in BOOKS if b.speaker == EVA_K]
RAMONA_DEININGER_BOOKS = [b for b in BOOKS if b.speaker == RAMONA_DEININGER]
REBECCA_BRAUNERT_PLUNKETT_BOOKS = [b for b in BOOKS if b.speaker == REBECCA_BRAUNERT_PLUNKETT]
KARLSSON_BOOKS = [b for b in BOOKS if b.speaker == KARLSSON]
DE_BOOKS = [b for b in BOOKS if b.dataset == DE_DATASET]


def m_ailabs_de_de_angela_merkel_speech_dataset(*args, books=ANGELA_MERKEL_BOOKS, **kwargs):
    return m_ailabs_de_de_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_de_de_eva_k_speech_dataset(*args, books=EVA_K_BOOKS, **kwargs):
    return m_ailabs_de_de_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_de_de_ramona_deininger_speech_dataset(*args, books=RAMONA_DEININGER_BOOKS, **kwargs):
    return m_ailabs_de_de_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_de_de_rebecca_braunert_plunkett_speech_dataset(
    *args, books=REBECCA_BRAUNERT_PLUNKETT_BOOKS, **kwargs
):
    return m_ailabs_de_de_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_de_de_karlsson_speech_dataset(*args, books=KARLSSON_BOOKS, **kwargs):
    return m_ailabs_de_de_speech_dataset(*args, books=books, **kwargs)  # type: ignore


def m_ailabs_de_de_speech_dataset(
    directory,
    url="https://data.solak.de/data/Training/stt_tts/de_DE.tgz",
    extracted_name=str(DE_DATASET),
    books=DE_BOOKS,
    language=Language.GERMAN,
    check_files=["de_DE/by_book/info.txt"],
    **kwargs,
):
    """Download, extract, and process the M-AILABS `de_DE` dataset.

    The dataset is 21GB compressed. The file to be downloaded is called `de_DE.tgz`. It contains
    237 hours of audio. When extracted, it creates a list of 32 books.
    """
    return m_ailabs_speech_dataset(
        directory, extracted_name, url, books, language, check_files, **kwargs
    )
