import logging
import re
import typing
from functools import lru_cache, partial

import config as cf
from third_party import LazyLoader

import lib
import run
from lib.text import grapheme_to_phoneme
from lib.utils import identity
from run.data._loader import Language

if typing.TYPE_CHECKING:  # pragma: no cover
    import google.cloud.speech_v1p1beta1 as google_speech
else:
    google_speech = LazyLoader("google_speech", globals(), "google.cloud.speech_v1p1beta1")

logger = logging.getLogger(__name__)


LANGUAGE = Language.ENGLISH

# fmt: off

_NON_ASCII_CHARS: typing.Dict[Language, frozenset] = {
    # Resources:
    # https://en.wikipedia.org/wiki/English_terms_with_diacritical_marks
    # https://en-academic.com/dic.nsf/enwiki/3894487
    Language.ENGLISH: frozenset([
        "â", "Â", "à", "À", "á", "Á", "ê", "Ê", "é", "É", "è", "È", "ë", "Ë", "î", "Î", "ï", "Ï",
        "ô", "Ô", "ù", "Ù", "û", "Û", "ç", "Ç", "ä", "ö", "ü", "Ä", "Ö", "Ü", "ñ", "Ñ",
    ]),
    Language.GERMAN: frozenset(["ß", "ä", "ö", "ü", "Ä", "Ö", "Ü"]),
    # Portuguese makes use of five diacritics: the cedilla (ç), acute accent (á, é, í, ó, ú),
    # circumflex accent (â, ê, ô), tilde (ã, õ), and grave accent (à, and rarely è, ì, ò, and ù).
    # src: https://en.wikipedia.org/wiki/Portuguese_orthography
    Language.PORTUGUESE: frozenset([
        "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "ç", "Ç", "â", "Â", "ê", "Ê", "ô", "Ô",
        "ã", "Ã", "õ", "Õ", "à", "À", "è", "È", "ì", "Ì", "ò", "Ò", "ù", "Ù"
    ]),
    # Spanish uses only the acute accent, over any vowel: ⟨á é í ó ú⟩. The only other diacritics
    # used are the tilde on the letter ⟨ñ⟩ and the diaeresis used in the sequences ⟨güe⟩ and ⟨güi⟩.
    # The special characters required are ⟨á⟩, ⟨é⟩, ⟨í⟩, ⟨ó⟩, ⟨ú⟩, ⟨ñ⟩, ⟨Ñ⟩, ⟨ü⟩, ⟨Ü⟩, ⟨¿⟩, ⟨¡⟩
    # and the uppercase ⟨Á⟩, ⟨É⟩, ⟨Í⟩, ⟨Ó⟩, and ⟨Ú⟩.
    # src: https://en.wikipedia.org/wiki/Spanish_orthography
    Language.SPANISH: frozenset([
        "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "ñ", "Ñ", "ü", "Ü"
    ]),
}
# fmt: on
_NON_ASCII_MARKS: typing.Dict[Language, frozenset] = {
    Language.ENGLISH: frozenset(),
    Language.GERMAN: frozenset(),
    Language.PORTUGUESE: frozenset(),
    Language.SPANISH: frozenset(["¿", "¡"]),
}
_NON_ASCII_ALL = {l: _NON_ASCII_CHARS[l].union(_NON_ASCII_MARKS[l]) for l in Language}


def normalize_vo_script(text: str, language: Language) -> str:
    return lib.text.normalize_vo_script(text, _NON_ASCII_ALL[language])


def is_normalized_vo_script(text: str, language: Language) -> bool:
    return lib.text.is_normalized_vo_script(text, _NON_ASCII_ALL[language])


def is_voiced(text: str, language: Language) -> bool:
    return lib.text.is_voiced(text, _NON_ASCII_CHARS[language])


_PUNCT_REGEXES = {l: re.compile(r"[^\w\s" + "".join(_NON_ASCII_CHARS[l]) + r"]") for l in Language}


def get_spoken_chars(text: str, language: Language) -> str:
    return lib.text.get_spoken_chars(text, _PUNCT_REGEXES[language])


def replace_punc(text: str, replace: str, language: Language) -> str:
    return _PUNCT_REGEXES[language].sub(replace, text)


SST_CONFIGS = None

try:
    # TODO: Integrate this with the new `Dialect`s data structure.
    _make_config = partial(
        google_speech.RecognitionConfig,
        model="command_and_search",
        use_enhanced=True,
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True,
    )
    STT_CONFIGS = {
        Language.ENGLISH: _make_config(language_code="en-US", model="video"),
        Language.GERMAN: _make_config(language_code="de-DE"),
        Language.PORTUGUESE: _make_config(language_code="pt-BR"),
        Language.SPANISH: _make_config(language_code="es-CO"),
    }
    LanguageCode = typing.Literal["en-US", "de-DE", "pt-BR", "es-CO"]
except ImportError:
    logger.info("Ignoring optional `google` import.")


@lru_cache(maxsize=2 ** 20)
def _grapheme_to_phoneme(grapheme: str) -> str:
    """Fast grapheme to phoneme implementation where punctuation is ignored.

    NOTE: Use private `_line_grapheme_to_phoneme` for performance...
    """
    return grapheme_to_phoneme([grapheme], separator="|")[0]


def _spoken_chars_and_de_eszett_transliteration(text: str) -> str:
    """
    NOTE: (Rhyan) The "ß" is used to denote a "ss" sound.
    """
    return get_spoken_chars(text, Language.GERMAN).replace("ß", "ss")


# NOTE: Phonetic rules to help determine if two words sound-a-like.
_SOUND_OUT = {
    Language.ENGLISH: _grapheme_to_phoneme,
    Language.GERMAN: _spoken_chars_and_de_eszett_transliteration,
}


def _remove_letter_casing(a: str) -> str:
    return a.lower()


@lru_cache(maxsize=2 ** 20)
def _is_sound_alike(a: str, b: str, language: Language) -> bool:
    a = normalize_vo_script(a, language)
    b = normalize_vo_script(b, language)
    spoken_chars = partial(get_spoken_chars, language=language)
    sound_out = _SOUND_OUT[language] if language in _SOUND_OUT else identity
    return any(func(a) == func(b) for func in (_remove_letter_casing, spoken_chars, sound_out))


def is_sound_alike(a: str, b: str, language: Language) -> bool:
    """Return `True` if `str` `a` and `str` `b` sound a-like.

    NOTE: If two words have same sounds are spoken in the same order, then they sound-a-like.

    Example:
        >>> is_sound_alike("Hello-you've", "Hello. You've", Language.ENGLISH)
        True
        >>> is_sound_alike('screen introduction', 'screen--Introduction,', Language.ENGLISH)
        True
        >>> is_sound_alike('twentieth', '20th', Language.ENGLISH)
        True
        >>> is_sound_alike('financingA', 'financing a', Language.ENGLISH)
        True
    """
    if a == b:
        return True

    return _is_sound_alike(a, b, language)


def configure(overwrite: bool = False):
    """Configure modules involved in processing text."""
    config = {
        run._config.data._include_passage: cf.Args(language=LANGUAGE),
        run.train._utils._get_dataset: cf.Args(language=LANGUAGE),
    }
    cf.add(config, overwrite)
