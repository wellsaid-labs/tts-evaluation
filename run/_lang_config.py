import logging
import re
import typing
from functools import lru_cache, partial

from hparams import HParams, add_config
from third_party import LazyLoader

import lib
from lib.text import _line_grapheme_to_phoneme, get_spoken_chars
from lib.utils import identity
from run.data._loader import Language

if typing.TYPE_CHECKING:  # pragma: no cover
    import google.cloud.speech_v1p1beta1 as google_speech
else:
    google_speech = LazyLoader("google_speech", globals(), "google.cloud.speech_v1p1beta1")

logger = logging.getLogger(__name__)

# NOTE: eSpeak doesn't have a dictionary of all the phonetic characters, so this is a dictionary
# of the phonetic characters we found in the English dataset.
# TODO: Remove this once `grapheme_to_phoneme` is deprecated
PHONEME_SEPARATOR = "|"
GRAPHEME_TO_PHONEME_RESTRICTED = list(lib.text.GRAPHEME_TO_PHONEME_RESTRICTED) + [PHONEME_SEPARATOR]
# fmt: off
ENGLISH_PHONETIC_CHARACTERS = (
    '\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '[', ']', '=', 'aɪ',
    'aɪə', 'aɪɚ', 'aɪʊ', 'aɪʊɹ', 'aʊ', 'b', 'd', 'dʒ', 'eɪ', 'f', 'h', 'i', 'iə', 'iː', 'j',
    'k', 'l', 'm', 'n', 'nʲ', 'n̩', 'oʊ', 'oː', 'oːɹ', 'p', 'r', 's', 't', 'tʃ', 'uː', 'v', 'w',
    'x', 'z', 'æ', 'æː', 'ð', 'ø', 'ŋ', 'ɐ', 'ɐː', 'ɑː', 'ɑːɹ', 'ɑ̃', 'ɔ', 'ɔɪ', 'ɔː', 'ɔːɹ',
    'ə', 'əl', 'ɚ', 'ɛ', 'ɛɹ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɪɹ', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ',
    'ʒ', 'ʔ', 'ˈ', 'ˌ', 'θ', 'ᵻ', 'ɬ'
)

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
    Language.PORTUGUESE_BR: frozenset([
        "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "ç", "Ç", "â", "Â", "ê", "Ê", "ô", "Ô",
        "ã", "Ã", "õ", "Õ", "à", "À", "è", "È", "ì", "Ì", "ò", "Ò", "ù", "Ù"
    ]),
    # Spanish uses only the acute accent, over any vowel: ⟨á é í ó ú⟩. The only other diacritics
    # used are the tilde on the letter ⟨ñ⟩ and the diaeresis used in the sequences ⟨güe⟩ and ⟨güi⟩.
    # The special characters required are ⟨á⟩, ⟨é⟩, ⟨í⟩, ⟨ó⟩, ⟨ú⟩, ⟨ñ⟩, ⟨Ñ⟩, ⟨ü⟩, ⟨Ü⟩, ⟨¿⟩, ⟨¡⟩
    # and the uppercase ⟨Á⟩, ⟨É⟩, ⟨Í⟩, ⟨Ó⟩, and ⟨Ú⟩.
    # src: https://en.wikipedia.org/wiki/Spanish_orthography
    Language.SPANISH_CO: frozenset([
        "á", "Á", "é", "É", "í", "Í", "ó", "Ó", "ú", "Ú", "ñ", "Ñ", "ü", "Ü"
    ]),
}
# fmt: on
_NON_ASCII_MARKS: typing.Dict[Language, frozenset] = {
    Language.ENGLISH: frozenset(),
    Language.GERMAN: frozenset(),
    Language.PORTUGUESE_BR: frozenset(),
    Language.SPANISH_CO: frozenset(["¿", "¡"]),
}
_NON_ASCII_ALL = {l: _NON_ASCII_CHARS[l].union(_NON_ASCII_MARKS[l]) for l in Language}


def normalize_vo_script(text: str, language: Language) -> str:
    return lib.text.normalize_vo_script(text, _NON_ASCII_ALL[language])


def is_normalized_vo_script(text: str, language: Language) -> bool:
    return lib.text.is_normalized_vo_script(text, _NON_ASCII_ALL[language])


def is_voiced(text: str, language: Language) -> bool:
    return lib.text.is_voiced(text, _NON_ASCII_CHARS[language])


SST_CONFIGS = None

try:
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
        Language.PORTUGUESE_BR: _make_config(language_code="pt-BR"),
        Language.SPANISH_CO: _make_config(language_code="es-CO"),
    }
    LanguageCode = typing.Literal["en-US", "de-DE", "pt-BR", "es-CO"]
except ImportError:
    logger.info("Ignoring optional `google` import.")


@lru_cache(maxsize=2 ** 20)
def _grapheme_to_phoneme(grapheme: str) -> str:
    """Fast grapheme to phoneme implementation where punctuation is ignored.

    NOTE: Use private `_line_grapheme_to_phoneme` for performance...
    """
    return _line_grapheme_to_phoneme([grapheme], separator="|")[0]


_PUNCT_REGEXES = {l: re.compile(r"[^\w\s" + "".join(_NON_ASCII_CHARS[l]) + r"]") for l in Language}


def _spoken_chars_and_de_eszett_transliteration(text: str) -> str:
    """
    NOTE: (Rhyan) The "ß" is used to denote a "ss" sound.
    """
    return get_spoken_chars(text, _PUNCT_REGEXES[Language.GERMAN]).replace("ß", "ss")


# NOTE: Phonetic rules to help determine if two words sound-a-like.
_SOUND_OUT = {
    Language.ENGLISH: _grapheme_to_phoneme,
    Language.GERMAN: _spoken_chars_and_de_eszett_transliteration,
}


def _remove_letter_casing(a: str) -> str:
    return a.lower()


@lru_cache(maxsize=2 ** 20)
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
    a = normalize_vo_script(a, language)
    b = normalize_vo_script(b, language)
    punc_regex = _PUNCT_REGEXES[language] if language in _PUNCT_REGEXES else None
    spoken_chars = partial(get_spoken_chars, punc_regex=punc_regex) if punc_regex else identity
    sound_out = _SOUND_OUT[language] if language in _SOUND_OUT else identity
    return any(func(a) == func(b) for func in (_remove_letter_casing, spoken_chars, sound_out))


def configure():
    """Configure modules involved in processing text."""
    config = {
        lib.text.grapheme_to_phoneme: HParams(separator=PHONEME_SEPARATOR),
    }
    add_config(config)
