import re
import typing
from functools import lru_cache, partial

from google.cloud.speech_v1p1beta1 import RecognitionConfig
from hparams import HParams, add_config

import lib
import run
from lib.text import _line_grapheme_to_phoneme, get_spoken_chars, normalize_vo_script
from lib.utils import identity
from run.data._loader import Language

# NOTE: eSpeak doesn't have a dictionary of all the phonetic characters, so this is a dictionary
# of the phonetic characters we found in the English dataset.
# TODO: Remove this once `grapheme_to_phoneme` is deprecated
PHONEME_SEPARATOR = "|"
GRAPHEME_TO_PHONEME_RESTRICTED = list(lib.text.GRAPHEME_TO_PHONEME_RESTRICTED) + [PHONEME_SEPARATOR]
# fmt: off
DATASET_PHONETIC_CHARACTERS = (
    '\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/', ':', ';', '?', '[', ']', '=', 'aɪ',
    'aɪə', 'aɪɚ', 'aɪʊ', 'aɪʊɹ', 'aʊ', 'b', 'd', 'dʒ', 'eɪ', 'f', 'h', 'i', 'iə', 'iː', 'j',
    'k', 'l', 'm', 'n', 'nʲ', 'n̩', 'oʊ', 'oː', 'oːɹ', 'p', 'r', 's', 't', 'tʃ', 'uː', 'v', 'w',
    'x', 'z', 'æ', 'æː', 'ð', 'ø', 'ŋ', 'ɐ', 'ɐː', 'ɑː', 'ɑːɹ', 'ɑ̃', 'ɔ', 'ɔɪ', 'ɔː', 'ɔːɹ',
    'ə', 'əl', 'ɚ', 'ɛ', 'ɛɹ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɪɹ', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʊ', 'ʊɹ', 'ʌ',
    'ʒ', 'ʔ', 'ˈ', 'ˌ', 'θ', 'ᵻ', 'ɬ'
)

NON_ASCII_CHARS: typing.Dict[Language, frozenset] = {
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
NON_ASCII_MARKS: typing.Dict[Language, frozenset] = {
    Language.ENGLISH: frozenset([]),
    Language.GERMAN: frozenset(["«", "»", "‹", "›"]),
    Language.PORTUGUESE_BR: frozenset(["«", "»", "‹", "›"]),
    Language.SPANISH_CO: frozenset(["¿", "¡", "«", "»", "‹", "›"]),
}
NON_ASCII_ALL = {l: NON_ASCII_CHARS[l].union(NON_ASCII_MARKS[l]) for l in Language}

_make_config = partial(
    RecognitionConfig,
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


@lru_cache(maxsize=2 ** 20)
def _grapheme_to_phoneme(grapheme: str):
    """Fast grapheme to phoneme implementation where punctuation is ignored.

    NOTE: Use private `_line_grapheme_to_phoneme` for performance...
    """
    return _line_grapheme_to_phoneme([grapheme], separator="|")[0]


_PUNCTUATION_REGEXES = {
    l: re.compile(r"[^\w\s" + "".join(NON_ASCII_CHARS[l]) + r"]") for l in Language
}


def _grapheme_to_phoneme_approx_de(text: str):
    """
    NOTE: (Rhyan) The "ß" is used to denote a "ss" sound.
    """
    return get_spoken_chars(text, _PUNCTUATION_REGEXES[Language.GERMAN]).replace("ß", "ss")


# NOTE: Phonetic rules to help determine if two words sound-a-like.
_GRAPHEME_TO_PHONEME = {
    Language.ENGLISH: _grapheme_to_phoneme,
    Language.GERMAN: _grapheme_to_phoneme_approx_de,
}


@lru_cache(maxsize=2 ** 20)
def is_sound_alike(a: str, b: str, language: Language) -> bool:
    """Return `True` if `str` `a` and `str` `b` sound a-like.

    NOTE: If two words have same sounds are spoken in the same order, then they sound-a-like.

    Example:
        >>> is_sound_alike("Hello-you've", "Hello. You've")
        True
        >>> is_sound_alike('screen introduction', 'screen--Introduction,')
        True
        >>> is_sound_alike('twentieth', '20th')
        True
        >>> is_sound_alike('financingA', 'financing a')
        True
    """
    a = normalize_vo_script(a, NON_ASCII_ALL[language])
    b = normalize_vo_script(b, NON_ASCII_ALL[language])
    spoken_chars = (
        partial(get_spoken_chars, punc_regex=_PUNCTUATION_REGEXES[language])
        if language in _PUNCTUATION_REGEXES
        else identity
    )
    grapheme_to_phoneme = (
        _GRAPHEME_TO_PHONEME[language] if language in _GRAPHEME_TO_PHONEME else identity
    )
    return (
        a.lower() == b.lower()
        or spoken_chars(a) == spoken_chars(b)
        or grapheme_to_phoneme(a) == grapheme_to_phoneme(b)
    )


def configure():
    """Configure modules involved in processing text."""
    config = {
        run._tts.encode_tts_inputs: HParams(seperator=PHONEME_SEPARATOR),
        lib.text.grapheme_to_phoneme: HParams(separator=PHONEME_SEPARATOR),
        run.train.spectrogram_model._data.InputEncoder.__init__: HParams(
            token_separator=PHONEME_SEPARATOR
        ),
    }
    add_config(config)
