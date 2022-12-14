import logging
import re
import typing
from functools import lru_cache, partial

import config as cf
import spacy.language
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

# TODO: We should consider adding other valid characters like "£" which can be found readily
# when using the English language much like "$".

_NON_ASCII_CHARS: typing.Dict[Language, frozenset] = {
    # Resources:
    # https://en.wikipedia.org/wiki/English_terms_with_diacritical_marks
    # https://en-academic.com/dic.nsf/enwiki/3894487
    Language.ENGLISH: frozenset(list("âÂàÀáÁêÊéÉèÈëËîÎïÏôÔùÙûÛçÇäöüÄÖÜñÑ")),
    Language.GERMAN: frozenset(list("ßäöüÄÖÜ")),
    # Portuguese makes use of five diacritics: the cedilla (ç), acute accent (á, é, í, ó, ú),
    # circumflex accent (â, ê, ô), tilde (ã, õ), and grave accent (à, and rarely è, ì, ò, and ù).
    # src: https://en.wikipedia.org/wiki/Portuguese_orthography
    Language.PORTUGUESE: frozenset(list("áÁéÉíÍóÓúÚçÇâÂêÊôÔãÃõÕàÀèÈìÌòÒùÙ")),
    # Spanish uses only the acute accent, over any vowel: ⟨á é í ó ú⟩. The only other diacritics
    # used are the tilde on the letter ⟨ñ⟩ and the diaeresis used in the sequences ⟨güe⟩ and ⟨güi⟩.
    # The special characters required are ⟨á⟩, ⟨é⟩, ⟨í⟩, ⟨ó⟩, ⟨ú⟩, ⟨ñ⟩, ⟨Ñ⟩, ⟨ü⟩, ⟨Ü⟩, ⟨¿⟩, ⟨¡⟩
    # and the uppercase ⟨Á⟩, ⟨É⟩, ⟨Í⟩, ⟨Ó⟩, and ⟨Ú⟩.
    # src: https://en.wikipedia.org/wiki/Spanish_orthography
    Language.SPANISH: frozenset(list("áÁéÉíÍóÓúÚñÑüÜ")),
}
_NON_ASCII_MARKS: typing.Dict[Language, frozenset] = {
    Language.ENGLISH: frozenset(),
    Language.GERMAN: frozenset(),
    Language.PORTUGUESE: frozenset(),
    Language.SPANISH: frozenset(list("¿¡")),
}
_NON_ASCII_ALL = {l: _NON_ASCII_CHARS[l].union(_NON_ASCII_MARKS[l]) for l in Language}


def normalize_vo_script(text: str, language: Language) -> str:
    return lib.text.normalize_vo_script(text, _NON_ASCII_ALL[language])


def is_normalized_vo_script(text: str, language: Language) -> bool:
    return lib.text.is_normalized_vo_script(text, _NON_ASCII_ALL[language])


def is_voiced(text: str, language: Language) -> bool:
    return lib.text.is_voiced(text, _NON_ASCII_CHARS[language])


def normalize_and_verbalize_text(text: str, language: Language) -> str:
    text = normalize_vo_script(text, language)
    if language == Language.ENGLISH:
        # TODO: Given that `verbalize_text` is language specific and it's WSL specific, I'd consider
        # moving it to `run` instead of `lib`.
        return lib.text.verbalize_text(text)
    return text


_PUNCT_REGEXES = {l: re.compile(r"[^\w\s" + "".join(_NON_ASCII_CHARS[l]) + r"]") for l in Language}


def get_spoken_chars(text: str, language: Language) -> str:
    return lib.text.get_spoken_chars(text, _PUNCT_REGEXES[language])


def replace_punc(text: str, replace: str, language: Language) -> str:
    return _PUNCT_REGEXES[language].sub(replace, text)


STT_CONFIGS = None
LanguageCode = typing.Literal["en-US", "de-DE", "pt-BR", "es-CO"]

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
except ImportError:
    logger.info("Ignoring optional `google` import.")


@lru_cache(maxsize=2**20)
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


@lru_cache(maxsize=2**20)
def _is_sound_alike(a: str, b: str, language: Language) -> bool:
    a = normalize_vo_script(a, language)
    b = normalize_vo_script(b, language)
    spoken_chars = partial(get_spoken_chars, language=language)
    sound_out = _SOUND_OUT[language] if language in _SOUND_OUT else identity
    return any(func(a) == func(b) for func in (_remove_letter_casing, spoken_chars, sound_out))


def is_sound_alike(a: str, b: str, language: Language) -> bool:
    """Return `True` if `str` `a` and `str` `b` sound a-like.

    NOTE: If two words have same sounds are spoken in the same order, then they sound-a-like.
    TODO: This does not support accents well, for example: `décor` and `decor` are not matching up.
          We should consider adding a check to just ensure the letters are the same, if so, that's
          enough.

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


_LANGUAGE_TO_SPACY = {
    Language.ENGLISH: "en_core_web_md",
    Language.GERMAN: "de_core_news_md",
    Language.SPANISH: "es_core_news_md",
    Language.PORTUGUESE: "pt_core_news_md",
}


def load_spacy_nlp(language: Language) -> spacy.language.Language:
    disable = ("ner", "tagger", "lemmatizer")
    return lib.text.load_spacy_nlp(_LANGUAGE_TO_SPACY[language], disable=disable)


# NOTE: This regex gets abbreviations that are multiple characters long that particularly have
# an impact on audio length. We do not match single letter initials like in "Big C", "c-suite",
# "u-boat", "t-shirt" and "rain-x" because they largely do not affect the audio length. We DO
# match acronyms like "U. S.", even so.
_LONG_ABBREV = re.compile(
    r"("
    # GROUP 2: Abbr separated with dots like "a.m.".
    r"\b([A-Za-z]\.){2,}\B"
    r"|"
    # GROUP 3: Upper-case abbr like "MiniUSA.com", "fMRI", "DirecTV", "PCI-DSS", "U. S.", "W-USA",
    #          "JCPenney", "PhD", "U. S.", etc.
    r"([A-Z0-9](?:[a-z]?[&\-\.\s*]*[A-Z0-9])+)"
    r"(?=\b|s|[A-Z][a-z])"
    r")"
)


def _get_long_abbrevs(text: str) -> typing.Tuple[str]:
    """Get a list of abbreviations that take a long time to speak in `text`."""
    return tuple(m[0] for m in _LONG_ABBREV.findall(text))


def predict_audio_length(text: str) -> float:
    """Predict the audio length given the text.

    NOTE: This approach counts the individual characters and assigns them with a seconds value. It
          was developed using this workbook
          `run/review/dataset_processing/text_audio_length_correlation.py`. It has a r=0.946
          correlation with audio length.
    """
    counts = {p: text.count(p) for p in ["-", "!", ",", ".", '"', " ", "'", "?"]}
    num_counted_punc = sum(counts.values())
    num_upper = sum(c.isupper() for c in text)
    num_lower = sum(c.islower() for c in text)
    abbreviations = "".join(_get_long_abbrevs(text))
    num_upper_initials = sum(c.isupper() for c in abbreviations)
    num_lower_initials = sum(c.islower() for c in abbreviations)
    num_initial_dots = sum(c == "." for c in abbreviations)
    counts = {
        "num_upper": num_upper - num_upper_initials,
        "num_lower": num_lower - num_lower_initials,
        "num_initials": num_upper_initials + num_lower_initials,
        **counts,
    }
    counts["."] = counts["."] - num_initial_dots
    counts["num_other_punc"] = (
        len(text) - num_upper - num_lower - num_counted_punc - num_initial_dots
    )
    seconds = (
        (0.2228, "num_initials"),
        (0.1288, "-"),
        (0.1112, "!"),
        (0.0952, ","),
        (0.0943, "num_upper"),
        (0.0815, "num_other_punc"),
        (0.0575, "num_lower"),
        (0.0487, "."),
        (0.0372, '"'),
        (0.0289, " "),
        (0.0000, "'"),
        (0.0000, "?"),
    )
    assert len(seconds) == len(counts)
    return sum(counts[feat] * val for val, feat in seconds) + 0.1561


def predict_max_audio_length(text: str) -> float:
    """Predict the maximum audio length given `text`.

    NOTE: Using speech segments in our data, this ensures that 99.97% of the time, the audio length
          is smaller than this maximum audio length, after analyzing 30k segments. The cases
          are buggy because extenuated pauses should not have been included in speech segments.
    TODO: Measure how well this aligns with spans, not just speech segments, which may include
          longer pauses. It should scale well because spans are longer, so, it'll tend toward
          the average much more.
    """
    return predict_audio_length(text) * 1.4 + 0.6


def configure(overwrite: bool = False):
    """Configure modules involved in processing text."""
    # NOTE: These are speakers reliable datasets, in the right language.
    debug_speakers = set(
        s for s in run._config.DEV_SPEAKERS if LANGUAGE is None or s.language == LANGUAGE
    )
    config = {
        run._utils._get_debug_datasets: cf.Args(speakers=debug_speakers),
        run._utils.get_unprocessed_dataset: cf.Args(language=LANGUAGE),
    }
    cf.add(config, overwrite)
