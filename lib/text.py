import itertools
import json
import logging
import os
import pathlib
import re
import shlex
import string
import subprocess
import typing
from collections import defaultdict
from functools import lru_cache, partial
from multiprocessing.pool import ThreadPool
from typing import get_args

import ftfy
import hparams
import unidecode
from third_party import LazyLoader
from tqdm import tqdm

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import en_core_web_md
    import Levenshtein
    import nltk
    import normalise
    import spacy
    import spacy.tokens
    from spacy.lang import en as spacy_en
else:
    en_core_web_md = LazyLoader("en_core_web_md", globals(), "en_core_web_md")
    Levenshtein = LazyLoader("Levenshtein", globals(), "Levenshtein")
    nltk = LazyLoader("nltk", globals(), "nltk")
    normalise = LazyLoader("normalise", globals(), "normalise")
    spacy = LazyLoader("spacy", globals(), "spacy")
    spacy_en = LazyLoader("spacy_en", globals(), "spacy.lang.en")


logger = logging.getLogger(__name__)


def _grapheme_to_phoneme_helper(
    grapheme: str,
    service: typing.Literal["espeak"] = "espeak",
    flags: typing.List[str] = ["--ipa=3", "-q", "-ven-us", "--stdin"],
    separator: str = "",
    service_separator: str = "_",
) -> str:
    """
    TODO: Support `espeak-ng` `service`, if needed.

    Args:
        grapheme
        service: The service used to compute phonemes.
        flags: The list of flags to add to the service.
        separator: The separator used to separate phonemes.
        service_separator: The separator used by the service between phonemes.
    """
    # NOTE: `espeak` can be inconsistent in it's handling of outer spacing; therefore, it's
    # recommended both the `espeak` output and input is trimmed.
    grapheme, stripped_left, stripped_right = strip(grapheme)

    # NOTE: The `--sep` flag is not supported by older versions of `espeak`.
    # NOTE: We recommend using `--stdin` otherwise `espeak` might misinterpret an input like
    # "--For this community," as a flag.
    command = "echo %s | %s %s" % (shlex.quote(grapheme), service, " ".join(flags))
    phoneme = subprocess.check_output(command, shell=True).decode("utf-8")
    assert (
        not separator or separator == service_separator or separator not in phoneme
    ), "The separator is not unique."

    phoneme = " ".join([s.strip() for s in phoneme.strip().split("\n")])

    if len(re.findall(r"\(.+?\)", phoneme)) > 0:
        logger.warning(
            '`%s` switched languages for phrase "%s" and outputed "%s".', service, grapheme, phoneme
        )

    # NOTE: Remove language flags like `(en-us)` or `(fr)` that might be included for text like:
    # Grapheme: “MON DIEU”
    # Phoneme: “m_ˈɑː_n (fr)_d_j_ˈø_(en-us)”
    phoneme = re.sub(r"\(.+?\)", "", phoneme)

    # NOTE: Replace multiple separators in a row without any phonemes in between with one separator.
    phoneme = re.sub(r"%s+" % re.escape(service_separator), service_separator, phoneme)
    phoneme = re.sub(r"%s+\s+" % re.escape(service_separator), " ", phoneme)
    phoneme = re.sub(r"\s+%s+" % re.escape(service_separator), " ", phoneme)
    phoneme = phoneme.strip()

    phoneme = stripped_left + phoneme + stripped_right
    phoneme = phoneme.replace(service_separator, separator)

    # NOTE: Add separators around stress tokens and words.
    phoneme = phoneme.replace(" ", separator + " " + separator)
    phoneme = phoneme.replace("ˈ", separator + "ˈ" + separator)
    phoneme = phoneme.replace("ˌ", separator + "ˌ" + separator)
    phoneme = (
        re.sub(r"%s+" % re.escape(separator), separator, phoneme) if len(separator) > 0 else phoneme
    )
    return phoneme.strip(separator)


def _grapheme_to_phoneme(grapheme: str, separator: str = "", **kwargs) -> str:
    # NOTE: `grapheme` is split on new lines because `espeak` is inconsistent in it's handling of
    # new lines.
    split = [
        _grapheme_to_phoneme_helper(s, separator=separator, **kwargs) for s in grapheme.split("\n")
    ]
    return_ = (separator + "\n" + separator).join(split)
    # NOTE: We need to remove double separators from when there are consecutive new lines like
    # "\n\n\n", for example.
    if len(separator) > 0:
        return_ = re.sub(r"%s+" % re.escape(separator), separator, return_).strip(separator)
    return return_


def _grapheme_to_phoneme_preserve_punctuation(
    doc: spacy.tokens.Doc, separator: str = "", **kwargs
) -> str:
    """Convert grapheme to phoneme while preserving punctuation.

    Args:
        doc
        separator: The separator used to separate phonemes, stress, and punctuation.
        **kwargs: Key-word arguments passed to `_grapheme_to_phoneme`.

    Returns:
        Phonemes with the original punctuation (as defined by spaCy).
    """
    assert not separator or all(
        separator not in t.text for t in doc
    ), "The separator is not unique."

    # NOTE: `is_punct` is not contextual while `pos == spacy.symbols.PUNCT` is, see:
    # https://github.com/explosion/spaCy/issues/998. This enables us to phonemize cases like:
    # - "form of non-linguistic representations"  (ADJ)
    # - "The psychopaths' empathic reaction"  (PART)
    # - "judgement, name & face memory" (CCONJ)
    # - "to public interest/national security" (SYM)
    # - "spectacular, grand // desco da" (SYM)
    return_ = []
    iterator = itertools.groupby(doc, lambda t: t.pos == spacy.symbols.PUNCT)  # type: ignore
    for is_punct, group in iterator:
        phrase = "".join([t.text_with_ws for t in group])
        is_alpha_numeric = any(c.isalpha() or c.isdigit() for c in phrase)
        if is_punct and is_alpha_numeric:
            logger.warning("Punctuation contains alphanumeric characters: %s" % phrase)
        if is_punct and not is_alpha_numeric:
            return_.extend(list(phrase))
        else:
            return_.append(_grapheme_to_phoneme(phrase, separator=separator, **kwargs))
    return separator.join([t for t in return_ if len(t) > 0])


# TODO: With `spacy` v3 and their type hints up, we can change these signatures to include
# `spacy.tokens.Doc`.


@typing.overload
def grapheme_to_phoneme(
    graphemes: typing.Union[typing.Tuple[str], typing.List[str]],
    chunk_size: int = 128,
    **kwargs,
) -> typing.List[str]:
    ...


@typing.overload
def grapheme_to_phoneme(
    graphemes: str,
    chunk_size: int = 128,
    **kwargs,
) -> str:
    ...


@hparams.configurable
def grapheme_to_phoneme(graphemes, chunk_size: int = 128, max_parallel=os.cpu_count(), **kwargs):
    """Convert graphemes into phonemes and preserve punctuation.

    NOTE: `espeak` can give different results for the same argument, sometimes. For example,
    "Fitness that's invigorating, not intimidating!" sometimes returns...
    1. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|ɚ|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...
    2. "f|ˈ|ɪ|t|n|ə|s| |ð|æ|t|s| |ɪ|n|v|ˈ|ɪ|ɡ|oː|ɹ|ˌ|eɪ|ɾ|ɪ|ŋ|,| "...

    TODO: Replace the eSpeak with a in-house solution including:
    - `lib.text.normalize_non_standard_words`
    - CMU dictionary or https://github.com/kylebgorman/wikipron for most words
    - spaCy for homographs similar to https://github.com/Kyubyong/g2p
    - A neural network trained on CMU dictionary for words not in the dictionaries.

    Args:
        graphemes: The graphemes to convert to phonemes.
        chunk_size: `chunk_size` parameter passed to `imap` for multiprocessing.
        max_parallel
        **kwargs: Key-word arguments passed to `_grapheme_to_phoneme_preserve_punctuation`.
    """
    assert chunk_size >= 1
    is_iterable = not isinstance(graphemes, (str, spacy.tokens.Doc, spacy.tokens.span.Span))
    graphemes = list(graphemes) if is_iterable else [graphemes]
    part = partial(_grapheme_to_phoneme_preserve_punctuation, **kwargs)

    if any(isinstance(g, str) for g in graphemes):
        items = {i: g for i, g in enumerate(graphemes) if isinstance(g, str)}
        nlp = load_en_core_web_md(disable=("parser", "ner"))
        for i, doc in zip(items.keys(), nlp.pipe(items.values())):
            graphemes[i] = doc

    if len(graphemes) < chunk_size:
        return_ = [part(g) for g in graphemes]
        return return_[slice(0, len(graphemes)) if is_iterable else 0]  # type: ignore

    logger.info("Getting phonemes for %d graphemes.", len(graphemes))
    with ThreadPool(min(max_parallel, len(graphemes))) as pool:
        return list(tqdm(pool.imap(part, graphemes, chunksize=chunk_size), total=len(graphemes)))


""" Learn more about pronunciation dictionarys:

CMU Dictionary: (https://github.com/cmusphinx/cmudict)
- Versioning: https://github.com/rhdunn/cmudict
- Data Loader: https://www.nltk.org/_modules/nltk/corpus/reader/cmudict.html
- Data Loader: https://github.com/prosegrinder/python-cmudict
- Data Loader: https://github.com/coolbutuseless/phon
- Web Tool: http://www.speech.cs.cmu.edu/cgi-bin/cmudict?stress=-s&in=WEIDE
- Web Code: http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/
- Based off CMUDict, the American English Pronunciation Dictionary:
  https://github.com/rhdunn/amepd
  https://github.com/MycroftAI/mimic1/issues/15
  https://github.com/MycroftAI/mimic1/pull/36
  https://github.com/rhdunn/amepd/commits/master?after=9e800706fbd76c129769a22196e96910c33c443b+384
- Data Cleaner: https://github.com/Alexir/CMUdict
- Data Cleaner: https://github.com/rhdunn/cmudict-tools

Grapheme-to-Phoneme based on CMU Dictionary:
- Python Tool: https://github.com/Kyubyong/g2p
- CMU Sphinx: https://github.com/cmusphinx/g2p-seq2seq
- Lexicon Tool: http://www.speech.cs.cmu.edu/tools/lextool.html

Other Dictionaries:
- Moby Project: https://en.wikipedia.org/wiki/Moby_Project#Pronunciator
- Wiktionary Pronunciation: https://github.com/kylebgorman/wikipron
- Received Pronunciation (British English) Pronunciation Dictionary:
  https://github.com/rhdunn/en-GB-x-rp.dict
- Scottish English Pronunciation Dictionary
  https://github.com/rhdunn/en-scotland.dict/blob/master/en-scotland.dict
- eSpeak dictionary
  https://github.com/espeak-ng/espeak-ng/blob/master/dictsource/en_list

Partial Dictionaries:
- Homographs:
  - https://en.wikipedia.org/wiki/Heteronym_(linguistics)
  - https://github.com/Kyubyong/g2p/blob/master/g2p_en/homographs.en
  - http://web.archive.org/web/20180816004508/http://www.minpairs.talktalk.net/graph.html
- Acronyms
  - https://github.com/rhdunn/cmudict/blob/master/acronym
  - https://github.com/allenai/scispacy#abbreviationdetector (Automatic)

Notable People:
- github.com/nshmyrev: Contributor to cmudict, pocketsphinx, kaldi, cmusphinx, alphacep
- github.com/rhdunn: Contributor to espeak-ng, cmudict-tools, amepd, cainteoir text-to-speech
"""

# NOTE: AmEPD based their POS tags off this:
# https://github.com/rhdunn/pos-tags/blob/master/cainteoir.ttl
AMEPD_PART_OF_SPEECH_COARSE = typing.Literal[
    "adj",  # adjective
    "adv",  # adverb
    "conj",  # conjunction
    "det",  # determiner
    "intj",  # interjection
    "noun",
    "num",  # number
    "prep",  # preposition
    "pron",  # pronoun
    "verb",
]

SPACY_TO_AMEPD_POS: typing.Dict[int, typing.Optional[AMEPD_PART_OF_SPEECH_COARSE]] = {
    spacy.symbols.ADJ: "adj",  # type: ignore # adjective -> adjective
    spacy.symbols.ADP: None,  # type: ignore # adposition
    spacy.symbols.ADV: "adv",  # type: ignore # adverb -> adverb
    spacy.symbols.AUX: "verb",  # type: ignore # auxiliary -> verb
    spacy.symbols.CONJ: "conj",  # type: ignore # conjunction -> conjunction
    spacy.symbols.CCONJ: "conj",  # type: ignore # coordinating conjunction -> conjunction
    spacy.symbols.DET: "det",  # type: ignore # determiner -> determiner
    spacy.symbols.INTJ: "intj",  # type: ignore # interjection -> interjection
    spacy.symbols.NOUN: "noun",  # type: ignore # noun -> noun
    spacy.symbols.NUM: "num",  # type: ignore # numeral -> numeral
    spacy.symbols.PART: None,  # type: ignore # particle
    spacy.symbols.PRON: "noun",  # type: ignore # pronoun -> noun
    spacy.symbols.PROPN: "noun",  # type: ignore # proper noun -> noun
    spacy.symbols.PUNCT: None,  # type: ignore # punctuation
    spacy.symbols.SCONJ: "conj",  # type: ignore # subordinating conjunction -> conjunction
    spacy.symbols.SYM: None,  # type: ignore # symbol
    spacy.symbols.VERB: "verb",  # type: ignore # verb -> verb
    spacy.symbols.X: None,  # type: ignore # other
    spacy.symbols.SPACE: None,  # type: ignore
}


# NOTE: AmEPD has only one word that uses "attr" or "pred", as October 2020.
AMEPD_PART_OF_SPEECH_FINE = typing.Literal[
    "attr",  # NOTE: Stands for: "attributive"
    "pred",  # NOTE: Stands for: "predicative"
    "past",  # NOTE: "past" tense
]

# fmt: off
AMEPD_ARPABET = typing.Literal[
    "N", "AX", "L", "S", "T", "R", "K", "IH0", "D", "M", "Z", "AXR", "IY0", "B", "EH1", "P", "AA1",
    "AE1", "IH1", "G", "F", "NG", "V", "IY1", "OW0", "EY1", "HH", "SH", "OW1", "W", "AO1", "AH1",
    "AY1", "UW1", "JH", "Y", "CH", "AA0", "ER1", "EH2", "AY2", "AE2", "EY2", "AA2", "TH", "IH2",
    "EH0", "AW1", "UW0", "AE0", "AO2", "UH1", "IY2", "OW2", "AO0", "AY0", "UW2", "AH2", "EY0",
    "OY1", "AH0", "AW2", "DH", "ZH", "ER2", "UH2", "AW0", "UH0", "OY2", "OY0", "ER0",
]
# fmt: on


class AmEPDPartOfSpeech(typing.NamedTuple):
    """
    Learn more about a two-step representation of part-of-speech:
    https://spacy.io/api/annotation#pos-tagging
    https://hpi.de/fileadmin/user_upload/fachgebiete/plattner/teaching/NaturalLanguageProcessing/NLP2017/NLP04_PartOfSpeechTagging.pdf

    Args:
        coarse: Coarse-grained part-of-speech tags (e.g. "verb" and "adv").
        fine: Fine-grained part-of-speech tags, like the tense.
    """

    coarse: typing.Literal[AMEPD_PART_OF_SPEECH_COARSE]
    fine: typing.Optional[typing.List[typing.Literal[AMEPD_PART_OF_SPEECH_FINE]]]


class AmEPDMetadata(typing.NamedTuple):
    """
    Args:
        name: For nouns, this provides an additional descriptor (e.g. "family" for "REILLY" or
            "place/palace" for "VERSAILLES").
        lang: The BCP 47 (RFC 5646) language tag for the pronunciation, representing the origin
            language (e.g. "fr" for "VERSAILLES").
        usage: This tag clarifies how this pronunciations meaning
            (e.g. "sound" or "fish" for "BASS").
        misspelling: The pronunciation is a misspelling of this (e.g. "THOU" for "THOUGH").
        root: The root word which this pronunciation is derived (e.g. "AXE" for "AXES").
        group: A tag used to group related pronunciations (e.g. "Harry Potter" for "HORCRUX").
        expanded: This pronunciation is an expansion of this word
            (e.g. "MPG" is pronounced like "MILES PER GALLON").
        disable_warnings: This metadata field is used by `cmudict-tools` for reviewing the
            pronunciation dictionary.
    """

    name: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    lang: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    usage: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    misspelling: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    root: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    group: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    expanded: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None
    disable_warnings: typing.Optional[typing.Union[typing.Tuple[str, ...], str]] = None


class AmEPDPronunciation(typing.NamedTuple):
    """
    Args:
        pronunciation: List of ARPABET characters representing a pronunciation
            (e.g. "R EH1 D" for "READ").
        pos: This pronunciation has this part-of-speech
            (e.g. "noun" or "verb" or "adj" or "adv" for "WICKED").
        metadata: Additional structured metadata about the pronunciation.
        note: Additional unstructured information about the pronunciation.
    """

    pronunciation: typing.Tuple[AMEPD_ARPABET, ...]
    pos: typing.Optional[AmEPDPartOfSpeech] = None
    metadata: AmEPDMetadata = AmEPDMetadata()
    note: typing.Optional[str] = None


def _assert_valid_amepd_word(word):
    assert all(
        c.isalpha() or c == "'" for c in word
    ), "Words may only be defined with letter(s) or apostrophe(s)."


@lru_cache(maxsize=None)
def _load_amepd(
    path: pathlib.Path = lib.environment.ROOT_PATH / "third_party" / "amepd" / "cmudict",
    comment_delimiter: str = ";;;",
) -> typing.Dict[str, typing.List[AmEPDPronunciation]]:
    """Load the American English Pronunciation Dictionary.

    TODO: Use `pydantic` for type checking the loaded data. Learn more:
    https://pydantic-docs.helpmanual.io/

    NOTE: Loanwords from other languages are ignored to ensure ASCII compatibility.
    """
    dictionary: typing.Dict[str, typing.List[AmEPDPronunciation]] = defaultdict(list)
    entries = [l.split(comment_delimiter, 1)[0].strip() for l in path.read_text().split("\n")]
    ignored_words = []
    for entry in entries:
        if len(entry) == 0:
            continue

        kwargs: typing.Dict[str, typing.Union[None, AmEPDPartOfSpeech, AmEPDMetadata, str]] = {}
        word, rest = tuple(entry.split("  ", 1))

        # NOTE: Handle cases like: `word = "BATHED(verb@past)"``
        if "(" in word:
            word, other = tuple(word.split("(", 1))
            assert other[-1] == ")", "Closing parentheses not found."
            other = other[:-1]
            if not other.isnumeric():
                coarse = other
                fine: typing.Optional[str] = None
                if "@" in other:
                    coarse, fine = tuple(other.split("@", 1))
                    assert fine in get_args(AMEPD_PART_OF_SPEECH_FINE), "Invalid part-of-speech."
                assert coarse in get_args(AMEPD_PART_OF_SPEECH_COARSE), "Invalid part-of-speech."
                kwargs["pos"] = AmEPDPartOfSpeech(coarse, fine)  # type: ignore

        if any(c not in string.ascii_uppercase and c != "'" for c in word):
            ignored_words.append(word)
            continue

        pronunciation = rest
        # NOTE: Handle cases like:
        # `rest = "B EY1 L AXR #@@{ "name": "family" }@@ Terence Bayler"``
        # `rest = "OW1 L # ol' = old"`
        if "#" in rest:
            pronunciation, rest = tuple([s.strip() for s in rest.split("#", 1)])
            if "@@" in rest:
                split = tuple([s for s in rest.split("@@") if len(s) > 0])
                assert len(split) == 1 or len(split) == 2
                if len(split) == 1:
                    (structured,) = split
                else:
                    structured, unstructured = split
                    kwargs["note"] = unstructured
                metadata = json.loads(structured)
                metadata = {
                    k.replace("-", "_"): (tuple(v) if isinstance(v, list) else v)
                    for k, v in metadata.items()
                }
                kwargs["metadata"] = AmEPDMetadata(**metadata)
            else:
                kwargs["note"] = rest

        arpabet = typing.cast(typing.Tuple[AMEPD_ARPABET, ...], tuple(pronunciation.split()))
        assert all(
            c in get_args(AMEPD_ARPABET) for c in arpabet
        ), "The pronunciation may only use ARPABET characters."

        assert word.isupper(), "A word in this dictionary must be uppercase."
        _assert_valid_amepd_word(word)
        assert word[-1] != "'", (
            "This dictionary does not include apostrophe's at the end of a word because they do not"
            "change the pronunciation of the word. Learn more here: "
            "https://github.com/rhdunn/amepd/commit/5fcd23a4424807e8b1c3f8736f19b38cd7e5abaf"
        )

        dictionary[word].append(AmEPDPronunciation(arpabet, **kwargs))  # type: ignore
    logger.warning("Non-ascii word(s) in AmEPD dictionary ignored: %s", ", ".join(ignored_words))
    return dictionary


def get_initialism_pronunciation(word: str) -> typing.Tuple[AMEPD_ARPABET, ...]:
    """Get the ARABET pronunciation for an initialism."""
    _assert_valid_amepd_word(word)
    dictionary = _load_amepd()
    pronunciation: typing.List[AMEPD_ARPABET] = []
    for character in word:
        lambda_ = lambda p: p.metadata.usage == "stressed" or p.metadata.usage is None
        character_pronunciations = list(filter(lambda_, dictionary[character.upper()]))
        assert len(set(p.pronunciation for p in character_pronunciations)) == 1
        pronunciation.extend(character_pronunciations[0].pronunciation)
    return tuple(pronunciation)


def get_pronunciation(
    word: str,
    pos_coarse: typing.Optional[typing.Literal[AMEPD_PART_OF_SPEECH_COARSE]] = None,
    pos_fine: typing.Optional[typing.Literal["past", "pres"]] = None,
) -> typing.Optional[typing.Tuple[AMEPD_ARPABET, ...]]:
    """Get the ARABET pronunciation for `word`, unless it's ambigious or not available.

    Args:
        word: English word spelled with only English letter(s) or apostrophe(s).
        pos_coarse: Coarse-grained part-of-speech tags (e.g. "verb" and "adv")
        pos_fine: Fine-grained part-of-speech tags, like the part-of-speech tense.
    """
    _assert_valid_amepd_word(word)
    dictionary = _load_amepd()

    # NOTE: This dictionary does not include apostrophe's at the end of a word because they do not
    # change the pronunciation of the word. Learn more here:
    # https://github.com/rhdunn/amepd/commit/5fcd23a4424807e8b1c3f8736f19b38cd7e5abaf
    word = word[:-1] if word[-1] == "'" else word

    pronunciations = dictionary[word.upper()]

    # NOTE: An abbreviation is not necessarily always expanded when voiced, and due to this
    # ambigiuty we do not return a pronunciation.
    is_maybe_an_abbreviation = any(p.metadata.expanded is not None for p in pronunciations)
    pronunciations = [] if is_maybe_an_abbreviation else pronunciations

    # NOTE: In descending order of part-of-speech specificity, look for pronunciations.
    pos = AmEPDPartOfSpeech(pos_coarse, pos_fine)  # type: ignore
    for rule in (
        lambda p: pos.fine and p.pos == pos,
        lambda p: pos.fine and p.pos and p.pos.coarse == pos.coarse and not p.pos.fine,
        lambda p: pos.coarse and p.pos and p.pos.coarse == pos.coarse,
    ):
        if any(rule(p) for p in pronunciations):
            pronunciations = list(filter(rule, pronunciations))
            break

    if len(pronunciations) > 1:
        lib.utils.call_once(
            logger.warning,  # type: ignore
            "Unable to disamgiuate pronunciation of '%s'.",
            word,
        )
    elif len(pronunciations) == 0:
        lib.utils.call_once(
            logger.warning,  # type: ignore
            "Unable to find pronunciation of '%s'.",
            word,
        )

    return pronunciations[0].pronunciation if len(pronunciations) == 1 else None


_is_initialism_default = lambda t: len(_load_amepd()[t.text.upper()]) == 0 and t.text.isupper()


def get_pronunciations(
    doc: spacy.tokens.Doc,
    is_initialism: typing.Callable[[spacy.tokens.token.Token], bool] = _is_initialism_default,
) -> typing.Tuple[typing.Optional[typing.Tuple[AMEPD_ARPABET, ...]], ...]:
    """Get the ARABET pronunciation for each token in `doc`.

    Args:
        doc
        is_initialism: This callable disambiguates a normal word from an initialism.
    """
    return_: typing.List[typing.Optional[typing.Tuple[AMEPD_ARPABET, ...]]] = []
    for token in doc:
        if not all(c.isalpha() or c == "'" for c in token.text):
            lib.utils.call_once(
                logger.warning,  # type: ignore
                "Words may only have letter(s) or apostrophe(s): '%s'",
                token.text,
            )
            return_.append(None)
        else:
            pos = lib.text.SPACY_TO_AMEPD_POS[token.pos]
            tense = spacy_en.TAG_MAP[token.tag_].get("Tense", None)
            prounciation = lib.text.get_pronunciation(token.text, pos, tense)
            if is_initialism(token):
                lib.utils.call_once(
                    logger.warning,  # type: ignore
                    "Guessing '%s' is an initialism.",
                    token.text,
                )
                prounciation = lib.text.get_initialism_pronunciation(token.text)
            return_.append(prounciation)
    return tuple(return_)


def natural_keys(text: str) -> typing.List[typing.Union[str, int]]:
    """Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r"(\d+)", str(text))]


def strip(text: str) -> typing.Tuple[str, str, str]:
    """Strip and return the stripped text.

    Returns:
        text: The stripped text.
        left: Text stripped from the left-side.
        right: Text stripped from the right-side.
    """
    input_ = text
    text = text.rstrip()
    right = input_[len(text) :]
    text = text.lstrip()
    left = input_[: len(input_) - len(right) - len(text)]
    return text, left, right


def normalize_vo_script(text: str, strip: bool = True) -> str:
    """Normalize a voice-over script such that only readable characters remain.

    References:
    - Generic package for text cleaning: https://github.com/jfilter/clean-text
    - ASCII characters: https://www.ascii-code.com/
    - `Unidecode` vs `unicodedata`:
      https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string
    """
    text = str(text)
    text = ftfy.fix_text(text)
    text = text.replace("\f", "\n")
    text = text.replace("\t", "  ")
    text = str(unidecode.unidecode(text))
    if strip:
        text = text.strip()
    return text


_READABLE_CHARACTERS = set(
    lib.utils.flatten([normalize_vo_script(chr(i), strip=False) for i in range(0, 128)])
)


def is_normalized_vo_script(text: str) -> bool:
    """ Return `True` if `text` has been normalized to a small set of characters. """
    return len(set(text) - _READABLE_CHARACTERS) == 0


def add_space_between_sentences(doc: spacy.tokens.Doc) -> str:
    """ Add spaces between sentences which are not seperated by a space. """
    if len(doc) == 1:
        return doc[0].text_with_ws
    text = ""
    for curr, next in zip(doc, doc[1:]):
        # NOTE: Add whitespace if there isn't one at the sentence start.
        prev = None if curr.i == 0 else doc[curr.i - 1]
        if (
            next.is_sent_start
            and len(curr.whitespace_) == 0
            and not curr.text[-1].isspace()
            # CASE: Don't split apart punctuation
            and not (next.is_punct and curr.is_punct)
            # CASE: Don't split apart if `curr` is already followed by a punctuation and a
            # white-space.
            and not (next.is_punct and len(next.whitespace_) > 0)
            # CASE: Don't split apart if there are two sentences back to back, for some reason.
            and not curr.is_sent_start
            # CASE: Don't split apart if `curr` is already preceded by a punctuation and a
            # white-space.
            and not (prev is not None and prev.is_punct and len(prev.whitespace_) > 0)
        ):
            text += curr.text + " "
        else:
            text += curr.text_with_ws
    text += next.text_with_ws  # type: ignore
    return text


@lru_cache(maxsize=None)
def _nltk_download(dependency):
    """ Run `nltk.download` but only once per process. """
    nltk.download(dependency)


@lru_cache(maxsize=None)
def load_en_core_web_md(*args, **kwargs) -> spacy_en.English:
    """ Load and cache in memory a spaCy `spacy_en.English` object. """
    return en_core_web_md.load(*args, **kwargs)


@lru_cache(maxsize=None)
def load_en_english(*args, **kwargs) -> spacy_en.English:
    """ Load and cache in memory a spaCy `spacy_en.English` object. """
    return spacy_en.English()


def normalize_non_standard_words(text: str, variety: str = "AmE", **kwargs) -> str:
    """Noramlize non-standard words (NSWs) into standard words.

    References:
      - Text Normalization Researcher, Richard Sproat:
        https://scholar.google.com/citations?hl=en&user=LNDGglkAAAAJ&view_op=list_works&sortby=pubdate
        https://rws.xoba.com/
      - Timeline:
        - Sproat & Jaitly Dataset (2020):
          https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
        - Zhang & Sproat Paper (2019):
          https://www.mitpressjournals.org/doi/full/10.1162/COLI_a_00349
        - Wu & Gorman & Sproat Code (2016):
            https://github.com/google/TextNormalizationCoveringGrammars
        - Ford & Flint `normalise` Paper (2017): https://www.aclweb.org/anthology/W17-4414.pdf
        - Ford & Flint `normalise` Code (2017): https://github.com/EFord36/normalise
        - Sproat & Jaitly Dataset (2017): https://github.com/rwsproat/text-normalization-data
        - Siri (2017): https://machinelearning.apple.com/research/inverse-text-normal
        - Sproat Kaggle Challenge (2017):
          https://www.kaggle.com/c/text-normalization-challenge-english-language/overview
        - Sproat Kaggle Dataset (2017): https://www.kaggle.com/google-nlu/text-normalization
        - Sproat TTS Tutorial (2016): https://github.com/rwsproat/tts-tutorial
        - Sproat & Jaitly Paper (2016): https://arxiv.org/pdf/1611.00068.pdf
        - Wu & Gorman & Sproat Paper (2016): https://arxiv.org/abs/1609.06649
        - Gorman & Sproat Paper (2016): https://transacl.org/ojs/index.php/tacl/article/view/897/213
        - Ebden and Sproat (2014) Code:
          https://github.com/google/sparrowhawk
          https://opensource.google/projects/sparrowhawk
          https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/39061#219939
        - Sproat Course (2011):
          https://web.archive.org/web/20181029032542/http://www.csee.ogi.edu/~sproatr/Courses/TextNorm/
      - Other:
        - MaryTTS text normalization:
          https://github.com/marytts/marytts/blob/master/marytts-languages/marytts-lang-en/src/main/java/marytts/language/en/Preprocess.java
        - ESPnet text normalization:
          https://github.com/espnet/espnet_tts_frontend/tree/master/tacotron_cleaner
        - Quora question on text normalization:
          https://www.quora.com/Is-it-possible-to-use-festival-toolkit-for-text-normalization
        - spaCy entity classification:
          https://explosion.ai/demos/displacy-ent
          https://prodi.gy/docs/named-entity-recognition#manual-model
          https://spacy.io/usage/examples#training
        - Dockerized installation of festival by Google:
          https://github.com/google/voice-builder

    TODO:
       - Following the state-of-the-art approach presented here:
         https://www.kaggle.com/c/text-normalization-challenge-english-language/discussion/43963
         Use spaCy to classify entities, and then use a formatter to clean up the strings. The
         dataset was open-sourced here:
         https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
         A formatter can be found here:
         https://www.kaggle.com/neerjad/class-wise-regex-functions-l-b-0-995
         We may need to train spaCy to detect new entities, if the ones already supported are not
         enough via prodi.gy:
         https://prodi.gy/docs/named-entity-recognition#manual-model
       - Adopt Google's commercial "sparrowhawk" or the latest grammar
         "TextNormalizationCoveringGrammars" for text normalization.
    """
    for dependency in (
        "brown",
        "names",
        "wordnet",
        "averaged_perceptron_tagger",
        "universal_tagset",
    ):
        _nltk_download(dependency)

    tokens = [[t.text, t.whitespace_] for t in load_en_english()(text)]
    merged = [tokens[0]]
    # TODO: Use https://spacy.io/usage/linguistic-features#retokenization
    for token, whitespace in tokens[1:]:
        # NOTE: For example, spaCy tokenizes "$29.95" as two tokens, and this undos that.
        if (merged[-1][0] == "$" or token == "$") and merged[-1][1] == "":
            merged[-1][0] += token
            merged[-1][1] = whitespace
        else:
            merged.append([token, whitespace])

    assert "".join(lib.utils.flatten(merged)) == text
    normalized = normalise.normalise([t[0] for t in merged], variety=variety, **kwargs)
    return "".join(lib.utils.flatten([[n.strip(), m[1]] for n, m in zip(normalized, merged)]))


def format_alignment(
    tokens: typing.List[str],
    other_tokens: typing.List[str],
    alignments: typing.List[typing.Tuple[int, int]],
) -> typing.Tuple[str, str]:
    """Format strings to be printed of the alignment.

    Example:

        >>> tokens = ['i','n','t','e','n','t','i','o','n']
        >>> other_tokens = ['e','x','e','c','u','t','i','o','n']
        >>> alignment = [(0, 0), (2, 1), (3, 2), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]
        >>> formatted = format_alignment(tokens, other_tokens, alignment)
        >>> formatted[0]
        'i n t e   n t i o n'
        >>> formatted[1]
        'e   x e c u t i o n'

    Args:
        tokens: Sequences of strings.
        other_tokens: Sequences of strings.
        alignment: Alignment between the tokens in both sequences. The alignment is a sequence of
            sorted tuples with a `tokens` index and `other_tokens` index.

    Returns:
        formatted_tokens: String formatted for printing including `tokens`.
        formatted_other_tokens: String formatted for printing including `other_tokens`.
    """
    tokens_index = 0
    other_tokens_index = 0
    alignments_index = 0
    tokens_string = ""
    other_tokens_string = ""
    while True and len(alignments) != 0:
        alignment = alignments[alignments_index]
        other_token = other_tokens[other_tokens_index]
        token = tokens[tokens_index]
        if tokens_index == alignment[0] and other_tokens_index == alignment[1]:
            padded_format = "{:" + str(max(len(token), len(other_token))) + "s} "
            tokens_string += padded_format.format(token)
            other_tokens_string += padded_format.format(other_token)
            tokens_index += 1
            other_tokens_index += 1
            alignments_index += 1
        elif tokens_index == alignment[0]:
            other_tokens_string += other_token + " "
            tokens_string += " " * (len(other_token) + 1)
            other_tokens_index += 1
        elif other_tokens_index == alignment[1]:
            tokens_string += token + " "
            other_tokens_string += " " * (len(token) + 1)
            tokens_index += 1

        if alignments_index == len(alignments):
            break

    return tokens_string.rstrip(), other_tokens_string.rstrip()


def _is_in_window(value: int, window: typing.Tuple[int, int]) -> bool:
    """ Check if `value` is in the range [`window[0]`, `window[1]`).  """
    return value >= window[0] and value < window[1]


def align_tokens(
    tokens: typing.Union[typing.List[str], str],
    other_tokens: typing.Union[typing.List[str], str],
    window_length: typing.Optional[int] = None,
    all_alignments: bool = False,
    allow_substitution: typing.Callable[[str, str], bool] = lambda a, b: True,
) -> typing.Tuple[int, typing.List[typing.Tuple[int, int]]]:
    """Compute the alignment between `tokens` and `other_tokens`.

    Base algorithm implementation: https://en.wikipedia.org/wiki/Levenshtein_distance

    This implements a modified version of the levenshtein distance algorithm. The modifications are
    as follows:
      - We do not assume each token is a character; therefore, the cost of substition is the
        levenshtein distance between tokens. The cost of deletion or insertion is the length
        of the token. In the case that the tokens are characters, this algorithms functions
        equivalently to the original.
      - The user may specify a `window_length`. Given that the window length is the same length as
        the smallest sequence, then this algorithm functions equivalently to the original;
        otherwise, not every token in both sequences is compared to compute the alignment. A user
        would use this parameter if the two sequences are mostly aligned. The runtime of the
        algorithm with a window length is
        O(`window_length` * `max(len(tokens), len(other_tokens))`).

    Args:
        tokens: Sequences of strings.
        other_tokens: Sequences of strings.
        window_length: Approximately the maximum number of consecutive insertions or deletions
            required to align two similar sequences.
        allow_substitution: Callable that returns `True` if the substitution is allowed between two
            tokens `a` and `b`.

    Returns:
        cost: The cost of alignment.
        alignment: The alignment consisting of a sequence of indicies.
    """
    # For `window_center` to be on a diagonal with an appropriate slope to align both sequences,
    # `tokens` needs to be the longer sequence.
    flipped = len(other_tokens) > len(tokens)
    if flipped:
        other_tokens, tokens = (tokens, other_tokens)

    if window_length is None:
        window_length = len(other_tokens)

    alignment_window_length = min(2 * window_length + 1, len(other_tokens) + 1)
    row_one = typing.cast(typing.List[typing.Optional[int]], [None] * alignment_window_length)
    row_two = typing.cast(typing.List[typing.Optional[int]], [None] * alignment_window_length)
    # NOTE: This operation copies a reference to the initial list `alignment_window_length` times.
    # Example:
    # >>> list_of_lists = [[]] * 10
    # >>> list_of_lists[0].append(1)
    # >>> list_of_lists
    # [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    row_one_paths: typing.List[typing.List[typing.List[typing.Tuple[int, int]]]]
    row_one_paths = [[[]]] * alignment_window_length
    row_two_paths: typing.List[typing.List[typing.List[typing.Tuple[int, int]]]]
    row_two_paths = [[[]]] * alignment_window_length

    # Number of edits to convert a substring of the `other_tokens` into the empty string.
    row_one[0] = 0
    for i in range(min(window_length, len(other_tokens))):
        assert row_one[i] is not None
        # Deletion of `other_tokens[:i + 1]`.
        row_one[i + 1] = typing.cast(int, row_one[i]) + len(other_tokens[i])

    # Both `row_one_window` and `row_two_window` are not inclusive of the maximum.
    row_one_window = (0, min(len(other_tokens) + 1, window_length + 1))

    for i in tqdm(range(len(tokens))):
        # TODO: Consider setting the `window_center` at the minimum index in `row_one`. There are
        # a number of considerations to make:
        # 1. The window no longer guarantees that the last window will have completed both
        # sequences.
        # 2. Smaller indicies in `row_one` have not completed as much of the sequences as larger
        # sequences in `row_one`.
        window_center = min(i, len(other_tokens))
        row_two_window = (
            max(0, window_center - window_length),
            min(len(other_tokens) + 1, window_center + window_length + 1),
        )
        if (row_two_window[1] - row_two_window[0]) < len(row_two) and (
            row_two_window[1] == len(other_tokens) + 1
        ):
            row_two = row_two[: (row_two_window[1] - row_two_window[0])]
            row_two_paths = row_two_paths[: (row_two_window[1] - row_two_window[0])]

        for j in range(*row_two_window):
            choices = []

            if _is_in_window(j, row_one_window):
                deletion_cost = typing.cast(int, row_one[j - row_one_window[0]]) + len(tokens[i])
                deletion_path = row_one_paths[j - row_one_window[0]]
                choices.append((deletion_cost, deletion_path))
            if _is_in_window(j - 1, row_two_window):
                assert row_two[j - 1 - row_two_window[0]] is not None
                insertion_cost = typing.cast(int, row_two[j - 1 - row_two_window[0]]) + len(
                    other_tokens[j - 1]
                )
                insertion_path = row_two_paths[j - 1 - row_two_window[0]]
                choices.append((insertion_cost, insertion_path))
            if _is_in_window(j - 1, row_one_window):
                token = tokens[i]
                other_token = other_tokens[j - 1]
                if token == other_token or allow_substitution(token, other_token):
                    substition_cost = Levenshtein.distance(token, other_token)  # type: ignore
                    substition_cost = row_one[j - 1 - row_one_window[0]] + substition_cost
                    alignment = (j - 1, i) if flipped else (i, j - 1)
                    substition_path = row_one_paths[j - 1 - row_one_window[0]]
                    substition_path = [p + [alignment] for p in substition_path]
                    choices.append((substition_cost, substition_path))

            # NOTE: `min` picks the first occurring minimum item in the iterable passed.
            # NOTE: Substition is put last on purpose to discourage substition if other options are
            # available that cost the same.
            min_cost, min_paths = min(choices, key=lambda p: p[0])

            row_two[j - row_two_window[0]] = min_cost
            row_two_paths[j - row_two_window[0]] = min_paths

        row_one = row_two[:]
        row_one_paths = row_two_paths[:]
        row_one_window = row_two_window

    return (
        typing.cast(int, row_one[-1]),
        row_one_paths[-1][0],
    )
