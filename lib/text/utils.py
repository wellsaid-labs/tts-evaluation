import enum
import functools
import logging
import pathlib
import re
import string
import subprocess
import typing
import unicodedata
import urllib.request
from collections import defaultdict
from typing import get_args

import ftfy
import spacy
import spacy.tokens
import unidecode
from spacy.lang import en as spacy_en
from spacy.language import Language
from third_party import LazyLoader
from tqdm import tqdm

import lib

if typing.TYPE_CHECKING:  # pragma: no cover
    import Levenshtein
else:
    Levenshtein = LazyLoader("Levenshtein", globals(), "Levenshtein")


logger = logging.getLogger(__name__)

GRAPHEME_TO_PHONEME_RESTRICTED = ("[[", "]]", "<", ">")


def grapheme_to_phoneme(
    graphemes: typing.List[str],
    service: typing.Literal["espeak"] = "espeak",
    flags: typing.List[str] = ["--ipa=3", "-q", "-ven-us", "--stdin", "-m"],
    separator: str = "",
    service_separator: str = "_",
    grapheme_batch_separator: str = "<break> [[_::_::_::_::_::_::_::_::_::_::]] <break>",
    phoneme_batch_separator: str = "\n _________\n",
    restricted: typing.Tuple[str, ...] = GRAPHEME_TO_PHONEME_RESTRICTED,
) -> typing.List[str]:
    """
    TODO: Support `espeak-ng` `service`, if needed.

    Args:
        graphemes: Batch of grapheme sequences.
        service: The service used to compute phonemes.
        flags: The list of flags to add to the service.
        separator: The separator used to separate phonemes.
        service_separator: The separator used by the service between phonemes.
        grapheme_batch_separator: The seperator used deliminate grapheme sequences.
        phoneme_batch_separator: The seperator used deliminate phoneme sequences.
        restricted: An `AssertionError` will be raised if these substrings are found in `graphemes`.
    """
    template = "Special character '%s' is not allowed."
    condition = all([grapheme_batch_separator not in g for g in graphemes])
    assert condition, template % grapheme_batch_separator
    for substring in restricted:
        assert all([substring not in g for g in graphemes]), template % substring

    # NOTE: `espeak` can be inconsistent in it's handling of outer spacing; therefore, it's
    # recommended both the `espeak` output and input is trimmed.
    stripped = [strip(g) for g in graphemes]

    # NOTE: The `--sep` flag is not supported by older versions of `espeak`.
    # NOTE: We recommend using `--stdin` otherwise `espeak` might misinterpret an input like
    # "--For this community," as a flag.
    inputs = [g for g, _, _ in stripped if len(g) > 0]
    if len(inputs) > 0:
        input_ = grapheme_batch_separator.join(inputs).encode()
        output = subprocess.check_output([service] + flags, input=input_).decode("utf-8")

        message = "The separator is not unique."
        assert not separator or separator == service_separator or separator not in output, message

        phonemes = output.split(phoneme_batch_separator)
        assert len(inputs) == len(phonemes)
    phonemes = [phonemes.pop(0) if len(g) > 0 else g for g, _, _ in stripped]  # type: ignore
    assert len(phonemes) == len(graphemes)

    return_ = []
    for (grapheme, stripped_left, stripped_right), phoneme in zip(stripped, phonemes):
        phoneme = " ".join([s.strip() for s in phoneme.strip().split("\n")])

        if len(re.findall(r"\(.+?\)", phoneme)) > 0:
            message = '`%s` switched languages for phrase "%s" and outputed "%s".'
            logger.warning(message, service, grapheme, phoneme)

        # NOTE: Remove language flags like `(en-us)` or `(fr)` that might be included for text like:
        # Grapheme: “MON DIEU”
        # Phoneme: “m_ˈɑː_n (fr)_d_j_ˈø_(en-us)”
        phoneme = re.sub(r"\(.+?\)", "", phoneme)

        # NOTE: Replace multiple separators in a row without any phonemes in between with one
        # separator.
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
        if len(separator) > 0:
            phoneme = re.sub(r"%s+" % re.escape(separator), separator, phoneme)

        return_.append(phoneme.strip(separator))
    return return_


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

Phonetic Syllabification:
- State-of-the-art
  - https://aclanthology.org/N09-1035.pdf
  - http://web.archive.org/web/20220121090130/https://webdocs.cs.ualberta.ca/~kondrak/cmudict.html
  -
  http://web.archive.org/web/20170120191217/https://webdocs.cs.ualberta.ca/~kondrak/cmudict/cmudict.rep
- Rule-based Approach:
  - https://en.wikipedia.org/wiki/Phonotactics
  - https://github.com/myorm00000000/syllabify-1
- Syllabificiation Dictionaries:
  - https://catalog.ldc.upenn.edu/LDC96L14
  - https://github.com/oliverbrehm/LinguisticTextAnnotation/tree/master/backend/celex2db/celex2
  - https://www.merriam-webster.com/dictionary/pronunciation
  -
  https://en.wikipedia.org/w/index.php?search=hastemplate%3Arespell&title=Special:Search&go=Go&ns0=1

Notable People:
- github.com/nshmyrev: Contributor to cmudict, pocketsphinx, kaldi, cmusphinx, alphacep
- github.com/rhdunn: Contributor to espeak-ng, cmudict-tools, amepd, cainteoir text-to-speech
- en.wikipedia.org/wiki/User:Nardog: Contributor to wikipedia respellings, and wikipedia
    pronunciations
"""

# fmt: off
ARPAbet = typing.Literal[
    "N", "L", "S", "T", "R", "K", "IH0", "D", "M", "Z", "IY0", "B", "EH1", "P", "AA1",
    "AE1", "IH1", "G", "F", "NG", "V", "IY1", "OW0", "EY1", "HH", "SH", "OW1", "W", "AO1", "AH1",
    "AY1", "UW1", "JH", "Y", "CH", "AA0", "ER1", "EH2", "AY2", "AE2", "EY2", "AA2", "TH", "IH2",
    "EH0", "AW1", "UW0", "AE0", "AO2", "UH1", "IY2", "OW2", "AO0", "AY0", "UW2", "AH2", "EY0",
    "OY1", "AH0", "AW2", "DH", "ZH", "ER2", "UH2", "AW0", "UH0", "OY2", "OY0", "ER0",
    # NOTE: These codes have been removed in further iterations of the dictionary...
    "IH", "ER"
    # NOTE: These codes were added in later iterations of the dictionary...
    # "AXR", "AX"
]
# fmt: on


def _is_valid_cmudict_syl_word(word: str):
    return len(word) > 0 and all(c.isalpha() or c == "'" for c in word)


Pronunciation = typing.Tuple[typing.Tuple[ARPAbet, ...], ...]
CMUDictSyl = typing.Dict[str, typing.List[typing.Tuple[typing.Tuple[ARPAbet, ...], ...]]]


@functools.lru_cache(maxsize=None)
def load_cmudict_syl(
    path: pathlib.Path = lib.environment.ROOT_PATH / "lib" / "cmudict.0.6d.syl",
    url: str = (
        "http://web.archive.org/web/20170120191217if_/"
        "http://webdocs.cs.ualberta.ca:80/~kondrak/cmudict/cmudict.rep"
    ),
    comment_delimiter: str = "##",
) -> CMUDictSyl:
    """Load the CMU Pronouncing Dictionary version 0.6 augmented with syllable boundaries
    (syllabified CMU). This was created for: https://aclanthology.org/N09-1035.pdf.

    TODO: Use `pydantic` for type checking the loaded data. Learn more:
    https://pydantic-docs.helpmanual.io/

    NOTE: Loanwords from other languages are ignored to ensure ASCII compatibility.
    """
    if not path.exists():
        urllib.request.urlretrieve(url, filename=path)

    dictionary: CMUDictSyl = defaultdict(list)
    entries = [l.split(comment_delimiter, 1)[0].strip() for l in path.read_text().split("\n")]
    invalid_chars = set()
    too_many_syl = set()

    for entry in entries:
        if len(entry) == 0:
            continue

        word, rest = tuple(entry.split("  ", 1))

        # NOTE: Handle cases like: `word = "ZEPA(2)"``
        if "(" in word and ")" in word:
            word, other = tuple(word.split("(", 1))
            assert other[-1] == ")", f"Closing parentheses not found: {word}"
            assert other[:-1].isnumeric()

        # NOTE: Remove apostrophe's at the end of a word because they do not
        # change the pronunciation of the word. Learn more here:
        # https://github.com/rhdunn/amepd/commit/5fcd23a4424807e8b1c3f8736f19b38cd7e5abaf
        if (
            any(c not in string.ascii_uppercase and c != "'" for c in word)
            or word[-1] == "'"
            or word[0] == "'"
        ):
            invalid_chars.add(word)
            continue

        if word in too_many_syl:
            continue

        pronunciation = rest

        syllables: typing.List[typing.Tuple[ARPAbet, ...]] = []
        for syllable in pronunciation.split(" - "):
            arpabet = typing.cast(typing.Tuple[ARPAbet, ...], tuple(syllable.split()))
            message = f"The pronunciation may only use ARPAbet characters: {word}"
            assert all(c in get_args(ARPAbet) for c in arpabet), message
            syllables.append(arpabet)

        # NOTE: Handle abbreviations like:
        # "AOL(2)  AH0 - M ER1 - IH0 - K AH0 - AA1 N - L AY2 N"
        if len(syllables) > len(word):
            too_many_syl.add(word)
            continue

        assert word.isupper(), "A word in this dictionary must be uppercase."
        assert _is_valid_cmudict_syl_word(word)

        dictionary[word].append(tuple(syllables))

    format = lambda s: ", ".join(sorted(list(s)))
    logger.warning(f"CMUDict word(s) ignored (invalid characters): {format(invalid_chars)}")
    logger.warning(f"CMUDict word(s) ignored (too many syllabels): {format(too_many_syl)}")

    return dictionary


def get_pronunciation(word: str, dictionary: CMUDictSyl) -> typing.Optional[Pronunciation]:
    """Get the syllabified CMU pronunciation for `word`, unless it's ambigious or not available.

    Args:
        word: English word spelled with only English letter(s) or apostrophe(s).
        ...
    """
    word = word.strip()

    if len(word) > 1 and word.isupper():  # NOTE: Acronyms are not supported.
        return None

    if not _is_valid_cmudict_syl_word(word):
        return None

    # NOTE: We do not include apostrophe's at the end of a word because they do not change the
    # pronunciation of the word. Learn more here:
    # https://github.com/rhdunn/amepd/commit/5fcd23a4424807e8b1c3f8736f19b38cd7e5abaf
    word = word[:-1] if word[-1] == "'" else word
    pronunciations = dictionary[word.upper()]
    return pronunciations[0] if len(pronunciations) == 1 else None


"""
This `RESPELLINGS` dictionary is a consolidation of ARPAbet:IPA and IPA:Wikipedia respellings.

Note the CMU-provided ARPAbet differs from "ARPAbet Wiki" (see link below) because it does not use
any of the following: "AX", "AXR", "IX", "UX", "DX", "EL", "EM", "EN", "NX", "Q", "WH". This means
ARPAbet defined herein is comprised of 17 vowel sounds and 24 consonant sounds. Wikipedia, on the
other hand, uses a system comprised of 40 vowel sounds and 28 consonant sounds, including some
sounds containing multiple phonemes, some sounds having multiple respellings AND the respelling
'y' being used for both the long i vowel sound and the y consonant sound. Because of this and
because the mapping of sounds is not one-to-one between the systems, some decisions had to be
made...

This is the set of 39 IPA phonemes represented in ARPAbet:
{'ɑ', 'æ', 'ʌ', 'ɔ', 'aʊ', 'aɪ', 'ɛ', 'ɝ', 'eɪ', 'ɪ', 'i', 'oʊ', 'ɔɪ', 'ʊ', 'u', 'b', 'tʃ',
'd', 'ð', 'f', 'ɡ', 'h', 'dʒ', 'k', 'l', 'm', 'n', 'ŋ', 'p', 'ɹ', 's', 'ʃ', 't', 'θ', 'v',
'w', 'j', 'z', 'ʒ'}

This is the set of 63 IPA phoneme and phoneme combinations represented in Wikipedia Respellings:
{'ɪər', 'æ', 'ɜːr', 'juː', 'ʒ', 'aʊ', 'θ', 'ɪ', 'z', 'f', 'l', 'ʃ', 'ŋ', 'h', 'ɡ', 'd', 'ə',
'u', 'ær', 'ɒ', 'r', 's', 'ɔːr', 'uː', 'ʌ', 'p', 'hw', 'n', 'v', 'ər', 'ʊər', 'ʌr', 'ɔː',
'ʊr', 'ɪr', 'ɛər', 'ŋk', 'ɛr', 'm', 'x', 'jʊər', 't', 'eɪ', 'ð', 'iː', 'tʃ', 'aʊər', 'aɪər',
'b', 'w', 'j', 'ɔɪ', 'i', 'ɛ', 'oʊ', 'ɔɪər', 'aɪ', 'ɑːr', 'ɑː', 'ɒr', 'k', 'dʒ', 'ʊ'}

These are the 4 ARPAbet IPA phonemes missing from Wikipedia Respelling phonemes:
{'ɹ', 'ɝ', 'ɔ', 'ɑ'}

These 28 Wikipedia Respelling phonemes missing from ARPAbet phonemes:
{'ɒ', 'ər', 'ɔː', 'juː', 'ɪr', 'ŋk', 'ə', 'r', 'ʊər', 'hw', 'aʊər', 'ɑː', 'ɛər', 'iː', 'ɔːr',
'ɑːr', 'ʌr', 'ɒr', 'x', 'ɔɪər', 'ɜːr', 'ɪər', 'ʊr', 'uː', 'aɪər', 'jʊər', 'ɛr', 'ær'}

We first worked toward ARPAbet coverage, of the 4 mentioned missing phonemes:
  - For 'R' (as in 'rye'),     we use 'r' because 'ɹ'  is nearly equivalent to 'r'
  - For 'ER' (as in 'bird'),   we use 'ur' because 'ɝ' is nearly equivalent to 'ɜːr'
  - For 'AO' (as in 'bought'), we use 'aw' because 'ɔ' is nearly equivalent to 'ɔː' **
  - For 'AA' (as in 'father'), we use 'ah' because 'ɑ' is nearly equivalent to 'ɑː'

** Keep in mind, ARPAbet is inconsistent with their use of 'AO':
  - sometimes used as a long O (oh) ['BOARD']
  - sometimes used as a short O (ah) ['BALL'], but uses 'AA' for 'FATHER'
  - 'WATER' uses 'AO' but 'SEAWATER' uses 'AA'

Next, we considered the missing Wikipedia phonemes and phoneme combinations:
  - 'iː'   can be approximated with 'IY':'ee'
  - 'ə'    can be approximated with 'AH':'uh'
  - 'uː'   can be approximated with 'UW':'oo'
  - 'r'    can be approximated with 'R':'r'
  - 'x'    can be approximated with 'K':'k', very rare
  - 'hw'   can be approximated with 'W':'w', very rare
  - 'juː'  can be approximated with 'Y UW':'yoo' as in 'beauty':'BYOO-tee'
  - 'ɪr'   can be approximated with 'IH R':'ihr' as in 'mirror':'MIHR-ur'
  - 'ʊər'  can be approximated with 'UH R':'uur' as in 'premature':'pree-muh-CHUUR'
  - 'ɔɪər' can be approximated with 'OY ER':'oyur' as in 'hoyer':'HOY-ur'
  - 'aʊər' can be approximated with 'AW ER':'owur' as in 'flower':'FLOW-ur'
  - 'aɪər' can be approximated with 'AY ER':'y-ur' as in 'higher':'HY-ur'
  - 'jʊər' can be approximated with 'Y UH R':'yuur' as in 'cure':'KYUUR'
  - 'ɒ', 'ɑː' can be approximated with 'AA':'ah'
  - 'ʊr', 'ər', 'ʌr', 'ɜːr' can be approximated with 'ER':'ur'

NOTE: Wikipedia uses 'y' for both the 'iy' vowel sound and the 'y' consonant sound. We've chosen to
do the same, as preliminary testing showed good results.

Next, we considered some final phoneme combinations for user-friendliness and to correct for
some ARPAbet errors:
  - 'ŋk'  without a special exception would be 'NG K':'ngk', we instead use 'nk'
  - 'ɑːr' without a special exception would be 'AA R':'ahr', we instead use 'ar'
  - 'ɔːr' without a special exception would be 'AO R':'awr', we instead use 'or'
      - NOTE: ARPAbet misuses 'AO':'aw' in these cases, when the sound should be 'OW':'oh', like in
        'STORY' and 'BOARD'.
      - NOTE: If separated by a hyphen, we use 'oh-r'.
  - 'ɪər' without a special exception would be 'IH IY R':'iheer', we instead use 'ar'
      - NOTE: ARPAbet uses 'IH R' in 'peer' and 'IY R' in 'peering' for the same sound.
      - NOTE: If separated by a hyphen, the combination becomes 'ee-r'.
  - 'æŋ' or 'æŋk' sounds are misassigned in ARPAbet and will be re-combined to 'ang' or 'ank',
    unfortunately.
      - NOTE: When offering pronunciations for words ending in 'ang' and 'ank', ARPAbet uses 'AE'
        (as in 'bat') when they mean 'EY' (as in 'bank'). In fact, it is nearly impossible to make a
        short A sound when an NG sound is followng it.
  - 'ɛər', 'ɛr', and 'ær' without a special exception would be 'EH R':'ehr', we instead use 'err'

Lastly, we considered short and long vowels:
- For simplicity, short vowels mid-syllable should be plain: a e i o u ✓
- For short vowels ending a syllable: [no aa], eh, ih, ah, uh ✓
- For long vowels: ay, ee, (i?)y, oh, oo ✓

Resources:
- CMUDict has a number of inconsistencies:
    - This documents how CMUDict is put together,
      http://www.cs.cmu.edu/~archan/presentation/new_cmudict.pdf
    - Many of these commits are fixes for CMUDict issues,
      https://github.com/rhdunn/amepd/commits/master
- ARPAbet Wiki: https://en.wikipedia.org/wiki/ARPABET
- Wiki Respelling Key: https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key
"""
# fmt: off
RESPELLINGS: typing.Dict[str, str] = {
    'AA': 'ah', 'AE': 'a', 'AH': 'uh', 'AO': 'aw', 'AW': 'ow', 'AY': 'y', 'EH': 'eh', 'EY': 'ay',
    'IY': 'ee', 'OW': 'oh', 'OY': 'oy', 'UH': 'uu', 'UW': 'oo', 'B': 'b', 'CH': 'ch', 'D': 'd',
    'DH': 'dh', 'F': 'f', 'G': 'g', 'HH': 'h', 'JH': 'j', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n',
    'NG': 'ng', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'sh', 'T': 't', 'TH': 'th', 'V': 'v', 'W': 'w',
    'Y': 'y', 'Z': 'z', 'ZH': 'zh', 'IH': 'ih', 'ER': 'ur'
}
RESPELLING_COMBOS: typing.Dict[str, str] = {
    'ang': 'ayng', 'angk': 'aynk', 'ngk': 'nk', 'ehr': 'err', 'ahr': 'ar', 'awr': 'or', 'ihr': 'eer'
}
RESPELLING_COMBOS__SYLLABIC_SPLIT: typing.Dict[typing.Tuple[str, str], str] = {
    ("aw", "r"): "oh", ("ih", "r"): "ee"
}
RESPELLING_ALPHABET: typing.Dict[str, str] = {
    "A": "ay", "B": "bee", "C": "see", "D": "dee", "E": "ee", "F": "ehf", "G": "jee", "H": "aych",
    "I": "y", "J": "jay", "K": "kay", "L": "ehl", "M": "ehm", "N": "ehn", "O": "oh", "P": "pee",
    "Q": "kyoo", "R": "ar", "S": "ehs", "T": "tee", "U": "yoo", "V": "vee", "W": "DUH-buhl-yoo",
    "X": "ehks", "Y": "wy", "Z": "zee",
}
# fmt: on


class ARPAbetStress(enum.Enum):
    NONE: typing.Final = "0"
    PRIMARY: typing.Final = "1"
    SECONDARY: typing.Final = "2"


_REMOVE_ARPABET_MARKINGS = {ord(m.value): None for m in ARPAbetStress}


def _remove_arpabet_markings(code: ARPAbet):
    return code.translate(_REMOVE_ARPABET_MARKINGS)


def respell(word: str, dictionary: CMUDictSyl, delim: str = "-") -> typing.Optional[str]:
    """Get the respelling for `word` using the syllabified CMU pronunciation, learn more about
    respellings: https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key

    Args:
        word: English word spelled with only English letter(s) or apostrophe(s).
        ...
    """
    pronunciation = get_pronunciation(word, dictionary)
    if pronunciation is None:
        return None

    syllables = []
    # NOTE: In words where primary stress precedes secondary stress, however, the secondary stress
    # should not be differentiated from unstressed syllables; for example, "motorcycle"
    # (/ˈmoʊtərˌsaɪkəl/) should be respelled as MOH-tər-sy-kəl because MOH-tər-SY-kəl would
    # incorrectly suggest the pronunciation /ˌmoʊtərˈsaɪkəl/.
    # Learn more:
    # https://en.wikipedia.org/wiki/Help:Pronunciation_respelling_key#Syllables_and_stress
    has_primary = False
    is_upper = []
    for syl in pronunciation:
        respellings = []
        upper = False
        for phoneme in syl:
            if ARPAbetStress.PRIMARY.value in phoneme:
                upper, has_primary = True, True
            if ARPAbetStress.SECONDARY.value in phoneme and has_primary is False:
                upper = True
            respellings.append(RESPELLINGS[_remove_arpabet_markings(phoneme)])
        syllable = "".join(respellings)

        # NOTE: Handle phoneme combinations
        for combo in RESPELLING_COMBOS.keys():
            if combo in syllable:
                syllable = syllable.replace(combo, RESPELLING_COMBOS[combo])

        is_upper.append(upper)
        syllables.append(syllable)

    # NOTE: Handle phoneme combinations across two syllables.
    for i, (curr, next) in enumerate(zip(syllables, syllables[1:])):
        for (vowel, cons), replacement in RESPELLING_COMBOS__SYLLABIC_SPLIT.items():
            if curr.endswith(vowel) and next.lower().startswith(cons):
                syllables[i] = curr.replace(vowel, replacement)

    syllables = [s.upper() if u else s for s, u in zip(syllables, is_upper)]
    return delim.join(syllables)


def respell_initialism(initialism: str, delim: str = "-") -> typing.Optional[str]:
    """Get an initialism respelling, with emphasis on the final character.

    NOTE: Final syllable is most commonly stressed, learn more:
    - https://www.confidentvoice.com/blog/2-tips-for-pronouncing-abbreviations
    -
    https://english.stackexchange.com/questions/88040/why-are-all-acronyms-accented-on-the-last-syllable
    """
    syllables = [RESPELLING_ALPHABET[l] for l in list(initialism.upper())]
    syllables[-1] = syllables[-1] if delim in syllables[-1] else syllables[-1].upper()
    return delim.join(syllables)


def natural_keys(text: str) -> typing.List[typing.Union[str, int]]:
    """Returns keys (`list`) for sorting in a "natural" order.

    Inspired by: http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [(int(char) if char.isdigit() else char) for char in re.split(r"(\d+)", str(text))]


def numbers_then_natural_keys(text: str) -> typing.List[typing.List[typing.Union[str, int]]]:
    """Returns keys (`list`) for sorting with numbers first, and then natural keys."""
    return [[int(i) for i in re.findall(r"\d+", text)], natural_keys(text)]


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


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace variations into standard characters. Formfeed `f` and carriage return
    `r` should be replace with new line `\n` and tab `\t` should be replaced with two spaces
    `  `."""
    text = text.replace("\f", "\n")
    text = text.replace("\t", "  ")
    return text


def _normalize_guillemets(text: str) -> str:
    """Guillemets [https://en.wikipedia.org/wiki/Guillemet] should be normalized to standard
    quotaton marks because they carry the same meaning in speech. Some keyboards rely on Guillemets
    in place of ", but for most internet usage in particular, it's common for these to be replaced
    with standard ". Some info here: https://coerll.utexas.edu/gg/gr/mis_01.html

    Note: `ftfy` will not fix these, and `unidecode` will convert them incorrectly to <<,>>. This
    function will ensure correct normalization after `ftfy` runs and before `unidecode` runs."""
    text = text.replace("«", '"')
    text = text.replace("»", '"')
    text = text.replace("‹", "'")
    text = text.replace("›", "'")
    return text


# Normalize all text to the same unicode form, NFC. Normal form C (NFC) first applies a canonical
# decomposition, then composes pre-combined characters again. More info:
# https://docs.python.org/3/howto/unicode.html#comparing-strings
# https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
# https://stackoverflow.com/questions/7931204/what-is-normalized-utf-8-all-about
_UNICODE_NORMAL_FORM = "NFC"


def normalize_vo_script(text: str, non_ascii: frozenset, strip: bool = True) -> str:
    """Normalize a voice-over script such that only readable characters remain.

    TODO: Use `unidecode.unidecode` in "strict" mode so that data isn't lost.
    TODO: Clarify that some characters like `«` will be normalized regardless of being in the
          `non_ascii` set.
    NOTE: `non_ascii` needs to be explicitly set so that text isn't processed incorrecly accidently.
    TODO: Double check datasets for ambigiously verbalized characters like "<" which can be
          "greater than" or "silent".
    TODO: This removes characters like ℃.
    TODO: Research the impact of normalizing backticks to single quotes.

    References:
    - Generic package for text cleaning: https://github.com/jfilter/clean-text
    - ASCII characters: https://www.ascii-code.com/
    - `Unidecode` vs `unicodedata`:
      https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-normalize-in-a-python-unicode-string

    Args:
        ...
        non_ascii: Allowed Non-ASCII characters.
        strip: If `True`, strip white spaces at the ends of `text`.

    Returns: Normalized string.
    """
    text = str(text)
    text = ftfy.fix_text(text)
    text = _normalize_whitespace(text)
    text = _normalize_guillemets(text)
    text = text.replace("`", "'")  # NOTE: Normalize backticks to single quotes
    text = "".join([c if c == " " or c in non_ascii else str(unidecode.unidecode(c)) for c in text])
    text = re.compile(r" +").sub(" ", text)
    if strip:
        text = text.strip()
    return text


_NORMALIZED_ASCII_CHARS = set(
    normalize_vo_script(chr(i), frozenset(), strip=False) for i in range(0, 128)
)


def is_normalized_vo_script(text: str, non_ascii: frozenset) -> bool:
    """Return `True` if `text` has been normalized to a small set of characters."""
    return (
        unicodedata.is_normalized(_UNICODE_NORMAL_FORM, text)
        and len(set(text) - _NORMALIZED_ASCII_CHARS - non_ascii) == 0
    )


ALPHANUMERIC_REGEX = re.compile(r"[a-zA-Z0-9@#$%&+=*]")


def is_voiced(text: str, non_ascii: frozenset) -> bool:
    """Return `True` if any of the text is spoken.

    NOTE: This isn't perfect. For example, this function assumes a "-" isn't voiced; however, it
    could be a minus sign.
    NOTE: This assumes that symbols are not dictated.
    """
    text = text.strip()
    if len(text) == 0:
        return False
    assert is_normalized_vo_script(text, non_ascii), "Text must be normalized."
    return ALPHANUMERIC_REGEX.search(text) is not None or any(c in non_ascii for c in text)


DIGIT_REGEX = re.compile(r"\d")


def has_digit(text: str) -> bool:
    return bool(DIGIT_REGEX.search(text))


SPACES_REGEX = re.compile(r"\s+")


@functools.lru_cache(maxsize=2**20)
def get_spoken_chars(text: str, punc_regex: re.Pattern) -> str:
    """Remove all unspoken characters from string including spaces, marks, casing, etc.

    Example:
        >>> get_spoken_chars('123 abc !.?')
        '123abc'
        >>> get_spoken_chars('Hello. You\'ve')
        'helloyouve'

    Args:
        ...
        punc_regex: Regex for selecting all marks in `text`.

    Returns: String without unspoken characters.
    """
    text = text.lower()
    text = punc_regex.sub(" ", text)
    text = text.strip()
    return SPACES_REGEX.sub("", text)


def add_space_between_sentences(doc: spacy.tokens.Doc) -> str:
    """Add spaces between sentences which are not separated by a space."""
    if len(doc) <= 2:
        return str(doc)
    text = doc[0].text_with_ws
    for prev, curr, next in zip(doc, doc[1:], doc[2:]):
        # NOTE: Add a whitespace after `curr` if it's wedged in between two words with no
        # white space following it.
        # NOTE: This approach avoids tricky cases involving multiple sequential punctuation marks.
        if (
            next.is_sent_start
            and len(curr.whitespace_) == 0
            # NOTE: This handles newlines and other special characters
            and not curr.text[-1].isspace()
            and prev.text.isalnum()
            and next.text.isalnum()
        ):
            text += curr.text + " "
        else:
            text += curr.text_with_ws
    text += doc[-1].text_with_ws
    return text


@functools.lru_cache(maxsize=None)
def load_spacy_nlp(name, *args, **kwargs) -> Language:
    logger.info(f"Loading spaCy model `{name}`...")
    nlp = spacy.load(name, *args, **kwargs)
    logger.info(f"Loaded spaCy model `{name}`.")
    return nlp


def load_en_core_web_sm(*args, **kwargs):
    return load_spacy_nlp("en_core_web_sm", *args, **kwargs)


def load_en_english(*args, **kwargs) -> Language:
    """Load and cache in memory a spaCy `Language` object."""
    return spacy_en.English(*args, **kwargs)


"""TODO: Noramlize non-standard words (NSWs) into standard words.

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
  - Look into NVIDIA's recent text normalization and denormalization:
    https://arxiv.org/abs/2104.05055
    https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/tools/text_normalization.html
    https://github.com/NVIDIA/NeMo/pull/1797/files
"""


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
    """Check if `value` is in the range [`window[0]`, `window[1]`)."""
    return value >= window[0] and value < window[1]


def align_tokens(
    tokens: typing.Union[typing.List[str], str],
    other_tokens: typing.Union[typing.List[str], str],
    window_length: typing.Optional[int] = None,
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
