import dataclasses
import logging
import pathlib
import re
import typing

from third_party import LazyLoader
from torchnlp.download import download_file_maybe_extract

import run
from run.data._loader.data_structures import (
    Language,
    Passage,
    Session,
    Speaker,
    UnprocessedPassage,
    make_en_speaker,
)
from run.data._loader.utils import conventional_dataset_loader, make_passages

if typing.TYPE_CHECKING:  # pragma: no cover
    import num2words
else:
    num2words = LazyLoader("num2words", globals(), "num2words")


logger = logging.getLogger(__name__)
LINDA_JOHNSON = make_en_speaker("linda_johnson")


def _get_session(passage: UnprocessedPassage) -> Session:
    """For the LJ speech dataset, we define each chapter as an individual recording session."""
    return Session(str(passage.audio_path.stem.rsplit("-", 1)[0]))


def lj_speech_dataset(
    directory: pathlib.Path,
    root_directory_name: str = "LJSpeech-1.1",
    check_files=["LJSpeech-1.1/metadata.csv"],
    url: str = "http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
    speaker: Speaker = LINDA_JOHNSON,
    verbalize: bool = True,
    metadata_text_column: typing.Union[str, int] = 1,
    add_tqdm: bool = False,
    get_session: typing.Callable[[UnprocessedPassage], Session] = _get_session,
    **kwargs,
) -> typing.List[Passage]:
    """Load the Linda Johnson (LJ) Speech dataset.

    This is a public domain speech dataset consisting of 13,100 short audio clips of a single
    speaker reading passages from 7 non-fiction books. A transcription is provided for each clip.
    Clips vary in length from 1 to 10 seconds and have a total length of approximately 24 hours.

    Reference:
        * Link to dataset source:
          https://keithito.com/LJ-Speech-Dataset/
        * Comparison of command line resampling libraries:
          http://www.mainly.me.uk/resampling/
        * Comparison of python resampling libraries:
          https://machinelearningmastery.com/resample-interpolate-time-series-data-python/

    Books:
        * [LJ001] Morris, William, et al. Arts and Crafts Essays. 1893.
        * [LJ002-LJ019] Griffiths, Arthur. The Chronicles of Newgate, Vol. 2. 1884.
        * [LJ020] Harland, Marion. Marion Harland's Cookery for Beginners. 1893.
        * [LJ021-LJ024] Roosevelt, Franklin D. The Fireside Chats of
          Franklin Delano Roosevelt. 1933-42.
        * [LJ025-LJ027] Rolt-Wheeler, Francis. The Science - History of the Universe,
          Vol. 5: Biology. 1910.
        * [LJ028] Banks, Edgar J. The Seven Wonders of the Ancient World. 1916.
        * [LJ029-LJ050] President's Commission on the Assassination of President Kennedy. Report of
          the President's Commission on the Assassination of President Kennedy. 1964.

    Args:
        directory: Directory to cache the dataset.
        root_directory_name: Name of the extracted dataset directory.
        check_files
        url: URL of the dataset `tar.gz` file.
        speaker
        verbalize: If `True`, verbalize the text.
        metadata_text_column
        add_tqdm
        get_session
        **kwargs: Key word arguments passed to `conventional_dataset_loader`.
    """
    logger.info(f'Loading "{root_directory_name}" speech dataset...')
    download_file_maybe_extract(url, str(directory.absolute()), check_files=check_files)
    passages = conventional_dataset_loader(
        directory / root_directory_name,
        speaker,
        **kwargs,
        metadata_text_column=metadata_text_column,
    )
    passages = [_process_text(p, verbalize) for p in passages]
    return list(make_passages(root_directory_name, [passages], add_tqdm, get_session))


"""
Modules for text preprocessing.

M-AILABS speech dataset mentioned: “numbers have been converted to words and some cleanup of
“foreign” characters (transliterations) have been applied.” The Tacotron 1 paper mentioned: “All
text in our datasets is spelled out. e.g., “16” is written as “sixteen”, i.e., our models are all
trained on normalized text.” LJ Speech dataset mentioned: “Normalized Transcription: transcription
with numbers, ordinals, and monetary units expanded into full words (UTF-8).” Tacotron 2 authors
mentioned: "We use verbalized data so numbers are extended. There is no need to lowercase or remove
punctuation.".
"""


def _process_text(passage: UnprocessedPassage, verbalize: bool) -> UnprocessedPassage:
    script = _normalize_whitespace(passage.script)
    script = _normalize_quotations(script)

    if verbalize:
        script = _verbalize_special_cases(passage.audio_path, script)
        script = _expand_abbreviations(script)
        script = _verbalize_time_of_day(script)
        script = _verbalize_ordinals(script)
        script = _verbalize_currency(script)
        script = _verbalize_serial_numbers(script)
        script = _verbalize_year(script)
        script = _verbalize_numeral(script)
        script = _verbalize_number(script)
        script = _verbalize_roman_number(script)

    # NOTE: Messes up pound sign (£); therefore, this is after `_verbalize_currency`
    script = run._config.normalize_vo_script(script, Language.ENGLISH)
    return dataclasses.replace(passage, script=script, transcript=script)


# Reference: https://keithito.com/LJ-Speech-Dataset/
# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations: typing.List[typing.Tuple[typing.Pattern, str]]
_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
        ("messrs", "messrs"),
        ("no", "number"),
    ]
]


def _match_case(source: str, target: str) -> str:
    """Match `source` letter case to `target` letter case.

    Args:
        source: Reference text for the letter case.
        target: Target text to transfer the letter case.

    Returns:
        Target text with source the letter case.
    """
    if source.isupper():
        return target.upper()

    if source.islower():
        return target.lower()

    if source.istitle():
        return target.title()

    return target


def _iterate_and_replace(
    regex: typing.Pattern[str],
    text: str,
    replace: typing.Callable[[str], str],
    group: int = 1,
) -> str:
    """Iterate over all `regex` matches in `text`, and replace the matched text.

    Args:
        regex: Pattern to match subtext.
        text: Source text to edit.
        replace: Given a match, return a replacement.
        group: Regex match group to select.

    Returns:
        Updated text.
    """
    matches = re.finditer(regex, text)
    offset = 0
    for match in matches:
        start = match.start(group) + offset
        end = match.end(group) + offset
        replacement = replace(match.group(group))
        offset += start - end + len(replacement)
        text = text[:start] + replacement + text[end:]
    return text


def _expand_abbreviations(text: str) -> str:
    """Expand abbreviations in `text`.

    Notes:
        * The supported abbreviations can be found at: `_abbreviations`.
        * Expanded abbreviations will maintain the same letter case as the initial abbreviation.

    Example:
        >>> _expand_abbreviations('Mr. Gurney')
        'Mister Gurney'
    """
    for regex, expansion in _abbreviations:
        text = _iterate_and_replace(regex, text, lambda s: _match_case(s, expansion), group=0)
    return text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")


def _normalize_whitespace(text: str) -> str:
    """Normalize white spaces in `text`.

    Ensure there is only one white space at a time and no white spaces at the end.

    Example:
        >>> _normalize_whitespace('Mr.     Gurney   ')
        'Mr. Gurney'
    """
    return re.sub(_whitespace_re, " ", text).strip()


def _normalize_quotations(text: str) -> str:
    """Normalize quotation marks from the text.

    Example:
        >>> _normalize_quotations('“sponge,”')
        '"sponge,"'
    """
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    return text


_special_cases = {
    "LJ044-0055": ("544 Camp Street New", "five four four Camp Street New"),
    "LJ028-0180": ("In the year 562", "In the year five sixty-two"),
    "LJ047-0063": ("602 Elsbeth Street", "six oh two Elsbeth Street"),
    "LJ047-0160": ("411 Elm Street", "four one one Elm Street"),
    "LJ047-0069": ("214 Neely Street", "two one four Neely Street"),
    "LJ040-0121": ("P.S. 117", "P.S. one seventeen"),
    "LJ032-0036": (
        "No. 2,202,130,462",
        "No. two two zero two one three zero four six two",
    ),
    "LJ029-0193": ("100 extra off-duty", "one hundred extra off-duty"),
}

_re_filename = re.compile("LJ[0-9]{3}-[0-9]{4}")


def _verbalize_special_cases(audio: pathlib.Path, text: str) -> str:
    """Uses `_special_cases` to verbalize text.

    Args:
        audio: Filename of the audio file (e.g. LJ044-0055)
        text: Text associated with audio file

    Returns:
        text: Text with special cases verbalized.
    """
    basename = audio.name[:10]  # Extract 10 characters similar to `LJ029-0193`
    assert _re_filename.match(basename)
    if basename in _special_cases:
        return text.replace(*_special_cases[basename])
    return text


_re_time_of_day = re.compile(r"([0-9]{1,2}:[0-9]{1,2})")


def _verbalize_time_of_day(text: str) -> str:
    """Verbalize time of day in text.

    Example:
        >>> _verbalize_time_of_day('San Antonio at 1:30 p.m.,')
        'San Antonio at one thirty p.m.,'
    """

    def _replace(match):
        split = match.split(":")
        assert len(split) == 2
        words = [num2words.num2words(int(num)) for num in split]
        ret = " ".join(words)
        return ret

    return _iterate_and_replace(_re_time_of_day, text, _replace)


_re_ordinals = re.compile(r"([0-9]+(st|nd|rd|th))")


def _verbalize_ordinals(text: str) -> str:
    """Verbalize ordinals in text.

    Example:
        >>> _verbalize_ordinals('between May 1st, 1827,')
        'between May first, 1827,'
    """

    def _replace(match):
        digit = "".join([c for c in match if c.isdigit()])
        ret = num2words.num2words(int(digit), ordinal=True)
        return ret

    return _iterate_and_replace(_re_ordinals, text, _replace)


_re_currency = re.compile(r"(\S*([$£]{1}[0-9\,\.]+\b))")


def _verbalize_currency(text: str) -> str:
    """Verbalize currencies in text.

    Example:
        >>> _verbalize_currency('inch BBL, unquote, cost $29.95.')
        'inch BBL, unquote, cost twenty-nine dollars, ninety-five cents.'
    """

    def _replace(match):
        digit = match[1:].replace(",", "")
        ret = num2words.num2words(digit, to="currency", currency="USD")
        ret = ret.replace(", zero cents", "")
        ret = ret.replace("hundred and", "hundred")
        if "£" in match:
            # num2words has bugs with their GBP currency
            ret = ret.replace("dollar", "pound")
            ret = ret.replace("cents", "pence")
            ret = ret.replace("cent", "penny")
        return ret

    return _iterate_and_replace(_re_currency, text, _replace)


_re_po_box = re.compile(r"([Bb]ox [0-9]+\b)")
_re_serial_number = re.compile(r"(\b[A-Za-z]+[0-9]+\b)")


def _verbalize_serial_numbers(text: str) -> str:
    """Verbalize serial numbers in text.

    Example:
        >>> _verbalize_serial_numbers('Post Office Box 2915, Dallas, Texas')
        'Post Office Box two nine one five, Dallas, Texas'
        >>> _verbalize_serial_numbers('serial No. C2766, which was also found')
        'serial No. C two seven six six, which was also found'
    """

    def _replace(match):
        split = match.split(" ")
        ret = " ".join([num2words.num2words(int(t)) if t.isdigit() else t for t in list(split[-1])])
        if len(split) == 2:
            ret = split[0] + " " + ret
        return ret

    for regex in [_re_po_box, _re_serial_number]:
        text = _iterate_and_replace(regex, text, _replace)
    return text


_re_year_thousand = re.compile(r"(\b[0-9]{4}\b)")
_re_year_hundred = re.compile(r"\b(?:in|In) ([0-9]{3})\b")
_re_year_bce = re.compile(r"\b([0-9]{3}) B\.C\b")


def _verbalize_year(text: str) -> str:
    """Verbalize years in text.

    Example:
        >>> _verbalize_year('Newgate down to 1818,')
        'Newgate down to eighteen eighteen,'
        >>> _verbalize_year('It was about 250 B.C., when the great')
        'It was about two fifty B.C., when the great'
        >>> _verbalize_year('In 606, Nineveh')
        'In six oh-six, Nineveh'
    """

    def _replace(match):
        ret = num2words.num2words(int(match), lang="en", to="year")
        return ret

    for regex in [_re_year_thousand, _re_year_hundred, _re_year_bce]:
        text = _iterate_and_replace(regex, text, _replace)
    return text


_re_numeral = re.compile(r"(?:Number|number) ([0-9]+)")


def _verbalize_numeral(text: str) -> str:
    """Verbalize numerals in text.

    Example:
        >>> _verbalize_numeral(_expand_abbreviations('Exhibit No. 143 as the'))
        'Exhibit Number one forty-three as the'
    """

    def _replace(match):
        ret = num2words.num2words(int(match), lang="en", to="year")
        return ret

    return _iterate_and_replace(_re_numeral, text, _replace)


_re_roman_number = re.compile(
    r"\b(?:George|Charles|Napoleon|Henry|Nebuchadnezzar|William) ([IV]+\.{0,})"
)


def _verbalize_roman_number(text: str) -> str:
    """Verbalize roman numers in text.

    Example:
        >>> _verbalize_roman_number('William IV. was also the victim')
        'William the fourth was also the victim'
    """

    def _replace(match):
        if match[-1] == ".":
            match = match[:-1]

        # 0 - 9 roman number to int
        if "V" not in match:
            num = len(match)
        elif "IV" == match:
            num = 4
        else:
            num = 5 + len(match) - 1

        ret = "the " + num2words.num2words(int(num), to="ordinal")
        return ret

    return _iterate_and_replace(_re_roman_number, text, _replace)


_re_number = re.compile(r"(\b[0-9]{1}[0-9\.\,]{0,}\b)")


def _verbalize_number(text: str) -> str:
    """Verbalize numbers in text.

    Example:
        >>> _verbalize_number('Chapter 4. The Assassin:')
        'Chapter four. The Assassin:'
        >>> _verbalize_number('was shipped on March 20, and the')
        'was shipped on March twenty, and the'
        >>> _verbalize_number('distance of 265.3 feet was, quote')
        'distance of two hundred sixty-five point three feet was, quote'
        >>> _verbalize_number('information on some 50,000 cases')
        'information on some fifty thousand cases'
        >>> _verbalize_number('PRS received items in 8,709 cases')
        'PRS received items in eight thousand, seven hundred nine cases'
    """

    def _replace(match):
        match = match.replace(",", "")
        ret = num2words.num2words(float(match))
        ret = ret.replace("hundred and", "hundred")
        return ret

    return _iterate_and_replace(_re_number, text, _replace)
