"""
TODO: Utilize spaCy 3.0 for entity identification [inconsistent, unreliable in 2.0 so far]
TODO: Handle...
  * Fractions ?
  * Nominals (digits, identification, zip codes?)
  * Generic, final catch-all numeral case
  * Abbreviations
      * Measurements/units ✔
      * Addresses/streets
  * Symbols
      * # (hashtag)
      * / (silent, slash, context)
      * @ (at)
      * : (to - ratio context ?)
TODO: Group together all the text related modules. For example, we could create a folder to
incapsulate `text.py`, `text_utils.py` and `non_standard_words.py`? Or we could just add everything
to `text.py`?
TODO: First, run a sentencizer before classifying and normalizing the text. This should help ensure
that end of sentence punctuation marks are preserved. For example, this will help in scenarios
where a dotted phone number or abbreviation is at the end of a sentence.
TODO: Add a classifier which checks if a string has any non-standard words. This should be
helpful for finding test cases, and ensuring that nothing was missed. For example, it could look
for the presence of numbers.
TODO: Accept a spaCy `Doc` as an arguement, use it's metadata to normalize non-standard words...

We'd need to normalize at the `Token` rather than `Span` level so that each word as an embedding.
It's also preferrable to use the embeddings prior to normalization because the text spaCy trained
on is likely not verbalized. Furthermore, if working with `Token`s, we'd need to create a mechanism
for rearranging them when verbalizing monetary values. Keep in mind, if we are using word embeddings
prior to verbalization, we should do the same thing during training.
"""
import logging
import re
import typing

import en_core_web_sm
from num2words import num2words as __num2words
from spacy.tokens import Span

from lib.non_standard_words import (
    ACRONYMS,
    CURRENCIES,
    HYPHENS,
    LARGE_FICTIOUS_NUMBERS,
    LARGE_NUMBERS,
    MONEY_ABBREVIATIONS,
    MONEY_SUFFIX,
    ORDINAL_SUFFIXES,
    PLUS_OR_MINUS_PREFIX,
    SYMBOLS_VERBALIZED,
    TITLES_PERSON,
    TITLES_PERSON_SFX,
    UNITS_ABBREVIATIONS,
)

logger = logging.getLogger(__name__)

# TODO: Don't load `en_core_web_md` globally.
SPACY = en_core_web_sm.load()


def _num2words(num: str, ignore_zeros: bool = True, **kwargs) -> str:
    """Normalize `num` into standard words.

    TODO: Should we consider just creating a basic function that voices each individual number
    without any fancy interpretation? Is that what we need?

    Args:
        ...
        ignore_zeros: If `False`, this verbalizes the leading and trailing zeros, as well.
    """
    if ignore_zeros:
        return __num2words(num, **kwargs)

    lstripped = num.lstrip("0")  # NOTE: Handle leading zeros
    out = ["zero" for _ in range(len(num) - len(lstripped))]
    if "." in num:
        stripped = lstripped.rstrip("0")  # NOTE: Handle trailing zeros
        zeros = ["zero" for _ in range(len(lstripped) - len(stripped))]
        if stripped != ".":
            out.append(__num2words(stripped, **kwargs))
        if stripped[-1] == ".":
            out.append("point")
        out.extend(zeros)
    elif len(lstripped) > 0:
        out.append(__num2words(lstripped, **kwargs))

    return " ".join(out)


def reg_ex_or(vals: typing.Iterable[str], delimiter: str = r"|") -> str:
    return delimiter.join(re.escape(v) for v in vals)


_HYPHEN_PATTERN = re.compile(reg_ex_or(HYPHENS))


def _normalize_text_from_pattern(
    text: str,
    regex: typing.Union[typing.Iterable[typing.Pattern[str]], typing.Pattern[str]],
    translate: typing.Callable[[typing.Match[str]], str],
    *args,
    **kwargs,
) -> str:
    """
    Args:
        text: Text string to be normalized.
        regex: Regex pattern(s) to search and find non-standard words.
        translate: Normalization function to translate non-standard words into standard words.
        *args: Arguments passed to `translate`.
        **kwargs: Key-word arguments passed to `translate`.

    Returns: Normalized text string.
    """
    for regex in [regex] if isinstance(regex, re.Pattern) else regex:
        for match in reversed(list(regex.finditer(text))):
            replacement_text = translate(match, *args, **kwargs)
            text = text[: match.start()] + replacement_text + text[match.end() :]
    return text


def _normalize_money(match: typing.Match[str]) -> str:
    """Take regex match of money pattern and translate it into standard words. Considers currency,
    unit abbreviations, trailing 'big money' amounts, and dollar/cent splits.

    Args:
        match: Matched text (e.g. $1.2 trillion, $5, £2.36, ¥1MM)
    """
    currency: str = match.group(1)  # (e.g. $, €, NZ$)
    money: str = match.group(2)  # (e.g. 1.2, 15,000)
    abbr: str = match.group(3)  # (e.g. M, K, k, BB)
    trail: typing.Optional[str] = match.group(4)  # (e.g. trillion)

    # CASE: Big Money ($1.2B, $1.2 billion, etc.)
    if abbr:
        abbr = abbr.upper()
        return " ".join([_num2words(money), MONEY_ABBREVIATIONS[abbr], CURRENCIES[currency][1]])
    elif trail:
        return _num2words(money) + trail + " " + CURRENCIES[currency][1]

    # CASE: Standard ($1.20, $4,000, etc.)
    tokens = money.split(".")
    assert len(tokens) == 2 or len(tokens) == 1, "Found strange number."
    dollars = int(tokens[0].replace(",", ""))
    cents = int(tokens[1]) if len(tokens) > 1 else 0
    normalized = _num2words(dollars) + " " + CURRENCIES[currency][0 if dollars == 1 else 1]
    if cents > 0:
        normalized += " and " + _num2words(cents)
        normalized += " " + CURRENCIES[currency][2 if cents == 1 else 3]
    return normalized


def _normalize_ordinals(match: typing.Match[str]) -> str:
    """Take regex match of ordinal pattern and translate it into standard words.

    Args:
        match: Matched text (e.g. 1st, 2nd)
    """
    numeral = "".join(match.group(1).split(","))
    return _num2words(numeral, ordinal=True)


def _normalize_times(match: typing.Match[str], o_clock: str = "") -> str:
    """Take regex match of time or verse pattern and translate it into standard words.

    Args:
        match: Matched text (e.g. 10:04PM, 2:13 a.m.)
        o_clock: Phrasing preference of 'o'clock' for on-the-hour times.
    """
    hour: str = match.group(2)
    minute: str = o_clock if match.group(3) == "00" else _num2words(match.group(3))
    suffix: typing.Optional[str] = match.group(4)
    suffix = "" if suffix is None else suffix.replace(".", "").strip().upper()
    segments = (s for s in (_num2words(hour), minute, suffix) if len(s) > 0)
    return " ".join(segments)


def _normalize_abbreviated_times(match: typing.Match[str]) -> str:
    """Take regex match of time or verse pattern and translate it into standard words.

    Args:
        match: Matched text (e.g. 10PM, 10 p.m.)
    """
    suffix = "" if match.group(2) is None else match.group(2).replace(".", "").strip().upper()
    return _num2words(match.group(1)) + " " + suffix


def _normalize_phone_numbers(match: typing.Match[str]) -> str:
    """Take regex match of phone number pattern and translate it into standard words.

    TODO: Support "oh" instead of "zero" because it's more typical based on this source:
    https://www.woodwardenglish.com/lesson/telephone-numbers-in-english/

    Args:
        match: Matched text (e.g. 1.800.573.1313, (123) 555-1212)
    """
    phone_number = match.group(0)
    numerals = re.split(r"\+|\(|\)|-|\.| ", phone_number)
    digits = []
    for n in numerals:
        if len(n) > 0:
            if n == "800":
                digits.append(_num2words(n))
            elif re.search(r"[a-zA-Z]", n) is not None:
                digits.append(n)
            else:
                digits.append(" ".join(list(map(_num2words, list(n)))))
    return ", ".join(digits)


num2year = lambda n: _num2words(int(n), lang="en", to="year")


def _normalize_years(match: typing.Match[str]) -> str:
    """Take regex match of year pattern and translate it into standard words.

    TODO: Make more robust. Consider spaCy's 'DATE' entity type in conjunction with the regex
    pattern. Don't need to find all years, just 3- and 4-digit years because these are pronounced
    differently from standard numerical pronunciations.

    Args:
        match: Matched text (e.g. 1900, 1776)
    """
    year = match.group(0)
    yr_range = _HYPHEN_PATTERN.split(year)
    return " to ".join(list(map(num2year, yr_range)))


def _normalize_decades(match: typing.Match[str]) -> str:
    """Take regex match of decade pattern and translate it into standard words.

    Args:
        match: Matched text (e.g. 1900s, 70s, '80s)
    """
    year = num2year(match.group(2))
    return "tie".join(year.rsplit("ty", 1)) + "s"


def _normalize_number_ranges(match: typing.Match[str], connector: str = "to") -> str:
    """Take regex match of numerical range pattern and translate it into standard words.

    Args:
        match: Matched text (e.g. 10-15, 35-75, 125-300, 50-50)
    """
    num_range = match.group(0)
    # Non-range special cases
    # TODO: Should 7-11 and 9-11 be included?
    special_cases = ["1-800", "50-50"]  # '7-11' and '9-11' are commonly spoken without 'to'
    connector = " " if num_range in special_cases else " %s " % connector
    return connector.join(list(map(_num2words, _HYPHEN_PATTERN.split(num_range))))


def _normalize_percents(match: typing.Match[str]) -> str:
    """Take regex match of percent or percent range pattern and translate it into standard words.

    TODO: Test examples that require a comma or quote.

    Args:
        match: Matched text (e.g. 75%, 15–25%)
    """
    num = "".join(re.split(r",|%|'|\"", match.group(0)))
    suffix = " %s" % SYMBOLS_VERBALIZED.get("%")
    return " to ".join(list(map(_num2words, _HYPHEN_PATTERN.split(num)))) + suffix


def _normalize_number_signs(match: typing.Match[str]) -> str:
    """Take regex match of percent or percent range pattern and translate it into standard words.

    TODO: # as 'hash', # as 'hashtag', # as 'pound' ?

    Args:
        match: Matched text (e.g. #1, #2-4)
    """
    num = "".join(re.split(r",|#|'|\"", match.group(0)))
    if any(h in num for h in HYPHENS):
        return "numbers " + " through ".join(list(map(_num2words, _HYPHEN_PATTERN.split(num))))
    return "number " + _num2words(num)


def _normalize_urls(match: typing.Match[str]) -> str:
    """Take regex match of URL (website, email, etc.) pattern and translate it into standard words.

    Args:
        match: Matched text
            (e.g. https://www.wellsaidlabs.com, wellsaidlabs.com, rhyan@wellsaidlabs.com)
    """
    return_ = ""
    if (
        match.group(1) is None
        and match.group(2) is None
        and match.group(3) is None
        and "@" in match.group(4)
    ):  # CASE: Email
        return_ += " at ".join(match.group(4).split("@")) + " " + " ".join(match.group(5, 6))
    else:  # CASE: Web Address
        # NOTE: Handle prefixes (e.g. "http://", "https://", "www.")
        return_ += " " + " ".join(" ".join(list(m)) for m in match.groups()[:2] if m is not None)
        return_ += " ".join(m for m in match.groups()[2:] if m is not None)

    for s in SYMBOLS_VERBALIZED:
        return_ = return_.replace(s, " %s " % SYMBOLS_VERBALIZED.get(s))

    for word in return_.split(" "):
        for char in word:
            if char.isdigit():
                return_ = return_.replace(char, " %s " % _num2words(char))

    return re.sub(" +", " ", return_).strip()


def _normalize_acronyms(match: typing.Match[str], end_of_sentence: bool = False) -> str:
    """Take regex match of acronym (all capitals) pattern and translate it into standard words.

    TODO: Consider how to to coordinate with Roman Numeral cases? Can you identify an ORG from a RN?

    Args:
        match: Matched text (e.g. RSVP, CEO, NASA, ASAP, NBA, A.S.A.P., C.T.O.)
        end_of_sentence: If the acronym ends in a dot, then that dot is assumed to be a period.
    """
    last_char = "." if end_of_sentence and match.group(0)[-1] == "." else ""
    acronym = "".join(match.group(0).split("."))
    acronym = ACRONYMS[acronym] if acronym in ACRONYMS else acronym
    return acronym + last_char


def _normalize_measurement_abbreviations(match: typing.Match[str]) -> str:
    """
    Take string and search for abbreviations, following numerals.

    TODO: Fix.
    TODO: Support cases likes "m/s".

    pattern_abbr = "(\+|-|±|\+\/-)?(\d{1,3}(,\d{3})*(\.\d+)?)( |-)?(%s)\\b" % "|".join(
        UNITS_ABBREVIATIONS.keys()
    )
    MEASUREMENT_ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(" + reg_ex_or(PLUS_OR_MINUS_PREFIX.keys()) + r")?"
        r"(\d{1,3}(,\d{3})*(\.\d+)?)"
        r"( |-)?"
        r"((" + reg_ex_or(UNITS_ABBREVIATIONS.keys()) + r"|\s)+)"
    )

    Args:
        match: Matched text (e.g. 48-kHz, 3,000 fl oz, 3,450.6Ω)

    Returns:
        normalized: text string now normalized with standard words
    """
    normalized = match.group(0)
    prefix = "" if match.group(1) is None else match.group(1)
    prefix = PLUS_OR_MINUS_PREFIX[prefix] if prefix in PLUS_OR_MINUS_PREFIX else prefix

    replacement_text = ""
    prefix = match.group(1)
    value = _num2words("".join(match.group(2).split(",")), ignore_zeros=False)
    units = UNITS_ABBREVIATIONS.get(match.group(6))

    if prefix is not None:
        replacement_text = PLUS_OR_MINUS_PREFIX.get(prefix) + " "  # type: ignore
    replacement_text += value + " " + (units[0] if value == "one" else units[1])  # type: ignore

    normalized = normalized.replace(match.group(0), replacement_text, 1) + match.group(8)

    return normalized
    # normalized = text

    # plus_or_minus = {
    #     "+": "plus ",
    #     "-": "minus ",
    #     "±": "plus or minus ",
    #     "+/-": "plus or minus ",
    # }

    # # CREATE REGEX PATTERN:
    # pattern_abbr = "(\+|-|±|\+\/-)?(\d{1,3}(,\d{3})*(\.\d+)?)( |-)?(%s)\\b" % "|".join(
    #     UNITS_ABBREVIATIONS.keys()
    # )
    # # NOTE: These special characters (°, ″, ′, Ω, and superscripts)
    # pattern_abbr_alt = "(\+|-|±|\+\/-)?(\d{1,3}(,\d{3})*(\.\d+)?)( |-)?(%s)" % "|".join(
    #     UNITS_ABBREVIATIONS_ALT.keys()
    # )

    # regex_abbr = re.compile(pattern_abbr)
    # regex_abbr_alt = re.compile(pattern_abbr_alt)

    # matches = re.finditer(regex_abbr, text)
    # matches_alt = re.finditer(regex_abbr_alt, text)

    # for match in chain(matches, matches_alt):
    #     replacement_text = ""
    #     prefix = match.group(1)
    #     value = my_num2words("".join(match.group(2).split(",")))
    #     units = {**UNITS_ABBREVIATIONS, **UNITS_ABBREVIATIONS_ALT}.get(match.group(6))

    #     if prefix is not None:
    #         replacement_text = plus_or_minus.get(prefix)
    #     replacement_text += value + " " + (units[0] if value == "one" else units[1])

    #     normalized = normalized.replace(match.group(0), replacement_text, 1)

    # return normalized


def _normalize_fraction(match: typing.Match[str]) -> str:
    # TODO: Document
    # TODO: Use the signature `match: typing.Match[str]) -> str:`
    normalized = ""
    whole = match.group(1)
    numerator = int(match.group(2))
    denominator = int(match.group(3))

    if whole is not None:
        normalized += _num2words(whole, ignore_zeros=False) + " and "

    if numerator == 1:
        if denominator == 2:
            return normalized + "one half"
        if denominator == 4:
            return normalized + "one quarter"

    normalized += _num2words(numerator) + " " + _num2words(denominator, ordinal=True)

    if numerator > 1:
        normalized += "s"

    return normalized


def _normalize_generic_digit(match: typing.Match[str]) -> str:
    digits = match.group(0)
    return _num2words("".join(digits.split(",")), ignore_zeros=False)


def get_person_title(span):
    """
    Utilizes spaCy's NER to first identify `PERSON` entities. Then find potential title cases
    from TITLES_PERSON dict and return the token text which captures the abbreviated title to
    be replaced. Examples: "Dr. Ruth Gainor" "Rev. Christopher Smith"

    TODO: Add typing.

    Args:
        span: spaCy Span object (entity) that might have a title token preceding it

    Returns:
        prev_token: the title token preceding the given PERSON Span (entity)
    """

    if span.label_ == "PERSON" and span.start != 0:
        prev_token = span.doc[span.start - 1].text

        # Try to capture titles which spaCy recognizes as end of sentence (keep "."):
        if prev_token == ".":
            prev_token = span.doc[span.start - 2].text + "."
        if prev_token.strip(".") in TITLES_PERSON.keys():
            return prev_token


def get_person_title_sfx(span):
    """
    Utilizes spaCy's NER to first identify `PERSON` entities. Then find potential title cases
    from TITLES_PERSON_SFX dict and return the token text which captures the abbreviated title to
    be replaced. Examples: "William Simmons, Sr." "William Simmons Jr."

    Args:
        span: spaCy Span object (entity) that might have a title token succeeding it

    Returns:
        prev_token: the title token succeeding the given PERSON Span (entity)
    """

    if span.label_ == "PERSON" and span.end < len(span.doc):
        end_token = span.doc[span.end - 1].text

        # Try to capture titles which spaCy recognizes as end of sentence (keep "."):
        if end_token == ".":
            end_token = span.doc[span.end - 2].text + "."
        if end_token.strip(".") in TITLES_PERSON_SFX.keys():
            return end_token

        next_token = span.doc[span.end].text
        if next_token == ",":
            if len(span.doc) > (span.end + 1):
                next_token = span.doc[span.end].nbor().text
            else:
                logging.exception("Reached end of doc!\t%s" % span.doc)
        if next_token.strip(".") in TITLES_PERSON_SFX.keys():
            return next_token


# TODO: Will this affect other modules using spaCy since this is registered in the global scope?
# Register the Span extensions to spaCy
if not Span.has_extension("person_title"):
    Span.set_extension("person_title", getter=get_person_title)
    Span.set_extension("person_title_sfx", getter=get_person_title_sfx)


# TODO: Ensure the signature for `normalize_title_abbreviations` is consistent with the rest of
# the functions. Can this be broken up into a classification step and a normalization step?
def _normalize_title_abbreviations(text: str) -> str:
    """
    Utilizes new Span extensions get_person_title and get_person_title_sfx to find if entity might
    have a preceding or succeeding title token, respectively. Then translates the abbreviated title
    to plain English using the TITLES_PERSON and TITLES_PERSON_SFX dicts.

    Args:
        text: text string to be normalized

    Returns:
        normalized: text string now normalized with standard words
    """

    normalized = text
    doc = SPACY(text)
    for ent in doc.ents:
        title = ent._.person_title
        if title:
            replacement_title = TITLES_PERSON.get(title.strip("."))
            normalized = (
                normalized.replace(title, replacement_title, 1) if replacement_title else text
            )
        suffix = ent._.person_title_sfx
        if suffix:
            replacement_suffix = TITLES_PERSON_SFX.get(suffix.strip("."))
            normalized = (
                normalized.replace(suffix, replacement_suffix, 1) if replacement_suffix else text
            )

    return normalized


def _roman_to_int(input):
    """Borrowed from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s24.html"""
    """ Convert a Roman numeral to an integer. """

    input = input.upper()
    # TODO: Move constant to `non_standard_words.py`
    digits = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}
    sum = 0
    for i in range(len(input)):
        try:
            value = digits[input[i]]
            if i + 1 < len(input) and digits[input[i + 1]] > value:
                sum -= value
            else:
                sum += value

        except KeyError:
            logger.warn("input is not a valid Roman numeral: %s" % input)

    return sum


def _normalize_roman_numerals(text: str) -> str:
    """
    Utilizes spaCy's NER to first identify `PERSON`, `EVENT`, etc. entities. Then uses regex to
    find a Roman Numeral match in the final word of the entity. Then translates the numeral to
    plain English - either in the Ordinal (`PERSON`) or Cardinal (`EVENT`, etc.).

    TODO: More complex Roman Numeral identification. This method ignores contexts such as page
    numbers or legal references. SpaCy has mis-identified Roman Numeral strings as `ORG` entities,
    a misunderstanding we want to avoid. Consider adding `LAW` entity type.

    Args:
        text: text string to be normalized
        regex: regex pattern to search + find in text

    Returns:
        normalized: text string now normalized with standard words
    """
    regex = RegExPatterns.ROMAN_NUMERALS
    assert regex is not None, "Could not locate regex for identifying Roman Numerals."

    normalized = text
    for entity in SPACY(text).ents:
        type = entity.label_
        phrase = entity.text
        words = entity.text.strip().split(" ")
        # TODO: Use `Spacy.symbols` constants instead of their string representations.
        if type in ["PERSON", "EVENT", "FAC", "CARDINAL", "ORDINAL"] and len(words) > 1:
            last_word = str(words[-1])
            matches = re.finditer(regex, last_word)
            for match in matches:
                translated = ""
                word = match.group(0)
                if type in ["PERSON", "ORDINAL"]:
                    # Ordinal tranlsation
                    translated = "the " + _num2words(_roman_to_int(word), ordinal=True).capitalize()
                else:
                    # Cardinal translation
                    translated = _num2words(_roman_to_int(word)).capitalize()

                if translated != "":
                    replacement_text = phrase.replace(word, translated, 1)
                    normalized = normalized.replace(phrase, replacement_text, 1)

    return normalized


class RegExPatterns:
    # TODO: Document where these regexes orginated from.
    # TODO: Breakdown each regex for clarity.

    MONEY: typing.Final[typing.Pattern[str]] = re.compile(
        r"(" + reg_ex_or(CURRENCIES.keys()) + ")"  # GROUP 1: Currency prefix
        r"([0-9\,\.]+)"  # GROUP 2: Numerical value
        r"([kmbt]{0,2})"  # GROUP 3: Unit
        # GROUP 4 - 5 (Optional): Currency suffix
        r"(\b\s(" + reg_ex_or(MONEY_SUFFIX + LARGE_NUMBERS + LARGE_FICTIOUS_NUMBERS) + r"))?"
        r"\b",  # Word boundary
        flags=re.IGNORECASE,
    )
    ORDINALS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(([0-9]{0,3})([,]{0,1}[0-9])+)"  # GROUP 1: Numerical value
        r"(" + reg_ex_or(ORDINAL_SUFFIXES) + ")"  # GROUP 2: Ordinal suffix
    )
    TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"((\d{1,2}):([0-5]{1}[0-9]{1}))"  # GROUP 1 - 3: Hours and minutes
        r"((\s?(a|p)\.?m\.?)?)",  # GROUP 4 - 6 (Optional): Time period (e.g. AM, PM)
        flags=re.IGNORECASE,
    )
    ABBREVIATED_TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(\d{1,2})"  # GROUP 1: Hours
        r"(\s?(a|p)\.?m\.?)",  # GROUP 2 - 3: Time period (e.g. AM, PM, a.m., p.m., am, pm)
        flags=re.IGNORECASE,
    )
    PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: This Regex was adapted from here:
        # https://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
        r"(?:\+?(\d{1,2})[-. ]{0,1})?"  # GROUP 1 (Optional): The Country Code
        r"(\(?(\d{3})\)?[-. ]{0,1})?"  # GROUP 2 (Optional): The Area Code
        r"(\d{3})"  # GROUP 3: The Exchange number
        r"[-. ]{0,1}"
        r"(\d{4})"  # Group 4: The Subscriber Number
        r"\b"
    )
    TOLL_FREE_PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(?:\+?(\d{1,2})[-. (]{1})"  # GROUP 1: The Country Code
        # TODO: For toll free numbers, consider being more restrictive, like so:
        # https://stackoverflow.com/questions/34586409/regular-expression-for-us-toll-free-number
        # https://en.wikipedia.org/wiki/Toll-free_telephone_number
        r"\(?(\d{3})\)?"  # GROUP 2: The Area Code
        r"[-. )]{1,2}"
        r"([a-zA-Z0-9-\.]{1,10})"  # GROUP 3: The Line Number
        r"\b"
    )
    YEARS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"([0-9]{4})"  # GROUP 1: Year (or start year in a range of years)
        r"(-([0-9]{1,4}))?"  # GROUP 2 (Optional): The end year in a range of years
        r"\b"
    )
    DECADES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(\')?"  # GROUP 1 (Optional): Contraction (e.g. '90)
        r"([0-9]{1,3}0)"  # GROUP 2: Year
        r"s"
    )
    NUMBER_RANGES: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"(\d+)"  # GROUP 1: First number in the range
        r"[" + reg_ex_or(HYPHENS, "") + r"]"
        r"(\d+)"  # GROUP 2: Second number in the range
        r"\b"
    )
    PERCENTS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(\d+)"  # GROUP 1: First percentage in the range or the whole part of a number
        r"[\." + reg_ex_or(HYPHENS, "") + r"]*"
        r"(\d*)"  # GROUP 2: Second number in the range or the fractional part of a number
        r"%"
    )
    NUMBER_SIGNS: typing.Final[typing.Pattern[str]] = re.compile(r"#((\d+))")
    URLS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: Learn more about the autonomy of a URL here:
        # https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL
        r"(https?:\/\/)?"  # GROUP 1 (Optional): Protocol
        r"(www\.)?"  # GROUP 2 (Optional): World wide web subdomain
        r"([a-zA-Z]+\.)?"  # Group 3 (Optional): Other subdomain
        r"([-a-zA-Z0-9@:%_\+~#=]{2,256})"  # Group 4: Domain name
        r"(\.)"
        r"([a-z]{2,4})"  # Group 6: Domain extension
        r"\b"
        r"([-a-zA-Z0-9@:%_\+.~#?&//=]*)"  # Group 7: Port, path, parameters and anchor
        r"\b"
    )
    FRACTIONS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"(\d+ )?"  # GROUP 1 (Optional): Whole Number
        r"(\d+)"  # GROUP 2: Numerator
        r"\/"
        r"(\d+)"  # GROUP 3: Denominator
        r"\b"
    )
    GENERIC_DIGITS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"\d{1,3}"
        r"(,\d{3})*"  # GROUP 1: Thousands separator
        r"(\.\d+)?"  # GROUP 2 (Optional): Decimal
        r"\b"
    )
    ACRONYMS: typing.Final[typing.Pattern[str]] = re.compile(
        r"([A-Z]\.?){2,}"  # GROUP 1: Upper case acronym
        r"|"
        r"([a-z]\.){2,}"  # GROUP 2: Lower case acronym
    )
    MEASUREMENT_ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(" + reg_ex_or(PLUS_OR_MINUS_PREFIX.keys()) + r")?"  # GROUP 1 (Optional): Prefix symbol
        r"(\d{1,3}(,\d{3})*(\.\d+)?)"  # GROUP 2 - 4: Number
        r"( |-)?"  # GROUP 5 (Optional): Delimiter
        r"(" + reg_ex_or(UNITS_ABBREVIATIONS.keys()) + r")"  # GROUP 6: Unit
        r"\b"
    )
    ROMAN_NUMERALS: typing.Final[typing.Pattern[str]] = re.compile(
        r"^(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))+$"
    )


def normalize_text(text_in: str) -> str:
    """
    Takes in a text string and, in an intentional and controlled manner, verbalizes numerals and
    non-standard-words in plain English. The order of events is important. Normalizing generic
    digits before normalizing money cases specifically, for example, will yield incomplete and
    inaccurate results.
    The order is: MONEY > ORDINALS > TIMES > PHONE NUMBERS > ALTERNATIVE PHONE NUMBERS >
                  DECADES > YEARS > NUMBER RANGES > PERCENTS > NUMBER SIGNS > MEASUREMENT
                  ABBREVIATIONS > GENERIC DIGITS > ROMAN NUMERALS > ACRONYMS > TITLE ABBREVIATIONS
    TODO: Add ADDRESS ABBREVIATIONS (Dr., St., Blvd., Apt., states??, ...)
          Add GENERAL ABBREVIATIONS (etc., misc., appt., ...)
    Args:
        text: text string to be normalized
        regex: regex pattern to search + find in text
        translate: normalization function to translate non-standard words into standard words
        translate_args: optional arguments for translate function
    Returns:
        normalized: text string now normalized with standard words
    """

    normalized = _normalize_text_from_pattern(text_in, RegExPatterns.MONEY, _normalize_money)
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.ORDINALS, _normalize_ordinals
    )
    normalized = _normalize_text_from_pattern(
        normalized,
        RegExPatterns.TIMES,
        _normalize_times,
        o_clock="o'clock",
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.PHONE_NUMBERS, _normalize_phone_numbers
    )
    normalized = _normalize_text_from_pattern(
        normalized,
        RegExPatterns.TOLL_FREE_PHONE_NUMBERS,
        _normalize_phone_numbers,
    )
    normalized = _normalize_text_from_pattern(normalized, RegExPatterns.DECADES, _normalize_decades)
    normalized = _normalize_text_from_pattern(normalized, RegExPatterns.YEARS, _normalize_years)
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.NUMBER_RANGES, _normalize_number_ranges
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.PERCENTS, _normalize_percents
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.NUMBER_SIGNS, _normalize_number_signs
    )
    normalized = _normalize_text_from_pattern(normalized, RegExPatterns.URLS, _normalize_urls)
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.MEASUREMENT_ABBREVIATIONS, _normalize_measurement_abbreviations
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.FRACTIONS, _normalize_fraction
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.GENERIC_DIGITS, _normalize_generic_digit
    )
    normalized = _normalize_text_from_pattern(
        normalized, RegExPatterns.ACRONYMS, _normalize_acronyms
    )
    normalized = _normalize_title_abbreviations(normalized)
    normalized = _normalize_roman_numerals(normalized)
    # TODO: Add `normalize_general_abbreviations`.
    return normalized


__all__ = [
    "_normalize_text_from_pattern",
    "_normalize_money",
    "_normalize_ordinals",
    "_normalize_times",
    "_normalize_phone_numbers",
    "_normalize_years",
    "_normalize_decades",
    "_normalize_number_ranges",
    "_normalize_percents",
    "_normalize_number_signs",
    "_normalize_urls",
    "_normalize_acronyms",
    "_normalize_abbreviated_times",
    "_normalize_measurement_abbreviations",
    "_normalize_fraction",
    "_normalize_generic_digit",
    "_normalize_title_abbreviations",
    "_normalize_roman_numerals",
    "normalize_text",
    "RegExPatterns",
]
