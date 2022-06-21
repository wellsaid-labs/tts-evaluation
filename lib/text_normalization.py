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
import functools
import logging
import re
import typing

from num2words import num2words as __num2words

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
    UNITS_ABBREVIATIONS,
)
from lib.text import load_en_english

logger = logging.getLogger(__name__)


def _num2words(num: str, ignore_zeros: bool = True, **kwargs) -> str:
    """Normalize `num` into standard words.

    Args:
        ...
        ignore_zeros: If `False`, this verbalizes the leading and trailing zeros, as well.
    """
    num = num.replace(",", "")

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
    iter_ = sorted(vals, key=len, reverse=True)  # type: ignore
    return f"(?:{delimiter.join(re.escape(v) for v in iter_)})"


def _verbalize_money(
    currency: str, money: str, abbr: typing.Optional[str], trail: typing.Optional[str]
) -> str:
    """Verbalize a monetary value (e.g. $1.2 trillion, $5, £2.36, ¥1MM).

    NOTE: This considers currency, unit abbreviations, trailing 'big money' amounts, and
    dollar/cent splits.

    Args:
        currency (e.g. $, €, NZ$)
        money (e.g. 1.2, 15,000)
        abbr (e.g. M, K, k, BB)
        trail (e.g. trillion)
    """
    # CASE: Big Money ($1.2B, $1.2 billion, etc.)
    if abbr:
        abbr = abbr.upper()
        return " ".join([_num2words(money), MONEY_ABBREVIATIONS[abbr], CURRENCIES[currency][1]])
    elif trail:
        return _num2words(money) + trail + " " + CURRENCIES[currency][1]

    # CASE: Standard ($1.20, $4,000, etc.)
    tokens = money.split(".")
    assert len(tokens) == 2 or len(tokens) == 1, "Found strange number."
    dollars = tokens[0].replace(",", "")
    cents = int(tokens[1]) if len(tokens) > 1 else 0
    normalized = _num2words(dollars) + " " + CURRENCIES[currency][0 if int(dollars) == 1 else 1]
    if cents > 0:
        normalized += " and " + _num2words(str(cents))
        normalized += " " + CURRENCIES[currency][2 if cents == 1 else 3]
    return normalized


_DIGIT_PATTERN = re.compile(r"\D")


def _verbalize_ordinal(value: str) -> str:
    """Verbalize an ordinal (e.g. "1st", "2nd").

    Args:
        value (e.g. "123,456", "1", "2", "100.000")
    """
    return _num2words(_DIGIT_PATTERN.sub("", value), ordinal=True)


def _verbalize_time_period(suffix: typing.Optional[str]):
    """
    Args:
        suffix (e.g. "PM", "a.m.")
    """
    return "" if suffix is None else suffix.replace(".", "").strip().upper()


def _verbalize_time(hour: str, minute: str, suffix: typing.Optional[str], o_clock: str = "") -> str:
    """Verbalize a time (e.g. "10:04PM", "2:13 a.m.").

    Args:
        hours (e.g. "10")
        minute (e.g. "04")
        suffix (e.g. "PM", "a.m.")
        o_clock: Phrasing preference of 'o'clock' for on-the-hour times.
    """
    minute = o_clock if minute == "00" else _num2words(minute)
    suffix = _verbalize_time_period(suffix)
    return " ".join((s for s in (_num2words(hour), minute, suffix) if len(s) > 0))


def _verbalize_abbreviated_time(hour: str, suffix: str) -> str:
    """Verbalize a abbreviated time (e.g. "10PM", "10 p.m.").

    Args:
        hours (e.g. "10")
        suffix (e.g. "PM", "a.m.")
    """
    return f"{_num2words(hour)} {_verbalize_time_period(suffix)}"


_LETTER_PATTERN = re.compile(r"[A-z]")
_NUMBER_PATTERN = re.compile(r"[0-9\.\,]+")
_ALPHANUMERIC_PATTERN = re.compile(r"[0-9A-z]+")


def _get_digits(text):
    return _NUMBER_PATTERN.findall(text)


def _verbalize_phone_number(phone_number: str) -> str:
    """Verbalize a phone numbers (e.g. 1.800.573.1313, (123) 555-1212)

    TODO: Support "oh" instead of "zero" because it's more typical based on this source:
    https://www.woodwardenglish.com/lesson/telephone-numbers-in-english/

    Args:
        phone_number (e.g. 1.800.573.1313, (123) 555-1212)
    """
    digits = []
    n: str
    for n in _ALPHANUMERIC_PATTERN.findall(phone_number):
        if len(n) > 0:
            if n == "800":
                digits.append(_num2words(n))
            elif _LETTER_PATTERN.search(n) is not None:
                digits.append(n)
            else:
                digits.append(" ".join(list(map(_num2words, list(n)))))
    return ", ".join(digits)


num2year = lambda n: _num2words(n, lang="en", to="year")


def _verbalize_year(year: str) -> str:
    """Verbalize a year or year range.

    TODO: Make more robust. Consider spaCy's 'DATE' entity type in conjunction with the regex
    pattern. Don't need to find all years, just 3- and 4-digit years because these are pronounced
    differently from standard numerical pronunciations.

    Args:
        year (e.g. "1900", "1776", "1880-1990")
    """
    return " to ".join(list(map(num2year, _get_digits(year))))


def _verbalize_decade(year: str) -> str:
    """Verbalize decade (e.g. 1900s, 70s, '80s).

    Args:
        year (e.g. "1900", "70", "80")
    """
    return "tie".join(num2year(year).rsplit("ty", 1)) + "s"


def _verbalize_percent(percent: str) -> str:
    """Verbalize a percentage or a percent range.

    Args:
        percent (e.g. 75%, 15–25%)
    """
    numbers = _get_digits(percent)
    return " to ".join(list(map(_num2words, numbers))) + f" {SYMBOLS_VERBALIZED['%']}"


def _verbalize_number_sign(number: str) -> str:
    """Verbalize number signs.

    TODO: # as 'hash', # as 'hashtag', # as 'pound' ?

    Args:
        number (e.g. #1, #2-4)
    """
    numbers = _get_digits(number)
    verbalized = " through ".join(list(map(_num2words, numbers)))
    return f"number{'' if len(numbers) < 2 else 's'} {verbalized}"


def _verbalize_url(
    protocol: typing.Optional[str],
    www_subdomain: typing.Optional[str],
    subdomain: typing.Optional[str],
    domain_name: str,
    domain_extension: str,
    rest: str,
) -> str:
    """Verbalize a URL (e.g. https://www.wellsaidlabs.com, wellsaidlabs.com, rhyan@wellsaidlabs.com)

    Args:
        protocol (e.g. "http://", "https://")
        www_subdomain (e.g. "www.")
        subdomain (e.g. "help.")
        domain_name (e.g. "wellsaidlabs")
        domain_extension (e.g. "com", "org")
        rest (e.g. "/things-to-do", ":80/path/to/myfile.html?key1=value1#SomewhereInTheDocument")
    """
    return_ = ""
    # CASE: Email
    if protocol is None and www_subdomain is None and subdomain is None and "@" in domain_name:
        return_ += " at ".join(domain_name.split("@")) + " " + " ".join([".", domain_extension])
    else:  # CASE: Web Address
        # NOTE: Handle prefixes (e.g. "http://", "https://", "www.")
        prefixes = [protocol, www_subdomain]
        return_ += " " + " ".join(" ".join(list(m)) for m in prefixes if m is not None)
        suffixes = [subdomain, domain_name, ".", domain_extension, rest]
        return_ += " ".join(m for m in suffixes if m is not None)

    for s in SYMBOLS_VERBALIZED:
        return_ = return_.replace(s, f" {SYMBOLS_VERBALIZED[s]} ")

    digits = (c for w in return_.split(" ") for c in w if c.isdigit())
    for char in digits:
        return_ = return_.replace(char, f" {_num2words(char)} ")

    return re.sub(" +", " ", return_).strip()


def _verbalize_measurement_abbreviation(prefix: typing.Optional[str], value: str, unit: str) -> str:
    """Verbalize a measurement abbreviation (e.g. 48-kHz, 3,000 fl oz, 3,450.6Ω).

    Args:
        prefix (e.g. "+", "-", "±")
        value (e.g. "48")
        unit (e.g. "fl oz", "kHz")
    """
    prefix = "" if prefix is None else prefix
    prefix = PLUS_OR_MINUS_PREFIX[prefix] if prefix in PLUS_OR_MINUS_PREFIX else prefix
    value = _num2words(value, ignore_zeros=False)
    if unit in UNITS_ABBREVIATIONS:
        singular, plural = UNITS_ABBREVIATIONS[unit]
        unit = singular if value == "1" else plural  # TODO: Test
    return f"{prefix} {value} {unit}".strip()


# TODO: What about swedish numbers 4 294 967 295,000
# Or Spanish numbers: 4.294.967.295,000
# TODO: What about an ordinal with "."
# TODO: Test a number range with decimal or space


def _verbalize_fraction(
    whole: typing.Optional[str],
    numerator: str,
    denominator: str,
    special_cases: typing.Dict[typing.Tuple[bool, str, str], str] = {
        (False, "1", "2"): "one half",
        (False, "1", "4"): "one quarter",
        (True, "1", "2"): "half",
        (True, "1", "4"): "quarter",
    },
) -> str:
    """Verbalize a fraction (e.g. "59 1/2").

    TODO: Test if `my_num2words` recieves 00 for the whole number.
    TODO: Test fractions with special formatting
    TODO: Test negative numbers

    Args:
        whole (e.g. "59")
        numerator (e.g. "1")
        denominator (e.g. "2")
    """
    verbalized = "" if whole is None else f"{_num2words(whole, ignore_zeros=False)} and "
    key = (whole is not None, numerator, denominator)
    if key in special_cases:
        return f"{verbalized}{special_cases[key]}".strip()
    verbalized += f"{_num2words(numerator)} {_num2words(denominator, ordinal=True)}"
    verbalized += "s" if int(numerator) > 1 else ""
    return verbalized.strip()


def _verbalize_generic_number(
    num_range: str, connector: str = "to", special_cases: typing.List[str] = ["1-800", "50-50"]
) -> str:
    """Verbalize a range of numbers.

    TODO: Handle special cases like 7-11 and 9-11, they are commonly spoken without 'to'?
    TODO: Handle number ranges with spaces?

    Args:
        num_range (e.g. 10-15, 35-75, 125-300, 50-50)
        ...
    """
    connector = " " if num_range in special_cases else " %s " % connector
    partial = functools.partial(_num2words, ignore_zeros=False)
    return connector.join(list(map(partial, _get_digits(num_range))))


def _verbalize_acronym(acronym: str) -> str:
    """Verbalize a acronym.

    TODO: Consider how to to coordinate with Roman Numeral cases? Can you identify an ORG from a RN?

    Args:
        acronym (e.g. RSVP, CEO, NASA, ASAP, NBA, A.S.A.P., C.T.O.)
    """
    acronym = "".join(acronym.split("."))
    return ACRONYMS[acronym] if acronym in ACRONYMS else acronym


# TODO: What about abbreviations with multiple periods?
# TODO: What about abbreviations not in the table?


def _verbalize_title_abbreviation(abbr: str) -> str:
    """Verbalize a title abbreviation.

    Args:
        abbr (e.g. "Mr", "Ms", etc.)
    """
    key = (abbr[:-1] if abbr[-1] == "." else abbr).lower()
    return TITLES_PERSON[key] if key in TITLES_PERSON else abbr


_TIME_PERIOD = r"(\s?[ap]\.?m\.?)"  # Time period (e.g. AM, PM, a.m., p.m., am, pm)
_PHONE_NUMBER_DELIMITER = r"[-. ]{0,1}"
# TODO: Is this just a generic digit?
_DIGIT = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
_NUMBER_RANGE_SUFFIX = rf"(?:\s?[{reg_ex_or(HYPHENS, '')}]\s?{_DIGIT})"
_MAYBE_NUMBER_RANGE = rf"(?:{_DIGIT}{_NUMBER_RANGE_SUFFIX}?)"
_NUMBER_RANGE = rf"(?:{_DIGIT}{_NUMBER_RANGE_SUFFIX})"

# TODO: Handle the various thousand seperators, like dots or spaces.


class RegExPatterns:

    MONEY: typing.Final[typing.Pattern[str]] = re.compile(
        rf"({reg_ex_or(CURRENCIES.keys())})"  # GROUP 1: Currency prefix
        rf"({_DIGIT})"  # GROUP 2: Numerical value
        r"([kmbt]{0,2})"  # GROUP 3: Unit
        # GROUP 4 (Optional): Currency suffix
        rf"(\b\s(?:{reg_ex_or(MONEY_SUFFIX + LARGE_NUMBERS + LARGE_FICTIOUS_NUMBERS)}))?"
        r"\b",  # Word boundary
        flags=re.IGNORECASE,
    )
    ORDINALS: typing.Final[typing.Pattern[str]] = re.compile(
        r"([0-9]{0,3}(?:[,]{0,1}[0-9])+)"  # GROUP 1: Numerical value
        rf"{reg_ex_or(ORDINAL_SUFFIXES)}"  # Ordinal suffix
    )
    TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(\d{1,2})"  # GROUP 1: Hours
        r":"
        r"([0-5]{1}[0-9]{1})"  # GROUP 2: Minutes
        rf"{_TIME_PERIOD}?",  # GROUP 3 (Optional): Time period
        flags=re.IGNORECASE,
    )
    ABBREVIATED_TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(\d{1,2})" rf"{_TIME_PERIOD}",  # GROUP 1: Hours, GROUP 2: Time period
        flags=re.IGNORECASE,
    )
    PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: This Regex was adapted from here:
        # https://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
        r"((?:\+?\d{1,2}"
        rf"{_PHONE_NUMBER_DELIMITER})?"  # (Optional) The Country Code
        r"(?:\(?\d{3}\)?"
        rf"{_PHONE_NUMBER_DELIMITER})?"  # (Optional) The Area Code
        r"\d{3}"  # The Exchange Number
        rf"{_PHONE_NUMBER_DELIMITER}"
        r"\d{4})"  # The Subscriber Number
        r"\b"
    )
    TOLL_FREE_PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        r"((?:\+?\d{1,2}[-. (]{1})"  # The Country Code
        # TODO: For toll free numbers, consider being more restrictive, like so:
        # https://stackoverflow.com/questions/34586409/regular-expression-for-us-toll-free-number
        # https://en.wikipedia.org/wiki/Toll-free_telephone_number
        r"\(?\d{3}\)?"  # (Optional) The Area Code
        r"[-. )]{1,2}"
        r"[a-zA-Z0-9-\.]{1,10})"  # The Line Number
        r"\b"
    )
    YEARS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"([0-9]{4}"  # Year (or start year in a range of years)
        r"(?:-[0-9]{1,4})?)"  # The end year in a range of years
        r"\b"
    )
    # fmt: off
    DECADES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(?:\')?"  # (Optional) Contraction (e.g. '90)
        r"([0-9]{1,3}0)"  # GROUP 1: Year
        r"s",
    )
    # fmt: on
    PERCENTS: typing.Final[typing.Pattern[str]] = re.compile(rf"\b({_MAYBE_NUMBER_RANGE}%)")
    NUMBER_SIGNS: typing.Final[typing.Pattern[str]] = re.compile(rf"(#{_MAYBE_NUMBER_RANGE}\b)")
    NUMBER_RANGE: typing.Final[typing.Pattern[str]] = re.compile(rf"(\b{_NUMBER_RANGE}\b)")
    URLS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: Learn more about the autonomy of a URL here:
        # https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL
        r"(https?:\/\/)?"  # GROUP 1 (Optional): Protocol
        r"(www\.)?"  # GROUP 2 (Optional): World wide web subdomain
        r"([a-zA-Z]+\.)?"  # Group 3 (Optional): Other subdomain
        r"([-a-zA-Z0-9@:%_\+~#=]{2,256})"  # Group 4: Domain name
        r"\."
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
    GENERIC_DIGIT: typing.Final[typing.Pattern[str]] = re.compile(rf"({_DIGIT})")
    # fmt: off
    ACRONYMS: typing.Final[typing.Pattern[str]] = re.compile(
        r"((?:[A-Z]\.?){2,}"  # Upper case acronym
        r"|"
        r"(?:[a-z]\.){2,})"  # Lower case acronym
    )
    # fmt: on
    MEASUREMENT_ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        r"(" + reg_ex_or(PLUS_OR_MINUS_PREFIX.keys()) + r")?"  # GROUP 1 (Optional): Prefix symbol
        rf"({_DIGIT})"  # GROUP 2: Number
        r"[ -]{0,}"  # Delimiter
        # GROUP 3: Unit
        r"(" + reg_ex_or(UNITS_ABBREVIATIONS.keys()) + r")"
    )
    TITLE_ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        rf"\b({reg_ex_or(TITLES_PERSON.keys())}(?:\.|\b))",
        flags=re.IGNORECASE,
    )


def _norm(
    text: str,
    pattern: typing.Pattern,
    verbalize: typing.Callable[..., str],
    space_out: bool = False,
    **kw,
):
    """Helper function for verbalizing `text` by matching and verbalizing non-standard words."""
    for match in reversed(list(pattern.finditer(text))):
        verbalized = verbalize(*match.groups(), **kw)
        left, right = text[: match.start()], text[match.end() :]
        if space_out:
            left = left + " " if len(left) > 0 and left[-1] != " " else left
            right = " " + right if len(right) > 0 and right[0] != " " else right
        text = f"{left}{verbalized}{right}"
    return text


def verbalize_text(text: str) -> str:
    """Takes in a text string and, in an intentional and controlled manner, verbalizes numerals and
    non-standard-words in plain English. The order of events is important. Normalizing generic
    digits before normalizing money cases specifically, for example, will yield incomplete and
    inaccurate results.

    TODO: Add ADDRESS ABBREVIATIONS (Dr., St., Blvd., Apt., states??, ...)
          Add GENERAL ABBREVIATIONS (etc., misc., appt., ...)
    """
    nlp = load_en_english()
    nlp.add_pipe("sentencizer")
    sents = []
    for span in nlp(text).sents:
        sent = span.text
        sent = _norm(sent, RegExPatterns.MONEY, _verbalize_money)
        sent = _norm(sent, RegExPatterns.ORDINALS, _verbalize_ordinal)
        sent = _norm(sent, RegExPatterns.TIMES, _verbalize_time, o_clock="o'clock")
        sent = _norm(sent, RegExPatterns.PHONE_NUMBERS, _verbalize_phone_number)
        sent = _norm(sent, RegExPatterns.TOLL_FREE_PHONE_NUMBERS, _verbalize_phone_number)
        sent = _norm(sent, RegExPatterns.DECADES, _verbalize_decade)
        sent = _norm(sent, RegExPatterns.YEARS, _verbalize_year)
        sent = _norm(sent, RegExPatterns.PERCENTS, _verbalize_percent)
        sent = _norm(sent, RegExPatterns.NUMBER_SIGNS, _verbalize_number_sign)
        sent = _norm(sent, RegExPatterns.NUMBER_RANGE, _verbalize_generic_number)
        sent = _norm(sent, RegExPatterns.URLS, _verbalize_url)
        sent = _norm(
            sent, RegExPatterns.MEASUREMENT_ABBREVIATIONS, _verbalize_measurement_abbreviation
        )
        sent = _norm(sent, RegExPatterns.ABBREVIATED_TIMES, _verbalize_abbreviated_time)
        sent = _norm(sent, RegExPatterns.FRACTIONS, _verbalize_fraction)
        sent = _norm(sent, RegExPatterns.GENERIC_DIGIT, _verbalize_generic_number, space_out=True)
        sent = _norm(sent, RegExPatterns.ACRONYMS, _verbalize_acronym)
        sent = _norm(sent, RegExPatterns.TITLE_ABBREVIATIONS, _verbalize_title_abbreviation)
        # NOTE: A period at the end of a sentence might get eaten.
        sent = sent + "." if span.text[-1] == "." and sent[-1] != "." else sent
        sents.extend([sent, span[-1].whitespace_])
    return "".join(sents)
