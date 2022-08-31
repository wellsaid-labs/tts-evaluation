"""
TODO: Utilize spaCy 3.0 for entity identification [inconsistent, unreliable in 2.0 so far]
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

from num2words import num2words

from lib.text import load_en_english
from lib.text.non_standard_words import (
    ACRONYMS,
    CURRENCIES,
    GENERAL_ABBREVIATIONS,
    HYPHENS,
    LARGE_FICTIOUS_NUMBERS,
    LARGE_NUMBERS,
    MONEY_ABBREVIATIONS,
    MONEY_SUFFIX,
    ORDINAL_SUFFIXES,
    PLUS_OR_MINUS_PREFIX,
    SYMBOLS_VERBALIZED,
    TIME_ZONES,
    UNITS_ABBREVIATIONS,
    VERBALIZED_SYMBOLS_VERBALIZED,
    WEB_INITIALISMS,
)

logger = logging.getLogger(__name__)


def _num2words(num: str, ignore_zeros: bool = True, **kwargs) -> str:
    """Normalize `num` into standard words.

    Args:
        ...
        ignore_zeros: If `False`, this verbalizes the leading and trailing zeros, as well.
    """
    num = num.replace(",", "")

    if ignore_zeros:
        return num2words(num, **kwargs)

    lstripped = num.lstrip("0")  # NOTE: Handle leading zeros
    out = ["zero" for _ in range(len(num) - len(lstripped))]
    if "." in num:
        stripped = lstripped.rstrip("0")  # NOTE: Handle trailing zeros
        zeros = ["zero" for _ in range(len(lstripped) - len(stripped))]
        if stripped != ".":
            # NOTE: num2words will still verbalize the leading zero, even when not given (e.g. ".2")
            out.append(num2words(stripped, **kwargs).replace("zero point", "point"))
        if stripped[-1] == ".":
            out.append("point")
        out.extend(zeros)
    elif len(lstripped) > 0:
        out.append(num2words(lstripped, **kwargs))

    return " ".join(out)


_WORD_CHARACTER = re.compile(r"^\w$")

_get_bound = lambda t: r"\B" if _WORD_CHARACTER.match(t) is None else r"\b"


def _add_bounds(vals: typing.Iterable[str], left: bool = False, right: bool = False):
    """Add the correct boundary characters to every sequence in `vals`."""
    vals = [_get_bound(v[0]) + v for v in vals] if left else vals
    return [v + _get_bound(v[-1]) for v in vals] if right else vals


def _reg_ex_or(vals: typing.Iterable[str], delimiter: str = r"|", **kwargs) -> str:
    iter_ = sorted(vals, key=len, reverse=True)  # type: ignore
    iter_ = (re.escape(v) for v in iter_)
    iter_ = _add_bounds(iter_, **kwargs)
    return f"(?:{delimiter.join(iter_)})"


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
        money = _num2words(money, ignore_zeros=False)
        return " ".join([money, MONEY_ABBREVIATIONS[abbr], CURRENCIES[currency][1]])
    elif trail:
        return _num2words(money, ignore_zeros=False) + trail + " " + CURRENCIES[currency][1]

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


def _verbalize_money__reversed(
    money: str, abbr: typing.Optional[str], trail: typing.Optional[str], currency: str
) -> str:
    """Verbalize a monetary value with the currency trailing (e.g. 24$, 4.2MM¥).

    Args:
        money (e.g. 1.2, 15,000)
        abbr (e.g. M, K, k, BB)
        trail (e.g. trillion)
        currency (e.g. $, €, NZ$)
    """
    return _verbalize_money(currency, money, abbr, trail)


_NON_DIGIT_PATTERN = re.compile(r"\D")


def _verbalize_ordinal(value: str) -> str:
    """Verbalize an ordinal (e.g. "1st", "2nd").

    Args:
        value (e.g. "123,456", "1", "2", "100,000")
    """
    return _num2words(_NON_DIGIT_PATTERN.sub("", value), ordinal=True)


def _verbalize_time_period(period: typing.Optional[str]):
    """
    Args:
        period (e.g. "PM", "a.m.")
    """
    return "" if period is None else period.replace(".", "").strip().upper()


def _verbalize_time(
    hour: str,
    minute: str,
    period: typing.Optional[str],
    zone: typing.Optional[str],
    o_clock: str = "oh clock",
) -> str:
    """Verbalize a time (e.g. "10:04PM", "2:13 a.m.", "11:15 PM (PST)").

    Args:
        hours (e.g. "10")
        minute (e.g. "04")
        period (e.g. "PM", "a.m.")
        zone (e.g. "PST")
        o_clock: Phrasing preference of 'o'clock' for on-the-hour times.
    """
    minute = o_clock if minute == "00" else _num2words(minute)
    period = _verbalize_time_period(period)
    zone = TIME_ZONES[zone.lower()] if zone else ""
    return " ".join((s for s in (_num2words(hour), minute, period, zone) if len(s) > 0))


def _verbalize_abbreviated_time(hour: str, period: str, zone: typing.Optional[str]) -> str:
    """Verbalize a abbreviated time (e.g. "10PM", "10 p.m.", "10PM PST").

    Args:
        hours (e.g. "10")
        period (e.g. "PM", "a.m.")
    """
    zone = " " + TIME_ZONES[zone.lower()] if zone else ""
    return f"{_num2words(hour)} {_verbalize_time_period(period)}{zone}"


_LETTER_PATTERN = re.compile(r"[A-Za-z]")
_NUMBER_PATTERN = re.compile(r"[0-9\.\,]+")
_ALPHANUMERIC_PATTERN = re.compile(r"[0-9A-Za-z]+")


def _get_digits(text):
    return _NUMBER_PATTERN.findall(text)


def _verbalize_phone_number(phone_number: str) -> str:
    """Verbalize a phone number (e.g. 1.800.573.1313, (123) 555-1212)

    TODO: Support "oh" instead of "zero" because it's more typical based on this source:
    https://www.woodwardenglish.com/lesson/telephone-numbers-in-english/

    Args:
        phone_number (e.g. 1.800.573.1313, (123) 555-1212)
    """
    digits = []
    n: str
    for n in _ALPHANUMERIC_PATTERN.findall(phone_number):
        if n == "800":
            digits.append(_num2words(n))
        elif _LETTER_PATTERN.search(n) is not None:
            digits.append(n)
        else:
            digits.append(" ".join(list(map(_num2words, list(n)))))
    return ", ".join(digits)


def _verbalize_alternative_phone_number(call_verb: str, phone_number: str) -> str:
    """Verbalize a phone number indicated by 'call' or 'dial' (e.g. calling 911)

    Args:
        call_verb (e.g. Call, dialing)
        phone_number (e.g. 911, 5-4332)
    """
    digits = []
    n: str
    for n in _ALPHANUMERIC_PATTERN.findall(phone_number):
        digits.append(" ".join(list(map(_num2words, list(n)))))
    return call_verb + ", ".join(digits)


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


_WHITE_SPACES = re.compile(r"\s+")


def _verbalize_url(
    protocol: typing.Optional[str],
    www_subdomain: typing.Optional[str],
    subdomain: typing.Optional[str],
    domain_name: str,
    domain_extension: str,
    remainder: str,
) -> str:
    """Verbalize a URL (e.g. https://www.wellsaidlabs.com, wellsaidlabs.com, rhyan@wellsaidlabs.com)

    Args:
        protocol (e.g. "http://", "https://")
        www_subdomain (e.g. "www.")
        subdomain (e.g. "help.")
        domain_name (e.g. "wellsaidlabs")
        domain_extension (e.g. "com", "org")
        remainder (e.g. "/things-to-do", ":80/path/to/myfile.html?key1=value1#SomewhereInTheFile")
    """
    return_ = ""
    # CASE: Email
    if protocol is None and www_subdomain is None and subdomain is None and "@" in domain_name:
        return_ += " at ".join(domain_name.split("@")) + " " + " ".join([".", domain_extension])
    else:  # CASE: Web Address
        # NOTE: Handle prefixes (e.g. "http://", "https://", "www.")
        prefixes = [protocol, www_subdomain]
        return_ += " " + " ".join(" ".join(list(m)) for m in prefixes if m is not None)
        suffixes = [subdomain, domain_name, ".", domain_extension, remainder]
        return_ += " ".join(m for m in suffixes if m is not None)

    for s in SYMBOLS_VERBALIZED:
        return_ = return_.replace(s, f" {SYMBOLS_VERBALIZED[s]} ")

    digits = (c for w in return_.split(" ") for c in w if c.isdigit())
    for char in digits:
        return_ = return_.replace(char, f" {_num2words(char)} ")

    # Pronounce "HTTP", "HTTPS", "WWW"
    for s in WEB_INITIALISMS.keys():
        return_ = return_.replace(s, WEB_INITIALISMS[s])

    return _WHITE_SPACES.sub(" ", return_).strip()


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
        unit = singular if value == "one" else plural
    return f"{prefix} {value} {unit}".strip()


def _verbalize_fraction(
    prefix: typing.Optional[str],
    whole: typing.Optional[str],
    numerator: str,
    denominator: str,
    special_cases: typing.Dict[typing.Tuple[bool, str, str], str] = {
        (False, "1", "2"): "one half",
        (False, "1", "4"): "one quarter",
        (True, "1", "2"): "a half",
        (True, "1", "4"): "a quarter",
    },
) -> str:
    """Verbalize a fraction (e.g. "59 1/2").

    Args:
        minus (e.g. "-" in "-1/2")
        whole (e.g. "59")
        numerator (e.g. "1")
        denominator (e.g. "2")
    """
    prefix = "" if prefix is None else prefix
    prefix = PLUS_OR_MINUS_PREFIX[prefix] + " " if prefix in PLUS_OR_MINUS_PREFIX else prefix
    verbalized = "" if whole is None else f"{_num2words(whole)} and "
    key = (whole is not None, numerator, denominator)
    if key in special_cases:
        return f"{prefix}{verbalized}{special_cases[key]}".strip()
    verbalized += f"{_num2words(numerator)} {_num2words(denominator, ordinal=True)}"
    verbalized += "s" if int(numerator) > 1 else ""
    return f"{prefix}{verbalized.strip()}"


def _verbalize_acronym(acronym: str) -> str:
    """Verbalize a acronym.

    TODO: Consider speaker dialect for "Z" as "zed" case

    Args:
        acronym (e.g. "RSVP", "CEO", "NASA", "ASAP", "NBA", "A.S.A.P.", "C.T.O.")
    """
    acronym = "".join(acronym.split("."))
    return ACRONYMS[acronym] if acronym in ACRONYMS else acronym


def _verbalize_abbreviation(abbr: str) -> str:
    """Verbalize an abbreviation.

    Args:
        abbr (e.g. "Mr", "Ms", etc.)
    """
    abbr = abbr.strip()
    key = (abbr[:-1] if abbr[-1] == "." else abbr).lower()
    return GENERAL_ABBREVIATIONS[key] if key in GENERAL_ABBREVIATIONS else abbr


def _verbalize_generic_number(
    prefix: typing.Optional[str],
    numbers: str,
    connector: str = "to",
    special_cases: typing.List[str] = ["50-50"],
) -> str:
    """Verbalize a generic number or range of numbers.

    TODO: Handle special cases like 7-11 and 9-11, they are commonly spoken without 'to'?

    Args:
        numbers (e.g. "1", "10-15", "35-75", "125-300", "50-50", "-9.2")
        ...
    """
    prefix = "" if prefix is None else prefix
    prefix = PLUS_OR_MINUS_PREFIX[prefix] + " " if prefix in PLUS_OR_MINUS_PREFIX else prefix
    connector = " " if numbers in special_cases else " %s " % connector
    partial = functools.partial(_num2words, ignore_zeros=False)
    return f"{prefix}{connector.join(list(map(partial, _get_digits(numbers))))}"


def _verbalize_generic_symbol(symbols: str) -> str:
    """Verbalize a generic symbol.

    TODO: Handle special cases like "and/or", "if/when", "if/then", "either/or"?

    Args:
        symbols (e.g. "@", "&")
    """
    return " ".join(VERBALIZED_SYMBOLS_VERBALIZED[s] for s in list(symbols))


_TIME_PERIOD = r"( ?[ap]\.?m\b(?:\.\B)?)"  # Time period (e.g. AM, PM, a.m., p.m., am, pm)
_TIME_ZONE = rf"(?: \(?({_reg_ex_or(TIME_ZONES)})(?:\)\B)?|\b)"  # Time period (PST, GST)
# TODO: Handle the various thousand separators, like dots or spaces.
_DIGIT = r"\d(?:\d|\,\d)*(?:\.[\d]+)?"
_NUMBER_RANGE_SUFFIX = rf"(?: ?[{_reg_ex_or(HYPHENS, '')}] ?{_DIGIT})"
_MAYBE_NUMBER_RANGE = rf"(?:{_DIGIT}{_NUMBER_RANGE_SUFFIX}?)"
_NUMBER_RANGE = rf"(?:{_DIGIT}{_NUMBER_RANGE_SUFFIX})"
_COUNTRY_CODE = r"\+?\b\d{1,2}[-. ]*"
_AREA_CODE = r"\(?\b\d{3}\b\)?[-. ]*"
_PHONE_DELIMITER = r"[-. ]{1}"


class RegExPatterns:
    # NOTE: Each RegEx ends with a `\b` or `\B` to ensure there is no partial matches.

    MONEY: typing.Final[typing.Pattern[str]] = re.compile(
        rf"({_reg_ex_or(CURRENCIES.keys(), left=True)})"  # GROUP 1: Currency prefix
        rf"({_DIGIT})"  # GROUP 2: Numerical value
        r"([kmbt]{0,2})"  # GROUP 3: Unit
        # GROUP 4 (Optional): Currency suffix
        rf"(\b\s(?:{_reg_ex_or(MONEY_SUFFIX + LARGE_NUMBERS + LARGE_FICTIOUS_NUMBERS)}))?"
        r"\b",  # Word boundary
        flags=re.IGNORECASE,
    )
    MONEY_REVERSED: typing.Final[typing.Pattern[str]] = re.compile(
        rf"({_DIGIT})" r"([kmbt]{0,2})"  # GROUP 1: Numerical value  # GROUP 2: Unit
        # GROUP 3 (Optional): Currency suffix
        rf"(\b\s(?:{_reg_ex_or(MONEY_SUFFIX + LARGE_NUMBERS + LARGE_FICTIOUS_NUMBERS)}))?"
        rf" ?({_reg_ex_or(CURRENCIES.keys(), right=True)})",  # GROUP 4: Currency, optional space
        flags=re.IGNORECASE,
    )
    ORDINALS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b([0-9]{0,3}(?:[,]{0,1}[0-9])+)"  # GROUP 1: Numerical value
        rf"{_reg_ex_or(ORDINAL_SUFFIXES)}\b"  # Ordinal suffix
    )
    TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b(\d{1,2})"  # GROUP 1: Hours
        r":"
        r"([0-5]{1}[0-9]{1})"  # GROUP 2: Minutes
        rf"(?:{_TIME_PERIOD}|\b)"  # GROUP 3 (Optional): Time period
        rf"{_TIME_ZONE}?",  # GROUP 4 (Optional): Time zone
        flags=re.IGNORECASE,
    )
    ABBREVIATED_TIMES: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b(\d{1,2})"  # GROUP 1: Hours
        rf"{_TIME_PERIOD}"  # GROUP 2: Time period
        rf"{_TIME_ZONE}?",  # GROUP 3 (Optional): Time zone
        flags=re.IGNORECASE,
    )
    # TODO: A phone number could have up to 15 characters
    # https://stackoverflow.com/questions/6478875/regular-expression-matching-e-164-formatted-phone-numbers
    PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: This Regex was adapted from here:
        # https://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
        r"("
        rf"(?:(?:{_COUNTRY_CODE})?(?:{_AREA_CODE}))?"
        r"\b\d{3}"  # The Exchange Number
        rf"{_PHONE_DELIMITER}"
        r"\d{4}"  # The Subscriber Number
        r")"
        r"\b"
    )
    TOLL_FREE_PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        r"("
        rf"{_COUNTRY_CODE}"  # The Country Code
        # TODO: For toll free numbers, consider being more restrictive, like so:
        # https://stackoverflow.com/questions/34586409/regular-expression-for-us-toll-free-number
        # https://en.wikipedia.org/wiki/Toll-free_telephone_number
        rf"{_AREA_CODE}"  # The Area Code
        r"[A-Za-z\d][A-Za-z\d\-\.]{3,9}"
        r")"  # The Line Number
        r"\b"
    )
    ALTERNATIVE_PHONE_NUMBERS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"("
        r"(?:call|dial)"  # Verb
        r"(?:ing)?"  # (Optional) Verb ending
        r"(?:\s)"
        r")"  # Call verb
        r"([0-9-\.]{1,10})"  # Phone Number
        r"\b",
        flags=re.IGNORECASE,
    )
    YEARS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b"
        r"([0-9]{4}"  # Year (or start year in a range of years)
        rf"(?:[{_reg_ex_or(HYPHENS, '')}]"
        r"[0-9]{1,4})?)"  # The end year in a range of years
        r"\b"
    )
    # fmt: off
    DECADES: typing.Final[typing.Pattern[str]] = re.compile(
        r"(?:\B\')?"  # (Optional) Contraction (e.g. '90)
        r"\b([0-9]{1,3}0)"  # GROUP 1: Year
        r"s\b",
    )
    # fmt: on
    PERCENTS: typing.Final[typing.Pattern[str]] = re.compile(rf"\b({_MAYBE_NUMBER_RANGE}%\B)")
    NUMBER_SIGNS: typing.Final[typing.Pattern[str]] = re.compile(rf"(\B#{_MAYBE_NUMBER_RANGE}\b)")
    NUMBER_RANGE: typing.Final[typing.Pattern[str]] = re.compile(
        rf"(\B{_reg_ex_or(PLUS_OR_MINUS_PREFIX.keys())})?"  # GROUP 1 (Optional): Prefix symbol
        rf"(\b{_NUMBER_RANGE}\b)"  # GROUP 2: Number Range
    )
    URLS: typing.Final[typing.Pattern[str]] = re.compile(
        # NOTE: Learn more about the autonomy of a URL here:
        # https://developer.mozilla.org/en-US/docs/Learn/Common_questions/What_is_a_URL
        r"\b"
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
        rf"(\B{_reg_ex_or(PLUS_OR_MINUS_PREFIX.keys())})?"  # GROUP 1 (Optional): Prefix symbol
        r"\b(\d+ )?"  # GROUP 2 (Optional): Whole Number
        r"(\d+)"  # GROUP 3: Numerator
        r"\/"
        r"(\d+)"  # GROUP 4: Denominator
        r"\b"
    )
    # fmt: off
    ACRONYMS: typing.Final[typing.Pattern[str]] = re.compile(
        r"\b((?:[A-Z]){2,}\b"  # Upper case acronym
        r"|"
        r"(?:[A-Za-z]\.){2,}\B)"  # Lower case acronym
    )
    # fmt: on
    MEASUREMENT_ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        rf"(\B{_reg_ex_or(PLUS_OR_MINUS_PREFIX.keys())})?"  # GROUP 1 (Optional): Prefix symbol
        rf"(\b{_DIGIT})"  # GROUP 2: Number
        r"[ -]{0,}"  # Delimiter
        # GROUP 3: Unit
        rf"({_reg_ex_or(UNITS_ABBREVIATIONS.keys(), right=True)})"
    )
    ABBREVIATIONS: typing.Final[typing.Pattern[str]] = re.compile(
        rf"\b({_reg_ex_or(GENERAL_ABBREVIATIONS.keys())}(?:\.))\B"
    )
    GENERIC_DIGIT: typing.Final[typing.Pattern[str]] = re.compile(
        rf"(\B{_reg_ex_or(PLUS_OR_MINUS_PREFIX.keys())})?"  # GROUP 1 (Optional): Prefix symbol
        rf"({_DIGIT})"  # GROUP 2: Number
    )
    ISOLATED_GENERIC_DIGIT: typing.Final[typing.Pattern[str]] = re.compile(
        rf"(\B{_reg_ex_or(PLUS_OR_MINUS_PREFIX.keys())})?"  # GROUP 1 (Optional): Prefix symbol
        rf"\b({_DIGIT})\b"  # GROUP 2: Number
    )
    GENERIC_VERBALIZED_SYMBOL: typing.Final[typing.Pattern[str]] = re.compile(
        rf"({_reg_ex_or(VERBALIZED_SYMBOLS_VERBALIZED.keys())})"
    )


def _apply(
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
            # TODO: `isalnum` in this case is playing the role of determining if a character is
            # voiced or not. It allows us to seperate out voiced characters. With that in mind,
            # should `lib.text.is_voiced` be used instead. Furthermore, since this verbalization, is
            # English centric, should it be moved into the `run` folder?
            left = left + " " if len(left) > 0 and left[-1].isalnum() else left
            right = " " + right if len(right) > 0 and right[0].isalnum() else right
        text = f"{left}{verbalized}{right}"
    return text


def ensure_period(sent: str, span_text: str) -> str:
    """Add a period to `span_text` if it doesn't have one and `sent` does.

    TODO: Incorperate spaCy more deeply so we know the intent of a period.
    """
    return sent + "." if span_text[-1] == "." and sent[-1] != "." else sent


def verbalize_text(text: str) -> str:
    """Takes in a text string and, in an intentional and controlled manner, verbalizes numerals and
    non-standard-words in plain English. The order of events is important. Normalizing generic
    digits before normalizing money cases specifically, for example, will yield incomplete and
    inaccurate results.
    """
    nlp = load_en_english()
    nlp.add_pipe("sentencizer")
    sents = []
    nlp_sents = nlp(text).sents
    for span in nlp_sents:
        sent = span.text
        sent = _apply(sent, RegExPatterns.MONEY, _verbalize_money)
        sent = _apply(sent, RegExPatterns.MONEY_REVERSED, _verbalize_money__reversed)
        sent = _apply(sent, RegExPatterns.ORDINALS, _verbalize_ordinal)
        sent = _apply(sent, RegExPatterns.TIMES, _verbalize_time)
        sent = ensure_period(sent, span.text)
        sent = _apply(sent, RegExPatterns.PHONE_NUMBERS, _verbalize_phone_number)
        sent = _apply(sent, RegExPatterns.TOLL_FREE_PHONE_NUMBERS, _verbalize_phone_number)
        sent = _apply(
            sent, RegExPatterns.ALTERNATIVE_PHONE_NUMBERS, _verbalize_alternative_phone_number
        )
        sent = _apply(sent, RegExPatterns.DECADES, _verbalize_decade)
        sent = _apply(sent, RegExPatterns.YEARS, _verbalize_year)
        sent = _apply(sent, RegExPatterns.PERCENTS, _verbalize_percent)
        sent = _apply(sent, RegExPatterns.NUMBER_SIGNS, _verbalize_number_sign)
        sent = _apply(sent, RegExPatterns.NUMBER_RANGE, _verbalize_generic_number)
        sent = _apply(sent, RegExPatterns.URLS, _verbalize_url)
        sent = _apply(
            sent, RegExPatterns.MEASUREMENT_ABBREVIATIONS, _verbalize_measurement_abbreviation
        )
        sent = _apply(sent, RegExPatterns.ABBREVIATED_TIMES, _verbalize_abbreviated_time)
        sent = ensure_period(sent, span.text)
        sent = _apply(sent, RegExPatterns.FRACTIONS, _verbalize_fraction)
        sent = _apply(sent, RegExPatterns.ACRONYMS, _verbalize_acronym)
        sent = ensure_period(sent, span.text)
        sent = _apply(sent, RegExPatterns.ABBREVIATIONS, _verbalize_abbreviation)
        sent = _apply(sent, RegExPatterns.GENERIC_DIGIT, _verbalize_generic_number, True)
        sent = _apply(sent, RegExPatterns.ISOLATED_GENERIC_DIGIT, _verbalize_generic_number, True)
        sent = _apply(
            sent, RegExPatterns.GENERIC_VERBALIZED_SYMBOL, _verbalize_generic_symbol, True
        )

        sents.extend([sent, span[-1].whitespace_])
    return "".join(sents)
