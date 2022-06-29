import typing
from functools import partial

from lib.text.utils import normalize_vo_script

_norm = partial(normalize_vo_script, non_ascii=frozenset())

###############
#   NUMBERS   #
###############

# See: https://en.wikipedia.org/wiki/Names_of_large_numbers
LARGE_NUMBERS: typing.Final[typing.Tuple[str, ...]] = (
    "Million",
    "Billion",
    "Trillion",
    "Quadrillion",
    "Quintillion",
    "Sextillion",
    "Septillion",
    "Octillion",
    "Nonillion",
    "Decillion",
    "Undecillion",
    "Duodecillion",
    "Tredecillion",
    "Quattuordecillion",
    "Quindecillion",
    "Sedecillion",
    "Septendecillion",
    "Octodecillion",
    "Novendecillion",
    "Vigintillion",
    "Unvigintillion",
    "Duovigintillion",
    "Tresvigintillion",
    "Quattuorvigintillion",
    "Quinvigintillion",
    "Sesvigintillion",
    "Septemvigintillion",
    "Octovigintillion",
    "Novemvigintillion",
    "Trigintillion",
    "Untrigintillion",
    "Duotrigintillion",
    "Trestrigintillion",
    "Quattuortrigintillion",
    "Quintrigintillion",
    "Sestrigintillion",
    "Septentrigintillion",
    "Octotrigintillion",
    "Noventrigintillion",
    "Quadragintillion",
    "Quinquagintillion",
    "Sexagintillion",
    "Septuagintillion",
    "Octogintillion",
    "Nonagintillion",
    "Centillion",
    "Uncentillion",
    "Decicentillion",
    "Undecicentillion",
    "Viginticentillion",
    "Unviginticentillion",
    "Trigintacentillion",
    "Quadragintacentillion",
    "Quinquagintacentillion",
    "Sexagintacentillion",
    "Septuagintacentillion",
    "Octogintacentillion",
    "Nonagintacentillion",
    "Ducentillion",
    "Trecentillion",
    "Quadringentillion",
    "Quingentillion",
    "Sescentillion",
    "Septingentillion",
    "Octingentillion",
    "Nongentillion",
    "Millinillion",
)

# See: https://en.wikipedia.org/wiki/Indefinite_and_fictitious_numbers
LARGE_FICTIOUS_NUMBERS: typing.Final[typing.Tuple[str, ...]] = (
    "zillion",
    "gazillion",
    "jillion",
    "bajillion",
    "squillion",
)

ORDINAL_SUFFIXES: typing.Final[typing.Tuple[str, ...]] = ("st", "nd", "rd", "th")

############
#   MONEY  #
############

_CURRENCIES: typing.Final[typing.Dict[str, typing.Tuple[str, str, str, str]]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "US$": ("US dollar", "US dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", "yen", "yen"),
    "£": ("pound", "pounds", "pence", "pence"),
    "A$": ("Australian dollar", "Australian dollars", "cent", "cents"),
    "C$": ("Canadian dollar", "Canadian dollars", "cent", "cents"),
    "元": ("Renminbi", "Renminbi", "jiao", "jiao"),
    "CHF": ("Swiss franc", "Swiss francs", "centime", "centimes"),
    "HK$": ("Hong Kong dollar", "Hong Kong dollars", "cent", "cents"),
    "NZ$": ("New Zealand dollar", "New Zealand dollars", "cent", "cents"),
    "MX$": ("Mexican Peso", "Mexican Pesos", "centavo", "centavos"),
    "DEM": ("Deutsche Mark", "Deutsche Marks", "pfennig", "pfennigs"),
    "R$": ("Brazilian real", "Brazilian real", "centavo", "centavos"),
}

CURRENCIES: typing.Final[typing.Dict[str, typing.Tuple[str, str, str, str]]]
CURRENCIES = {_norm(k): v for k, v in _CURRENCIES.items()}

MONEY_SUFFIX: typing.Final[typing.Tuple[str, ...]] = (
    "hundred",
    "thousand",
    "million",
    "billion",
    "trillion",
)

MONEY_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    "K": MONEY_SUFFIX[1],
    "M": MONEY_SUFFIX[2],
    "MM": MONEY_SUFFIX[2],
    "B": MONEY_SUFFIX[3],
    "BB": MONEY_SUFFIX[3],
    "T": MONEY_SUFFIX[4],
    "TT": MONEY_SUFFIX[4],
}

################
# MEASUREMENTS #
################

# NOTE: These abbreviations must follow a time so as not to be confused with alternative
# abbreviations such as "established" (est) or "piedmont" (pdt).

TIME_ZONES: typing.Final[typing.Dict[str, str]] = {
    # TODO: include global timezones
    "est": "eastern standard time",
    "cst": "central standard time",
    "mst": "mountain standard time",
    "pst": "pacific standard time",
    "edt": "eastern daylight time",
    "cdt": "central daylight time",
    "mdt": "mountain daylight time",
    "pdt": "pacific daylight time",
    "gmt": "greenwich mean time",
}

################
# MEASUREMENTS #
################

# NOTE: These abbreviations must follow a digit to be considered a unit of measurement. Text
#       Normalization will not blindly replace all instances of "in" with "inches", e.g.
#       I've left out rare single letter abbreviations. They are rarely used by our users
#       to denote a measurement, more often are expected to be read as a letter: 1A, 4K, 5V, etc.


LENGTH: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "in. ": ("inch", "inches"),  # inches is tricky because 'in' is a word
    "ft": ("foot", "feet"),
    "yd": ("yard", "yards"),
    "mi": ("mile", "miles"),
    "mm": ("millimeter", "millimeters"),
    "cm": ("centimeter", "centimeters"),
    "m": ("meter", "meters"),
    "km": ("kilometer", "kilometers"),
}

AREA: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "sq in": ("square inch", "square inches"),
    "sq ft": ("square foot", "square feet"),
    "sq yd": ("square yard", "square yards"),
    "sq mi": ("square mile", "square miles"),
    "ac": ("acre", "acres"),
    "ha": ("hectare", "hectares"),
}

VOLUME: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "cu mi": ("cubic mile", "cubic miles"),
    "cu yd": ("cubic yard", "cubic yards"),
    "cu ft": ("cubic foot", "cubic feet"),
    "cu in": ("cubic inch", "cubic inches"),
}

LIQUID_VOLUME: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "tsp": ("teaspoon", "teaspoons"),
    # "t": ("teaspoon", "teaspoons"),       # RARE
    "tbs": ("tablespoon", "tablespoons"),
    "tbsp": ("tablespoon", "tablespoons"),
    # "T": ("tablespoon", "tablespoons"),   # RARE, Watch out for 'Tons'
    # "c": ("cup", "cups"),                 # RARE
    "fl oz": ("fluid ounce", "fluid ounces"),
    "qt": ("quart", "quarts"),
    "pt": ("pint", "pints"),
    "gal": ("gallon", "gallons"),
    "mL": ("milliliter", "milliliters"),
    "L": ("liter", "liters"),
    "kL": ("kiloliter", "kiloliters"),
}

WEIGHT_MASS: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "lb": ("pound", "pounds"),
    "oz": ("ounce", "ounces"),
    "mg": ("milligram", "milligrams"),
    "g": ("gram", "grams"),
    "kg": ("kilogram", "kilograms"),
    "MT": ("metric ton", "metric tons"),
    "ct": ("carat", "carats"),
    "kt": ("karat", "karats"),
}

SPEED: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "mph": ("mile per hour", "miles per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "rpm": ("revolution per minute", "revolutions per minute"),
    "kn": ("knot", "knots"),
    "m/s": ("meter per second", "meters per second"),
    "ft/s": ("foot per second", "feet per second"),
}

TEMPERATURE: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "°F": ("degree Fahrenheit", "degrees Fahrenheit"),
    "°C": ("degree Celsius", "degrees Celsius"),
    # "K": ("Kelvin", "Kelvin"),           # Too rare; K is much more often associated with 4K, 401K
}

ENERGY: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    # "W": ("watt", "watts"),         # RARE
    "mW": ("megawatt", "megawatts"),
    "kW": ("kilowatt", "kilowatts"),
    "kWh": ("kilowatt-hour", "kilowatt-hours"),
    # "A": ("ampere", "amperes"),     # Too vague at this point, often used for labeling "1A, 1B..."
    "Hz": ("hertz", "hertz"),
    "mHz": ("megahertz", "megahertz"),
    "kHz": ("kilohertz", "kilohertz"),
    "GHz": ("gigahertz", "gigahertz"),
    # "N": ("newton", "newtons"),     # RARE
    "Pa": ("pascal", "pascals"),
    # "J": ("joule", "joules"),       # RARE
    # "V": ("volt", "volts"),         # RARE
    "cal": ("calorie", "calories"),
    "lb-ft": ("pount-foot", "pound-feet"),
}

MEMORY: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "b": ("bit", "bits"),
    # "B": ("byte", "bytes"),         # Too vague at this point, often used for labeling "1A, 1B..."
    "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"),
    "GB": ("gigabyte", "gigabytes"),
}

TIME: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "ms": ("millisecond", "milliseconds"),
    "sec": ("second", "seconds"),
    "min": ("minute", "minutes"),
    "h": ("hour", "hours"),
    "hr": ("hour", "hours"),
    "wk": ("week", "weeks"),
    "mo": ("month", "months"),
    "yr": ("year", "years"),
    "yo": ("year-old", "year-old"),
}

ETCETERA: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "doz": ("dozen", "dozens"),
    "rad": ("radian", "radians"),
    "dB": ("decibel", "decibels"),
}

SUPERSCRIPT: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    "in²": ("square inch", "square inches"),
    "ft²": ("square foot", "square feet"),
    "yd²": ("square yard", "square yards"),
    "mi²": ("square mile", "square miles"),
    "cm²": ("square centimeter", "square centimeters"),
    "m²": ("square meter", "square meters"),
    "km²": ("square kilometer", "square kilometers"),
    "mi³": ("cubic mile", "cubic miles"),
    "yd³": ("cubic yard", "cubic yards"),
    "ft³": ("cubic foot", "cubic feet"),
    "in³": ("cubic inch", "cubic inches"),
    "cm³": ("cubic centimeter", "cubic centimeters"),
    "m³": ("cubic meter", "cubic meters"),
    "km³": ("cubic kilometer", "cubic kilometers"),
    "m/s²": ("meter per second squared", "meters per second squared"),
    "ft/s²": ("foot per second squared", "feet per second squared"),
}

OTHER: typing.Final[typing.Dict[str, typing.Tuple[str, str]]] = {
    # '"': ("inch", "inches"),              # Too vague, quotes are often used for emphasis instead.
    # "″": ("inch", "inches"),
    # "'": ("foot", "feet"),                # Too vague, quotes are often used for emphasis instead.
    # "′": ("foot", "feet"),
    "°": ("degree", "degrees"),
    "Ω": ("ohm", "ohms"),
}

_UNITS_ABBREVIATIONS: typing.Dict[str, typing.Tuple[str, str]] = {
    **LENGTH,
    **AREA,
    **VOLUME,
    **LIQUID_VOLUME,
    **WEIGHT_MASS,
    **SPEED,
    **TEMPERATURE,
    **ENERGY,
    **MEMORY,
    **TIME,
    **ETCETERA,
    **SUPERSCRIPT,
    **OTHER,
}

UNITS_ABBREVIATIONS: typing.Final[typing.Dict[str, typing.Tuple[str, str]]]
UNITS_ABBREVIATIONS = {_norm(k): v for k, v in _UNITS_ABBREVIATIONS.items()}

_PLUS_OR_MINUS_PREFIX = {
    "+": "plus",
    "-": "minus",
    "±": "plus or minus",
    "+/-": "plus or minus",
}
PLUS_OR_MINUS_PREFIX: typing.Final[typing.Dict[str, str]]
PLUS_OR_MINUS_PREFIX = {_norm(k): v for k, v in _PLUS_OR_MINUS_PREFIX.items()}

#################
# ABBREVIATIONS #
#################

TITLES_PERSON_PRX: typing.Final[typing.Dict[str, str]] = {
    "mr": "Mister",
    "ms": "Miz",
    "mrs": "Missus",
    "dr": "Doctor",
    "prof": "Professor",
    "rev": "Reverend",
    "fr": "Father",
    "pr": "Pastor",
    # "br": "Brother",  # Rare, 'br' is more commonly denoting 'Brazil'
    # "sr": "Sister",   # Rare, 'sr' is more commonly denoting 'Senior'
    # "st": "Saint",    # Rare, 'st' is more commonly denoting 'street'
    "pres": "President",
    "vp": "Vice President",
    "hon": "Honorable",
    # "rep": "Representative",  # rep is a common word used in fitness
    "sen": "Senator",
    "adm": "Admiral",
    "capt": "Captain",
    # "gen": "General",  # Rare, more commonly gen is denoting 'Generation' but can be spoken "gen"
    "col": "Colonel",
    "lt": "Lieutenant",
    "maj": "Major",
    "supt": "Superintendent",
}

TITLES_PERSON_SFX: typing.Final[typing.Dict[str, str]] = {
    "sr": "Senior",
    "jr": "Junior",
    "esq": "Esquire",
}

MONTH_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    # "may": "may",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}

# TODO: Include Days of the week abbreviations; they currently confuse the sentencizer.
DAY_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    # "sun": "sunday",
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tueseday",
    "wed": "wednesday",
    "thur": "thursday",
    "thurs": "thursday",
    "fri": "friday",
    # "sat": "saturday",
}

GEOGRAPHY_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    "ave": "avenue",
    "blvd": "boulevard",
    "cyn": "canyon",
    # "dr": "drive",
    "ln": "lane",
    "rd": "road",
    "st": "street",
    "hwy": "highway",
    "fwy": "freeway",
    "rte": "route",
    # "apt": "apartment",  'apt' is an English word
    # "no": "number",
}


PART_OF_SPEECH_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    "adj": "adjective",
    "adjs": "adjectives",
    "adv": "adverb",
    "advb": "adverb",
    "pron": "pronoun",
    "prov": "proverb",
    "subj": "subject",
    "subord": "subordinate",
    "vb": "verb",
    "vbl": "verbal",
    "vbs": "verbs",
}

_OTHER_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    "approx": "approximately",
    "appt": "appointment",
    "cal": "calendar",
    # "cent": "century", # Rare, 'cent' is more commonly denoting 'cent'
    "conj": "conjunction",
    "dept": "department",
    "dept": "department",
    "dict": "dictionary",
    "doc": "document",
    "docs": "documents",
    # "ed": "edition",  # Ed is a person's name
    "eds": "editions",
    "est": "established",
    "etc": "et cetera",
    "fig": "figure",
    "govt": "government",
    "inc": "incorporated",
    "misc": "miscellaneous",
    "num": "numbers",
    "std": "standard",
    "tel": "telephone",
    "trig": "trigonometry",
    "vs": "versus",
}

_GENERAL_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {}
for dict_ in (
    _OTHER_ABBREVIATIONS,
    TITLES_PERSON_PRX,
    TITLES_PERSON_SFX,
    MONTH_ABBREVIATIONS,
    # DAY_ABBREVIATIONS,
    GEOGRAPHY_ABBREVIATIONS,
    PART_OF_SPEECH_ABBREVIATIONS,
):
    for key, value in dict_.items():
        assert key not in _GENERAL_ABBREVIATIONS, f"Duplicate key found: {key} "
        _GENERAL_ABBREVIATIONS[key] = value

GENERAL_ABBREVIATIONS: typing.Final[typing.Dict[str, str]] = {
    **{k.lower(): v for k, v in _GENERAL_ABBREVIATIONS.items()},
    **{k.capitalize(): v for k, v in _GENERAL_ABBREVIATIONS.items()},
}

############
# ACRONYMS #
############

ORGANIZATIONS: typing.Final[typing.Dict[str, str]] = {
    "AAA": "Triple A",
    "DARE": "Dare",
    "EPCOT": "Epcot",
    "FIFA": "Fifa",
    "G20": "G twenty",
    "GEICO": "Geico",
    "HUD": "Hud",
    "IEEE": "I triple E",
    "IMAX": "I Max",
    "MADD": "Mad",
    "NCAA": "N C double A",
    "NAACP": "N double A C P",
    "NASA": "Nasa",
    "NATO": "Nato",
    "NAFTA": "Nafta",
    "NERF": "Nerf",
    "OSHA": "Osha",
    "OPEC": "Opec",
    "UNESCO": "Unesco",
    "UNICEF": "Unicef",
    "UNIFEM": "Unifem",
}

EVENTS: typing.Final[typing.Dict[str, str]] = {
    "WWI": "World War One",
    "WWII": "World War Two",
    "Y2K": "Y two K",
}

FINANCIAL: typing.Final[typing.Dict[str, str]] = {
    "401K": "four oh one K",
    "401(k)": "four oh one K",
    "403(b)": "four oh three B",
    "AMEX": "Am X",
    "FICO": "Fi-co",
    "NASDAQ": "Nazdack",
}

TECH_TERMS: typing.Final[typing.Dict[str, str]] = {
    "2D": "two D",
    "3D": "three D",
    "CAPTCHA": "Captcha",
    "CD-ROM": "C D Rom",
    "GIF": "gif",
    "iOS": "I O S",
    "JPEG": "J peg",
    "SQL": "Sequel",
    "STEM": "Stem",
    "WYSIWYG": "wizzy wig",
}

SLANG: typing.Final[typing.Dict[str, str]] = {
    "FOMO": "fo mo",
    "YOLO": "yo low",
    "ROFL": "roffle",
}

STAR_WARS: typing.Final[typing.Dict[str, str]] = {
    "BB-8": "B B eight",
    "C-3PO": "C three P O",
    "R2D2": "R two D two",
}

GENERAL: typing.Final[typing.Dict[str, str]] = {
    "AIDS": "Aids",
    "HVAC": "H Vac",
    "SCUBA": "scuba",
    "SNAFU": "snafu",
    "SWAT": "Swat",
}

DIRECTIONS: typing.Final[typing.Dict[str, str]] = {
    "E": "east",
    "N": "north",
    "NE": "northeast",
    "NW": "northwest",
    "S": "south",
    "SE": "southeast",
    "SW": "southwest",
    "W": "west",
}

ACRONYMS: typing.Final[typing.Dict[str, str]] = {
    **ORGANIZATIONS,
    **EVENTS,
    **FINANCIAL,
    **TECH_TERMS,
    **SLANG,
    **STAR_WARS,
    **DIRECTIONS,
    **GENERAL,
}

############
# SYMBOLS  #
############

_SYMBOLS_VERBALIZED: typing.Final[typing.Dict[str, str]] = {
    "!": "exclamation point",
    "@": "at",
    "#": "hash",
    "$": "dollar",
    "%": "percent",
    "^": "carat",
    "&": "ampersand",
    "*": "asterisk",
    "(": "left parenthesis",
    ")": "right parenthesis",
    "-": "dash",
    "_": "underscore",
    "+": "plus",
    "=": "equals",
    "~": "tilde",
    "`": "back tick",
    "{": "left curly brace",
    "}": "right curly brace",
    "[": "left bracket",
    "]": "right bracket",
    "|": "pipe",
    "\\": "backslash",
    ":": "colon",
    ";": "semi-colon",
    '"': "quote",
    "'": "apostrophe",
    "<": "left chevron",
    ">": "right chevron",
    ",": "comma",
    ".": "dot",
    "?": "question mark",
    "/": "slash",
    "±": "plus or minus",
}
SYMBOLS_VERBALIZED: typing.Final[typing.Dict[str, str]]
SYMBOLS_VERBALIZED = {_norm(k): v for k, v in _SYMBOLS_VERBALIZED.items()}

HYPHENS: typing.Tuple[str, ...] = tuple(_norm(t) for t in ("-", "—", "–"))
