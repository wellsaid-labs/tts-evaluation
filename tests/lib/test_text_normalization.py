import typing

from lib.text_normalization import (
    RegExPatterns,
    _norm,
    _num2words,
    _verbalize_abbreviated_time,
    _verbalize_acronym,
    _verbalize_alternative_phone_number,
    _verbalize_decade,
    _verbalize_fraction,
    _verbalize_generic_number,
    _verbalize_measurement_abbreviation,
    _verbalize_money,
    _verbalize_number_sign,
    _verbalize_ordinal,
    _verbalize_percent,
    _verbalize_phone_number,
    _verbalize_time,
    _verbalize_title_abbreviation,
    _verbalize_url,
    _verbalize_year,
    verbalize_text,
)


def assert_verbalized(
    tests: typing.List[typing.Tuple[str, str]],
    patterns: typing.Union[typing.Pattern, typing.List[typing.Pattern]],
    verbalize: typing.Callable[..., str],
    **kwargs,
):
    for text_in, text_out in tests:
        for pattern in patterns if isinstance(patterns, list) else [patterns]:
            text_in = _norm(text_in, pattern, verbalize, **kwargs)
        assert text_in == text_out


def test__num2words():
    """Test `_num2words` for numbers with leading or trailing zeros."""
    tests_ = [
        ("", ""),
        ("0", "zero"),
        ("00", "zero zero"),
        ("00.", "zero zero point"),
        (".00", "point zero zero"),
        ("00.00", "zero zero point zero zero"),
        ("003", "zero zero three"),
        ("003.", "zero zero three point"),
        ("003.00", "zero zero three point zero zero"),
        ("003.10", "zero zero three point one zero"),
        ("0030.10", "zero zero thirty point one zero"),
        ("0030.1010", "zero zero thirty point one zero one zero"),
    ]
    for text_in, text_out in tests_:
        assert _num2words(text_in, ignore_zeros=False) == text_out


def test__money():
    """Test `RegExPatterns.MONEY` matching and `_verbalize_money` verbalization."""
    tests = [
        (
            "You can earn from $5 hundred to $5 million.",
            "You can earn from five hundred dollars to five million dollars.",
        ),
        ("How to turn $200 into a grand.", "How to turn two hundred dollars into a grand."),
        (
            "It is estimated that by 2010, $1.20 trillion will be spent on education reform.",
            "It is estimated that by 2010, one point two zero trillion dollars will be spent on "
            "education reform.",
        ),
        (
            "Inside the drawer was a fake passport and $15,000 cash.",
            "Inside the drawer was a fake passport and fifteen thousand dollars cash.",
        ),
        (
            "All this for only $9.99 a month!",
            "All this for only nine dollars and ninety-nine cents a month!",
        ),
        (
            "And it's only €1.01 for general admission and €3.00 for the weekend pass.",
            "And it's only one euro and one cent for general admission and three euros for the "
            "weekend pass.",
        ),
        (
            "£212,512 was spent on management last year",
            "two hundred and twelve thousand, five hundred and twelve pounds was spent on "
            "management last year",
        ),
        (
            "The team will accrue NZ$150M to NZ$180M this month.",
            "The team will accrue one hundred and fifty million New Zealand dollars to one "
            "hundred and eighty million New Zealand dollars this month.",
        ),
        ("$5.", "five dollars."),
        ("You won $5 zillion. Wow. Thanks.", "You won five zillion dollars. Wow. Thanks."),
        (
            "This guide will tell you how to retire on $5 Million for the rest of your life, "
            "guaranteed.",
            "This guide will tell you how to retire on five Million dollars for the rest of your "
            "life, guaranteed.",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.MONEY, _verbalize_money)


def test___ordinals():
    """Test `RegExPatterns.ORDINALS` matching and `_verbalize_ordinal` verbalization."""
    tests = [
        (
            "On November 22nd, 1963 President John F. Kennedy arrived.",
            "On November twenty-second, 1963 President John F. Kennedy arrived.",
        ),
        (
            "Jesus of Prague, the 13th-century Cathedral of St. Matthew",
            "Jesus of Prague, the thirteenth-century Cathedral of St. Matthew",
        ),
        ("what he wanted for his 15th birthday.", "what he wanted for his fifteenth birthday."),
        (
            "be recycled, while the 3rd booster - travelling much too fast",
            "be recycled, while the third booster - travelling much too fast",
        ),
        (
            "celebrate the 400th anniversary of Rembrandt's birth?",
            "celebrate the four hundredth anniversary of Rembrandt's birth?",
        ),
        (
            "1127 days, from June 25th 1950 through July 27th 1953",
            "1127 days, from June twenty-fifth 1950 through July twenty-seventh 1953",
        ),
        (
            "In the early-20th century, people like Ordway Tead,",
            "In the early-twentieth century, people like Ordway Tead,",
        ),
        (
            "2000 (MM) was a century leap year starting on Saturday of the Gregorian calendar, "
            "the 2000th year of the Common Era (CE) and Anno Domini (AD) designations, the 1000th "
            "and last year of the 2nd millennium, the 100th and last year of the 20th century, "
            "and the 1st year of the 2000s decade.",
            "2000 (MM) was a century leap year starting on Saturday of the Gregorian calendar, "
            "the two thousandth year of the Common Era (CE) and Anno Domini (AD) designations, "
            "the one thousandth and last year of the second millennium, the one hundredth and "
            "last year of the twentieth century, and the first year of the 2000s decade.",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.ORDINALS, _verbalize_ordinal)


def test___times():
    """Test `RegExPatterns.TIMES` matching and `_verbalize_time` verbalization."""
    tests = [
        ("Today at 12:44 PM", "Today at twelve forty-four PM"),
        (
            "set session times for 7:00 p.m. but show up hours later",
            "set session times for seven oh clock PM but show up hours later",
        ),
        (
            "The Love Monkeys from 8:15 to midnight!",
            "The Love Monkeys from eight fifteen to midnight!",
        ),
        (
            'Proverbs 16:21 includes: "The wise of heart is called perceptive.',
            'Proverbs sixteen twenty-one includes: "The wise of heart is called perceptive.',
        ),
        (
            "and then he naps at 12:15 every afternoon",
            "and then he naps at twelve fifteen every afternoon",
        ),
        (
            "They should be arriving between 9:45 and 11:00 tomorrow night!",
            "They should be arriving between nine forty-five and eleven oh clock tomorrow night!",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.TIMES, _verbalize_time, o_clock="oh clock")


def test___abbreviated_times():
    """Test `RegExPatterns.ABBREVIATED_TIMES` matching and `_verbalize_abbreviated_time`
    verbalization."""
    tests = [
        (
            "Don't forget to feed him at 9AM, 1 p.m., and 5 pm!",
            "Don't forget to feed him at nine AM, one PM, and five PM!",
        )
    ]
    assert_verbalized(tests, RegExPatterns.ABBREVIATED_TIMES, _verbalize_abbreviated_time)


def test_reg_ex_patterns_phone_numbers():
    """Test `RegExPatterns.PHONE_NUMBERS` against a number of cases."""
    # NOTE: These test cases were adapted from here:
    # https://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
    match = RegExPatterns.PHONE_NUMBERS.fullmatch
    assert match("1 800 555 1234")
    assert match("+1 800 555-1234")
    assert match("+86 800 555 1234")
    assert match("1-800-555-1234")
    assert match("1 (800) 555-1234")
    assert match("(800)555-1234")
    assert match("(800) 555-1234")
    assert match("800-555-1234")
    assert match("800.555.1234")
    assert match("555-6482")  # No area code
    assert match("86 800 555 1212")  # Non-NA country code doesn't have +
    assert match("1 (800)  555-1234")  # Too many spaces
    assert not match("180055512345")  # Too many digits
    assert not match("(800)5551234")
    assert not match("18005551234")
    assert not match("4 967 295,000")  # Canadian Number
    assert not match("800 555 1234x5678")  # Extension not supported
    assert not match("8005551234 x5678")  # Extension not supported
    assert not match("1 800 5555 1234")  # Prefix code too long
    assert not match("+1 800 555x1234")  # Invalid delimiter
    assert not match("+867 800 555 1234")  # Country code too long
    assert not match("1-800-555-1234p")  # Invalid character
    assert not match("800x555x1234")  # Invalid delimiter


def test_reg_ex_patterns_toll_free_phone_numbers():
    """Test `RegExPatterns.TOLL_FREE_PHONE_NUMBERS` against a number of cases."""
    match = RegExPatterns.TOLL_FREE_PHONE_NUMBERS.fullmatch

    # Numeric phone numbers
    assert match("1-800-555-1234")
    assert match("1 (800) 555-1234")
    assert not match("+86 800 555 1234")
    assert not match("1 800 555 1234x567")  # An extension code
    assert not match("800 5551234 x567")  # An extension code
    assert not match("8005551234 x567")
    assert not match("18005551234")
    assert not match("800x555x1234")  # Invalid delimiter

    # Alpha numeric phone numbers
    assert match("1-800 XFINITY")
    assert match("1 (800) XFINITY")
    assert match("+1 (800) XFINITY")
    assert match("+1 (800) XFINITY")
    assert match("1(800)Xfinity")
    assert match("1-800-ski-green")
    assert match("1-800-4MY-HOME")
    assert not match("1800-XFINITY")  # No delimiter between Area Code and Line Number
    assert not match("1800XFINITY")  # No delimiter between Area Code and Line Number
    assert not match("1-800XFINITY")


def test_reg_ex_patterns_alternative_phone_numbers():
    """Test `RegExPatterns.ALTERNATIVE_PHONE_NUMBERS` against a number of cases."""
    match = RegExPatterns.ALTERNATIVE_PHONE_NUMBERS.fullmatch

    assert match("dialing 911")
    assert match("Call 5-4356")


def test___phone_numbers():
    """Test `RegExPatterns.PHONE_NUMBERS` and `RegExPatterns.TOLL_FREE_PHONE_NUMBERS` matching and
    `_verbalize_phone_number` verbalization."""
    tests = [
        ("123-456-7890", "one two three, four five six, seven eight nine zero"),
        ("(123) 456-7890", "one two three, four five six, seven eight nine zero"),
        ("123 456 7890", "one two three, four five six, seven eight nine zero"),
        ("1.123.456.7890", "one, one two three, four five six, seven eight nine zero"),
        ("+91 (123) 456-7890", "nine one, one two three, four five six, seven eight nine zero"),
        (
            "Contact Arizona tourism at 1-800-925-6689 or www.ArizonaGuide.com.",
            "Contact Arizona tourism at one, eight hundred, nine two five, six six eight nine or "
            "www.ArizonaGuide.com.",
        ),
        (
            "Call Jeannie at 555-9875 to start looking",
            "Call Jeannie at five five five, nine eight seven five to start looking",
        ),
        (
            "go to crystalcruises.com or call 1 800 340 1300.",
            "go to crystalcruises.com or call one, eight hundred, three four zero, one three "
            "zero zero.",
        ),
        ("Give us a call at 555-6482.", "Give us a call at five five five, six four eight two."),
        (
            "Call telecharge today at 212-947-8844. Some restrictions apply.",
            "Call telecharge today at two one two, nine four seven, eight eight four four. Some "
            "restrictions apply.",
        ),
        (
            "Call 1-800-SANDALS now and save 50%.",
            "Call one, eight hundred, SANDALS now and save 50%.",
        ),
        (
            "representative at 1-888-Comcast for further assistance.",
            "representative at one, eight eight eight, Comcast for further assistance.",
        ),
        (
            "Largest Hotel Chain, call 1-800-Western today.",
            "Largest Hotel Chain, call one, eight hundred, Western today.",
        ),
        (
            "Discover New York. Call 1-800-I love NY.",
            "Discover New York. Call one, eight hundred, I love NY.",
        ),
        (
            "For more information, call 1-800-ski-green. That‘s 1-800-ski-green.",
            "For more information, call one, eight hundred, ski, green. That‘s one, eight "
            "hundred, ski, green.",
        ),
        (
            "your travel agent, or call 1-800 USVI-INFO.",
            "your travel agent, or call one, eight hundred, USVI, INFO.",
        ),
        (
            "personal escape packages call 1-800-ASK-4-SPA or visit our website",
            "personal escape packages call one, eight hundred, ASK, four, SPA or visit our website",
        ),
    ]
    patterns = [RegExPatterns.PHONE_NUMBERS, RegExPatterns.TOLL_FREE_PHONE_NUMBERS]
    assert_verbalized(tests, patterns, _verbalize_phone_number)


def test___alternative_phone_numbers():
    """Test `RegExPatterns.ALTERNATIVE_PHONE_NUMBERS` matching and
    `_verbalize_alternative_phone_number` verbalization."""
    tests = [
        (
            "you can reach me by calling 5-4332.",
            "you can reach me by calling five, four three three two.",
        ),
        (
            "If this is an emergency, please hang up and dial 911.",
            "If this is an emergency, please hang up and dial nine one one.",
        ),
    ]
    assert_verbalized(
        tests, RegExPatterns.ALTERNATIVE_PHONE_NUMBERS, _verbalize_alternative_phone_number
    )


def test___years():
    """Test `RegExPatterns.YEARS` and `_verbalize_year` verbalization.

    TODO: With more robust year searches, include additional test cases, especially around more
    historic periods.
    """
    tests = [
        ("Monday, March 16, 1908.", "Monday, March 16, nineteen oh-eight."),
        (
            "It is 1959: in this remote corner of India.",
            "It is nineteen fifty-nine: in this remote corner of India.",
        ),
        (
            "Twenty years ago, in 1990, Declan Treacy invented",
            "Twenty years ago, in nineteen ninety, Declan Treacy invented",
        ),
        ("A 2004 CIBC survey suggests", "A two thousand and four CIBC survey suggests"),
        (
            "with a run of hit songs during the mid-1960's, including \"Surfin' U.S.A.\" (1963)",
            "with a run of hit songs during the mid-nineteen sixty's, including "
            '"Surfin\' U.S.A." (nineteen sixty-three)',
        ),
        (
            "If you find a 1776-1976 quarter",
            "If you find a seventeen seventy-six to nineteen seventy-six quarter",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.YEARS, _verbalize_year)


def test___decades():
    """Test `RegExPatterns.DECADES` matching and `_verbalize_decade` verbalization."""
    tests = [
        (
            'knowledge worker" in the 1950s. He described how',
            'knowledge worker" in the nineteen fifties. He described how',
        ),
        ("leaders in the mid-1980s.", "leaders in the mid-nineteen eighties."),
        (
            "In the '80s there was a strong underground influence",
            "In the eighties there was a strong underground influence",
        ),
        ("--most in their 50s and 60s--", "--most in their fifties and sixties--"),
        (
            "In the 2020s, there has been an increase in online sales",
            "In the twenty twenties, there has been an increase in online sales",
        ),
        (
            "Corporate scandals in the earlier 2000s",
            "Corporate scandals in the earlier two thousands",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.DECADES, _verbalize_decade)


def test___percents():
    """Test `RegExPatterns.PERCENTS` matching and `_verbalize_percent` verbalization."""
    tests = [
        ("Over 88% of smokers start young", "Over eighty-eight percent of smokers start young"),
        ("Book now and save 50%.", "Book now and save fifty percent."),
        (
            "Corps makes up only 10.8% of the total Department",
            "Corps makes up only ten point eight percent of the total Department",
        ),
        (
            "since most are only 2-3% over the amount you will be earning",
            "since most are only two to three percent over the amount you will be earning",
        ),
        (
            "by as much as 15–25% every time cumulative",
            "by as much as fifteen to twenty-five percent every time cumulative",
        ),
        (
            "Everyone's Use of Percentages is 1,000% Out of Control. You are going to 10,000% "
            "agree with this take.",
            "Everyone's Use of Percentages is one thousand percent Out of Control. You are going "
            "to ten thousand percent agree with this take.",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.PERCENTS, _verbalize_percent)


def test___number_signs():
    """Test `RegExPatterns.NUMBER_SIGNS` matching and `_verbalize_number_sign` verbalization."""
    tests = [
        (
            "#1: The aesthetic dimension of human experience",
            "number one: The aesthetic dimension of human experience",
        ),
        (
            "#2: Green spaces are easily taken for granted",
            "number two: Green spaces are easily taken for granted",
        ),
        (
            "Section #3: In the next few minutes, you will learn...",
            "Section number three: In the next few minutes, you will learn...",
        ),
        (
            "They were ranked #4 on Vault's twenty eighteen list",
            "They were ranked number four on Vault's twenty eighteen list",
        ),
        (
            "#2-4",
            "numbers two through four",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.NUMBER_SIGNS, _verbalize_number_sign)


def test___urls():
    """Test `RegExPatterns.URLS` matching and `_verbalize_url` verbalization."""
    tests = [
        ("staging.wellsaidlabs.com", "staging dot wellsaidlabs dot com"),
        (
            "you just need to visit www.ArizonaGuide.com to see for yourself",
            "you just need to visit w w w dot ArizonaGuide dot com to see for yourself",
        ),
        ("find us at Dianetics.org", "find us at Dianetics dot org"),
        (
            "For inquiries, please email info@website.com and include your order number.",
            "For inquiries, please email info at website dot com and include your order number.",
        ),
        (
            "and many more activities are listed at visitdallas.com/things-to-do",
            "and many more activities are listed at visitdallas dot com slash things dash to "
            "dash do",
        ),
        ("www.gov.nf.ca/tourism", "w w w dot gov dot nf dot ca slash tourism"),
        ("https://www.foxwoods.com", "h t t p s colon slash slash w w w dot foxwoods dot com"),
        (
            "http://www.example.com:80/path/to/myfile.html?key1=value1#SomewhereInTheFile",
            "h t t p colon slash slash w w w dot example dot com colon eight zero slash path "
            "slash to slash myfile dot html question mark key one equals value one hash "
            "SomewhereInTheFile",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.URLS, _verbalize_url)


def test___acronyms():
    """Test `RegExPatterns.ACRONYMS` matching and `_verbalize_acronym` verbalization for cases
    involving initialisms, word acronyms, and pseudo-blends."""
    tests = [
        (
            "The mission of the NAACP is to ensure the political, educational, social, and "
            "economic equality of rights of all persons and to eliminate race-based "
            "discrimination.",
            "The mission of the N double A C P is to ensure the political, educational, "
            "social, and economic equality of rights of all persons and to eliminate "
            "race-based discrimination.",
        ),
        (
            "The NCAA tournament ended last night in a thrilling upset from the underdog team.",
            "The N C double A tournament ended last night in a thrilling upset from the underdog "
            "team.",
        ),
        ("Please R.S.V.P. ASAP, thank you!", "Please RSVP ASAP, thank you!"),
        (
            "investing in the NASDAQ is a WYSIWYG situation, if you will",
            "investing in the Nazdack is a wizzy wig situation, if you will",
        ),
        (
            "you can always call AAA after pulling your car safely to the side of the road",
            "you can always call Triple A after pulling your car safely to the side of the road",
        ),
        (
            "she was so athletic, she could have played in the WNBA, the LPGA, the NPF, and "
            "the NWSL -- at the same time!",
            "she was so athletic, she could have played in the WNBA, the LPGA, the NPF, and "
            "the NWSL -- at the same time!",
        ),
        # TODO: This test case requires sentence context.
        # (
        #     "The United States of America (U.S.A. or USA), commonly known as the United States "
        #     "(U.S. or US) or America, is a country primarily located in North America.",
        #     'The phrase "United States" was originally plural in American usage. It described '
        #     'a collection of states—EG, "the United States are." The singular form became '
        #     "popular after the end of the Civil War and is now standard usage in the US. A "
        #     'citizen of the United States is an "American". "United States", "American" and '
        #     '"US" refer to the country adjectivally ("American values", "US forces"). In '
        #     'English, the word "American" rarely refers to topics or subjects not directly '
        #     "connected with the United States.",
        # ),
    ]
    assert_verbalized(tests, RegExPatterns.ACRONYMS, _verbalize_acronym)


def test___measurement_abbreviations():
    """Test `RegExPatterns.MEASUREMENT_ABBREVIATIONS` matching and
    `_verbalize_measurement_abbreviation` verbalization for cases involving initialisms, word
    acronyms, and pseudo-blends."""
    tests = [
        (
            "We request 48-kHz or 44.1Hz audio for 120min.",
            "We request forty-eight kilohertz or forty-four point one hertz audio for one "
            "hundred and twenty minutes.",
        ),
        (
            "The vehicle measures 6ft by 10.5ft and moves at around 65mph.",
            "The vehicle measures six feet by ten point five feet and moves at around "
            "sixty-five miles per hour.",
        ),
        (
            "and it can carry 3,000 fl oz and maintain temperatures below 30°F!",
            "and it can carry three thousand fluid ounces and maintain temperatures below "
            "thirty degrees Fahrenheit!",
        ),
        (
            "toddlers betwen the 2yo and 3yo age range tend to consume 1,200g of food per day",
            "toddlers betwen the two year-old and three year-old age range tend to consume "
            "one thousand, two hundred grams of food per day",
        ),
        (
            "use 3,450.6Ω of resistance for any load above 28kg ",
            "use three thousand, four hundred and fifty point six ohms of resistance for any "
            "load above twenty-eight kilograms ",
        ),
        (
            "and you should expect temperatures at around -20°F, ±1°.",
            "and you should expect temperatures at around minus twenty degrees Fahrenheit, plus "
            "or minus one degree.",
        ),
        ("the lot size is 18.7km³ ", "the lot size is eighteen point seven cubic kilometers "),
        (
            "For example, if the velocity of a particle moving in a straight line changes "
            "uniformly (at a constant rate of change) from 2 m/s to 5 m/s over one second, then "
            "its constant acceleration is 3 m/s².",
            "For example, if the velocity of a particle moving in a straight line changes "
            "uniformly (at a constant rate of change) from two meters per second to five meters "
            "per second over one second, then its constant acceleration is three meters per "
            "second squared.",
        ),
    ]
    assert_verbalized(
        tests, RegExPatterns.MEASUREMENT_ABBREVIATIONS, _verbalize_measurement_abbreviation
    )


def test___fractions():
    """Test `RegExPatterns.FRACTIONS` matching and `_verbalize_fraction`."""
    tests = [
        (
            "Turn air on by turning isolation valve switch 1/4 turn counter-clockwise. Retrieve "
            "Keys from lotto Box.",
            "Turn air on by turning isolation valve switch one quarter turn counter-clockwise. "
            "Retrieve Keys from lotto Box.",
        ),
        (
            "Evelyn is 59 1/2 years old.",
            "Evelyn is fifty-nine and a half years old.",
        ),
        (
            "It was 37 3/4 years ago...",
            "It was thirty-seven and three fourths years ago...",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.FRACTIONS, _verbalize_fraction)


def test___generic_numbers():
    """Test `RegExPatterns.GENERIC_DIGITS` and `RegExPatterns.NUMBER_RANGE` matching and
    `_verbalize_generic_number` verbalization for leftover, standalone numeral cases."""
    tests = [
        (
            "it had been 4,879 days and took 53 boats to relocate all 235 residents just 1.20 "
            "miles south",
            "it had been four thousand, eight hundred and seventy-nine days and took fifty-three "
            "boats to relocate all two hundred and thirty-five residents just one point two zero "
            "miles south",
        ),
        (
            "find us between the hours of 2 and 7 every afternoon.",
            "find us between the hours of two and seven every afternoon.",
        ),
        ("2.3 revolutions in 10 minutes", "two point three revolutions in ten minutes"),
        (
            "children from grades 6-12, and compared",
            "children from grades six to twelve, and compared",
        ),
        (
            "An estimated 30-40 million Americans bought tickets",
            "An estimated thirty to forty million Americans bought tickets",
        ),
        (
            "that you should only use 1-2 search terms",
            "that you should only use one to two search terms",
        ),
        (
            "Children ages 8-12 require tickets; children ages 3-7 do not.",
            "Children ages eight to twelve require tickets; children ages three to seven do not.",
        ),
        ("This may mean an even 50-50 split", "This may mean an even fifty fifty split"),
    ]
    assert_verbalized(
        tests,
        [RegExPatterns.NUMBER_RANGE, RegExPatterns.GENERIC_DIGIT],
        _verbalize_generic_number,
    )


def test__verbalize_title_abbreviations():
    tests = [
        (
            "Let's meet with Dr. Ruth Flintstone later today",
            "Let's meet with Doctor Ruth Flintstone later today",
        ),
        ("Mrs. Robinson will join us later", "Missus Robinson will join us later"),
        (
            "Rev. Silvester Beaman offered a benediction at the inauguration of President Biden.",
            "Reverend Silvester Beaman offered a benediction at the inauguration of President "
            "Biden.",
        ),
        (
            "William Simmons Sr. and Billy Simmons Jr. arrived late.",
            "William Simmons Senior and Billy Simmons Junior arrived late.",
        ),
        ("I didn't see Capt. Clark at the ceremony", "I didn't see Captain Clark at the ceremony"),
        (
            "Mr. and Mrs. Frizzle are out for the day.",
            "Mister and Missus Frizzle are out for the day.",
        ),
        (
            "Jain - Mr Johnson (Lyrics Video) - YouTube",
            "Jain - Mister Johnson (Lyrics Video) - YouTube",
        ),
    ]
    assert_verbalized(tests, RegExPatterns.TITLE_ABBREVIATIONS, _verbalize_title_abbreviation)


def test_verbalize_text():
    """Basic integration test for testing text normalization for text containing multiple
    non-standard-word cases."""
    tests = [
        (
            "Do you want to learn how to turn $200 into a grand!? Call 1-800-MONEY-4-ME now or "
            "visit www.money4me.com/now!",
            "Do you want to learn how to turn two hundred dollars into a grand!? Call one, eight "
            "hundred, MONEY, four, ME now or visit w w w dot money four me dot com slash now!",
        ),
        (
            "It is estimated that by 2010, $1.2 trillion will be spent on education reform for "
            "250,000,000 students",
            "It is estimated that by twenty ten, one point two trillion dollars will be spent on "
            "education reform for two hundred and fifty million students",
        ),
        (
            "The 60s saw a 24% rise in political engagement among the 30-40 year old age group",
            "The sixties saw a twenty-four percent rise in political engagement among the thirty "
            "to forty year old age group",
        ),
        (
            "How is the NCAA managing the fall out from yesterday's 7:00 game? Tune in to News at "
            "5 later tonight!",
            "How is the N C double A managing the fall out from yesterday's seven o'clock game? "
            "Tune in to News at five later tonight!",
        ),
        (
            "November 27th, 1934 came with the discovery of a diamond that was 48-kt!",
            "November twenty-seventh, nineteen thirty-four came with the discovery of a diamond "
            "that was forty-eight karats!",
        ),
        (
            "7:25AM. Run 13mi, eat 2,000cal, nap for 10-15 minutes, and eat dinner with Dr. "
            "Amelia Fern at 7:00 tonight.",
            "seven twenty-five AM. Run thirteen miles, eat two thousand calories, nap for ten to "
            "fifteen minutes, and eat dinner with Doctor Amelia Fern at seven o'clock tonight.",
        ),
        (
            "Growth Plan: #1: The WSL team will accrue NZ$150M this month, #2: we will continue "
            "contributing $1K in the 1st and 2nd quarters, #3: we will double that by 2024",
            "Growth Plan: number one: The WSL team will accrue one hundred and fifty million New "
            "Zealand dollars this month, number two: we will continue contributing one thousand "
            "dollars in the first and second quarters, number three: we will double that by "
            "twenty twenty-four",
        ),
        (
            "It took 3 years, but 2018 marked the beginning of a new AAA era when Dr. Ruth Kinton "
            "became the 1st woman to take over the prestigious position of C.E.O.",
            "It took three years, but twenty eighteen marked the beginning of a new Triple A era "
            "when Doctor Ruth Kinton became the first woman to take over the prestigious position "
            "of CEO.",
        ),
        (
            "For example, if you have a current of 2 A and a voltage of 5 V, the power is "
            "2A * 5V = 10W.",
            "For example, if you have a current of two A and a voltage of five V, the power is "
            "two A * five V = ten W.",
        ),
        (
            "Bacara utilized a DC-15A blaster rifle.",
            "Bacara utilized a DC- fifteen A blaster rifle.",
        ),
        # TODO: Support different number systems...
        # https://docs.oracle.com/cd/E19455-01/806-0169/overview-9/index.html
        (
            "Canadian (English and French) 4 967 295,000 German 4 967.295,000 Italian "
            "4.967.295,000 US-English 4,967,295.00",
            "Canadian (English and French) four, nine six seven, two nine five, zero zero zero "
            "German four, nine six seven, two nine five, zero zero zero Italian four, nine six "
            "seven, two nine five, zero zero zero US-English four million, nine hundred and "
            "sixty-seven thousand, two hundred and ninety-five point zero zero",
        ),
        # (
        #     "[2-Pack, 1ft] Short USB Type C Cable, etguuds 4.2A Fast Charging USB-A to USB-C "
        #     "Charger Cord Braided Compatible with Samsung Galaxy S20 S10 S9 S8 Plus S10E Note "
        #     "20 10 9 8, A10e A20 A50 A51, Moto G7 G8",
        #     "[two pack, one foot] Short USB Type C Cable, etguuds four point two A Fast "
        #     "Charging USB-A to USB-C Charger Cord Braided Compatible with Samsung Galaxy S "
        #     "twenty S ten S nine S eight Plus S ten E Note twenty ten nine eight, A ten e "
        #     "A twenty A fifty A fifty-one , Moto G seven G eight",
        # ),
        # TODO: Fix title abbreviation coverage so it doesn't convert "st." to "Saint" when not
        #       referring to a PERSON entity
        # (
        #     "I live at 324 south st. in lincoln, nebraska.",
        #     "I live at 324 south street in lincoln, nebraska.",
        # ),
    ]
    for text_in, text_out in tests:
        assert verbalize_text(str(text_in)) == text_out
