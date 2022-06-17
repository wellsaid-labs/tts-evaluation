"""
TODO: Add failure cases, so we can keep track of cases which are not yet supported.
TODO: Write a unit test for every function in `text_utils.py`.
"""

import typing

from lib.text_normalization import (
    RegExPatterns,
    _normalize_abbreviated_times,
    _normalize_acronyms,
    _normalize_decades,
    _normalize_fraction,
    _normalize_generic_digit,
    _normalize_measurement_abbreviations,
    _normalize_money,
    _normalize_number_ranges,
    _normalize_number_signs,
    _normalize_ordinals,
    _normalize_percents,
    _normalize_phone_numbers,
    _normalize_roman_numerals,
    _normalize_text_from_pattern,
    _normalize_times,
    _normalize_title_abbreviations,
    _normalize_urls,
    _normalize_years,
    _num2words,
    normalize_text,
)


def assert_normalized(
    in_: typing.List[str],
    out: typing.List[str],
    pattern: typing.Union[typing.Iterable[typing.Pattern[str]], typing.Pattern[str]],
    normalize: typing.Callable[[typing.Match[str]], str],
    **kwargs,
):
    for text_in, text_out in zip(in_, out):
        assert _normalize_text_from_pattern(text_in, pattern, normalize, **kwargs) == text_out


def test__num2words():
    """Test `_num2words` for numbers with leading or trailing zeros."""
    in_ = [
        "",
        "0",
        "00",
        "00.",
        "00.00",
        "003",
        "003.",
        "003.00",
        "003.10",
        "0030.10",
        "0030.1010",
    ]
    out = [
        "",
        "zero",
        "zero zero",
        "zero zero point",
        "zero zero point zero zero",
        "zero zero three",
        "zero zero three point",
        "zero zero three point zero zero",
        "zero zero three point one zero",
        "zero zero thirty point one zero",
        "zero zero thirty point one zero one zero",
    ]
    for text_in, text_out in zip(in_, out):
        assert _num2words(text_in, ignore_zeros=False) == text_out


def test___normalize_text_from_pattern__money():
    """Test `__normalize_text_from_pattern` for number cases involving money."""
    in_ = [
        "You can earn from $5 hundred to $5 million.",
        "How to turn $200 into a grand.",
        "It is estimated that by 2010, $1.2 trillion will be spent on education reform.",
        "Inside the drawer was a fake passport and $15,000 cash.",
        "All this for only $9.99 a month!",
        "And it's only €1.01 for general admission and €3.00 for the weekend pass.",
        "£212,512 was spent on management last year",
        "The team will accrue NZ$150M to NZ$180M this month.",
        "$5.",
        "You won $5 zillion. Wow. Thanks.",
        "This guide will tell you how to retire on $5 Million "
        "for the rest of your life, guaranteed.",
    ]
    out = [
        "You can earn from five hundred dollars to five million dollars.",
        "How to turn two hundred dollars into a grand.",
        "It is estimated that by 2010, one point two trillion dollars will be spent on education "
        "reform.",
        "Inside the drawer was a fake passport and fifteen thousand dollars cash.",
        "All this for only nine dollars and ninety-nine cents a month!",
        "And it's only one euro and one cent for general admission and three euros for the "
        "weekend pass.",
        "two hundred and twelve thousand, five hundred and twelve pounds was spent on management "
        "last year",
        "The team will accrue one hundred and fifty million New Zealand dollars to one hundred "
        "and eighty million New Zealand dollars this month.",
        "five dollars.",
        "You won five zillion dollars. Wow. Thanks.",
        "This guide will tell you how to retire on five Million dollars "
        "for the rest of your life, guaranteed.",
    ]
    assert_normalized(in_, out, RegExPatterns.MONEY, _normalize_money)


def test___normalize_text_from_pattern__ordinals():
    """Test `__normalize_text_from_pattern` for number cases involving ordinals."""
    in_ = [
        "On November 22nd, 1963 President John F. Kennedy arrived.",
        "Jesus of Prague, the 13th-century Cathedral of St. Matthew",
        "what he wanted for his 15th birthday.",
        "be recycled, while the 3rd booster - travelling much too fast",
        "celebrate the 400th anniversary of Rembrandt’s birth?",
        "1127 days, from June 25th 1950 through July 27th 1953",
        "In the early-20th century, people like Ordway Tead,",
        "2000 (MM) was a century leap year starting on Saturday of the Gregorian calendar, the "
        "2000th year of the Common Era (CE) and Anno Domini (AD) designations, the 1000th and "
        "last year of the 2nd millennium, the 100th and last year of the 20th century, and the "
        "1st year of the 2000s decade.",
    ]
    out = [
        "On November twenty-second, 1963 President John F. Kennedy arrived.",
        "Jesus of Prague, the thirteenth-century Cathedral of St. Matthew",
        "what he wanted for his fifteenth birthday.",
        "be recycled, while the third booster - travelling much too fast",
        "celebrate the four hundredth anniversary of Rembrandt’s birth?",
        "1127 days, from June twenty-fifth 1950 through July twenty-seventh 1953",
        "In the early-twentieth century, people like Ordway Tead,",
        "2000 (MM) was a century leap year starting on Saturday of the Gregorian calendar, the "
        "two thousandth year of the Common Era (CE) and Anno Domini (AD) designations, the one "
        "thousandth and last year of the second millennium, the one hundredth and last year of the "
        "twentieth century, and the first year of the 2000s decade.",
    ]
    assert_normalized(in_, out, RegExPatterns.ORDINALS, _normalize_ordinals)


def test___normalize_text_from_pattern__times():
    """Test `_normalize_text_from_pattern` for number cases involving times."""
    in_ = [
        "Today at 12:44 PM",
        "set session times for 7:00 p.m. but show up hours later",
        "The Love Monkeys from 8:15 to midnight!",
        'Proverbs 16:21 includes: "The wise of heart is called perceptive.',
        "and then he naps at 12:15 every afternoon",
        "They should be arriving between 9:45 and 11:00 tomorrow night!",
    ]
    out = [
        "Today at twelve forty-four PM",
        "set session times for seven oh clock PM but show up hours later",
        "The Love Monkeys from eight fifteen to midnight!",
        'Proverbs sixteen twenty-one includes: "The wise of heart is called perceptive.',
        "and then he naps at twelve fifteen every afternoon",
        "They should be arriving between nine forty-five and eleven oh clock tomorrow night!",
    ]
    assert_normalized(in_, out, RegExPatterns.TIMES, _normalize_times, o_clock="oh clock")


def test___normalize_text_from_pattern__abbreviated_times():
    """Test `_normalize_text_from_pattern` for number cases involving times."""
    in_ = ["Don't forget to feed him at 9AM, 1 p.m., and 5 pm!"]
    out = ["Don't forget to feed him at nine AM, one PM, and five PM!"]
    assert_normalized(in_, out, RegExPatterns.ABBREVIATED_TIMES, _normalize_abbreviated_times)


def test_reg_ex_patterns_phone_numbers():
    """Test `RegExPatterns.PHONE_NUMBERS` against a number of cases."""
    # NOTE: These test cases were adapted from here:
    # https://stackoverflow.com/questions/16699007/regular-expression-to-match-standard-10-digit-phone-number
    match = RegExPatterns.PHONE_NUMBERS.fullmatch
    assert match("18005551234")
    assert match("1 800 555 1234")
    assert match("+1 800 555-1234")
    assert match("+86 800 555 1234")
    assert match("1-800-555-1234")
    assert match("1 (800) 555-1234")
    assert match("(800)555-1234")
    assert match("(800) 555-1234")
    assert match("(800)5551234")
    assert match("800-555-1234")
    assert match("800.555.1234")
    assert match("555-6482")  # No area code
    assert match("180055512345")  # Too many digits
    assert match("86 800 555 1212")  # Non-NA country code doesn't have +
    assert not match("800 555 1234x5678")  # Extension not supported
    assert not match("8005551234 x5678")  # Extension not supported
    assert not match("1 800 5555 1234")  # Prefix code too long
    assert not match("+1 800 555x1234")  # Invalid delimiter
    assert not match("+867 800 555 1234")  # Country code too long
    assert not match("1-800-555-1234p")  # Invalid character
    assert not match("1 (800)  555-1234")  # Too many spaces
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
    assert match("1800-XFINITY")
    assert match("1(800)Xfinity")
    assert match("1-800-ski-green")
    assert match("1-800-4MY-HOME")
    assert not match("1800XFINITY")  # No delimiter between Area Code and Line Number
    assert not match("1-800XFINITY")


def test___normalize_text_from_pattern__phone_numbers():
    """Test `_normalize_text_from_pattern` for number cases involving phone numbers."""
    in_ = [
        "123-456-7890",
        "(123) 456-7890",
        "123 456 7890",
        "1.123.456.7890",
        "+91 (123) 456-7890",
        "Contact Arizona tourism at 1-800-925-6689 or www.ArizonaGuide.com.",
        "Call Jeannie at 555-9875 to start looking",
        "go to crystalcruises.com or call 1 800 340 1300.",
        "Give us a call at 555-6482.",
        "Call telecharge today at 212-947-8844. Some restrictions apply.",
        "Call 1-800-SANDALS now and save 50%.",
        "representative at 1-888-Comcast for further assistance.",
        "Largest Hotel Chain, call 1-800-Western today.",
        "Discover New York. Call 1-800-I love NY.",
        "For more information, call 1-800-ski-green. That‘s 1-800-ski-green.",
        "your travel agent, or call 1-800 USVI-INFO.",
        "personal escape packages call 1-800-ASK-4-SPA or visit our website",
    ]
    out = [
        "one two three, four five six, seven eight nine zero",
        "one two three, four five six, seven eight nine zero",
        "one two three, four five six, seven eight nine zero",
        "one, one two three, four five six, seven eight nine zero",
        "nine one, one two three, four five six, seven eight nine zero",
        "Contact Arizona tourism at one, eight hundred, nine two five, six six eight nine "
        "or www.ArizonaGuide.com.",
        "Call Jeannie at five five five, nine eight seven five to start looking",
        "go to crystalcruises.com or call one, eight hundred, three four zero, one three "
        "zero zero.",
        "Give us a call at five five five, six four eight two.",
        "Call telecharge today at two one two, nine four seven, eight eight four four. "
        "Some restrictions apply.",
        "Call one, eight hundred, SANDALS now and save 50%.",
        "representative at one, eight eight eight, Comcast for further assistance.",
        "Largest Hotel Chain, call one, eight hundred, Western today.",
        "Discover New York. Call one, eight hundred, I love NY.",
        "For more information, call one, eight hundred, ski, green. That‘s one, eight hundred, "
        "ski, green.",
        "your travel agent, or call one, eight hundred, USVI, INFO.",
        "personal escape packages call one, eight hundred, ASK, four, SPA or visit our website",
    ]
    patterns = [RegExPatterns.PHONE_NUMBERS, RegExPatterns.TOLL_FREE_PHONE_NUMBERS]
    assert_normalized(in_, out, patterns, _normalize_phone_numbers)


def test___normalize_text_from_pattern__years():
    """Test `_normalize_text_from_pattern` for number cases involving years.

    TODO: With more robust year searches, include additional test cases.
    """
    in_ = [
        "Monday, March 16, 1908.",
        "It is 1959: in this remote corner of India.",
        "Twenty years ago, in 1990, Declan Treacy invented",
        "A 2004 CIBC survey suggests",
        "with a run of hit songs during the mid-1960’s, including “Surfin’ U.S.A.” (1963)",
        "If you find a 1776-1976 quarter",
    ]
    out = [
        "Monday, March 16, nineteen oh-eight.",
        "It is nineteen fifty-nine: in this remote corner of India.",
        "Twenty years ago, in nineteen ninety, Declan Treacy invented",
        "A two thousand and four CIBC survey suggests",
        "with a run of hit songs during the mid-nineteen sixty’s, including “Surfin’ U.S.A.” "
        "(nineteen sixty-three)",
        "If you find a seventeen seventy-six to nineteen seventy-six quarter",
    ]
    assert_normalized(in_, out, RegExPatterns.YEARS, _normalize_years)


def test___normalize_text_from_pattern__decades():
    """Test `_normalize_text_from_pattern` for number cases involving decades."""
    in_ = [
        'knowledge worker" in the 1950s. He described how',
        "leaders in the mid-1980s.",
        "In the '80s there was a strong underground influence",
        "--most in their 50s and 60s--",
        "In the 2020s, there has been an increase in online sales",
        "Corporate scandals in the earlier 2000s",
    ]
    out = [
        'knowledge worker" in the nineteen fifties. He described how',
        "leaders in the mid-nineteen eighties.",
        "In the eighties there was a strong underground influence",
        "--most in their fifties and sixties--",
        "In the twenty twenties, there has been an increase in online sales",
        "Corporate scandals in the earlier two thousands",
    ]
    assert_normalized(in_, out, RegExPatterns.DECADES, _normalize_decades)


def test___normalize_text_from_pattern__ranges():
    """Test `_normalize_text_from_pattern` for number cases involving ranges."""
    in_ = [
        "children from grades 6-12, and compared",
        "An estimated 30-40 million Americans bought tickets",
        "that you should only use 1-2 search terms",
        "Children ages 8-12 require tickets; children ages 3-7 do not.",
        "This may mean an even 50-50 split",
    ]
    out = [
        "children from grades six to twelve, and compared",
        "An estimated thirty to forty million Americans bought tickets",
        "that you should only use one to two search terms",
        "Children ages eight to twelve require tickets; children ages three to seven do not.",
        "This may mean an even fifty fifty split",
    ]
    assert_normalized(in_, out, RegExPatterns.NUMBER_RANGES, _normalize_number_ranges)


def test___normalize_text_from_pattern__percents():
    """Test `_normalize_text_from_pattern` for number cases involving percents."""
    in_ = [
        "Over 88% of smokers start young",
        "Book now and save 50%.",
        "Corps makes up only 10.8% of the total Department",
        "since most are only 2-3% over the amount you will be earning",
        "by as much as 15–25% every time cumulative",
    ]
    out = [
        "Over eighty-eight percent of smokers start young",
        "Book now and save fifty percent.",
        "Corps makes up only ten point eight percent of the total Department",
        "since most are only two to three percent over the amount you will be earning",
        "by as much as fifteen to twenty-five percent every time cumulative",
    ]
    assert_normalized(in_, out, RegExPatterns.PERCENTS, _normalize_percents)


def test___normalize_text_from_pattern__number_signs():
    """Test `_normalize_text_from_pattern` for number cases involving the number sign (#)."""
    in_ = [
        "#1: The aesthetic dimension of human experience",
        "#2: Green spaces are easily taken for granted",
        "Section #3: In the next few minutes, you will learn...",
        "They were ranked #4 on Vault's twenty eighteen list",
    ]
    out = [
        "number one: The aesthetic dimension of human experience",
        "number two: Green spaces are easily taken for granted",
        "Section number three: In the next few minutes, you will learn...",
        "They were ranked number four on Vault's twenty eighteen list",
    ]
    assert_normalized(in_, out, RegExPatterns.NUMBER_SIGNS, _normalize_number_signs)


def test___normalize_text_from_pattern__urls():
    """Test `_normalize_text_from_pattern` for number cases involving urls."""
    in_ = [
        "staging.wellsaidlabs.com",
        "you just need to visit www.ArizonaGuide.com to see for yourself",
        "find us at Dianetics.org",
        "For inquiries, please email info@website.com and include your order number.",
        "and many more activities are listed at visitdallas.com/things-to-do",
        "www.gov.nf.ca/tourism",
        "https://www.foxwoods.com",
        "http://www.example.com:80/path/to/myfile.html?key1=value1#SomewhereInTheDocument",
    ]
    out = [
        "staging dot wellsaidlabs dot com",
        "you just need to visit w w w dot ArizonaGuide dot com to see for yourself",
        "find us at Dianetics dot org",
        "For inquiries, please email info at website dot com and include your order number.",
        "and many more activities are listed at visitdallas dot com slash things dash to dash do",
        "w w w dot gov dot nf dot ca slash tourism",
        "h t t p s colon slash slash w w w dot foxwoods dot com",
        "h t t p colon slash slash w w w dot example dot com colon eight zero slash path slash to "
        "slash myfile dot html question mark key one equals value one hash "
        "SomewhereInTheDocument",
    ]
    assert_normalized(in_, out, RegExPatterns.URLS, _normalize_urls)


def test___normalize_text_from_pattern__acronyms():
    """Test `_normalize_text_from_pattern` for number cases involving initialisms,
    word acronyms, and pseudo-blends."""
    in_ = [
        "The mission of the NAACP is to ensure the political, educational, social, and economic "
        "equality of rights of all persons and to eliminate race-based discrimination.",
        "The NCAA tournament ended last night in a thrilling upset from the underdog team.",
        "Please R.S.V.P. ASAP, thank you!",
        "investing in the NASDAQ is a WYSIWYG situation, if you will",
        "you can always call AAA after pulling your car safely to the side of the road",
        "she was so athletic, she could have played in the WNBA, the LPGA, the NPF, and the NWSL "
        "-- at the same time!",
        "The United States of America (U.S.A. or USA), commonly known as the United States (U.S. "
        "or US) or America, is a country primarily located in North America.",
        # TODO: This test case requires sentence context.
        # 'The phrase "United States" was originally plural in American usage. It described a '
        # 'collection of states—e.g., "the United States are." The singular form became popular '
        # "after the end of the Civil War and is now standard usage in the U.S. A citizen of the "
        # 'United States is an "American". "United States", "American" and "U.S." refer to the '
        # 'country adjectivally ("American values", "U.S. forces"). In English, the word '
        # '"American" rarely refers to topics or subjects not directly connected with the '
        # "United States.",
    ]
    out = [
        "The mission of the N double A C P is to ensure the political, educational, social, and "
        "economic equality of rights of all persons and to eliminate race-based discrimination.",
        "The N C double A tournament ended last night in a thrilling upset from the underdog team.",
        "Please RSVP ASAP, thank you!",
        "investing in the Nazdack is a wizzy wig situation, if you will",
        "you can always call Triple A after pulling your car safely to the side of the road",
        "she was so athletic, she could have played in the WNBA, the LPGA, the NPF, and the NWSL "
        "-- at the same time!",
        # 'The phrase "United States" was originally plural in American usage. It described a '
        # 'collection of states—EG, "the United States are." The singular form became popular '
        # "after the end of the Civil War and is now standard usage in the US. A citizen of the "
        # 'United States is an "American". "United States", "American" and "US" refer to the '
        # 'country adjectivally ("American values", "US forces"). In English, the word "American" '
        # "rarely refers to topics or subjects not directly connected with the United States.",
    ]
    assert_normalized(in_, out, RegExPatterns.ACRONYMS, _normalize_acronyms)


def testtest___normalize_text_from_pattern__abbreviations():
    """Test `normalize_measurement_abbreviations` for numeral cases involving abbreviated units."""
    in_ = [
        "We request 48-kHz or 44.1Hz audio for 120min.",
        "The vehicle measures 6ft by 10.5ft and moves at around 65mph.",
        "and it can carry 3,000 fl oz and maintain temperatures below 30°F!",
        "toddlers betwen the 2yo and 3yo age range tend to consume 1,200g of food per day",
        "use 3,450.6Ω of resistance for any load above 28kg ",
        "and you should expect temperatures at around -20°F, ±5°.",
        "the lot size is 18.7km³ ",
        "For example, if the velocity of a particle moving in a straight line changes uniformly "
        "(at a constant rate of change) from 2 m/s to 5 m/s over one second, then its constant "
        "acceleration is 3 m/s².",
    ]
    out = [
        "We request forty-eight kilohertz or forty-four point one hertz audio for one hundred and "
        "twenty minutes.",
        "The vehicle measures six feet by ten point five feet and moves at around sixty-five "
        "miles per hour.",
        "and it can carry three thousand fluid ounces and maintain temperatures below thirty "
        "degrees Fahrenheit!",
        "toddlers betwen the two year-old and three year-old age range tend to consume one "
        "thousand, two hundred grams of food per day",
        "use three thousand, four hundred and fifty point six ohms of resistance for any load "
        "above twenty-eight kilograms ",
        "and you should expect temperatures at around minus twenty degrees Fahrenheit, plus or "
        "minus five degrees.",
        "the lot size is eighteen point seven cubic kilometers ",
        "For example, if the velocity of a particle moving in a straight line changes uniformly "
        "(at a constant rate of change) from two meters per second to five meters per second over "
        "one second, then its constant acceleration is three meters per second squared.",
    ]

    assert_normalized(
        in_, out, RegExPatterns.MEASUREMENT_ABBREVIATIONS, _normalize_measurement_abbreviations
    )


def test___normalize_text_from_pattern__generic_digit():
    """Test `_normalize_text_from_pattern` for leftover, standalone numeral cases."""
    in_ = [
        "it had been 4,879 days and took 53 boats to relocate all 235 residents just 1.2 miles "
        "south",
        "find us between the hours of 2 and 7 every afternoon.",
        "2.3 revolutions in 10 minutes",
    ]
    out = [
        "it had been four thousand, eight hundred and seventy-nine days and took fifty-three "
        "boats to relocate all two hundred and thirty-five residents just one point two miles south",
        "find us between the hours of two and seven every afternoon.",
        "two point three revolutions in ten minutes",
    ]
    # assert_normalized(in_, out, RegExPatterns.GENERIC_DIGITS, _normalize_generic_digit)


def test___normalize_text_from_pattern__fraction():
    """Test `_normalize_text_from_pattern` for fraction cases."""
    in_ = [
        "Turn air on by turning isolation valve switch 1/4 turn counter-clockwise. Retrieve "
        "Keys from lotto Box.",
        "Evelyn is 59 1/2 years old.",
        "It was 37 3/4 years ago...",
    ]
    out = [
        "it had been four thousand, eight hundred and seventy-nine days and took fifty-three "
        "boats to relocate all two hundred and thirty-five residents just one point two miles south",
        "find us between the hours of two and seven every afternoon.",
        "two point three revolutions in ten minutes",
    ]
    # assert_normalized(in_, out, RegExPatterns.FRACTIONS, _normalize_fraction)


def test__normalize_title_abbreviations():
    in_ = [
        "Let's meet with Dr. Ruth Flintstone later today",
        "Mrs. Robinson will join us later",
        "Rev. Silvester Beaman offered a benediction at the inauguration of President Biden.",
        "William Simmons Sr. and Billy Simmons Jr. arrived late.",
        "I didn't see Capt. Clark at the ceremony",
        # "Mr. and Mrs. Frizzle are out for the day.",    # "Mr." not captured by spaCy
    ]
    out = [
        "Let's meet with Doctor Ruth Flintstone later today",
        "Missus Robinson will join us later",
        "Reverend Silvester Beaman offered a benediction at the inauguration of President Biden.",
        "William Simmons Senior and Billy Simmons Junior arrived late.",
        "I didn't see Captain Clark at the ceremony",
        # "Mister and Missus Frizzle are out for the day.",
    ]
    for text_in, text_out in zip(in_, out):
        assert _normalize_title_abbreviations(str(text_in)) == text_out


def test__normalize_roman_numerals():
    """Test `normalize_roman_numerals`."""
    in_ = [
        "the detonation over Nagasaki - ended World War II. Yet the shocking human effects soon ",
        "from Auburn’s Cam Newton, Baylor’s Robert Griffin III and the Aggies Johnny Football",
        "home to such English royalty as King Edward I, Henry V and Henry VIII.",
        "with experimental oxygen bottles from Camp IV high on Mount Everest.",
        "the Marqués de San Miguel de Aguayo proposed to King Philip V, of Spain that 400 families",
    ]
    out = [
        "the detonation over Nagasaki - ended World War Two. Yet the shocking human effects soon ",
        "from Auburn’s Cam Newton, Baylor’s Robert Griffin the Third and the Aggies Johnny "
        "Football",
        "home to such English royalty as King Edward the First, Henry the Fifth and Henry the Eighth.",
        "with experimental oxygen bottles from Camp Four high on Mount Everest.",
        "the Marqués de San Miguel de Aguayo proposed to King Philip the Fifth, of Spain that "
        "400 families",
    ]
    for text_in, text_out in zip(in_, out):

        assert _normalize_roman_numerals(str(text_in)) == text_out


def test_normalize_text():
    """Basic integration test for testing text normalization for text containing multiple non-standard-word cases."""
    in_ = [
        "Do you want to learn how to turn $200 into a grand!? Call 1-800-MONEY-4-ME now or visit www.money4me.com/now!",
        "It is estimated that by 2010, $1.2 trillion will be spent on education reform for 250,000,000 students",
        "The 60s saw a 24% rise in political engagement among the 30-40 year old age group",
        "How is the NCAA managing the fall out from yesterday's 7:00 game? Tune in to News at 5 later tonight!",
        "November 27th, 1934 came with the discovery of a diamond that was 48-kt!",
        "7:25AM. Run 13mi, eat 2,000cal, nap for 10-15 minutes, and eat dinner with Dr. Amelia Fern at 7:00 tonight.",
        "Growth Plan: #1: The WSL team will accrue NZ$150M this month, #2: we will continue contributing $1K in the 1st and 2nd quarters, #3: we will double that by 2024",
        "It took 3 years, but 2018 marked the beginning of a new AAA era when Dr. Ruth Kinton became the 1st woman to take over the prestigious position of C.E.O.",
    ]
    out = [
        "Do you want to learn how to turn two hundred dollars into a grand!? Call one, eight hundred, MONEY, four, ME now or visit w w w dot money four me dot com slash now!",
        "It is estimated that by twenty ten, one point two trillion dollars will be spent on education reform for two hundred and fifty million students",
        "The sixties saw a twenty-four percent rise in political engagement among the thirty to forty year old age group",
        "How is the N C double A managing the fall out from yesterday's seven o'clock game? Tune in to News at five later tonight!",
        "November twenty-seventh, nineteen thirty-four came with the discovery of a diamond that was forty-eight karats!",
        "seven twenty-five AM. Run thirteen miles, eat two thousand calories, nap for ten to fifteen minutes, and eat dinner with Doctor Amelia Fern at seven o'clock tonight.",
        "Growth Plan: number one: The WSL team will accrue one hundred and fifty million New Zealand dollars this month, number two: we will continue contributing one thousand dollars in the first and second quarters, number three: we will double that by twenty twenty-four",
        "It took three years, but twenty eighteen marked the beginning of a new Triple A era when Doctor Ruth Kinton became the first woman to take over the prestigious position of CEO.",
    ]

    for text_in, text_out in zip(in_, out):
        assert normalize_text(str(text_in)) == text_out


# def test_normalize_text__user_clips():
#     """Basic integration test for testing text normalization for text containing multiple non-standard-word cases."""
#     CLIPS = "disk/data/1000_Recent_Clips.csv"
#     user_clips = pd.read_csv(CLIPS, names=["text", "created_at"])

#     for i, row in user_clips.iterrows():
#         in_ = row["text"]
#         out = normalize_text(str(in_))

#         row["normalized"] = out if out is not in_ else ""

#     user_clips.to_csv("disk/data/1000_Recent_Clips__normalized.csv")

# for text_in, text_out in zip(in_, out):
#     assert normalize_text(str(text_in)) == text_out
