import random
import re

from generate_audio._test_cases.parity_test_cases import QUESTIONS, INITIALISMS
from generate_audio._test_cases.slurring import SLURRING
from generate_audio._test_cases.test_cases import (
    ABBREVIATIONS_WITH_VOWELS,
    HARD_SCRIPTS,
    HARD_SCRIPTS_2,
    QUESTIONS_WITH_UPWARD_INFLECTION,
    QUESTIONS_WITH_VARIED_INFLECTION,
    SLOW_SCRIPTS,
    VARIOUS_INITIALISMS,
)
from generate_audio._test_cases.v11_test_cases import (
    DIFFICULT_USER_INITIALISMS,
    DIFFICULT_USER_QUESTIONS,
    DIFFICULT_USER_URLS,
)

REPORT_CARD_TEST_CASES = (
    SLURRING
    + DIFFICULT_USER_QUESTIONS
    + DIFFICULT_USER_URLS
    + DIFFICULT_USER_INITIALISMS
    + ABBREVIATIONS_WITH_VOWELS
    + SLOW_SCRIPTS
    + HARD_SCRIPTS
    + HARD_SCRIPTS_2
    + QUESTIONS
    + QUESTIONS_WITH_VARIED_INFLECTION
    + QUESTIONS_WITH_UPWARD_INFLECTION
    + VARIOUS_INITIALISMS
)


def remove_xml(string):
    """Remove xml surrounding original word from string"""
    xml = re.compile(r"(<[^<>]+>)")
    xml_parts = xml.findall(string)
    for p in xml_parts:
        string = string.replace(p, "")
    return string


REPORT_CARD_TEST_CASES = list(set(remove_xml(i) for i in REPORT_CARD_TEST_CASES))
random.shuffle(REPORT_CARD_TEST_CASES)
