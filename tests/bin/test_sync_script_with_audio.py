import re

from src.bin.sync_script_with_audio import _get_speech_context
from src.bin.sync_script_with_audio import format_differences
from src.bin.sync_script_with_audio import format_ratio
from src.bin.sync_script_with_audio import natural_keys
from src.bin.sync_script_with_audio import remove_punctuation
from src.bin.sync_script_with_audio import ScriptToken
from src.bin.sync_script_with_audio import SttToken
from src.environment import COLORS
from src.utils import flatten


def test_natural_keys():
    list_ = ['name 0', 'name 1', 'name 10', 'name 11']
    assert sorted(list_, key=natural_keys) == list_


def test_remove_punctuation():
    assert remove_punctuation('123 abc !.?') == '123 abc'


def test_format_ratio():
    assert format_ratio(1, 100) == '1.000000% [1 of 100]'


def test__get_speech_context():
    assert set(_get_speech_context('a b c d e f g h i j',
                                   5).phrases) == set(['a b c', 'd e f', 'g h i', 'j'])


def _get_script_tokens(scripts):
    return flatten(
        [[ScriptToken(m.group(0), m.start(), m.end(), i)
          for m in re.finditer(r'\S+', script)]
         for i, script in enumerate(scripts)])


def _get_stt_tokens(stt_results):
    return flatten([[
        SttToken(m.group(0), float('nan'), float('nan')) for m in re.finditer(r'\S+', stt_result)
    ] for stt_result in stt_results])


def test_format_differences():
    scripts = ['Home to more than 36 HUNDRED native trees.']
    stt_results = ['Home to more than 3,600 native trees.']
    alignments = [(0, 0), (1, 1), (2, 2), (3, 3), (6, 5), (7, 6)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert format_differences(scripts, alignments, tokens, stt_tokens) == '\n'.join([
        'Home to more than',
        COLORS['red'] + '--- "36 HUNDRED "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "3,600"' + COLORS['reset_all'],
        'native trees.',
    ])


def test_format_differences__skip_script():
    """ Test that `format_differences` does not print scripts with no differences. """
    scripts = ['I love short sentences.']
    stt_results = ['I love short sentences.']
    alignments = [(0, 0), (1, 1), (2, 2), (3, 3)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert format_differences(scripts, alignments, tokens, stt_tokens) == ''


def test_format_differences__two_scripts__ends_unaligned():
    """ Test that `format_differences` is able to handle multiple scripts with the ends not aligned.
    """
    scripts = ['I love short sentences.', 'I am here.']
    stt_results = ['You love distort sentences.', 'I am there.']
    alignments = [(1, 1), (3, 3), (4, 4), (5, 5)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert format_differences(scripts, alignments, tokens, stt_tokens) == '\n'.join([
        COLORS['red'] + '--- "I "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "You"' + COLORS['reset_all'],
        'love',
        COLORS['red'] + '--- "short "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "distort"' + COLORS['reset_all'],
        'sentences.',
        '=' * 100,
        'I am',
        COLORS['red'] + '--- "here."' + COLORS['reset_all'],
        COLORS['green'] + '+++ "there."' + COLORS['reset_all'],
    ])


def test_format_differences__all_unaligned():
    """ Test that `format_differences` is able to handle a complete unalignment between multiple
    scripts.
    """
    scripts = ['I love short sentences.', 'I am here.', 'I am here again.']
    stt_results = ['You distort attendance.', 'You are there.']
    alignments = []
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert format_differences(scripts, alignments, tokens, stt_tokens) == '\n'.join([
        '-' * 100,
        COLORS['red'] + '--- "I love short sentences."' + COLORS['reset_all'],
        COLORS['red'] + '--- "I am here."' + COLORS['reset_all'],
        COLORS['red'] + '--- "I am here again."' + COLORS['reset_all'],
        COLORS['green'] + '+++ "You distort attendance. You are there."' + COLORS['reset_all'],
        '-' * 100,
    ])


def test_format_differences__unalignment_between_scripts():
    """ Test that `format_differences` is able to handle unalignment between two scripts.
    """
    scripts = ['I love short sentences.', 'I am here.']
    stt_results = ['I love short attendance.', 'You are here.']
    alignments = [(0, 0), (1, 1), (2, 2), (6, 6)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert format_differences(scripts, alignments, tokens, stt_tokens) == '\n'.join([
        'I love short',
        '-' * 100,
        COLORS['red'] + '--- "sentences."' + COLORS['reset_all'],
        COLORS['red'] + '--- "I am "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "attendance. You are"' + COLORS['reset_all'],
        '-' * 100,
        'here.',
    ])
