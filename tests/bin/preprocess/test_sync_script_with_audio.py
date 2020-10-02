import re

from src.bin.sync_script_with_audio import _fix_alignments
from src.bin.sync_script_with_audio import _get_speech_context
from src.bin.sync_script_with_audio import _normalize_text
from src.bin.sync_script_with_audio import _remove_punctuation
from src.bin.sync_script_with_audio import format_differences
from src.bin.sync_script_with_audio import format_ratio
from src.bin.sync_script_with_audio import ScriptToken
from src.bin.sync_script_with_audio import SttToken
from src.environment import COLORS
from src.utils import flatten


def test__normalize_text():
    assert _normalize_text(' Testing… \f ® ™ — coöperation ') == 'Testing...    - cooperation'


def test__remove_punctuation():
    assert _remove_punctuation('123 abc !.?') == '123 abc'
    assert _remove_punctuation('Hello. You\'ve') == 'Hello You ve'
    assert _remove_punctuation('Hello. \n\fYou\'ve') == 'Hello You ve'


def test_format_ratio():
    assert format_ratio(1, 100) == '1.000000% [1 of 100]'


def test__get_speech_context():
    assert set(_get_speech_context('a b c d e f g h i j', 5,
                                   0.0).phrases) == set(['a b c', 'd e f', 'g h i', 'j'])


def test__get_speech_context__overlap():
    assert set(_get_speech_context('a b c d e f g h i j', 5,
                                   0.2).phrases) == set(['a b c', 'c d e', 'e f g', 'g h i', 'i j'])


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
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        'Home to more than',
        COLORS['red'] + '--- " 36 HUNDRED "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "3,600"' + COLORS['reset_all'],
        'native trees.',
    ])


def test_format_differences__one_word():
    """ Test that `format_differences` is able to handle one aligned word.
    """
    scripts = ['Home']
    stt_results = ['Home']
    alignments = [(0, 0)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == ''


def test_format_differences__one_word_not_aligned():
    """ Test that `format_differences` is able to handle one unaligned word.
    """
    scripts = ['Home']
    stt_results = ['Tom']
    alignments = []
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        '\n' + COLORS['red'] + '--- "Home"' + COLORS['reset_all'],
        COLORS['green'] + '+++ "Tom"' + COLORS['reset_all'] + '\n',
    ])


def test_format_differences__extra_words():
    """ Test that `format_differences` is able to handle extra words on the edges and middle.
    """
    scripts = ['to than 36 HUNDRED native']
    stt_results = ['Home to more than 3,600 native trees.']
    alignments = [(0, 1), (1, 3), (4, 5)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        '\n' + COLORS['red'] + '--- ""' + COLORS['reset_all'],
        COLORS['green'] + '+++ "Home"' + COLORS['reset_all'],
        'to',
        COLORS['red'] + '--- " "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "more"' + COLORS['reset_all'],
        'than',
        COLORS['red'] + '--- " 36 HUNDRED "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "3,600"' + COLORS['reset_all'],
        'native',
        COLORS['red'] + '--- ""' + COLORS['reset_all'],
        COLORS['green'] + '+++ "trees."' + COLORS['reset_all'] + '\n',
    ])


def test_format_differences__skip_script():
    """ Test that `format_differences` is able to handle a perfect alignment.
    """
    scripts = ['I love short sentences.']
    stt_results = ['I love short sentences.']
    alignments = [(0, 0), (1, 1), (2, 2), (3, 3)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == ''


def test_format_differences__two_scripts__ends_unaligned():
    """ Test that `format_differences` is able to handle multiple scripts with the ends not aligned.
    """
    scripts = ['I love short sentences.', 'I am here.']
    stt_results = ['You love distort sentences.', 'I am there.']
    alignments = [(1, 1), (3, 3), (4, 4), (5, 5)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    print(list(format_differences(scripts, alignments, tokens, stt_tokens)))
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        '\n' + COLORS['red'] + '--- "I "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "You"' + COLORS['reset_all'],
        'love',
        COLORS['red'] + '--- " short "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "distort"' + COLORS['reset_all'],
        'sentences.',
        '=' * 100,
        'I am',
        COLORS['red'] + '--- " here."' + COLORS['reset_all'],
        COLORS['green'] + '+++ "there."' + COLORS['reset_all'] + '\n',
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
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        '\n' + COLORS['red'] + '--- "I love short sentences."' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "I am here."' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "I am here again."' + COLORS['reset_all'],
        COLORS['green'] + '+++ "You distort attendance. You are there."' + COLORS['reset_all'] +
        '\n',
    ])


def test_format_differences__unaligned_and_aligned():
    """ Test that `format_differences` is able to transition between unaligned and aligned scripts.
    """
    scripts = ['a b c', 'a b c', 'a b c', 'a b c', 'a b c']
    stt_results = ['x y z', 'x y z', 'a b c', 'x y z', 'x y z']
    alignments = [(6, 6), (7, 7), (8, 8)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        '\n' + COLORS['red'] + '--- "a b c"' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "a b c"' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- ""' + COLORS['reset_all'],
        COLORS['green'] + '+++ "x y z x y z"' + COLORS['reset_all'],
        'a b c',
        COLORS['red'] + '--- ""' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "a b c"' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "a b c"' + COLORS['reset_all'],
        COLORS['green'] + '+++ "x y z x y z"' + COLORS['reset_all'] + '\n',
    ])


def test_format_differences__unalignment_between_scripts():
    """ Test that `format_differences` is able to handle unalignment between two scripts.
    """
    scripts = ['I love short sentences.', 'I am here.']
    stt_results = ['I love short attendance.', 'You are here.']
    alignments = [(0, 0), (1, 1), (2, 2), (6, 6)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert ''.join(format_differences(scripts, alignments, tokens, stt_tokens)) == '\n'.join([
        'I love short',
        COLORS['red'] + '--- " sentences."' + COLORS['reset_all'],
        '=' * 100,
        COLORS['red'] + '--- "I am "' + COLORS['reset_all'],
        COLORS['green'] + '+++ "attendance. You are"' + COLORS['reset_all'],
        'here.',
    ])


def test__fix_alignments():
    scripts = ['a b c d e']
    stt_results = ['a b c d e']
    alignments = [(0, 0), (4, 4)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ['a', 'b c d', 'e']
    assert [s.text for s in updated_tokens] == ['a', 'b c d', 'e']


def test__fix_alignments__edges():
    scripts = ['a b c d e']
    stt_results = ['a b c d e']
    alignments = [(2, 2)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ['a b', 'c', 'd e']
    assert [s.text for s in updated_tokens] == ['a b', 'c', 'd e']


def test__fix_alignments__stt_edges():
    scripts = ['a-b c d-e']
    stt_results = ['a b c d e']
    alignments = [(1, 2)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    _, updated_stt_tokens, updated_alignments = _fix_alignments(scripts, alignments, tokens,
                                                                stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ['a b', 'c', 'd e']


def test__fix_alignments__script_edges():
    scripts = ['a b c d e']
    stt_results = ['a-b c d-e']
    alignments = [(2, 1)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, _, updated_alignments = _fix_alignments(scripts, alignments, tokens, stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_tokens] == ['a b', 'c', 'd e']


def test__fix_alignments__between_scripts():
    """ Test that `_fix_alignments` doesn't align tokens between two scripts. """
    scripts = ['a b c', 'd e']
    stt_results = ['a b c', 'd e']
    alignments = [(0, 0), (1, 1), (4, 4)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (4, 4)]
    assert [s.text for s in updated_tokens] == ['a', 'b', 'c', 'd', 'e']
    assert [s.text for s in updated_stt_tokens] == ['a', 'b', 'c', 'd', 'e']
