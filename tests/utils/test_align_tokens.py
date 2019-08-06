from src.utils.align_tokens import align_tokens
from src.utils.align_tokens import format_alignment


def _align_and_format(tokens, other, **kwargs):
    cost, alignment = align_tokens(tokens, other, **kwargs)
    return format_alignment(tokens, other, alignment)


def test_align_tokens():
    # Basic cases
    assert align_tokens('', '')[0] == 0
    assert align_tokens('a', '')[0] == 1
    assert align_tokens('', 'a')[0] == 1
    assert align_tokens('abc', '')[0] == 3
    assert align_tokens('', 'abc')[0] == 3
    assert align_tokens('', 'abc', window_length=1)[0] == 3

    # Should just add "a" to the beginning.
    assert align_tokens('abc', 'bc', window_length=1)[0] == 1
    assert align_tokens('abc', 'bc', allow_substitution=lambda a, b: False)[0] == 5
    assert (_align_and_format('abc', 'bc') == (
        'a b c',
        '  b c',
    ))

    # Should just add I to the beginning.
    assert align_tokens('islander', 'slander')[0] == 1
    assert align_tokens('islander', 'slander', window_length=1)[0] == 1
    assert (_align_and_format('islander', 'slander') == (
        'i s l a n d e r',
        '  s l a n d e r',
    ))

    # Needs to substitute M by K, T by M and add an A to the end
    assert align_tokens('mart', 'karma')[0] == 3

    # Substitute K by S, E by I and insert G at the end.
    assert align_tokens('kitten', 'sitting')[0] == 3

    # Should add 4 letters FOOT at the beginning.
    assert align_tokens('ball', 'football')[0] == 4

    assert align_tokens('ball', 'football', window_length=1)[0] == 7
    assert (_align_and_format('ball', 'football', window_length=1) == (
        'b a l       l',
        'f o o t b a l',
    ))
    assert align_tokens('ball', 'football', window_length=2)[0] == 6
    assert (_align_and_format('ball', 'football', window_length=2) == (
        'b a         l l',
        'f o o t b a l l',
    ))
    assert align_tokens('ball', 'football', window_length=3)[0] == 4
    assert (_align_and_format('ball', 'football', window_length=3) == (
        '        b a l l',
        'f o o t b a l l',
    ))

    # Should delete 4 letters FOOT at the beginning.
    assert align_tokens('football', 'foot')[0] == 4

    # Needs to substitute the first 5 chars: INTEN by EXECU
    assert align_tokens('intention', 'execution')[0] == 5

    # Subtitution of words
    assert align_tokens(['Hey', 'There'], ['Hey', 'There'])[0] == 0
    assert align_tokens(['Hey', 'There'], ['Hi', 'There'])[0] == 2
    assert align_tokens(['Hey', 'There'], ['Hi', 'The'])[0] == 4

    # Deletion of word
    assert align_tokens(['Hey', 'There', 'You'], ['Hey', ',', 'There'])[0] == 4
    assert (_align_and_format(['Hey', 'There', 'You'], ['Hey', ',', 'There']) == (
        'Hey   There',
        'Hey , There',
    ))
