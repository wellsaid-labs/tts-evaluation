import mock
import pathlib

import pytest

from src.bin.chunk_wav_and_text import Alignment
from src.bin.chunk_wav_and_text import Nonalignment
from src.bin.chunk_wav_and_text import average_silence_delimiter
from src.bin.chunk_wav_and_text import _chunks_to_spans
from src.bin.chunk_wav_and_text import _cut_text
from src.bin.chunk_wav_and_text import _find
from src.bin.chunk_wav_and_text import align_wav_and_scripts
from src.bin.chunk_wav_and_text import chunk_alignments
from src.bin.chunk_wav_and_text import GENTLE_SUCCESS_CASE

GENTLE_NOT_FOUND_IN_AUDIO_CASE = 'not-found-in-audio'


class MockAlignment(Alignment):

    def __init__(self, start_text, end_text, next_alignment=None, last_alignment=None):
        super().__init__(
            start_audio=start_text,
            end_audio=end_text,
            start_text=start_text,
            end_text=end_text,
            next_alignment=next_alignment,
            last_alignment=last_alignment)


def test_average_silence_delimiter():
    next_alignment = MockAlignment(start_text=4, end_text=5)
    alignment = MockAlignment(start_text=1, end_text=3, next_alignment=next_alignment)
    nonalignment = Nonalignment(start_text=1, end_text=3, next_alignment=next_alignment)

    assert average_silence_delimiter(alignment) == 1.0
    with pytest.raises(ValueError):
        average_silence_delimiter(next_alignment)
    assert average_silence_delimiter(nonalignment) == -1.0


def test__chunks_to_spans():
    script = 'A basic test.'
    token_test = MockAlignment(start_text=len('A basic '), end_text=len('A basic test'))
    token_basic = MockAlignment(
        start_text=len('A '), end_text=len('A basic'), next_alignment=token_test)
    token_a = MockAlignment(start_text=0, end_text=len('A'), next_alignment=token_basic)
    token_test.last_alignment = token_basic
    token_basic.last_alignment = token_a

    chunks = [[token_a, token_basic, token_test]]
    spans = _chunks_to_spans(script, chunks)
    assert len(spans) == 1
    assert spans[0]['text'] == (0, len(script))

    # Test additional punctuation
    script = 'A basic test..'
    spans = _chunks_to_spans(script, chunks)
    assert len(spans) == 1
    assert spans[0]['text'] == (0, len(script))


def test__chunks_to_spans_last_alignment():
    script = 'A. Basic test.'
    token_test = MockAlignment(start_text=len('A. Basic '), end_text=len('A. Basic test'))
    token_basic = MockAlignment(
        start_text=len('A. '), end_text=len('A. Basic'), next_alignment=token_test)
    token_a = Nonalignment(start_text=0, end_text=len('A'), next_alignment=token_basic)
    token_test.last_alignment = token_basic
    token_basic.last_alignment = token_a

    spans = _chunks_to_spans(script, [[token_basic, token_test]])
    assert len(spans) == 1
    assert spans[0]['text'] == (2, len(script))


def test__find():
    assert _find('Hihihhi', 'hi') == [2, 5]


def test__cut_text():
    assert _cut_text('test." Test', inclusive_stop_tokens=['."', '.']) == len('test."')
    assert _cut_text('test. " Test', inclusive_stop_tokens=['."', '.']) == len('test.')
    assert _cut_text('test.." Test', inclusive_stop_tokens=['."', '.']) == len('test.."')
    assert _cut_text('test(Test', exclusive_stop_tokens=['(']) == len('test')
    assert _cut_text(
        'test.(Test', exclusive_stop_tokens=['('], inclusive_stop_tokens=['.']) == len('test.')
    assert _cut_text(
        'test.(.Test', exclusive_stop_tokens=['('], inclusive_stop_tokens=['.']) == len('test.')
    assert _cut_text('test "Test', exclusive_stop_tokens=[' "']) == len('test')
    assert _cut_text('test Test', inclusive_stop_tokens=['."', '.']) is None


@mock.patch('src.bin.chunk_wav_and_text._request_gentle')
def test_align_wav_and_scripts(_request_gentle_mock):
    _request_gentle_mock.return_value = {
        'transcript':
            'Script 1. Script 2.',
        'words': [{
            'alignedWord': 'script',
            'word': 'Script',
            'case': GENTLE_SUCCESS_CASE,
            'start': 0,
            'end': len('Script'),
            'startOffset': 0,
            'endOffset': len('Script'),
        }, {
            'word': '1',
            'case': GENTLE_NOT_FOUND_IN_AUDIO_CASE,
            'start': len('Script '),
            'end': len('Script 1'),
            'startOffset': len('Script '),
            'endOffset': len('Script 1'),
        }, {
            'alignedWord': 'script',
            'word': 'Script',
            'case': GENTLE_SUCCESS_CASE,
            'start': len('Script 1. '),
            'end': len('Script 1. Script'),
            'startOffset': len('Script 1. '),
            'endOffset': len('Script 1. Script'),
        }, {
            'word': '2',
            'case': GENTLE_NOT_FOUND_IN_AUDIO_CASE,
            'start': len('Script 1. Script '),
            'end': len('Script 1. Script 2'),
            'startOffset': len('Script 1. Script '),
            'endOffset': len('Script 1. Script 2'),
        }]
    }
    wav_path = pathlib.Path('.')
    gentle_cache_directory = pathlib.Path('.')
    sample_rate = 44100
    scripts = ['Script 1.', 'Script 2.']
    output = align_wav_and_scripts(wav_path, scripts, gentle_cache_directory, sample_rate)
    expected_output = [[
        Alignment(start_text=0, end_text=len('Script'), start_audio=0, end_audio=len('Script')),
        Nonalignment(start_text=len('Script '), end_text=len('Script 1'))
    ], [
        Alignment(
            start_text=0,
            end_text=len('Script'),
            start_audio=len('Script 1. '),
            end_audio=len('Script 1. Script')),
        Nonalignment(start_text=len('Script '), end_text=len('Script 2'))
    ]]
    for script_alignments, expected_script_alignments in zip(output, expected_output):
        for alignment, expected_alignment in zip(script_alignments, expected_script_alignments):
            assert alignment.start_text == expected_alignment.start_text
            assert alignment.end_text == expected_alignment.end_text
            if isinstance(alignment, Alignment):
                assert alignment.start_audio == expected_alignment.start_audio * sample_rate
                assert alignment.end_audio == expected_alignment.end_audio * sample_rate


def test_chunk_alignments():
    script = 'A basic test.'
    token_test = MockAlignment(start_text=len('A basic '), end_text=len('A basic test'))
    token_basic = MockAlignment(
        start_text=len('A '), end_text=len('A basic'), next_alignment=token_test)
    token_a = MockAlignment(start_text=0, end_text=len('A'), next_alignment=token_basic)
    token_test.last_alignment = token_basic
    token_basic.last_alignment = token_a

    # Extra silence
    token_test.start_audio = len('A basic ') + 1
    token_test.end_audio = len('A basic test') + 1

    alignment = [token_a, token_basic, token_test]
    spans, unaligned_substrings = chunk_alignments(
        alignment, script, max_chunk_length=len('A basic'))

    assert unaligned_substrings == []
    assert len(spans) == 2
    assert script[slice(*spans[0]['text'])] == 'A basic '
    assert script[slice(*spans[1]['text'])] == 'test.'


def test_chunk_alignments_unable_to_cut():
    script = 'Yup, a basic test.'
    token_test = MockAlignment(start_text=len('Yup, a basic '), end_text=len('Yup, a basic test'))
    token_basic = Nonalignment(
        start_text=len('Yup, a '), end_text=len('Yup, a basic'), next_alignment=token_test)
    token_a = Nonalignment(
        start_text=len('Yup, '), end_text=len('Yup, a'), next_alignment=token_basic)
    token_yup = MockAlignment(start_text=0, end_text=len('Yup'), next_alignment=token_a)
    token_test.last_alignment = token_basic
    token_basic.last_alignment = token_a
    token_a.last_alignment = token_yup

    alignment = [token_yup, token_a, token_basic, token_test]
    spans, unaligned_substrings = chunk_alignments(alignment, script, max_chunk_length=4)

    assert unaligned_substrings == [script]
    assert len(spans) == 0


def test_chunk_alignments_unable_to_cut_two():
    script = 'Yup, a basic test today.'
    token_today = MockAlignment(
        start_text=len('Yup, a basic test '), end_text=len('Yup, a basic test today'))
    token_test = MockAlignment(
        start_text=len('Yup, a basic '),
        end_text=len('Yup, a basic test'),
        next_alignment=token_today)
    token_basic = Nonalignment(
        start_text=len('Yup, a '), end_text=len('Yup, a basic'), next_alignment=token_test)
    token_a = MockAlignment(
        start_text=len('Yup, '), end_text=len('Yup, a'), next_alignment=token_basic)
    token_yup = MockAlignment(start_text=0, end_text=len('Yup'), next_alignment=token_a)
    token_today.last_alignment = token_test
    token_test.last_alignment = token_basic
    token_basic.last_alignment = token_a
    token_a.last_alignment = token_yup

    alignment = [token_yup, token_a, token_basic, token_test, token_today]
    spans, unaligned_substrings = chunk_alignments(alignment, script, max_chunk_length=6)

    assert unaligned_substrings == [' a basic test ']
    assert len(spans) == 2
