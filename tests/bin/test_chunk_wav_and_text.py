import os
import pathlib

import pytest

from src.audio import read_audio
from src.bin.chunk_wav_and_text import align_wav_and_scripts
from src.bin.chunk_wav_and_text import Alignment
from src.bin.chunk_wav_and_text import allow_substitution
from src.bin.chunk_wav_and_text import average_silence_delimiter
from src.bin.chunk_wav_and_text import chunk_alignments
from src.bin.chunk_wav_and_text import get_samples_per_character
from src.bin.chunk_wav_and_text import main
from src.bin.chunk_wav_and_text import natural_keys
from src.bin.chunk_wav_and_text import Nonalignment
from src.bin.chunk_wav_and_text import normalize_text
from src.bin.chunk_wav_and_text import remove_punctuation
from src.bin.chunk_wav_and_text import review_chunk_alignments
from src.bin.chunk_wav_and_text import samples_to_seconds
from src.bin.chunk_wav_and_text import seconds_to_samples

CHUNKS_PATH = pathlib.Path('tests/_test_data/bin/test_chunk_wav_and_text/lj_speech_chunks')


def MockAlignment(start_text, end_text):
    return Alignment(
        start_audio=start_text, end_audio=end_text, start_text=start_text, end_text=end_text)


def test_get_samples_per_character():
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=100, start_text=0, end_text=100)) == 1.0
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=100, start_text=0, end_text=10)) == 10.0
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=10, start_text=0, end_text=100)) == 0.1


def test_natural_keys():
    list_ = ['name 0', 'name 1', 'name 10', 'name 11']
    assert sorted(list_, key=natural_keys) == list_


def test_seconds_to_samples():
    assert seconds_to_samples('0.1s', 24000) == 2400


def test_samples_to_seconds():
    assert samples_to_seconds(2400, 24000) == 0.1


def test_allow_substitution():
    # Regression tests on real use-cases.
    assert not allow_substitution('information."Public', 'Public')
    assert not allow_substitution('information."Public', 'information.')
    assert not allow_substitution('tax-deferred', 'deferred')
    assert not allow_substitution('tax-deferred', 'tax')
    assert allow_substitution('A', 'a')
    assert allow_substitution('In', 'An')
    assert allow_substitution('poietes', 'Pleiades')
    assert allow_substitution('Luchins', 'lucian\'s')
    assert allow_substitution('"raise"', 'raised')


def test_remove_punctuation():
    assert remove_punctuation('123 abc !.?') == '123 abc '


def test_review_chunk_alignments():
    with pytest.raises(ValueError):
        review_chunk_alignments('abcdefg', [{
            'text': (0, 3),
            'audio': (0, 3)
        }, {
            'text': (0, 3),
            'audio': (0, 3)
        }])

    with pytest.raises(ValueError):
        review_chunk_alignments('abcdefg', [{
            'text': (0, 3),
            'audio': (0, 0)
        }, {
            'text': (4, 7),
            'audio': (4, 7)
        }])

    with pytest.raises(ValueError):
        review_chunk_alignments('abcdefg', [{
            'text': (0, 0),
            'audio': (0, 3)
        }, {
            'text': (4, 7),
            'audio': (4, 7)
        }])

    assert review_chunk_alignments('abcdefghi', [{
        'text': (0, 3),
        'audio': (0, 3)
    }, {
        'text': (4, 7),
        'audio': (4, 7)
    }]) == ['d', 'hi']


def test_main__no_csv(capsys):
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        main(
            'tests/_test_data/bin/test_chunk_wav_and_text/rate(lj_speech,24000).wav',
            str(CHUNKS_PATH),
            max_chunk_seconds=2)
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_2.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_3.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_4.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_5.wav').exists()
    assert (CHUNKS_PATH / 'metadata.csv').exists()
    assert (CHUNKS_PATH / ('stderr.%s.log' % os.getpid())).exists()
    assert (CHUNKS_PATH / ('stdout.%s.log' % os.getpid())).exists()

    assert ((CHUNKS_PATH / 'metadata.csv').read_text().strip() == """Content,WAV Filename
The examination and testimony,"rate(lj_speech,24000)/script_0_chunk_0.wav"
of the experts,"rate(lj_speech,24000)/script_0_chunk_1.wav"
enabled the commission,"rate(lj_speech,24000)/script_0_chunk_2.wav"
to conclude,"rate(lj_speech,24000)/script_0_chunk_3.wav"
that five shots may,"rate(lj_speech,24000)/script_0_chunk_4.wav"
have been fired.,"rate(lj_speech,24000)/script_0_chunk_5.wav" """.strip())


def test_main__normalize_audio(capsys):
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        main(
            'tests/_test_data/bin/test_chunk_wav_and_text/lj_speech.wav',
            str(CHUNKS_PATH),
            csv_pattern='tests/_test_data/bin/test_chunk_wav_and_text/lj_speech.csv',
            max_chunk_seconds=2)

    with pytest.raises(AssertionError):  # The original audio file was not supported.
        read_audio('tests/_test_data/bin/test_chunk_wav_and_text/lj_speech.wav')

    # Ensure chunks are supported by this repository.
    read_audio(CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav')
    read_audio(CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_2.wav')


def test_main(capsys):
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        main(
            'tests/_test_data/bin/test_chunk_wav_and_text/rate(lj_speech,24000).wav',
            str(CHUNKS_PATH),
            csv_pattern='tests/_test_data/bin/test_chunk_wav_and_text/lj_speech.csv',
            max_chunk_seconds=2)
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_2.wav').exists()
    assert (CHUNKS_PATH / 'metadata.csv').exists()
    assert (CHUNKS_PATH / ('stderr.%s.log' % os.getpid())).exists()
    assert (CHUNKS_PATH / ('stdout.%s.log' % os.getpid())).exists()

    assert ((CHUNKS_PATH / 'metadata.csv').read_text().strip() == """Content,WAV Filename
The examination and,"rate(lj_speech,24000)/script_0_chunk_0.wav"
of the experts enabled,"rate(lj_speech,24000)/script_0_chunk_1.wav"
The SUBSTITUTE_WORD to conclude,"rate(lj_speech,24000)/script_1_chunk_0.wav"
that ADDED_WORD five sAhots may,"rate(lj_speech,24000)/script_1_chunk_1.wav"
have been fired.,"rate(lj_speech,24000)/script_1_chunk_2.wav" """.strip())


def test_average_silence_delimiter():
    next_alignment = MockAlignment(start_text=4, end_text=5)
    alignment = MockAlignment(start_text=1, end_text=3)
    nonalignment = Nonalignment(start_text=1, end_text=3)

    assert average_silence_delimiter(alignment, next_alignment) == 1.0
    assert average_silence_delimiter(nonalignment, next_alignment) == -1.0


def test_align_wav_and_scripts():
    sst_results = [
        {
            'transcript':
                'Script 1.',
            'confidence':
                1.0,
            'words': [
                {
                    'startTime': '0.0s',
                    'endTime': '%ds' % len('Script'),
                    'word': 'Script',
                },
                {
                    'startTime': '%ds' % len('Script '),
                    'endTime': '%ds' % len('Script 1.'),
                    'word': '1.',
                },
                {
                    'startTime': '%ds' % len('Script 1.'),
                    'endTime': '%ds' % len('Script 1.'),
                    'word': '',
                },
            ]
        },
        {
            'transcript':
                'Script.',
            'confidence':
                1.0,
            'words': [{
                'startTime': '%ds' % len('Script 1. '),
                'endTime': '%ds' % len('Script 1. Script 2.'),
                'word': 'Script.',
            }]
        },
    ]
    sample_rate = 44100
    scripts = ['Script 1.', 'Script 2.']
    output = align_wav_and_scripts(sst_results, scripts, sample_rate=sample_rate)
    expected_output = [
        [
            Alignment(
                start_text=0,
                end_text=len('Script'),
                start_audio=0,
                end_audio=len('Script') * sample_rate),
            Alignment(
                start_text=len('Script '),
                end_text=len('Script 1.'),
                start_audio=len('Script ') * sample_rate,
                end_audio=len('Script 1.') * sample_rate),
            Nonalignment(start_text=len('Script 1.'), end_text=len('Script 1.')),
        ],
        [
            Alignment(
                start_text=0,
                end_text=len('Script'),
                start_audio=len('Script 1. ') * sample_rate,
                end_audio=len('Script 1. Script 2.') * sample_rate),
            Nonalignment(start_text=len('Script '), end_text=len('Script 2.'))
        ],
    ]

    for script_alignments, expected_script_alignments in zip(output, expected_output):
        for alignment, expected_alignment in zip(script_alignments, expected_script_alignments):
            assert alignment.start_text == expected_alignment.start_text
            assert alignment.end_text == expected_alignment.end_text
            if isinstance(alignment, Alignment):
                assert alignment.start_audio == expected_alignment.start_audio
                assert alignment.end_audio == expected_alignment.end_audio


def test_chunk_alignments():
    script = 'A basic test.'
    token_test = MockAlignment(start_text=len('A basic '), end_text=len('A basic test.'))
    token_basic = MockAlignment(start_text=len('A '), end_text=len('A basic'))
    token_a = MockAlignment(start_text=0, end_text=len('A'))

    # Extra silence
    token_test = token_test._replace(start_audio=len('A basic ') + 1)
    token_test = token_test._replace(end_audio=len('A basic test.') + 1)

    alignment = [token_a, token_basic, token_test]
    spans = chunk_alignments(alignment, script, max_chunk_seconds=len('A basic'), sample_rate=1)
    unaligned_substrings = review_chunk_alignments(script, spans)

    assert unaligned_substrings == []
    assert len(spans) == 2
    assert script[slice(*spans[0]['text'])] == 'A basic'
    assert script[slice(*spans[1]['text'])] == 'test.'


def test_chunk_alignments_unable_to_cut():
    script = 'Yup, a basic test.'
    token_test = MockAlignment(start_text=len('Yup, a basic '), end_text=len('Yup, a basic test.'))
    token_basic = Nonalignment(start_text=len('Yup, a '), end_text=len('Yup, a basic'))
    token_a = Nonalignment(start_text=len('Yup, '), end_text=len('Yup, a'))
    token_yup = MockAlignment(start_text=0, end_text=len('Yup,'))

    alignment = [token_yup, token_a, token_basic, token_test]
    spans = chunk_alignments(alignment, script, max_chunk_seconds=4, sample_rate=1)
    unaligned_substrings = review_chunk_alignments(script, spans)

    assert unaligned_substrings == [script]
    assert len(spans) == 0


def test_chunk_alignments_unable_to_cut_two():
    script = 'Yup, a basic test today.'
    token_today = MockAlignment(
        start_text=len('Yup, a basic test '), end_text=len('Yup, a basic test today.'))
    token_test = MockAlignment(start_text=len('Yup, a basic '), end_text=len('Yup, a basic test'))
    token_basic = Nonalignment(start_text=len('Yup, a '), end_text=len('Yup, a basic'))
    token_a = MockAlignment(start_text=len('Yup, '), end_text=len('Yup, a'))
    token_yup = MockAlignment(start_text=0, end_text=len('Yup,'))

    alignment = [token_yup, token_a, token_basic, token_test, token_today]
    spans = chunk_alignments(alignment, script, max_chunk_seconds=6, sample_rate=1)
    unaligned_substrings = review_chunk_alignments(script, spans)

    assert unaligned_substrings == ['a basic test']
    assert len(spans) == 2


def test_chunk_alignments_no_delimitations():
    script = 'A basic test.'
    token_test = MockAlignment(start_text=len('A basic '), end_text=len('A basic test.'))
    token_basic = MockAlignment(start_text=len('A '), end_text=len('A basic'))
    token_a = MockAlignment(start_text=0, end_text=len('A'))

    alignment = [token_a, token_basic, token_test]
    spans = chunk_alignments(
        alignment,
        script,
        max_chunk_seconds=len('A basic'),
        sample_rate=1,
        delimiter=lambda *args: 0)
    unaligned_substrings = review_chunk_alignments(script, spans)

    assert unaligned_substrings == []
    assert len(spans) == 2
    assert script[slice(*spans[0]['text'])] == 'A basic'
    assert script[slice(*spans[1]['text'])] == 'test.'


def test_normalize_text():
    # Regression tests on real use-cases.
    assert (normalize_text('Dexpan® Non-Explosive Controlled Demolition Agent ') ==
            'Dexpan Non-Explosive Controlled Demolition Agent')
    assert normalize_text('to the environment.The draft') == 'to the environment. The draft'
    assert (normalize_text('V. D.. Retrieved March 30, 2016, from medscape.com</ref>') ==
            'V. D.. Retrieved March 30, 2016, from medscape.com')
    assert (normalize_text('•\tIt can act as buffer, or temporary storage area,') ==
            '*  It can act as buffer, or temporary storage area,')
