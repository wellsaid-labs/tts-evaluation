from copy import deepcopy

import pytest

from src.audio import read_audio
from src.bin.chunk_wav_and_text import align_wav
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
from src.environment import ROOT_PATH
from src.environment import TEST_DATA_PATH

TEST_DATA_PATH_LOCAL = TEST_DATA_PATH / 'bin' / 'test_chunk_wav_and_text'

CHUNKS_PATH = TEST_DATA_PATH_LOCAL / 'lj_speech_chunks'

MOCK_SST_RESULTS = [{
    'words': [{
        'startTime': '0.400s',
        'endTime': '1s',
        'word': 'Welcome'
    }, {
        'startTime': '1s',
        'endTime': '1.100s',
        'word': 'to'
    }, {
        'startTime': '1.100s',
        'endTime': '1.700s',
        'word': 'Morton'
    }, {
        'startTime': '1.700s',
        'endTime': '2.600s',
        'word': 'Arboretum'
    }]
}]

MOCK_SCRIPTS = ['Welcome to', 'Morton Arboretum']


def test_natural_keys():
    list_ = ['name 0', 'name 1', 'name 10', 'name 11']
    assert sorted(list_, key=natural_keys) == list_


def test_seconds_to_samples():
    assert seconds_to_samples('0.1s', 24000) == 2400


def test_samples_to_seconds():
    assert samples_to_seconds(2400, 24000) == 0.1


def test_get_samples_per_character():
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=100, start_text=0, end_text=100)) == 1.0
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=100, start_text=0, end_text=10)) == 10.0
    assert get_samples_per_character(
        Alignment(start_audio=0, end_audio=10, start_text=0, end_text=100)) == 0.1


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


def MockAlignment(start_text, end_text):
    return Alignment(
        start_audio=start_text, end_audio=end_text, start_text=start_text, end_text=end_text)


def test_average_silence_delimiter():
    next_alignment = MockAlignment(start_text=4, end_text=5)
    alignment = MockAlignment(start_text=1, end_text=3)
    nonalignment = Nonalignment(start_text=1, end_text=3)

    assert average_silence_delimiter(alignment, next_alignment) == 1.0
    assert average_silence_delimiter(nonalignment, next_alignment) == -1.0


def assert_equal_alignments(alignments, expected_alignments):
    for alignment, expected_alignment in zip(alignments, expected_alignments):
        assert isinstance(alignment, Alignment) or isinstance(alignment, Nonalignment)

        assert alignment.start_text == expected_alignment.start_text
        assert alignment.end_text == expected_alignment.end_text

        if isinstance(alignment, Alignment):
            assert alignment.start_audio == expected_alignment.start_audio
            assert alignment.end_audio == expected_alignment.end_audio


MockAlignmentTwo = lambda a, b, c, d: Alignment(a, b, seconds_to_samples(c), seconds_to_samples(d))


def test_align_wav():
    results = align_wav(MOCK_SST_RESULTS, 0.02)

    assert results[0][0] == 'Welcome to Morton Arboretum'
    assert_equal_alignments(results[1][0], [
        MockAlignmentTwo(0, len('Welcome'), '0.4s', '1.0s'),
        MockAlignmentTwo(len('Welcome '), len('Welcome to'), '1.0s', '1.1s'),
        MockAlignmentTwo(len('Welcome to '), len('Welcome to Morton'), '1.1s', '1.7s'),
        MockAlignmentTwo(
            len('Welcome to Morton '), len('Welcome to Morton Arboretum'), '1.7s', '2.6s'),
    ])


def test_align_wav__min_seconds_per_character():
    results = align_wav(MOCK_SST_RESULTS, 0.075)

    assert results[0][0] == 'Welcome to Morton Arboretum'
    assert_equal_alignments(results[1][0], [
        MockAlignmentTwo(0, len('Welcome'), '0.4s', '1.0s'),
        Nonalignment(len('Welcome '), len('Welcome to')),
        MockAlignmentTwo(len('Welcome to '), len('Welcome to Morton'), '1.1s', '1.7s'),
        MockAlignmentTwo(
            len('Welcome to Morton '), len('Welcome to Morton Arboretum'), '1.7s', '2.6s'),
    ])


def test_align_wav_and_scripts():
    scripts_alignments = align_wav_and_scripts(MOCK_SST_RESULTS, MOCK_SCRIPTS, 0.02)
    assert len(scripts_alignments) == 2
    assert_equal_alignments(scripts_alignments[0], [
        MockAlignmentTwo(0, len('Welcome'), '0.4s', '1.0s'),
        MockAlignmentTwo(len('Welcome '), len('Welcome to'), '1.0s', '1.1s'),
    ])
    assert_equal_alignments(scripts_alignments[1], [
        MockAlignmentTwo(0, len('Morton'), '1.1s', '1.7s'),
        MockAlignmentTwo(len('Morton '), len('Morton Arboretum'), '1.7s', '2.6s'),
    ])


def test_align_wav_and_scripts__missing_tokens():
    mock_sst_results = deepcopy(MOCK_SST_RESULTS)
    mock_sst_results[0]['words'].pop(1)
    mock_scripts = ['Welcome to', 'Morton']

    scripts_alignments = align_wav_and_scripts(mock_sst_results, mock_scripts, 0.02)
    assert len(scripts_alignments) == 2
    assert_equal_alignments(scripts_alignments[0], [
        MockAlignmentTwo(0, len('Welcome'), '0.4s', '1.0s'),
        Nonalignment(len('Welcome '), len('Welcome to')),
    ])
    assert_equal_alignments(scripts_alignments[1], [
        MockAlignmentTwo(0, len('Morton'), '1.1s', '1.7s'),
    ])


def test_align_wav_and_scripts__misspellings():
    mock_scripts = ['Welome to', 'Morton Aboretum.']

    scripts_alignments = align_wav_and_scripts(MOCK_SST_RESULTS, mock_scripts, 0.02)
    assert len(scripts_alignments) == 2
    assert_equal_alignments(scripts_alignments[0], [
        MockAlignmentTwo(0, len('Welome'), '0.4s', '1.0s'),
        MockAlignmentTwo(len('Welome '), len('Welome to'), '1.0s', '1.1s'),
    ])
    assert_equal_alignments(scripts_alignments[1], [
        MockAlignmentTwo(0, len('Morton'), '1.1s', '1.7s'),
        MockAlignmentTwo(len('Morton '), len('Morton Aboretum.'), '1.7s', '2.6s'),
    ])


def test_align_wav_and_scripts__min_seconds_per_character():
    scripts_alignments = align_wav_and_scripts(MOCK_SST_RESULTS, MOCK_SCRIPTS, 0.075)
    assert len(scripts_alignments) == 2
    assert_equal_alignments(scripts_alignments[0], [
        MockAlignmentTwo(0, len('Welcome'), '0.4s', '1.0s'),
        Nonalignment(len('Welcome '), len('Welcome to')),
    ])
    assert_equal_alignments(scripts_alignments[1], [
        MockAlignmentTwo(0, len('Morton'), '1.1s', '1.7s'),
        MockAlignmentTwo(len('Morton '), len('Morton Arboretum'), '1.7s', '2.6s'),
    ])


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


def test_main__no_csv(capsys):
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        main(
            str((TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav').relative_to(ROOT_PATH)),
            str(CHUNKS_PATH),
            max_chunk_seconds=2)
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_2.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_3.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_4.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_5.wav').exists()
    assert (CHUNKS_PATH / 'metadata.csv').exists()
    assert len(list(CHUNKS_PATH.glob('*.log'))) == 1

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
            str((TEST_DATA_PATH_LOCAL / 'lj_speech.wav').relative_to(ROOT_PATH)),
            str(CHUNKS_PATH),
            csv_pattern=str((TEST_DATA_PATH_LOCAL / 'lj_speech.csv').relative_to(ROOT_PATH)),
            max_chunk_seconds=2)

    with pytest.raises(AssertionError):  # The original audio file was not supported.
        read_audio(TEST_DATA_PATH_LOCAL / 'lj_speech.wav')

    # Ensure chunks are supported by this repository.
    read_audio(CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav')
    read_audio(CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_2.wav')


def test_main(capsys):
    with capsys.disabled():  # Required for the test to pass (could be a bug with PyTest).
        main(
            str((TEST_DATA_PATH_LOCAL / 'rate(lj_speech,24000).wav').relative_to(ROOT_PATH)),
            str(CHUNKS_PATH),
            csv_pattern=str((TEST_DATA_PATH_LOCAL / 'lj_speech.csv').relative_to(ROOT_PATH)),
            max_chunk_seconds=2)
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_0_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_0.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_1.wav').exists()
    assert (CHUNKS_PATH / 'wavs' / 'rate(lj_speech,24000)' / 'script_1_chunk_2.wav').exists()
    assert (CHUNKS_PATH / 'metadata.csv').exists()
    assert len(list(CHUNKS_PATH.glob('*.log'))) == 1

    assert ((CHUNKS_PATH / 'metadata.csv').read_text().strip() == """Content,WAV Filename
The examination and,"rate(lj_speech,24000)/script_0_chunk_0.wav"
of the experts enabled,"rate(lj_speech,24000)/script_0_chunk_1.wav"
The SUBSTITUTE_WORD to conclude,"rate(lj_speech,24000)/script_1_chunk_0.wav"
that ADDED_WORD five sAhots may,"rate(lj_speech,24000)/script_1_chunk_1.wav"
have been fired.,"rate(lj_speech,24000)/script_1_chunk_2.wav" """.strip())
