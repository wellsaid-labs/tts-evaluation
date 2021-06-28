import re
import typing

from lib.environment import AnsiCodes
from lib.utils import flatten_2d
from run.data.sync_script_with_audio import (
    ScriptToken,
    SttToken,
    _fix_alignments,
    _get_speech_context,
    _remove_punctuation,
    format_differences,
    format_ratio,
    is_sound_alike,
)


def test__remove_punctuation():
    """Test `_remove_punctuation` removes punctuation and fixes spacing."""
    assert _remove_punctuation("123 abc !.?") == "123 abc"
    assert _remove_punctuation("Hello. You've") == "Hello You ve"
    assert _remove_punctuation("Hello. \n\fYou've") == "Hello You ve"


def test_is_sound_alike():
    """Test `is_sound_alike` if determines if two phrase sound-alike."""
    assert not is_sound_alike("Hello", "Hi")
    assert is_sound_alike("financingA", "financing a")
    assert is_sound_alike("twentieth", "20th")
    assert is_sound_alike("screen introduction", "screen--Introduction,")
    assert is_sound_alike("Hello-you've", "Hello. You've")
    assert is_sound_alike("'is...'", "is")
    assert is_sound_alike("Pre-game", "pregame")
    assert is_sound_alike("Dreamfields.", "dream Fields")
    assert is_sound_alike(" — ", "")

    # NOTE: These cases are not supported, yet,
    assert not is_sound_alike("fifteen", "15")
    assert not is_sound_alike("forty", "40")


def test_format_ratio():
    """Test `format_ratio` formats a ratio."""
    assert format_ratio(1, 100) == "1.0% [1 of 100]"


def test__get_speech_context():
    """Test `_get_speech_context` creates a speech context with limited length phrases."""
    assert set(_get_speech_context("a b c d e f g h i j", 5, 0.0).phrases) == set(  # type: ignore
        ["a b c", "d e f", "g h i", "j"]
    )


def test__get_speech_context__overlap():
    """Test `_get_speech_context` creates a speech context with limited length overlapping
    phrases."""
    assert set(_get_speech_context("a b c d e f g h i j", 5, 0.2).phrases) == set(  # type: ignore
        ["a b c", "c d e", "e f g", "g h i", "i j"]
    )


def test__get_speech_context__continuous():
    """Test `_get_speech_context` ignores long continuous sequences of text longer than the
    `max_phrase_length`."""
    assert set(_get_speech_context("abcdef g h i j", 5, 0.2).phrases) == set(  # type: ignore
        ["g h i", "i j"]
    )


def _get_script_tokens(scripts: typing.List[str]) -> typing.List[ScriptToken]:
    """Create a list of `ScriptToken`s for testing."""
    tokens = [
        [ScriptToken(i, m.group(0), (m.start(), m.end())) for m in re.finditer(r"\S+", script)]
        for i, script in enumerate(scripts)
    ]
    return flatten_2d(tokens)


def _get_stt_tokens(stt_results: typing.List[str]) -> typing.List[SttToken]:
    """Create a list of `SttToken`s for testing."""
    tokens = [
        [
            SttToken(m.group(0), (float("nan"), float("nan")), (m.start(), m.end()))
            for m in re.finditer(r"\S+", stt_result)
        ]
        for stt_result in stt_results
    ]
    return flatten_2d(tokens)


def test_format_differences():
    """Test `format_differences` formats the alignment in a readable way."""
    scripts = ["Home to more than 36 HUNDRED native trees."]
    stt_results = ["Home to more than 3,600 native trees."]
    alignments = [(0, 0), (1, 1), (2, 2), (3, 3), (6, 5), (7, 6)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "Home to more than",
            AnsiCodes.RED + '--- " 36 HUNDRED "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "3,600"' + AnsiCodes.RESET_ALL,
            "native trees.",
        ]
    )


def test_format_differences__one_word():
    """Test that `format_differences` is able to handle one aligned word."""
    scripts = ["Home"]
    stt_results = ["Home"]
    alignments = [(0, 0)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == ""


def test_format_differences__one_word_not_aligned():
    """Test that `format_differences` is able to handle one unaligned word."""
    scripts = ["Home"]
    stt_results = ["Tom"]
    alignments: typing.List[typing.Tuple[int, int]] = []
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "\n" + AnsiCodes.RED + '--- "Home"' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "Tom"' + AnsiCodes.RESET_ALL + "\n",
        ]
    )


def test_format_differences__extra_words():
    """Test that `format_differences` is able to handle extra words on the edges and middle."""
    scripts = ["to than 36 HUNDRED native"]
    stt_results = ["Home to more than 3,600 native trees."]
    alignments = [(0, 1), (1, 3), (4, 5)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "\n" + AnsiCodes.RED + '--- ""' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "Home"' + AnsiCodes.RESET_ALL,
            "to",
            AnsiCodes.RED + '--- " "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "more"' + AnsiCodes.RESET_ALL,
            "than",
            AnsiCodes.RED + '--- " 36 HUNDRED "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "3,600"' + AnsiCodes.RESET_ALL,
            "native",
            AnsiCodes.RED + '--- ""' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "trees."' + AnsiCodes.RESET_ALL + "\n",
        ]
    )


def test_format_differences__skip_script():
    """Test that `format_differences` is able to handle a perfect alignment."""
    scripts = ["I love short sentences."]
    stt_results = ["I love short sentences."]
    alignments = [(0, 0), (1, 1), (2, 2), (3, 3)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == ""


def test_format_differences__two_scripts__ends_unaligned():
    """Test that `format_differences` is able to handle multiple scripts with the ends not
    aligned."""
    scripts = ["I love short sentences.", "I am here."]
    stt_results = ["You love distort sentences.", "I am there."]
    alignments = [(1, 1), (3, 3), (4, 4), (5, 5)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "\n" + AnsiCodes.RED + '--- "I "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "You"' + AnsiCodes.RESET_ALL,
            "love",
            AnsiCodes.RED + '--- " short "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "distort"' + AnsiCodes.RESET_ALL,
            "sentences.",
            "=" * 100,
            "I am",
            AnsiCodes.RED + '--- " here."' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "there."' + AnsiCodes.RESET_ALL + "\n",
        ]
    )


def test_format_differences__all_unaligned():
    """Test that `format_differences` is able to handle a complete unalignment between multiple
    scripts.
    """
    scripts = ["I love short sentences.", "I am here.", "I am here again."]
    stt_results = ["You distort attendance.", "You are there."]
    alignments: typing.List[typing.Tuple[int, int]] = []
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "\n" + AnsiCodes.RED + '--- "I love short sentences."' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "I am here."' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "I am here again."' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN
            + '+++ "You distort attendance. You are there."'
            + AnsiCodes.RESET_ALL
            + "\n",
        ]
    )


def test_format_differences__unaligned_and_aligned():
    """Test that `format_differences` is able to transition between unaligned and aligned
    scripts."""
    scripts = ["a b c", "a b c", "a b c", "a b c", "a b c"]
    stt_results = ["x y z", "x y z", "a b c", "x y z", "x y z"]
    alignments = [(6, 6), (7, 7), (8, 8)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "\n" + AnsiCodes.RED + '--- "a b c"' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "a b c"' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- ""' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "x y z x y z"' + AnsiCodes.RESET_ALL,
            "a b c",
            AnsiCodes.RED + '--- ""' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "a b c"' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "a b c"' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "x y z x y z"' + AnsiCodes.RESET_ALL + "\n",
        ]
    )


def test_format_differences__unalignment_between_scripts():
    """Test that `format_differences` is able to handle unalignment between two scripts."""
    scripts = ["I love short sentences.", "I am here."]
    stt_results = ["I love short attendance.", "You are here."]
    alignments = [(0, 0), (1, 1), (2, 2), (6, 6)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    assert "".join(format_differences(scripts, alignments, tokens, stt_tokens)) == "\n".join(
        [
            "I love short",
            AnsiCodes.RED + '--- " sentences."' + AnsiCodes.RESET_ALL,
            "=" * 100,
            AnsiCodes.RED + '--- "I am "' + AnsiCodes.RESET_ALL,
            AnsiCodes.GREEN + '+++ "attendance. You are"' + AnsiCodes.RESET_ALL,
            "here.",
        ]
    )


def test__fix_alignments():
    """Test `_fix_alignments` can alignment multiple tokens to multiple tokens."""
    scripts = ["a b c d e"]
    stt_results = ["a b c d e"]
    alignments = [(0, 0), (4, 4)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens
    )
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ["a", "b c d", "e"]
    assert [s.text for s in updated_tokens] == ["a", "b c d", "e"]


def test__fix_alignments__edges():
    """Test `_fix_alignments` can alignment multiple tokens to multiple tokens on the edges."""
    scripts = ["a b c d e"]
    stt_results = ["a b c d e"]
    alignments = [(2, 2)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens
    )
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ["a b", "c", "d e"]
    assert [s.text for s in updated_tokens] == ["a b", "c", "d e"]


def test__fix_alignments__stt_edges():
    """Test `_fix_alignments` can alignment multiple tokens to multiple tokens on the edges of a
    speech-to-text result."""
    scripts = ["a-b c d-e"]
    stt_results = ["a b c d e"]
    alignments = [(1, 2)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    _, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens
    )
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_stt_tokens] == ["a b", "c", "d e"]


def test__fix_alignments__script_edges():
    """Test `_fix_alignments` can alignment multiple tokens to multiple tokens on the edges of a
    script."""
    scripts = ["a b c d e"]
    stt_results = ["a-b c d-e"]
    alignments = [(2, 1)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, _, updated_alignments = _fix_alignments(scripts, alignments, tokens, stt_tokens)
    assert updated_alignments == [(0, 0), (1, 1), (2, 2)]
    assert [s.text for s in updated_tokens] == ["a b", "c", "d e"]


def test__fix_alignments__between_scripts():
    """Test that `_fix_alignments` doesn't align tokens between two scripts."""
    scripts = ["a b c", "d e"]
    stt_results = ["a b c", "d e"]
    alignments = [(0, 0), (1, 1), (4, 4)]
    tokens = _get_script_tokens(scripts)
    stt_tokens = _get_stt_tokens(stt_results)
    updated_tokens, updated_stt_tokens, updated_alignments = _fix_alignments(
        scripts, alignments, tokens, stt_tokens
    )
    assert updated_alignments == [(0, 0), (1, 1), (4, 4)]
    assert [s.text for s in updated_tokens] == ["a", "b", "c", "d", "e"]
    assert [s.text for s in updated_stt_tokens] == ["a", "b", "c", "d", "e"]
