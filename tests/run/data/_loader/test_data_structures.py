import functools
import pickle
import typing
from unittest import mock

import numpy as np
import pytest

import lib
from lib.utils import Interval, Timeline
from run.data import _loader
from run.data._loader.data_structures import (
    Alignment,
    IntInt,
    Passage,
    Span,
    Speaker,
    UnprocessedPassage,
    _check_updated_script,
    _filter_non_speech_segments,
    _make_speech_segments,
    _maybe_normalize_vo_script,
    has_a_mistranscription,
    make_passages,
)
from run.data._loader.utils import get_non_speech_segments_and_cache
from tests._utils import TEST_DATA_PATH, make_passage, make_unprocessed_passage

TEST_DATA_LJ = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"


def test__maybe_normalize_vo_script():
    """ Test `_maybe_normalize_vo_script` against some basic cases. """
    normal_script = "abc"
    assert _maybe_normalize_vo_script(normal_script) == normal_script
    script = "áƀć"
    assert _maybe_normalize_vo_script(script) == normal_script


def test__filter_non_speech_segments():
    """Test `_filter_non_speech_segments` against various word alignment and pausing intervals."""
    alignments = ((0, 1), (1, 2), (1.9, 3), (4, 5), (4.8, 5), (5, 6))
    non_speech_segments = {
        # CASE: Pause envelopes 0 alignment(s) and overlaps 0 alignment(s)
        (3.1, 3.9): True,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 1 alignment(s) on the left
        (2.9, 3.9): True,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 1 alignment(s) on the right
        (3.1, 4.1): True,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 2 alignment(s)
        (2.9, 4.1): True,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 2 alignment(s)
        (1.95, 3.1): False,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 3 alignment(s)
        (4.9, 5.1): False,
        # CASE: Pause envelopes 0 alignment(s) and overlaps 3 alignment(s)
        (1.95, 4.1): False,
        # CASE: Pause envelopes 1 alignment(s) and overlaps 0 alignment(s)
        (3.9, 5.1): False,
        # CASE: Pause envelopes 1 alignment(s) and overlaps 1 alignment(s)
        (2.9, 5.1): False,
        # CASE: Pause envelopes 1 alignment(s) and overlaps 2 alignment(s)
        (1.9, 4.1): False,
        # CASE: Pause envelopes 2 alignment(s) and overlaps 0 alignment(s)
        (1.9, 5): False,
        # CASE: 1 alignment(s) envelopes pause
        (4.1, 4.9): False,
        # CASE: 1 alignment(s) envelopes pause and overlaps 1 alignment(s)
        (1.9, 3): False,
        # CASE: 2 alignment(s) envelopes pause
        (1.9, 2): False,
    }
    script = "".join([str(i) for i in range(max(a[1] for a in alignments))])
    passage = make_passage(
        script=script,
        transcript=script,
        alignments=Alignment.stow(
            [Alignment((int(a[0]), int(a[1])), a, (int(a[0]), int(a[1]))) for a in alignments]
        ),
        nonalignments=Alignment.stow([]),
    )
    timeline = Timeline([Interval(a.audio, (i, a)) for i, a in enumerate(passage.alignments)])
    slices = [slice(*k) for k in non_speech_segments.keys()]
    filtered = _filter_non_speech_segments(timeline, slices)
    results = set(((f.start, f.stop) for f in filtered))
    assert results == {k for k, v in non_speech_segments.items() if v}


def test__make_speech_segments():
    """Test `_make_speech_segments` against various word alignment and pausing intervals."""
    script = (
        "The examination and testimony of the experts enabled the Commission to conclude"
        " that five shots may have been fired,"
    )
    audio_alignments = [
        (0, 0.2),  # The
        (0.2, 0.8),  # examination
        (0.9, 1.1),  # and
        (1.1, 1.7),  # testimony
        (1.7, 1.8),  # of
        (1.8, 1.9),  # the
        (1.9, 2.6),  # experts
        (3.0, 3.4),  # enabled
        (3.5, 3.6),  # the
        (3.6, 4.1),  # Commission
        (4.1, 4.2),  # to
        (4.2, 4.9),  # conclude
        (5.3, 5.5),  # that
        (5.5, 5.8),  # five
        (5.8, 6.2),  # shots
        (6.2, 6.4),  # may
        (6.4, 6.6),  # have
        (6.6, 6.8),  # been
        (6.9, 7.4),  # fired,
    ]
    offset = 0
    script_alignments = []
    for split in script.split():
        script_alignments.append((offset, offset + len(split)))
        offset += len(split) + 1
    passage = Passage(
        audio_file=lib.audio.get_audio_metadata(TEST_DATA_LJ),
        speaker=_loader.LINDA_JOHNSON,
        script=script,
        transcript=script,
        alignments=Alignment.stow(
            [Alignment(s, a, s) for a, s in zip(audio_alignments, script_alignments)]
        ),
    )
    speech_segments = (
        passage.span(slice(0, 7), slice(0.0, 2.775)),
        passage.span(slice(7, 12), slice(2.9599583333333332, 4.91)),
        passage.span(slice(12, len(script_alignments)), slice(5.264958333333333, 7.405)),
    )

    timeline = get_non_speech_segments_and_cache(passage.audio_file)
    assert _make_speech_segments(passage, timeline) == speech_segments

    timeline = get_non_speech_segments_and_cache(passage.audio_file, threshold=-1000)
    assert _make_speech_segments(passage, timeline) == (
        passage.span(slice(0, len(script_alignments))),
    )

    timeline = get_non_speech_segments_and_cache(passage.audio_file, threshold=0)
    assert _make_speech_segments(passage, timeline) == (
        passage.span(slice(0, len(script_alignments))),
    )

    timeline = get_non_speech_segments_and_cache(passage.audio_file, threshold=-40)
    assert _make_speech_segments(passage, timeline) == (
        passage.span(slice(0, 5), slice(0.0, 1.77)),
        passage.span(slice(5, 7), slice(1.7949583333333334, 2.62)),
        passage.span(slice(7, 8), slice(2.9649583333333336, 3.42)),
        passage.span(slice(8, 9), slice(3.4599583333333332, 3.55)),
        passage.span(slice(9, 11), slice(3.6149583333333335, 4.195)),
        passage.span(slice(11, 12), slice(4.279958333333333, 4.825)),
        passage.span(slice(12, 13), slice(5.279958333333333, 5.48)),
        passage.span(slice(13, 17), slice(5.549958333333333, 6.5600000000000005)),
        passage.span(slice(17, 18), slice(6.599958333333333, 6.755000000000001)),
        passage.span(slice(18, 19), slice(6.884958333333333, 7.36)),
    )


@mock.patch("run.data._loader.data_structures.logger.error")
def test__check_updated_script(mock_error):
    """ Test `_check_updated_script` against some basic cases. """
    passage = make_unprocessed_passage(script="abc", transcript="abc", alignments=tuple())
    _check_updated_script("", passage, "abc", "abc")
    assert mock_error.called == 0

    passage = make_unprocessed_passage(script="áƀć", transcript="áƀć", alignments=tuple())
    _check_updated_script("", passage, "abc", "abc")
    assert mock_error.called == 1

    with pytest.raises(AssertionError):
        passage = make_unprocessed_passage(script="ab\f", transcript="ab", alignments=tuple())
        _check_updated_script("", passage, "ab", "ab")


def test_passage_span__identity():
    """Test `Passage` and `Span` are the same after a identity operation."""
    audio_path = TEST_DATA_LJ
    metadata = lib.audio.get_audio_metadata(audio_path)
    script = (
        "The examination and testimony of the experts enabled the Commission to conclude "
        "that five shots may have been fired,"
    )
    alignment = Alignment((0, len(script)), (0.0, metadata.length), (0, len(script)))
    passage = Passage(
        audio_file=metadata,
        speaker=_loader.LINDA_JOHNSON,
        script=script,
        transcript=script,
        alignments=Alignment.stow([alignment]),
        other_metadata={"chapter": 37},
    )
    nonalignments = [
        Alignment((0, 0), (0.0, 0.0), (0, 0)),
        Alignment(
            (len(script), len(script)),
            (metadata.length, metadata.length),
            (len(script), len(script)),
        ),
    ]
    object.__setattr__(passage, "nonalignments", Alignment.stow(nonalignments))
    object.__setattr__(passage, "speech_segments", passage.span(slice(0, 1)))
    span = passage[:]
    assert passage.script == span.script
    assert passage.transcript == span.transcript
    assert passage.alignments == span.alignments
    assert passage.speaker == span.speaker
    assert passage.audio_file == span.audio_file
    assert passage.other_metadata == span.other_metadata
    assert passage.aligned_audio_length() == span.audio_length
    assert passage[-1] == span[-1]
    assert passage[0:0] == span[0:-1]
    span.check_invariants()
    passage.check_invariants()
    np.testing.assert_almost_equal(passage.audio(), span.audio())
    pickle.dumps(passage)
    pickle.dumps(span)


_find = lambda a, b: (a.index(b), a.index(b) + 1)


def _make_unprocessed_passage_helper(
    script: str,
    tokens: typing.List[str],
    transcript: str,
    find_transcript: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
    find_script: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
):
    """ Helper function for `test_passage_span__unaligned*`. """
    found = [(find_script(script, t), find_transcript(transcript, t)) for t in tokens]
    return UnprocessedPassage(
        audio_path=TEST_DATA_LJ,
        speaker=Speaker(""),
        script=script,
        transcript=transcript,
        alignments=tuple(Alignment(s, (0.0, 0.0), t) for s, t in found),
    )


def _get_nonaligned(
    span: typing.Union[Span, Passage]
) -> typing.List[typing.Optional[typing.Tuple[str, str, typing.Tuple[float, float]]]]:
    return [
        None if s is None else (s.script, s.transcript, (s.audio_start, s.audio_stop))
        for s in span.nonalignment_spans().spans
    ]


def test_passage_span__nonalignment_spans():
    """Test `Passage` and `Span` get the correct nonalignments under a variety of
    circumstances.
    """
    script = "abcdefghijklmnopqrstuvwxyz"
    make = functools.partial(_make_unprocessed_passage_helper, transcript=script)
    unprocessed_passages = []

    # TEST: Largely no issues, except one in the middle.
    split, script = script[:6], script[6:]
    unprocessed_passages.append(make(split, ["a", "b", "c", "e", "f"]))  # NOTE: split='abcdef'

    # TEST: Right edge has an issue, along with one in the middle.
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["g", "i"]))  # NOTE: split='ghi'

    # TEST: Left edge has an issue.
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["l"]))  # NOTE: split='jkl'

    # TEST: Right edge has an issue.
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["m"]))  # NOTE: split='mno'

    # TEST: Both edges have an issue, and there is no rightward passage.
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["q"]))  # NOTE: split='pqr'

    kwargs = {"script": False, "transcript": True, "audio": True}
    passages = list(make_passages("", [unprocessed_passages], **kwargs))

    a = (0.0, 0.0)
    empty = ("", "", a)
    expected = [empty, empty, empty, ("d", "d", a), empty, empty]
    assert _get_nonaligned(passages[0]) == expected
    assert _get_nonaligned(passages[0][:]) == _get_nonaligned(passages[0])
    assert _get_nonaligned(passages[1]) == [empty, ("h", "h", a), ("", "jk", a)]
    assert _get_nonaligned(passages[1][:]) == _get_nonaligned(passages[1])
    assert _get_nonaligned(passages[2]) == [("jk", "jk", a), empty]
    assert _get_nonaligned(passages[3]) == [empty, ("no", "nop", a)]
    assert _get_nonaligned(passages[4]) == [
        ("p", "nop", a),
        ("r", "rstuvwxyz", (0.0, 7.583958148956299)),
    ]

    # TEST: Test `spans` get the correct span.
    assert _get_nonaligned(passages[0][2:4][:]) == [empty, ("d", "d", a), empty]
    assert _get_nonaligned(passages[1][1][:]) == [("h", "h", a), ("", "jk", a)]


def test_passage_span__nonalignment_spans__zero_alignments():
    """Test `Passage` and `Span` get the correct nonalignments if one of the
    passages has zero alignments."""
    script = "abcdef"
    make = functools.partial(_make_unprocessed_passage_helper, transcript=script)
    unprocessed_passages = []

    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["b"]))  # NOTE: split='abc'
    unprocessed_passages.append(make("", []))
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["e"]))  # NOTE: split='def'

    kwargs = {"script": False, "transcript": True, "audio": True}
    passages = list(make_passages("", [unprocessed_passages], **kwargs))

    a = (0.0, 0.0)
    assert _get_nonaligned(passages[0]) == [("a", "a", a), ("c", "cd", a)]
    assert _get_nonaligned(passages[1]) == [("", "cd", a)]
    assert _get_nonaligned(passages[2]) == [("d", "cd", a), ("f", "f", (0.0, 7.583958148956299))]


def _has_a_mistranscription(
    args: typing.Sequence[typing.Tuple[str, str, typing.Sequence[typing.Tuple[IntInt, IntInt]]]],
    **kwargs,
) -> typing.List[bool]:
    """ Helper function for `test_has_a_mistranscription`. """
    unprocessed_passages = [
        make_unprocessed_passage(
            audio_path=TEST_DATA_LJ,
            script=s,
            transcript=t,
            alignments=tuple([Alignment(a, (0.0, 0.0), b) for a, b in a]),
        )
        for s, t, a in args
    ]
    passages = make_passages("", [unprocessed_passages], **kwargs)
    return [has_a_mistranscription(p) for p in passages]


def test_has_a_mistranscription():
    """Test `has_a_mistranscription` against a couple of basic cases."""
    passages = [("a", "a", (((0, 1), (0, 1)),))]
    assert not all(_has_a_mistranscription(passages))

    passages = [("a.", "a!", (((0, 1), (0, 1)),))]
    assert not all(_has_a_mistranscription(passages))

    passages = [("a1", "a!", (((0, 1), (0, 1)),))]
    assert all(_has_a_mistranscription(passages))

    passages = [("a.", "a1", (((0, 1), (0, 1)),))]
    assert all(_has_a_mistranscription(passages))

    passages = [("ac", "abc", (((0, 1), (0, 1)), ((1, 2), (2, 3))))]
    assert all(_has_a_mistranscription(passages))

    passages = [("abc", "ac", (((0, 1), (0, 1)), ((2, 3), (1, 2))))]
    assert all(_has_a_mistranscription(passages))


def test_has_a_mistranscription__multiple_passages():
    """Test `has_a_mistranscription` with multiple passages."""
    passages = [
        ("", "abc", tuple()),
        ("b", "abc", (((0, 1), (1, 2)),)),
        ("c", "abc", (((0, 1), (2, 3)),)),
    ]
    assert _has_a_mistranscription(passages, transcript=True) == [True, True, False]

    passages = [
        ("a", "abc", (((0, 1), (0, 1)),)),
        ("c", "abc", (((0, 1), (2, 3)),)),
    ]
    assert all(_has_a_mistranscription(passages, transcript=True))

    passages = [
        ("a", "ac", (((0, 1), (0, 1)),)),
        ("b", "ac", tuple()),
        ("c", "ac", (((0, 1), (1, 2)),)),
    ]
    assert all(_has_a_mistranscription(passages, transcript=True))
