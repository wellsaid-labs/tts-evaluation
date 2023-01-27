import functools
import pathlib
import pickle
import typing
from unittest import mock

import config as cf
import numpy as np
import pytest

import lib
from lib.utils import Timeline
from run._config import normalize_vo_script
from run.data._loader.english.lj_speech import LINDA_JOHNSON
from run.data._loader.structures import (
    Alignment,
    IntInt,
    IsLinked,
    Language,
    Passage,
    Session,
    Span,
    UnprocessedPassage,
    UnprocessedSession,
    _check_updated_script,
    _filter_non_speech_segments,
    _is_stand_abbrev_consistent,
    _make_speech_segments_helper,
    _maybe_normalize_vo_script,
    _process_sessions,
    _remove_ambiguous_abbrev,
    has_a_mistranscription,
    make_passages,
)
from run.data._loader.utils import get_non_speech_segments_and_cache
from tests._utils import TEST_DATA_PATH
from tests.run._utils import (
    make_passage,
    make_session,
    make_unprocessed_session,
    script_to_alignments,
)

TEST_DATA_LJ = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"


def make_unprocessed_passage(
    audio_path: pathlib.Path = pathlib.Path("."),
    session: UnprocessedSession = make_unprocessed_session(),
    script: str = "",
    transcript: str = "",
    alignments: typing.Optional[typing.Tuple[Alignment, ...]] = None,
    **kwargs,
) -> UnprocessedPassage:
    """Make a `UnprocessedPassage` for testing."""
    return UnprocessedPassage(audio_path, session, script, transcript, alignments, **kwargs)


def test__maybe_normalize_vo_script():
    """Test `_maybe_normalize_vo_script` against some basic cases."""
    normal_script = "abc"
    assert _maybe_normalize_vo_script(normal_script, Language.ENGLISH) == normal_script
    script = "áƀć°€¹"
    assert _maybe_normalize_vo_script(script, Language.ENGLISH) == "ábcde'"


def test__filter_non_speech_segments():
    """Test `_filter_non_speech_segments` against various word alignment and pausing intervals."""
    alignments = [(0, 0), (0, 1), (1, 1), (1, 2), (1.9, 3), (4, 5), (4.8, 5), (5, 6)]
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
        (0, 0.1): True,
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
        # CASE: Pause envelopes 0 alignment(s) and overlaps 3 alignment(s)
        (1, 1.1): False,
    }
    slices = [slice(*k) for k in non_speech_segments.keys()]
    filtered = _filter_non_speech_segments(alignments, Timeline(alignments), slices)
    results = set(((f.start, f.stop) for f in filtered))
    assert results == {k for k, v in non_speech_segments.items() if v}


def test__make_speech_segments_helper():
    """Test `_make_speech_segments_helper` against various word alignment and pausing intervals."""
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
    prev_alignment = (0, 0)
    audio_file = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    next_alignment = (audio_file.length, audio_file.length)
    args = (audio_alignments, prev_alignment, next_alignment, audio_file.length)
    make = lambda t: _make_speech_segments_helper(*args, t, **cf.get())

    speech_segments = (
        (slice(0, 7), slice(0.0, 2.775)),
        (slice(7, 12), slice(2.9599583333333337, 4.91)),
        (slice(12, len(audio_alignments)), slice(5.264958333333333, 7.405)),
    )
    get_nss_and_cache = cf.partial(get_non_speech_segments_and_cache)
    timeline = get_nss_and_cache(audio_file)
    assert make(timeline) == speech_segments
    timeline = get_nss_and_cache(audio_file, threshold=-1000)
    assert make(timeline) == tuple()
    timeline = get_nss_and_cache(audio_file, threshold=0)
    assert make(timeline) == tuple()
    timeline = get_nss_and_cache(audio_file, threshold=-40)
    assert make(timeline) == (
        (slice(0, 5, None), slice(0.009958333333333326, 1.7699999999999998, None)),
        (slice(5, 7, None), slice(1.7949583333333334, 2.62, None)),
        (slice(7, 8, None), slice(2.9649583333333336, 3.42, None)),
        (slice(8, 9, None), slice(3.4599583333333337, 3.5500000000000003, None)),
        (slice(9, 11, None), slice(3.6399583333333334, 4.195, None)),
        (slice(11, 12, None), slice(4.279958333333333, 4.825, None)),
        (slice(12, 13, None), slice(5.279958333333333, 5.48, None)),
        (slice(13, 17, None), slice(5.5499583333333335, 6.5600000000000005, None)),
        (slice(17, 18, None), slice(6.599958333333333, 6.755000000000001, None)),
        (slice(18, 19, None), slice(6.8849583333333335, 7.36, None)),
    )


def test__make_speech_segments_helper__partial():
    """Test `_make_speech_segments_helper` where `non_speech_segments` overlap an alignment
    partially."""
    speech_segments = _make_speech_segments_helper(
        alignments=[(0, 1), (1, 2)],
        prev_alignment=(0, 0),
        next_alignment=(2, 2),
        max_length=2,
        nss_timeline=Timeline([(0.0, 0.0), (0.0, 0.0)]),
        **cf.get(),
    )
    assert speech_segments == tuple()


def test__make_speech_segments_helper__overlap():
    """Test `_make_speech_segments_helper` where `non_speech_segments` overlap each other."""
    pad = 25 / 1000
    speech_segments = _make_speech_segments_helper(
        alignments=[(0.25, 1), (2, 2.75)],
        prev_alignment=(0, 0),
        next_alignment=(3, 3),
        max_length=3,
        nss_timeline=Timeline([(0, 0.25), (0.5, 1.75), (1.25, 2.5), (2.75, 3)]),
        pad=pad,
    )
    assert speech_segments == (
        (slice(0, 1, None), slice(0.25 - pad, 0.5 + pad, None)),
        (slice(1, 2, None), slice(2.5 - pad, 2.75 + pad, None)),
    )


def test__make_speech_segments_helper__overlap__regression():
    """Test `_make_speech_segments_helper` where `non_speech_segment` is on the border of an
    `alignment`.
    """
    pad = 25 / 1000
    speech_segments = _make_speech_segments_helper(
        alignments=[(17.4, 17.8), (20.7, 20.9), (24.0, 24.1)],
        prev_alignment=(0, 0),
        next_alignment=(26, 26),
        max_length=26,
        nss_timeline=Timeline([(16.77, 17.5), (20.7, 20.81), (24.0, 25.275)]),
        pad=pad,
    )
    assert speech_segments == ((slice(1, 3, None), slice(20.785, 24.025, None)),)


def test__make_speech_segments_helper__prev_alignment():
    """Test `_make_speech_segments_helper` where `alignments` and `prev_alignment` overlap."""
    speech_segments = _make_speech_segments_helper(
        alignments=[(0.25, 1)],
        prev_alignment=(0, 0.5),
        next_alignment=(1.5, 1.5),
        max_length=1.5,
        nss_timeline=Timeline([(0.0, 0.2), (0.3, 0.75), (1.0, 1.5)]),
        **cf.get(),
    )
    assert speech_segments == tuple()


def test__make_speech_segments_helper__next_alignment():
    """Test `_make_speech_segments_helper` where there is no pause between `alignments` and
    `next_alignment`."""
    speech_segments = _make_speech_segments_helper(
        alignments=[(0.25, 1)],
        prev_alignment=(0, 0),
        next_alignment=(1.5, 1.75),
        max_length=2.0,
        nss_timeline=Timeline([(0.0, 0.2), (1.75, 2.0)]),
        **cf.get(),
    )
    assert speech_segments == tuple()


def test__make_speech_segments_helper__padding():
    """Test `_make_speech_segments_helper` where padding goes past the edges."""
    speech_segments = _make_speech_segments_helper(
        alignments=[(0.1, 0.9)],
        prev_alignment=(0, 0),
        next_alignment=(1, 1),
        max_length=1.0,
        nss_timeline=Timeline([(0.0, 0.1), (0.9, 1.0)]),
        pad=2.0,
    )
    assert speech_segments == ((slice(0, 1, None), slice(0.0, 1.0, None)),)


@mock.patch("run.data._loader.structures.logger.error")
def test__check_updated_script(mock_error):
    """Test `_check_updated_script` against some basic cases."""
    passage = make_unprocessed_passage(script="abc", transcript="abc", alignments=tuple())
    _check_updated_script("", passage, "abc", "abc")
    assert mock_error.called == 0

    passage = make_unprocessed_passage(script="áƀćde'", transcript="áƀć°€¹", alignments=tuple())
    _check_updated_script("", passage, "abcde'", "abcde'")
    assert mock_error.called == 1

    with pytest.raises(AssertionError):
        passage = make_unprocessed_passage(script="áƀćde'", transcript="áƀć°€¹", alignments=tuple())
        _check_updated_script("", passage, "abcde'", "abcdegeurone")

    with pytest.raises(AssertionError):
        passage = make_unprocessed_passage(script="ab\f", transcript="ab", alignments=tuple())
        _check_updated_script("", passage, "ab", "ab")


@mock.patch("run.data._loader.structures.logger.error")
def test__check_updated_script__with__maybe_normalize_vo_script(mock_error):
    script = "áƀćde'"
    transcript = "áƀć°€¹"
    passage = make_unprocessed_passage(script=script, transcript=transcript, alignments=tuple())
    updated_script = _maybe_normalize_vo_script(script, language=Language.ENGLISH)
    updated_transcript = _maybe_normalize_vo_script(transcript, language=Language.ENGLISH)
    _check_updated_script("", passage, updated_script, updated_transcript)
    assert mock_error.called == 1

    with pytest.raises(AssertionError):
        script = "áƀćde'"
        transcript = "áƀć°€¹"
        passage = make_unprocessed_passage(script=script, transcript=transcript, alignments=tuple())
        updated_script = normalize_vo_script(script, Language.ENGLISH)
        updated_transcript = normalize_vo_script(transcript, Language.ENGLISH)
        _check_updated_script("", passage, updated_script, updated_transcript)


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
        session=make_session(LINDA_JOHNSON, audio_path.name),
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
    object.__setattr__(passage, "passages", [passage])
    object.__setattr__(passage, "index", 0)
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


def _make_unprocessed_passage(
    script: str,
    tokens: typing.List[str],
    transcript: str,
    find_transcript: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
    find_script: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
    **kwargs,
):
    """Helper function for `test_passage_span__unaligned*`."""
    found = [(find_script(script, t), find_transcript(transcript, t)) for t in tokens]
    return UnprocessedPassage(
        audio_path=TEST_DATA_LJ,
        session=make_unprocessed_session(),
        script=script,
        transcript=transcript,
        alignments=tuple(Alignment(s, (0.0, 0.0), t) for s, t in found),
        **kwargs,
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

    is_linked = IsLinked(transcript=True, audio=True)
    make = functools.partial(_make_unprocessed_passage, transcript=script, is_linked=is_linked)
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

    passages = list(make_passages("", [unprocessed_passages]))

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
    is_linked = IsLinked(transcript=True, audio=True)
    make = functools.partial(_make_unprocessed_passage, transcript=script, is_linked=is_linked)
    unprocessed_passages = []

    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["b"]))  # NOTE: split='abc'
    unprocessed_passages.append(make("", []))
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["e"]))  # NOTE: split='def'

    passages = list(make_passages("", [unprocessed_passages]))

    a = (0.0, 0.0)
    assert _get_nonaligned(passages[0]) == [("a", "a", a), ("c", "cd", a)]
    assert _get_nonaligned(passages[1]) == [("", "cd", a)]
    assert _get_nonaligned(passages[2]) == [("d", "cd", a), ("f", "f", (0.0, 7.583958148956299))]


def test_passage_linking():
    """Test `Passage` links `prev`, `next`, `_prev_alignment` and `_next_alignment`."""
    unprocessed_passages = [
        make_unprocessed_passage(
            audio_path=TEST_DATA_LJ,
            script=s,
            transcript="abc",
            alignments=(Alignment((0, 1), a, a),),
            is_linked=IsLinked(transcript=True, audio=True),
        )
        for s, a in (("a", (0, 1)), ("b", (1, 2)), ("c", (2, 3)))
    ]
    passages = make_passages("", [unprocessed_passages])
    assert len(passages[0].passages) == 3
    assert passages[0].prev is None
    assert passages[0].next == passages[1]
    assert passages[0]._prev_alignment() == Alignment((0, 0), (0.0, 0.0), (0, 0))
    assert passages[0]._next_alignment() == Alignment((1, 1), (1, 2), (1, 2))
    assert len(passages[1].passages) == 3
    assert passages[1].prev == passages[0]
    assert passages[1].next == passages[2]
    assert passages[1]._prev_alignment() == Alignment((0, 0), (0, 1), (0, 1))
    assert passages[1]._next_alignment() == Alignment((1, 1), (2, 3), (2, 3))
    assert len(passages[2].passages) == 3
    assert passages[2].prev == passages[1]
    assert passages[2].next is None
    assert passages[2]._prev_alignment() == Alignment((0, 0), (1, 2), (1, 2))
    length = passages[2].audio_file.length
    assert passages[2]._next_alignment() == Alignment((1, 1), (length, length), (3, 3))


def test_passage_linking__no_links():
    """Test `Passage` links `prev`, `next`, `_prev_alignment` and `_next_alignment` if no passages
    are linked."""
    unprocessed_passages = [
        make_unprocessed_passage(
            audio_path=TEST_DATA_LJ,
            script=s,
            transcript=s,
            alignments=(Alignment((0, 1), a, a),),
        )
        for s, a in (("a", (0, 1)), ("b", (0, 1)))
    ]
    passages = make_passages("", [[p] for p in unprocessed_passages])
    length = passages[0].audio_file.length
    for passage in passages:
        assert len(passage.passages) == 1
        assert passage.prev is None
        assert passage.next is None
        assert passage._prev_alignment() == Alignment((0, 0), (0, 0), (0, 0))
        assert passage._next_alignment() == Alignment((1, 1), (length, length), (1, 1))


def _has_a_mistranscription_helper(
    args: typing.Sequence[typing.Tuple[str, str, typing.Sequence[typing.Tuple[IntInt, IntInt]]]],
    **kwargs,
) -> typing.List[Passage]:
    """Helper function for `_has_a_mistranscription`."""
    unprocessed_passages = [
        make_unprocessed_passage(
            audio_path=TEST_DATA_LJ,
            script=s,
            transcript=t,
            alignments=tuple([Alignment(a, (0.0, 0.0), b) for a, b in a]),
            **kwargs,
        )
        for s, t, a in args
    ]
    return make_passages("", [unprocessed_passages])


def _has_a_mistranscription(*args, **kwargs) -> typing.List[bool]:
    """Helper function for `test_has_a_mistranscription`."""
    return [has_a_mistranscription(p) for p in _has_a_mistranscription_helper(*args, **kwargs)]


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
    is_linked = IsLinked(transcript=True)
    passages = [
        ("", "abc", tuple()),
        ("b", "abc", (((0, 1), (1, 2)),)),
        ("c", "abc", (((0, 1), (2, 3)),)),
    ]
    assert _has_a_mistranscription(passages, is_linked=is_linked) == [True, True, False]

    passages = [
        ("a", "abc", (((0, 1), (0, 1)),)),
        ("c", "abc", (((0, 1), (2, 3)),)),
    ]
    assert all(_has_a_mistranscription(passages, is_linked=is_linked))

    passages = [
        ("a", "ac", (((0, 1), (0, 1)),)),
        ("b", "ac", tuple()),
        ("c", "ac", (((0, 1), (1, 2)),)),
    ]
    assert all(_has_a_mistranscription(passages, is_linked=is_linked))


def test_has_a_mistranscription__span():
    """Test `has_a_mistranscription` with spans."""
    is_linked = IsLinked(transcript=True)
    passages = [
        ("abc", "abcdef", (((0, 1), (0, 1)), ((1, 2), (1, 2)), ((2, 3), (2, 3)))),
        ("def", "abcdef", (((1, 2), (4, 5)), ((2, 3), (5, 6)))),
    ]
    passages = _has_a_mistranscription_helper(passages, is_linked=is_linked)
    assert has_a_mistranscription(passages[0][:])
    assert not has_a_mistranscription(passages[0][:-1])
    assert has_a_mistranscription(passages[0][1:])
    assert has_a_mistranscription(passages[1][:])
    assert not has_a_mistranscription(passages[1][1:])


def test_spacy_context():
    """Test that `spacy_context` gets the full context."""
    script = "Give it back! He pleaded."
    passage = make_passage(script=script)
    assert str(passage[2:4].spacy_context(10)) == "Give it back! He pleaded."
    assert str(passage[2:4].spacy_context(0)) == "back! He"
    assert str(passage[2:4].spacy) == "back! He"


def test__is_stand_abbrev_consistent():
    """Test that `_is_stand_abbrev_consistent` can verify that an abbreviation is consistent
    between a script and transcript."""
    assert not _is_stand_abbrev_consistent("ABC", "abc")
    assert not _is_stand_abbrev_consistent("NDAs", "nda's")
    assert not _is_stand_abbrev_consistent("HAND-CUT", "hand cut")
    assert not _is_stand_abbrev_consistent("NOVA/national", "Nova National")
    assert not _is_stand_abbrev_consistent("I V As?", "ivas")
    assert not _is_stand_abbrev_consistent("I.V.A.", "iva")
    assert not _is_stand_abbrev_consistent("information...ELEVEN", "information 11")
    assert not _is_stand_abbrev_consistent("(JNA)", "JN a")
    assert not _is_stand_abbrev_consistent("PwC", "PWC")
    assert not _is_stand_abbrev_consistent("JC PENNEY", "JCPenney")
    assert not _is_stand_abbrev_consistent("DIRECTV", "DirecTV")
    assert not _is_stand_abbrev_consistent("M*A*C", "Mac")
    assert not _is_stand_abbrev_consistent("fMRI", "fmri")
    assert not _is_stand_abbrev_consistent("RuBP.", "rubp")
    assert not _is_stand_abbrev_consistent("MiniUSA.com,", "mini usa.com.")
    assert not _is_stand_abbrev_consistent("7UP", "7-Up")
    assert _is_stand_abbrev_consistent("NDT", "ND T")
    assert _is_stand_abbrev_consistent("L.V.N,", "LVN")
    assert _is_stand_abbrev_consistent("I", "i")
    assert _is_stand_abbrev_consistent("PM", "p.m.")
    assert _is_stand_abbrev_consistent("place...where", "place where")
    assert _is_stand_abbrev_consistent("Smucker's.", "Smuckers")
    assert _is_stand_abbrev_consistent("DVD-Players", "DVD players")
    assert _is_stand_abbrev_consistent("PCI-DSS,", "PCI DSS.")
    assert _is_stand_abbrev_consistent("UFOs", "UFO's,")
    assert _is_stand_abbrev_consistent("most[JT5]", "most JT 5")
    assert _is_stand_abbrev_consistent("NJ--at", "NJ. At")
    assert _is_stand_abbrev_consistent("U. S.", "u.s.")
    assert _is_stand_abbrev_consistent("ADHD.Some", "ADHD some")
    assert _is_stand_abbrev_consistent("W-USA", "WUSA.")
    assert _is_stand_abbrev_consistent("P-S-E-C-U", "PSECU")
    assert _is_stand_abbrev_consistent("J. V.", "JV")
    assert _is_stand_abbrev_consistent("PM", "P.m.")
    assert _is_stand_abbrev_consistent("Big-C", "Big C.")
    assert _is_stand_abbrev_consistent("U-Boats", "U-boats")
    assert _is_stand_abbrev_consistent("Me...obsessive?...I", "me obsessive. I")
    assert _is_stand_abbrev_consistent("apiece,", "A piece")
    assert _is_stand_abbrev_consistent("well.I'll,", "well. I'll")
    assert _is_stand_abbrev_consistent("Rain-x-car", "Rain-X car")
    assert _is_stand_abbrev_consistent("L.L.Bean", "LL Bean")
    assert _is_stand_abbrev_consistent("WBGP -", "W BG P.")
    assert _is_stand_abbrev_consistent("KRCK", "K RC K")
    assert _is_stand_abbrev_consistent("DVD-L10", "DVD L10")
    assert _is_stand_abbrev_consistent("DVD-L10", "DVD, L10")
    assert _is_stand_abbrev_consistent("t-shirt", "T-shirt")
    assert _is_stand_abbrev_consistent("3-D", "3D")


def test__remove_ambiguous_casing():
    """Test that `_remove_ambiguous_abbrev` removes any capitalized words or
    ambiguous abbreviations."""
    script, transcript, normalized = (
        "THIS is A test. This. TEST. urls. URLs. 111. Dash-dash. si punc. Dash-Dash. U.S. P-C-I.",
        "This is a TEST. This. TEST. URLs. urls. one. dash-dash. y p. dash-dash. u.s. PCI.",
        "     is A       This. TEST.             111. Dash-dash. si punc. Dash-Dash. U.S. P-C-I.",
    )
    iter_ = zip(script_to_alignments(script), script_to_alignments(transcript))
    alignments = tuple([Alignment(s, s, t) for s, t in iter_])
    passage = make_unprocessed_passage(script=script, transcript=transcript, alignments=alignments)
    passage = _remove_ambiguous_abbrev("", passage)
    assert passage.alignments is not None
    assert [a.script for a in passage.alignments] == list(script_to_alignments(normalized))


def test__process_sessions():
    """Test `_process_sessions` handles a basic case, with multiple passages and
    optional alignments."""
    meta = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    passage = make_unprocessed_passage(
        audio_path=TEST_DATA_LJ,
        script="abcd",
        transcript="abcd",
        alignments=(Alignment((0, 3), (0, 1), (0, 1)),),
    )
    other_audio_file = TEST_DATA_PATH / "audio" / "hilary.wav"
    other_meta = lib.audio.get_audio_metadata(other_audio_file)
    other_passage = make_unprocessed_passage(
        audio_path=other_audio_file,
        session=make_unprocessed_session(name="no alignments"),
        script="abcd",
        transcript="abcd",
    )
    kwargs = dict(
        meta={TEST_DATA_LJ: meta, other_audio_file: other_meta},
        get_loudness=lambda l: sum([n.mean() for n in l]),
        get_tempo=lambda s, f: f / len(s),
        no_tqdm=True,
    )
    processed = cf.call(_process_sessions, [[passage], [other_passage]], **kwargs)
    assert len(processed) == 2
    assert processed[passage.session] == Session(
        *passage.session,
        1.240387064171955e-06,
        1 / len("abc"),
        (other_meta.len + 1) / len("abc abcd"),
    )
    assert processed[other_passage.session] == Session(
        *other_passage.session,
        3.3791247915360145e-06,
        other_meta.len / len("abcd"),
        (other_meta.len + 1) / len("abc abcd"),
    )
