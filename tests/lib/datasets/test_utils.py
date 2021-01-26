import functools
import math
import pickle
import typing
from collections import Counter
from unittest import mock

import numpy as np
import pytest

import lib
from lib.datasets import Alignment, IsConnected, Passage
from lib.utils import flatten
from tests._utils import (
    TEST_DATA_PATH,
    assert_uniform_distribution,
    get_audio_metadata_side_effect,
    make_passage,
    subprocess_run_side_effect,
)


def _make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """ Make an `Alignment` for testing. """
    return lib.datasets.Alignment(script, audio, transcript)


def _make_alignments(
    alignments=typing.Tuple[typing.Tuple[int, int]]
) -> typing.Tuple[lib.datasets.Alignment, ...]:
    """ Make a tuple of `Alignment`(s) for testing. """
    return tuple([_make_alignment(a, a, a) for a in alignments])


def test_span_generator():
    """Test `lib.datasets.SpanGenerator` samples uniformly given a uniform distribution of
    alignments."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3))))]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__empty():
    """ Test `lib.datasets.SpanGenerator` handles an empty list. """
    iterator = lib.datasets.SpanGenerator([], max_seconds=10)
    assert next(iterator, None) is None


def test_span_generator__zero():
    """Test `lib.datasets.SpanGenerator` handles alignments of zero length."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((1, 1),))),
        make_passage(_make_alignments(((1, 2),))),
    ]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    alignments_ = [dataset[0].alignments, dataset[-1].alignments]
    alignments = [list(typing.cast(typing.Tuple[Alignment], a)) for a in alignments_]
    assert set(counter.keys()) == set(flatten(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__singular():
    """Test `lib.datasets.SpanGenerator` handles multiple passages with a singular alignment
    of varying lengths."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((0, 10),))),
        make_passage(_make_alignments(((0, 5),))),
    ]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__multiple_multiple():
    """Test `lib.datasets.SpanGenerator` handles multiple scripts with a uniform alignment
    distribution."""
    dataset = [
        make_passage(_make_alignments(((0, 1), (1, 2), (2, 3)))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__pause():
    """ Test `lib.datasets.SpanGenerator` samples uniformly despite a large pause. """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41))))]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=4)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.02)


def test_span_generator__multiple_unequal_passages__large_max_seconds():
    """Test `lib.datasets.SpanGenerator` samples uniformly despite unequal passage sizes,
    and large max seconds.

    NOTE: With a large enough `max_seconds`, the entire passage should be sampled most of the time.
    """
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=1000000)

    alignments_counter: typing.Counter[Alignment] = Counter()
    spans_counter: typing.Counter[typing.Tuple[Alignment, ...]] = Counter()
    num_passages = 10000
    for _ in range(num_passages):
        span = next(iterator)
        slice_ = span.passage.alignments[span.span]
        alignments_counter.update(slice_)
        spans_counter[slice_] += 1

    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(alignments_counter.keys()) == set(flatten(alignments))
    assert_uniform_distribution(alignments_counter, abs=0.015)

    for passage in dataset:
        assert spans_counter[passage.alignments] / num_passages == pytest.approx(
            1 / len(dataset), abs=0.015
        )


def test_span_generator__unequal_alignment_sizes():
    """ Test `lib.datasets.SpanGenerator` samples uniformly despite unequal alignment sizes. """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 5), (5, 20))))]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=20)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__unequal_alignment_sizes__boundary_bias():
    """Test `lib.datasets.SpanGenerator` is biased at the boundary, in some scenarios.

    NOTE: The `max_seconds` filtering introduces bias by predominately filtering out this sequence:
    `[(1, 2), (2, 3), (3, 11), (11, 12)]`. That leads to an oversampling of `(11, 12)`.
    """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (3, 11), (11, 12))))]
    iterator = lib.datasets.SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.span])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    with pytest.raises(AssertionError):
        assert_uniform_distribution(counter, abs=0.015)


@mock.patch("lib.audio.get_audio_metadata")
@mock.patch("lib.datasets.utils.subprocess.run", return_value=None)
def test_dataset_loader(mock_run, mock_get_audio_metadata):
    """ Test `lib.datasets.dataset_loader` loads a dataset. """
    mock_get_audio_metadata.side_effect = get_audio_metadata_side_effect
    mock_run.side_effect = functools.partial(subprocess_run_side_effect, _command="gsutil")
    passages = lib.datasets.dataset_loader(
        TEST_DATA_PATH / "datasets",
        "hilary_noriega",
        "",
        lib.datasets.HILARY_NORIEGA,
    )
    alignments = [
        Alignment(a[0], a[1], a[2])
        for a in [
            ((0, 6), (0.0, 0.6), (0, 6)),
            ((7, 9), (0.6, 0.8), (7, 9)),
            ((10, 13), (0.8, 0.8), (10, 13)),
            ((14, 20), (0.8, 1.4), (14, 20)),
            ((21, 27), (1.4, 1.8), (21, 26)),
            ((28, 34), (1.8, 2.5), (27, 33)),
            ((35, 42), (2.5, 2.6), (34, 40)),
            ((43, 47), (2.6, 3.3), (41, 45)),
        ]
    ]

    path = TEST_DATA_PATH / "datasets/hilary_noriega/recordings/Script 1.wav"
    assert passages[0].audio_file == lib.audio.get_audio_metadata([path])[0]
    assert passages[0].speaker == lib.datasets.HILARY_NORIEGA
    assert passages[0].script == "Author of the danger trail, Philip Steels, etc."
    assert passages[0].transcript == (
        "author of the danger Trail Philip Steels Etc. Not at this particular case Tom "
        "apologized Whitmore for the 20th time that evening the two men shook hands"
    )
    assert passages[0].alignments == tuple(alignments)
    assert passages[0].other_metadata == {"Index": 0, "Source": "CMU", "Title": "CMU"}
    assert passages[0].index == 0
    assert passages[0].passages == passages
    assert passages[0].is_connected == IsConnected(False, True, True)
    assert passages[1].script == "Not at this particular case, Tom, apologized Whittemore."


def test_passage_span__identity():
    """Test `lib.datasets.Passage` and `lib.datasets.Span` are the same after a identity
    operation."""
    audio_path = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    script = (
        "The examination and testimony of the experts enabled the Commission to conclude "
        "that five shots may have been fired,"
    )
    alignment = lib.datasets.Alignment((0, len(script)), (0.0, metadata.length), (0, len(script)))
    passage = Passage(
        audio_file=metadata,
        speaker=lib.datasets.LINDA_JOHNSON,
        script=script,
        transcript=script,
        alignments=(alignment,),
        other_metadata={"chapter": 37},
    )
    span = passage[:]
    assert passage.script == span.script
    assert passage.transcript == span.transcript
    assert passage.alignments == span.alignments
    assert passage.speaker == span.speaker
    assert passage.audio_file == span.audio_file
    assert passage.unaligned == span.unaligned
    assert passage.other_metadata == span.other_metadata
    assert passage.aligned_audio_length() == span.audio_length
    assert (
        passage.to_string("audio_file", "other_metadata")[len(passage.__class__.__name__) :]
        == span.to_string("audio_file", "other_metadata")[len(span.__class__.__name__) :]
    )
    np.testing.assert_almost_equal(passage.audio(), span.audio())
    pickle.dumps(passage)
    pickle.dumps(span)


_find = lambda a, b: (a.index(b), a.index(b) + 1)


def _add_passage(
    script: str,
    tokens: typing.List[str],
    passages: typing.List[lib.datasets.Passage],
    transcript: str,
    find_transcript: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
    find_script: typing.Callable[[str, str], typing.Tuple[int, int]] = _find,
    **kwargs,
):
    """ Helper function for `test_passage_span__unaligned*`. """
    found = [(find_script(script, t), find_transcript(transcript, t)) for t in tokens]
    passages.append(
        make_passage(
            script=script,
            alignments=tuple(_make_alignment(*arg) for arg in found),
            transcript=transcript,
            index=len(passages),
            passages=passages,
            **kwargs,
        )
    )


def test_passage_span__unaligned():
    """Test `lib.datasets.Passage` and `lib.datasets.Span` get the correct unalignments under a
    variety of circumstances."""
    passages = []
    script = "abcdefghijklmnopqrstuvwxyz"
    is_connected = IsConnected(False, True, True)
    add_passage = functools.partial(
        _add_passage, passages=passages, transcript=script, is_connected=is_connected
    )

    # TEST: Largely no issues, except one in the middle.
    split, script = script[:6], script[6:]
    add_passage(split, ["a", "b", "c", "e", "f"])  # NOTE: split='abcdef'

    # TEST: Right edge has an issue, along with one in the middle.
    split, script = script[:3], script[3:]
    add_passage(split, ["g", "i"])  # NOTE: split='ghi'

    # TEST: Left edge has an issue.
    split, script = script[:3], script[3:]
    add_passage(split, ["l"])  # NOTE: split='jkl'

    # TEST: Right edge has an issue.
    split, script = script[:3], script[3:]
    add_passage(split, ["m"])  # NOTE: split='mno'

    # TEST: Both edges have an issue, and there is no rightward passage.
    split, script = script[:3], script[3:]
    add_passage(split, ["q"])  # NOTE: split='pqr'

    a = (0.0, 0.0)
    expected = [("", "", a), ("", "", a), ("", "", a), ("d", "d", a), ("", "", a), ("", "", a)]
    assert passages[0].unaligned == expected
    assert passages[0][:].unaligned == passages[0].unaligned
    assert passages[1].unaligned == [("", "", a), ("h", "h", a), ("", "jk", a)]
    assert passages[1][:].unaligned == passages[1].unaligned
    assert passages[2].unaligned == [("jk", "jk", a), ("", "", a)]
    assert passages[2][:].unaligned == passages[2].unaligned
    assert passages[3].unaligned == [("", "", a), ("no", "nop", a)]
    assert passages[3][:].unaligned == passages[3].unaligned
    assert passages[4].unaligned == [("p", "nop", a), ("r", "rstuvwxyz", (0.0, math.inf))]
    assert passages[4][:].unaligned == passages[4].unaligned

    # TEST: Test `spans` get the correct span.
    assert passages[0][2:4][:].unaligned == [("", "", a), ("d", "d", a), ("", "", a)]
    assert passages[1][1][:].unaligned == [("h", "h", a), ("", "jk", a)]


def test_passage_span__unaligned__zero_alignments():
    """Test `lib.datasets.Passage` and `lib.datasets.Span` get the correct unalignments if one
    of the passages has zero alignments."""
    passages = []
    script = "abcdef"
    is_connected = IsConnected(False, True, True)
    add_passage = functools.partial(
        _add_passage, passages=passages, transcript=script, is_connected=is_connected
    )

    split, script = script[:3], script[3:]
    add_passage(split, ["b"])  # NOTE: split='abc'
    add_passage("", [])
    split, script = script[:3], script[3:]
    add_passage(split, ["e"])  # NOTE: split='abc'

    a = (0.0, 0.0)
    assert passages[0].unaligned == [("a", "a", a), ("c", "cd", a)]
    assert passages[1].unaligned == [("", "cd", a)]
    assert passages[2].unaligned == [("d", "cd", a), ("f", "f", (0.0, math.inf))]


def test__overlap():
    """ Test `_overlap` computes the percentage of overlap between two spans correctly. """
    assert lib.datasets.SpanGenerator._overlap((0, 1), (1, 2)) == 0.0
    assert lib.datasets.SpanGenerator._overlap((0, 1), (0, 1)) == 1.0
    assert lib.datasets.SpanGenerator._overlap((0, 1), (0.5, 0.5)) == 1.0
    assert lib.datasets.SpanGenerator._overlap((0, 1), (-1, 0)) == 0.0
    assert lib.datasets.SpanGenerator._overlap((0, 1), (-0.5, 0.5)) == 0.5
    assert lib.datasets.SpanGenerator._overlap((0, 1), (0.5, 1.5)) == 0.5
    assert lib.datasets.SpanGenerator._overlap((0.5, 2), (0, 1)) == 0.5
    assert lib.datasets.SpanGenerator._overlap((0, 2), (0, 1)) == 1.0
    assert lib.datasets.SpanGenerator._overlap((-1.5, 0.5), (0, 1)) == 0.5
    assert lib.datasets.SpanGenerator._overlap((-1, 1), (0, 1)) == 1.0


# NOTE: `lib.datasets.conventional_dataset_loader` is tested in `lj_speech` as part of loading the
# dataset.
