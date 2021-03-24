import functools
import pickle
import typing
from collections import Counter
from unittest import mock

import numpy as np
import pytest

import lib
import run.data._loader
from lib.utils import flatten_2d
from run.data._loader import Alignment, Passage, Span, SpanGenerator
from run.data._loader.utils import UnprocessedPassage, make_passages
from tests._utils import (
    TEST_DATA_PATH,
    assert_uniform_distribution,
    get_audio_metadata_side_effect,
    make_passage,
    subprocess_run_side_effect,
)


def _make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """ Make an `Alignment` for testing. """
    return Alignment(script, audio, transcript)


def _make_alignments(alignments=typing.Tuple[typing.Tuple[int, int]]) -> lib.utils.Tuple[Alignment]:
    """ Make a tuple of `Alignment`(s) for testing. """
    return Alignment.stow([_make_alignment(a, a, a) for a in alignments])


def test_span_generator():
    """Test `SpanGenerator` samples uniformly given a uniform distribution of
    alignments."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3))))]
    iterator = SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__empty():
    """ Test `SpanGenerator` handles an empty list. """
    iterator = SpanGenerator([], max_seconds=10)
    assert next(iterator, None) is None


def test_span_generator__zero():
    """Test `SpanGenerator` handles alignments of zero length."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((1, 1),))),
        make_passage(_make_alignments(((1, 2),))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments_ = [dataset[0].alignments, dataset[-1].alignments]
    alignments = [list(typing.cast(typing.Tuple[Alignment], a)) for a in alignments_]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__singular():
    """Test `SpanGenerator` handles multiple passages with a singular alignment
    of varying lengths."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((0, 10),))),
        make_passage(_make_alignments(((0, 5),))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__multiple_multiple():
    """Test `SpanGenerator` handles multiple scripts with a uniform alignment
    distribution."""
    dataset = [
        make_passage(_make_alignments(((0, 1), (1, 2), (2, 3)))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__pause():
    """ Test `SpanGenerator` samples uniformly despite a large pause. """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41))))]
    iterator = SpanGenerator(dataset, max_seconds=4)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.02)


def test_span_generator__multiple_unequal_passages__large_max_seconds():
    """Test `SpanGenerator` samples uniformly despite unequal passage sizes,
    and large max seconds.

    NOTE: With a large enough `max_seconds`, the entire passage should be sampled most of the time.
    """
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=1000000)

    alignments_counter: typing.Counter[Alignment] = Counter()
    spans_counter: typing.Counter[lib.utils.Tuple[Alignment]] = Counter()
    num_passages = 10000
    for _ in range(num_passages):
        span = next(iterator)
        slice_ = span.passage.alignments[span.slice]
        alignments_counter.update(slice_)
        spans_counter[slice_] += 1

    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(alignments_counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(alignments_counter, abs=0.015)

    for passage in dataset:
        assert spans_counter[passage.alignments] / num_passages == pytest.approx(
            1 / len(dataset), abs=0.015
        )


def test_span_generator__unequal_alignment_sizes():
    """ Test `SpanGenerator` samples uniformly despite unequal alignment sizes. """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 5), (5, 20))))]
    iterator = SpanGenerator(dataset, max_seconds=20)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__unequal_alignment_sizes__boundary_bias():
    """Test `SpanGenerator` is biased at the boundary, in some scenarios.

    NOTE: The `max_seconds` filtering introduces bias by predominately filtering out this sequence:
    `[(1, 2), (2, 3), (3, 11), (11, 12)]`. That leads to an oversampling of `(11, 12)`.
    """
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (3, 11), (11, 12))))]
    iterator = SpanGenerator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    with pytest.raises(AssertionError):
        assert_uniform_distribution(counter, abs=0.015)


@mock.patch("lib.audio.get_audio_metadata")
@mock.patch("run.data._loader.utils.subprocess.run", return_value=None)
def test_dataset_loader(mock_run, mock_get_audio_metadata):
    """ Test `run.data._loader.dataset_loader` loads a dataset. """
    mock_get_audio_metadata.side_effect = get_audio_metadata_side_effect
    mock_run.side_effect = functools.partial(subprocess_run_side_effect, _command="gsutil")
    passages = run.data._loader.dataset_loader(
        TEST_DATA_PATH / "datasets",
        "hilary_noriega",
        "",
        run.data._loader.HILARY_NORIEGA,
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
    nonalignments = [
        Alignment(a[0], a[1], a[2])
        for a in [
            ((0, 0), (0.0, 0.0), (0, 0)),
            ((6, 7), (0.6, 0.6), (6, 7)),
            ((9, 10), (0.8, 0.8), (9, 10)),
            ((13, 14), (0.8, 0.8), (13, 14)),
            ((20, 21), (1.4, 1.4), (20, 21)),
            ((27, 28), (1.8, 1.8), (26, 27)),
            ((34, 35), (2.5, 2.5), (33, 34)),
            ((42, 43), (2.6, 2.6), (40, 41)),
            ((47, 47), (3.3, 4.3), (45, 46)),
        ]
    ]

    path = TEST_DATA_PATH / "datasets/hilary_noriega/recordings/Script 1.wav"
    assert passages[0].audio_file == lib.audio.get_audio_metadata([path])[0]
    assert passages[0].speaker == run.data._loader.HILARY_NORIEGA
    assert passages[0].script == "Author of the danger trail, Philip Steels, etc."
    assert passages[0].transcript == (
        "author of the danger Trail Philip Steels Etc. Not at this particular case Tom "
        "apologized Whitmore for the 20th time that evening the two men shook hands"
    )
    assert passages[0].alignments == Alignment.stow(alignments)
    assert passages[0].nonalignments == Alignment.stow(nonalignments)
    assert passages[0].other_metadata == {"Index": 0, "Source": "CMU", "Title": "CMU"}
    assert passages[1].script == "Not at this particular case, Tom, apologized Whittemore."


def test_passage_span__identity():
    """Test `run.data._loader.Passage` and `run.data._loader.Span` are the same after a identity
    operation."""
    audio_path = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata(audio_path)
    script = (
        "The examination and testimony of the experts enabled the Commission to conclude "
        "that five shots may have been fired,"
    )
    alignment = Alignment((0, len(script)), (0.0, metadata.length), (0, len(script)))
    passage = Passage(
        audio_file=metadata,
        speaker=run.data._loader.LINDA_JOHNSON,
        script=script,
        transcript=script,
        alignments=Alignment.stow([alignment]),
        nonalignments=Alignment.stow([alignment]),
        other_metadata={"chapter": 37},
    )
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
):
    """ Helper function for `test_passage_span__unaligned*`. """
    found = [(find_script(script, t), find_transcript(transcript, t)) for t in tokens]
    return UnprocessedPassage(
        audio_path=TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav",
        speaker=run.data._loader.Speaker(""),
        script=script,
        transcript=transcript,
        alignments=tuple(_make_alignment(*arg) for arg in found),
    )


def _get_nonaligned(
    span: typing.Union[Span, Passage]
) -> typing.List[typing.Tuple[str, str, typing.Tuple[float, float]]]:
    return [
        (s.script, s.transcript, (s.audio_slice.start, s.audio_slice.stop))
        for s in span.nonalignment_spans()
    ]


def test_passage_span__nonalignment_spans():
    """Test `run.data._loader.Passage` and `run.data._loader.Span` get the correct nonalignments
    under a variety of circumstances."""
    script = "abcdefghijklmnopqrstuvwxyz"
    make = functools.partial(_make_unprocessed_passage, transcript=script)
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
    passages = list(make_passages([unprocessed_passages], **kwargs))

    a = (0.0, 0.0)
    expected = [("", "", a), ("", "", a), ("", "", a), ("d", "d", a), ("", "", a), ("", "", a)]
    assert _get_nonaligned(passages[0]) == expected
    assert _get_nonaligned(passages[0][:]) == _get_nonaligned(passages[0])
    assert _get_nonaligned(passages[1]) == [("", "", a), ("h", "h", a), ("", "jk", a)]
    assert _get_nonaligned(passages[1][:]) == _get_nonaligned(passages[1])
    assert _get_nonaligned(passages[2]) == [("jk", "jk", a), ("", "", a)]
    assert _get_nonaligned(passages[3]) == [("", "", a), ("no", "nop", a)]
    assert _get_nonaligned(passages[4]) == [
        ("p", "nop", a),
        ("r", "rstuvwxyz", (0.0, 7.583958148956299)),
    ]

    # TEST: Test `spans` get the correct span.
    assert _get_nonaligned(passages[0][2:4][:]) == [("", "", a), ("d", "d", a), ("", "", a)]
    assert _get_nonaligned(passages[1][1][:]) == [("h", "h", a), ("", "jk", a)]


def test_passage_span__nonalignment_spans__zero_alignments():
    """Test `run.data._loader.Passage` and `run.data._loader.Span` get the correct nonalignments if
    one of the passages has zero alignments."""
    script = "abcdef"
    make = functools.partial(_make_unprocessed_passage, transcript=script)
    unprocessed_passages = []

    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["b"]))  # NOTE: split='abc'
    unprocessed_passages.append(make("", []))
    split, script = script[:3], script[3:]
    unprocessed_passages.append(make(split, ["e"]))  # NOTE: split='abc'

    kwargs = {"script": False, "transcript": True, "audio": True}
    passages = list(make_passages([unprocessed_passages], **kwargs))

    a = (0.0, 0.0)
    assert _get_nonaligned(passages[0]) == [("a", "a", a), ("c", "cd", a)]
    assert _get_nonaligned(passages[1]) == [("", "cd", a)]
    assert _get_nonaligned(passages[2]) == [("d", "cd", a), ("f", "f", (0.0, 7.583958148956299))]


def test__overlap():
    """ Test `_overlap` computes the percentage of overlap between two spans correctly. """
    assert SpanGenerator._overlap(1, 2, 0, 1) == 0.0
    assert SpanGenerator._overlap(0, 1, 0, 1) == 1.0
    assert SpanGenerator._overlap(0.5, 0.5, 0, 1) == 1.0
    assert SpanGenerator._overlap(-1, 0, 0, 1) == 0.0
    assert SpanGenerator._overlap(-0.5, 0.5, 0, 1) == 0.5
    assert SpanGenerator._overlap(0.5, 1.5, 0, 1) == 0.5
    assert SpanGenerator._overlap(0, 1, 0.5, 2) == 0.5
    assert SpanGenerator._overlap(0, 1, 0, 2) == 1.0
    assert SpanGenerator._overlap(0, 1, -1.5, 0.5) == 0.5
    assert SpanGenerator._overlap(0, 1, -1, 1) == 1.0


# NOTE: `run.data._loader.conventional_dataset_loader` is tested in `lj_speech` as part of loading
# the dataset.
