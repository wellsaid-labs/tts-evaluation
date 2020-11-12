import pathlib
import typing
from collections import Counter
from unittest import mock

import numpy as np
import pytest

import lib
from lib.datasets import Alignment
from lib.datasets.utils import _overlap
from lib.utils import flatten
from tests import _utils


def _make_example(
    alignments=(),
    audio_path=pathlib.Path("."),
    speaker=lib.datasets.Speaker(""),
    script="",
    metadata={},
) -> lib.datasets.Example:
    """ Make a `lib.datasets.Example` for testing. """
    alignments = tuple([Alignment(a, a, a) for a in alignments])
    return lib.datasets.Example(audio_path, speaker, script, script, alignments, metadata)


def test_span_generator():
    """Test `lib.datasets.span_generator` samples uniformly given a uniform distribution of
    alignments."""
    dataset = [_make_example(((0, 1), (1, 2), (2, 3)))]
    iterator = lib.datasets.span_generator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    _utils.assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__empty():
    """ Test `lib.datasets.span_generator` handles an empty list. """
    iterator = lib.datasets.span_generator([], max_seconds=10)
    assert next(iterator, None) is None


def test_span_generator__zero():
    """Test `lib.datasets.span_generator` handles alignments of zero length."""
    dataset = [
        _make_example(((0, 1),)),
        _make_example(((1, 1),)),
        _make_example(((1, 2),)),
    ]
    iterator = lib.datasets.span_generator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    alignments_ = [dataset[0].alignments, dataset[-1].alignments]
    alignments = [list(typing.cast(typing.Tuple[Alignment], a)) for a in alignments_]
    assert set(counter.keys()) == set(flatten(alignments))
    _utils.assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__singular():
    """Test `lib.datasets.span_generator` handles multiple examples with a singular alignment
    of varying lengths."""
    dataset = [
        _make_example(((0, 1),)),
        _make_example(((0, 10),)),
        _make_example(((0, 5),)),
    ]
    iterator = lib.datasets.span_generator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten(alignments))
    _utils.assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__multiple_multiple():
    """Test `lib.datasets.span_generator` handles multiple scripts with a uniform alignment
    distribution."""
    dataset = [
        _make_example(((0, 1), (1, 2), (2, 3))),
        _make_example(((3, 4), (4, 5), (5, 6))),
    ]
    iterator = lib.datasets.span_generator(dataset, max_seconds=10)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten(alignments))
    _utils.assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__pause():
    """ Test `lib.datasets.span_generator` samples uniformly despite a large pause. """
    dataset = [_make_example(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41)))]
    iterator = lib.datasets.span_generator(dataset, max_seconds=4)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    _utils.assert_uniform_distribution(counter, abs=0.02)


def test_span_generator__multiple_unequal_examples__large_max_seconds():
    """Test `lib.datasets.span_generator` samples uniformly despite unequal example sizes,
    and large max seconds.

    NOTE: With a large enough `max_seconds`, the entire example should be sampled most of the time.
    """
    dataset = [_make_example(((0, 1),)), _make_example(((3, 4), (4, 5), (5, 6)))]
    iterator = lib.datasets.span_generator(dataset, max_seconds=1000000)

    alignments_counter: typing.Counter[Alignment] = Counter()
    spans_counter: typing.Counter[typing.Tuple[Alignment, ...]] = Counter()
    num_examples = 10000
    for _ in range(num_examples):
        span = next(iterator)
        slice_ = span.example.alignments[slice(*span.slice)]
        alignments_counter.update(slice_)
        spans_counter[slice_] += 1

    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(alignments_counter.keys()) == set(flatten(alignments))
    _utils.assert_uniform_distribution(alignments_counter, abs=0.015)

    for example in dataset:
        assert spans_counter[example.alignments] / num_examples == pytest.approx(
            1 / len(dataset), abs=0.015
        )


def test_span_generator__unequal_alignment_sizes():
    """ Test `lib.datasets.span_generator` samples uniformly despite unequal alignment sizes. """
    dataset = [_make_example(((0, 1), (1, 5), (5, 20)))]
    iterator = lib.datasets.span_generator(dataset, max_seconds=20)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.example.alignments[slice(*span.slice)])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten(alignments))
    _utils.assert_uniform_distribution(counter, abs=0.015)


@mock.patch("lib.datasets.utils.subprocess.run", return_value=None)
def test_dataset_loader(_):
    """ Test `lib.datasets.dataset_loader` loads a dataset. """
    examples = lib.datasets.dataset_loader(
        _utils.TEST_DATA_PATH / "datasets",
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
    assert examples[0] == lib.datasets.Example(
        audio_path=_utils.TEST_DATA_PATH / "datasets/hilary_noriega/recordings/Script 1.wav",
        speaker=lib.datasets.HILARY_NORIEGA,
        script="Author of the danger trail, Philip Steels, etc.",
        transcript=(
            "author of the danger Trail Philip Steels Etc. Not at this particular case Tom "
            "apologized Whitmore for the 20th time that evening the two men shook hands"
        ),
        alignments=tuple(alignments),
        other_metadata={"Index": 0, "Source": "CMU", "Title": "CMU"},
    )


def test_example_span__identity():
    """Test `lib.datasets.Example` and `lib.datasets.Span` are the same after a identity
    operation."""
    audio_path = _utils.TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
    metadata = lib.audio.get_audio_metadata([audio_path])[0]
    script = (
        "The examination and testimony of the experts enabled the Commission to conclude "
        "that five shots may have been fired,"
    )
    example = lib.datasets.Example(
        audio_path=audio_path,
        speaker=lib.datasets.LINDA_JOHNSON,
        script=script,
        transcript=script,
        alignments=(
            lib.datasets.Alignment((0, len(script)), (0.0, metadata.length), (0, len(script))),
        ),
        other_metadata={"chapter": 37},
    )
    span = lib.datasets.Span(example, (0, 1))
    assert example.script == span.script
    assert example.transcript == span.transcript
    assert example.alignments == span.alignments
    assert example.speaker == span.speaker
    assert example.audio_path == span.audio_path
    assert example.other_metadata == span.other_metadata
    np.testing.assert_almost_equal(example.audio, span.audio)


def test__overlap():
    """ Test `_overlap` computes the percentage of overlap between two spans correctly. """
    assert _overlap((0, 1), (1, 2)) == 0.0
    assert _overlap((0, 1), (0, 1)) == 1.0
    assert _overlap((0, 1), (0.5, 0.5)) == 1.0
    assert _overlap((0, 1), (-1, 0)) == 0.0
    assert _overlap((0, 1), (-0.5, 0.5)) == 0.5
    assert _overlap((0, 1), (0.5, 1.5)) == 0.5


# NOTE: `lib.datasets.conventional_dataset_loader` is tested in `lj_speech` as part of loading the
# dataset.
