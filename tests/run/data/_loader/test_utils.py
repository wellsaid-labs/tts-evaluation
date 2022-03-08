import functools
import math
import pathlib
import typing
from collections import Counter
from unittest import mock

import numpy as np
import pytest

import lib
from lib.utils import flatten_2d
from run.data import _loader
from run.data._loader.data_structures import Alignment
from run.data._loader.english import HILARY_NORIEGA
from run.data._loader.utils import SpanGenerator, get_non_speech_segments_and_cache
from tests._utils import (
    TEST_DATA_PATH,
    assert_uniform_distribution,
    get_audio_metadata_side_effect,
    subprocess_run_side_effect,
)
from tests.run._utils import make_passage

TEST_DATA_LJ = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
TEST_DATA_LJ_16_BIT = TEST_DATA_PATH / "audio" / "rate(lj_speech,24000).wav"


def _make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """Make an `Alignment` for testing."""
    return Alignment(script, audio, transcript)


def _make_alignments(
    alignments: typing.Tuple[typing.Tuple[int, int]]
) -> lib.utils.Tuple[Alignment]:
    """Make a tuple of `Alignment`(s) for testing."""
    return Alignment.stow([_make_alignment(a, a, a) for a in alignments])


def test_read_audio():
    """Test `_loader.read_audio` against a basic test case."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = _loader.read_audio(metadata)
    assert audio.dtype == np.float32
    assert audio.shape == (metadata.sample_rate * metadata.length,)


def test_read_audio__16_bit():
    """Test `_loader.read_audio` loads off-spec audio."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ_16_BIT)
    audio = _loader.read_audio(metadata)
    assert audio.dtype == np.float32
    assert audio.shape == (metadata.sample_rate * metadata.length,)


def test_normalize_audio_suffix():
    """Test `_loader.normalize_audio_suffix` normalizes suffix."""
    expected = pathlib.Path("directory/test.wav")
    input_ = pathlib.Path("directory/test.mp3")
    assert _loader.normalize_audio_suffix(input_, ".wav") == expected


def test_is_normalized_audio_file():
    """Test `_loader.is_normalized_audio_file` checks is audio is normalized."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    format_ = lib.audio.AudioFormat(
        sample_rate=24000,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
    )
    suffix = ".wav"
    assert _loader.is_normalized_audio_file(metadata, format_, suffix)
    metadata_16_bit = lib.audio.get_audio_metadata(TEST_DATA_LJ_16_BIT)
    assert not _loader.is_normalized_audio_file(metadata_16_bit, format_, suffix)


def test__cache_path():
    """Test `_cache_path` against basic cases."""
    path = _loader.utils._cache_path(TEST_DATA_LJ, "cache", ".json", ".tts_cache", test=1)
    expected = TEST_DATA_LJ.parent / ".tts_cache" / f"cache({TEST_DATA_LJ.stem},test=1).json"
    assert path == expected


def test_get_non_speech_segments_and_cache():
    """Test that `get_non_speech_segments_and_cache` uses the cache."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    kwargs = dict(low_cut=300, frame_length=50, hop_length=5, threshold=-60)
    segments = get_non_speech_segments_and_cache(metadata, **kwargs)
    intervals = [
        (0, 0.009958333333333333),
        (2.75, 2.9849583333333336),
        (4.885, 5.289958333333334),
        (7.38, 7.569958333333333),
        (7.55, 7.583958333333333),
    ]
    assert segments.intervals() == intervals

    with mock.patch("lib.audio.get_non_speech_segments") as module:
        # NOTE: `get_non_speech_segments` doesn't need to be called because the result is cached.
        module.return_value = None
        cached_segments = get_non_speech_segments_and_cache(metadata, **kwargs)
        assert segments.intervals() == cached_segments.intervals()

        # NOTE: `get_non_speech_segments` must be called because the `kwargs` changed.
        with pytest.raises(ValueError):
            _kwargs = {**kwargs, "low_cut": 0}
            cached_segments = get_non_speech_segments_and_cache(metadata, **_kwargs)


def test__maybe_normalize_audio_and_cache():
    """Test that `_maybe_normalize_audio_and_cache` uses the cache."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    kwargs = dict(
        suffix=".wav",
        data_type=lib.audio.AudioDataType.FLOATING_POINT,
        bits=16,
        sample_rate=metadata.sample_rate,
        num_channels=metadata.num_channels,
        encoding=metadata.encoding,
        bit_rate=metadata.bit_rate,
        precision=metadata.precision,
    )
    path = _loader.utils.maybe_normalize_audio_and_cache(metadata, **kwargs)
    assert path == metadata.path

    other_metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ_16_BIT)
    other_path = _loader.utils.maybe_normalize_audio_and_cache(other_metadata, **kwargs)
    assert other_path != other_metadata, "File wasn't normalized."

    with mock.patch("lib.audio.normalize_audio") as module:
        # NOTE: `normalize_audio` doesn't need to be called because the result is cached.
        module.return_value = None
        another_path = _loader.utils.maybe_normalize_audio_and_cache(other_metadata, **kwargs)
        assert other_path == another_path


def test_span_generator():
    """Test `SpanGenerator` samples uniformly given a uniform distribution of alignments."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3))))]
    iterator = SpanGenerator(dataset, max_seconds=10, max_pause=math.inf)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__empty():
    """Test `SpanGenerator` handles an empty list."""
    iterator = SpanGenerator([], max_seconds=10, max_pause=math.inf)
    assert next(iterator, None) is None


def test_span_generator__zero():
    """Test `SpanGenerator` handles alignments of zero length."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((1, 1),))),
        make_passage(_make_alignments(((1, 2),))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10, max_pause=math.inf)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments_ = [dataset[0].alignments, dataset[-1].alignments]
    alignments = [list(typing.cast(typing.Tuple[Alignment], a)) for a in alignments_]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__singular():
    """Test `SpanGenerator` handles multiple passages with a singular alignment of varying
    lengths."""
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((0, 10),))),
        make_passage(_make_alignments(((0, 5),))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10, max_pause=math.inf)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__multiple_multiple():
    """Test `SpanGenerator` handles multiple scripts with a uniform alignment distribution."""
    dataset = [
        make_passage(_make_alignments(((0, 1), (1, 2), (2, 3)))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=10, max_pause=math.inf)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    alignments = [list(typing.cast(typing.Tuple[Alignment], d.alignments)) for d in dataset]
    assert set(counter.keys()) == set(flatten_2d(alignments))
    assert_uniform_distribution(counter, abs=0.015)


def test_span_generator__pause():
    """Test `SpanGenerator` samples uniformly despite a large pause."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41))))]
    iterator = SpanGenerator(dataset, max_seconds=4, max_pause=math.inf)
    counter: typing.Counter[Alignment] = Counter()
    for _ in range(10000):
        span = next(iterator)
        counter.update(span.passage.alignments[span.slice])
    assert set(counter.keys()) == set(typing.cast(typing.Tuple[Alignment], dataset[0].alignments))
    assert_uniform_distribution(counter, abs=0.02)


def test_span_generator__ignore_pause():
    """Test `SpanGenerator` `max_pause` can ignore pauses."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41))))]
    iterator = SpanGenerator(dataset, max_seconds=50, max_pause=1)
    for _ in range(1000):
        span = next(iterator)
        assert span.slice.stop - span.slice.start <= 3

    with pytest.raises(AssertionError):
        iterator = SpanGenerator(dataset, max_seconds=50, max_pause=50)
        for _ in range(1000):
            span = next(iterator)
            assert span.slice.stop - span.slice.start <= 3


def test_span_generator__multiple_unequal_passages__large_max_seconds():
    """Test `SpanGenerator` samples uniformly despite unequal passage sizes, and large max seconds.

    NOTE: With a large enough `max_seconds`, the entire passage should be sampled most of the time.
    """
    dataset = [
        make_passage(_make_alignments(((0, 1),))),
        make_passage(_make_alignments(((3, 4), (4, 5), (5, 6)))),
    ]
    iterator = SpanGenerator(dataset, max_seconds=1000000, max_pause=math.inf)

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
    """Test `SpanGenerator` samples uniformly despite unequal alignment sizes."""
    dataset = [make_passage(_make_alignments(((0, 1), (1, 5), (5, 20))))]
    iterator = SpanGenerator(dataset, max_seconds=20, max_pause=math.inf)
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
    iterator = SpanGenerator(dataset, max_seconds=10, max_pause=math.inf)
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
    """Test `_loader.dataset_loader` loads a dataset."""
    mock_get_audio_metadata.side_effect = get_audio_metadata_side_effect
    mock_run.side_effect = functools.partial(subprocess_run_side_effect, _command="gsutil")
    passages = _loader.dataset_loader(
        TEST_DATA_PATH / "datasets",
        "hilary_noriega",
        "",
        HILARY_NORIEGA,
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
    assert passages[0].audio_file.sample_rate == 24000
    assert passages[0].audio_file.encoding == lib.audio.AudioEncoding.PCM_FLOAT_32_BIT
    assert path.stem in passages[0].audio_file.path.stem
    assert passages[0].speaker == HILARY_NORIEGA
    assert passages[0].script == "Author of the danger trail, Philip Steels, etc."
    assert passages[0].transcript == (
        "author of the danger Trail Philip Steels Etc. Not at this particular case Tom "
        "apologized Whitmore for the 20th time that evening the two men shook hands"
    )
    assert passages[0].alignments == Alignment.stow(alignments)
    assert passages[0].nonalignments == Alignment.stow(nonalignments)
    assert passages[0].other_metadata == {"Index": 0, "Source": "CMU", "Title": "CMU"}
    assert passages[1].script == "Not at this particular case, Tom, apologized Whittemore."


def test__overlap():
    """Test `_overlap` computes the percentage of overlap between two spans correctly."""
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


# NOTE: `_loader.conventional_dataset_loader` is tested in `lj_speech` as part of loading
# the dataset.
