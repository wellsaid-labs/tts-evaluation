import functools
import pathlib
import pickle
import tempfile
import typing
from collections import Counter
from unittest import mock

import hparams
import numpy as np
import pytest
from hparams import HParams

import lib
from lib.utils import Interval, Timeline, flatten_2d
from run.data import _loader
from run.data._loader import Alignment, Passage, Span, SpanGenerator
from run.data._loader.utils import (
    IntInt,
    UnprocessedPassage,
    _check_updated_script,
    _make_speech_segments,
    get_non_speech_segments_and_cache,
    has_a_mistranscription,
    make_passages,
)
from tests._utils import (
    TEST_DATA_PATH,
    assert_uniform_distribution,
    get_audio_metadata_side_effect,
    make_passage,
    make_unprocessed_passage,
    subprocess_run_side_effect,
)

TEST_DATA_LJ = TEST_DATA_PATH / "audio" / "bit(rate(lj_speech,24000),32).wav"
TEST_DATA_LJ_16_BIT = TEST_DATA_PATH / "audio" / "rate(lj_speech,24000).wav"


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration. """
    suffix = ".wav"
    data_type = lib.audio.AudioDataType.FLOATING_POINT
    bits = 32
    format_ = lib.audio.AudioFormat(
        sample_rate=24000,
        num_channels=1,
        encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
        bit_rate="768k",
        precision="25-bit",
    )
    non_speech_segment_frame_length = 50
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name)
    config = {
        _loader.utils.normalize_audio_suffix: HParams(suffix=suffix),
        _loader.utils.normalize_audio: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            sample_rate=format_.sample_rate,
            num_channels=format_.num_channels,
        ),
        _loader.utils.is_normalized_audio_file: HParams(audio_format=format_, suffix=suffix),
        _loader.utils._cache_path: HParams(cache_dir=temp_dir_path),
        # NOTE: `get_non_speech_segments` parameters are set based on `vad_workbook.py`. They
        # are applicable to most datasets with little to no noise.
        _loader.utils.get_non_speech_segments_and_cache: HParams(
            low_cut=300, frame_length=non_speech_segment_frame_length, hop_length=5, threshold=-60
        ),
        _loader.utils._make_speech_segments: HParams(
            padding=(non_speech_segment_frame_length / 2) / 1000
        ),
        _loader.utils.maybe_normalize_audio_and_cache: HParams(
            suffix=suffix,
            data_type=data_type,
            bits=bits,
            sample_rate=format_.sample_rate,
            num_channels=format_.num_channels,
        ),
    }
    hparams.add_config(config)
    yield
    hparams.clear_config()


def _make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """ Make an `Alignment` for testing. """
    return Alignment(script, audio, transcript)


def _make_alignments(alignments=typing.Tuple[typing.Tuple[int, int]]) -> lib.utils.Tuple[Alignment]:
    """ Make a tuple of `Alignment`(s) for testing. """
    return Alignment.stow([_make_alignment(a, a, a) for a in alignments])


def test_read_audio():
    """ Test `_loader.read_audio` against a basic test case. """
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    audio = _loader.read_audio(metadata)
    assert audio.dtype == np.float32
    assert audio.shape == (metadata.sample_rate * metadata.length,)


def test_read_audio__16_bit():
    """ Test `_loader.read_audio` loads off-spec audio. """
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
        (2.75, 2.984958333333333),
        (4.885, 5.289958333333334),
        (7.38, 7.569958333333333),
    ]
    assert segments.intervals() == set([Interval(s, s) for s in intervals])

    with mock.patch("lib.audio.get_non_speech_segments") as module:
        # NOTE: `get_non_speech_segments` doesn't need to be called because the result is cached.
        module.return_value = None
        cached_segments = get_non_speech_segments_and_cache(metadata, **kwargs)
        assert segments.intervals() == cached_segments.intervals()

        # NOTE: `get_non_speech_segments` must be called because the `kwargs` changed.
        with pytest.raises(ValueError):
            _kwargs = {**kwargs, "low_cut": 0}
            cached_segments = get_non_speech_segments_and_cache(metadata, **_kwargs)


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
    filtered = _loader.utils._filter_non_speech_segments(timeline, slices)
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
    passage = _loader.utils.Passage(
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


def test__maybe_normalize_vo_script():
    """ Test `_maybe_normalize_vo_script` against some basic cases. """
    normal_script = "abc"
    assert _loader.utils._maybe_normalize_vo_script(normal_script) == normal_script
    script = "áƀć"
    assert _loader.utils._maybe_normalize_vo_script(script) == normal_script


@mock.patch("run.data._loader.utils.logger.error")
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


def test__maybe_normalize_audio_and_cache():
    """Test that `_maybe_normalize_audio_and_cache` uses the cache."""
    metadata = lib.audio.get_audio_metadata(TEST_DATA_LJ)
    kwargs = dict(
        suffix=".wav",
        data_type=lib.audio.AudioDataType.FLOATING_POINT,
        bits=16,
        sample_rate=metadata.sample_rate,
        num_channels=metadata.num_channels,
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
    """Test `SpanGenerator` handles multiple passages with a singular alignment of varying
    lengths."""
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
    """Test `SpanGenerator` handles multiple scripts with a uniform alignment distribution."""
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
    """Test `SpanGenerator` samples uniformly despite unequal passage sizes, and large max seconds.

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
    """ Test `_loader.dataset_loader` loads a dataset. """
    mock_get_audio_metadata.side_effect = get_audio_metadata_side_effect
    mock_run.side_effect = functools.partial(subprocess_run_side_effect, _command="gsutil")
    passages = _loader.dataset_loader(
        TEST_DATA_PATH / "datasets",
        "hilary_noriega",
        "",
        _loader.HILARY_NORIEGA,
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
    assert passages[0].speaker == _loader.HILARY_NORIEGA
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
    """Test `_loader.Passage` and `_loader.Span` are the same after a identity operation."""
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
        speaker=_loader.Speaker(""),
        script=script,
        transcript=transcript,
        alignments=tuple(_make_alignment(*arg) for arg in found),
    )


def _get_nonaligned(
    span: typing.Union[Span, Passage]
) -> typing.List[typing.Optional[typing.Tuple[str, str, typing.Tuple[float, float]]]]:
    return [
        None if s is None else (s.script, s.transcript, (s.audio_start, s.audio_stop))
        for s in span.nonalignment_spans().spans
    ]


def test_passage_span__nonalignment_spans():
    """Test `_loader.Passage` and `_loader.Span` get the correct nonalignments under a variety of
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
    """Test `_loader.Passage` and `_loader.Span` get the correct nonalignments if one of the
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


# NOTE: `_loader.conventional_dataset_loader` is tested in `lj_speech` as part of loading
# the dataset.
