import collections
import contextlib
import logging
import math
import random
import time
import typing

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tqdm
from torchnlp.random import fork_rng

import lib
from lib.audio import amp_to_db, signal_to_rms
from lib.utils import clamp, flatten_2d, seconds_to_str
from run._config import Dataset
from run._streamlit import (
    dataset_passages,
    fast_grapheme_to_phoneme,
    make_interval_chart,
    make_signal_chart,
    read_wave_audio,
    span_audio,
)
from run.data._loader import Passage, Span, voiced_nonalignment_spans

logger = logging.getLogger(__name__)

ALIGNMENT_PRECISION = 0.1


@contextlib.contextmanager
def st_expander(label):
    with st.beta_expander(label):
        try:
            start = time.time()
            logger.info("Visualizing '%s'...", label)
            yield label
            elapsed = seconds_to_str(time.time() - start)
            logger.info("`%s` ran for %s", label, elapsed)
        except GeneratorExit:
            pass


_RandomSampleVar = typing.TypeVar("_RandomSampleVar")


def random_sample(
    list_: typing.List[_RandomSampleVar], max_samples: int, seed: int = 123
) -> typing.List[_RandomSampleVar]:
    """ Deterministic random sample. """
    with fork_rng(seed):
        return random.sample(list_, min(len(list_), max_samples))


def _ngrams(list_: typing.Sequence, n: int) -> typing.Iterator[slice]:
    """ Learn more: https://en.wikipedia.org/wiki/N-gram. """
    yield from (slice(i, i + n) for i in range(len(list_) - n + 1))


def _signal_to_db_rms_level(signal: np.ndarray) -> float:
    """ Get the dB RMS level of `signal`."""
    if signal.shape[0] == 0:
        return math.nan
    return round(amp_to_db(float(signal_to_rms(signal))), 1)


def _signal_to_loudness(signal: np.ndarray, sample_rate: int, block_size: float = 0.4) -> float:
    """Get the loudness in LUFS of `signal`."""
    meter = lib.audio.get_pyloudnorm_meter(block_size=block_size, sample_rate=sample_rate)
    if signal.shape[0] >= lib.audio.sec_to_sample(block_size, sample_rate):
        return round(meter.integrated_loudness(signal), 1)
    return math.nan


def bucket_and_chart(
    values: typing.Union[typing.List[float], typing.List[int]],
    labels: typing.List[str],
    bucket_size: float = 1,
    x: str = "Buckets",
    y: str = "Count",
    ndigits: int = 7,
) -> alt.Chart:
    """ Bucket `values` and create a bar chart. """
    assert len(values) == len(labels)
    data = [(v, l) for v, l in zip(values, labels) if not math.isnan(v)]
    if len(data) != len(values):
        logger.warning("Ignoring %d NaNs...", len(values) - len(data))
    buckets = collections.defaultdict(int)
    for value, label in data:
        buckets[(round(lib.utils.round_(value, bucket_size), ndigits), label)] += 1
    data = sorted([(l, b, v) for (b, l), v in buckets.items()])
    df = pd.DataFrame(data, columns=["label", "bucket", "count"])
    # NOTE: x-axis is "quantitative" due to this warning by Vega-Lite, in the online editor:
    # "[Warning] Scale bindings are currently only supported for scales with unbinned,
    #  continuous domains."
    # NOTE: We perform aggregation outside of `altair` due to this warning by Vega-Lite, in the
    # online editor:
    # "[Warning] Cannot project a selection on encoding channel "y" as it uses an aggregate
    #  function ("sum")."
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("bucket", type="quantitative", title=x + " (binned)"),
            y=alt.Y(field="count", type="quantitative", title=y, stack=True),
            color=alt.Color(field="label", type="nominal", title="Label"),
            tooltip=["label", "bucket", "count"],
        )
        .interactive()
    )


def dataset_audio_files(dataset: Dataset) -> typing.Set[lib.audio.AudioMetadata]:
    return set(flatten_2d([[p.audio_file for p in v] for v in dataset.values()]))


def dataset_total_aligned_audio(dataset: Dataset) -> float:
    return sum(flatten_2d([[p.aligned_audio_length() for p in v] for v in dataset.values()]))


def dataset_total_audio(dataset: Dataset) -> float:
    return sum([m.length for m in dataset_audio_files(dataset)])


def passage_alignment_speech_segments(passage: Passage) -> typing.List[Span]:
    """Get `passage` speech segments seperated by alignment pauses."""
    start = 0
    segments: typing.List[Span] = []
    spans, spans_is_voiced = voiced_nonalignment_spans(passage)
    for i, (span, is_voiced) in enumerate(zip(spans.spans[1:-1], spans_is_voiced[1:-1])):
        if not is_voiced and span.audio_length > 0:
            segments.append(passage.span(slice(start, i + 1)))
            start = i + 1
    segments.append(passage.span(slice(start, len(passage.alignments))))

    # NOTE: Verify the output is correct.
    expected = list(range(0, len(passage.alignments)))
    message = "Every alignment must be included."
    assert [i for s in segments for i in range(s.slice.start, s.slice.stop)] == expected, message
    if len(segments) > 0:
        message = "Every speech segment needs to be seperated by a non speech segment."
        pairs = zip(segments, segments[1:])
        assert all(b.audio_start - a.audio_stop > 0 for a, b in pairs), message

    return segments


def passages_alignment_ngrams(passages: typing.List[Passage], n: int = 1) -> typing.Iterator[Span]:
    """ Get ngram `Span`s with `n` alignments. """
    for passage in tqdm.tqdm(passages):
        passage: Passage
        yield from (passage.span(s) for s in _ngrams(passage.alignments, n=n))


def dataset_num_alignments(dataset: Dataset) -> int:
    """ Get number of `Alignment`s in `dataset`. """
    return sum([len(p.alignments) for p in dataset_passages(dataset)])


def dataset_coverage(dataset: Dataset, spans: typing.List[Span]) -> float:
    """ Get the percentage of the `dataset` these `spans` cover. """
    logger.info("Getting span coverage of dataset...")
    alignments = set()
    passage_ids = set(id(p) for d in dataset.values() for p in d)
    for span in spans:
        passage_id = id(span.passage)
        assert passage_id in passage_ids, "Passage not found in `dataset`."
        alignments.update((passage_id, i) for i in range(span.slice.start, span.slice.stop))
    return len(alignments) / dataset_num_alignments(dataset)


def span_mistranscriptions(span: Span) -> typing.List[typing.Tuple[str, str]]:
    """ Get a slices of script and transcript that were not aligned. """
    spans, spans_is_voiced = voiced_nonalignment_spans(span)
    return [(s.script, s.transcript) for s, i in zip(spans.spans, spans_is_voiced) if i]


def span_pauses(span: Span) -> typing.List[float]:
    """ Get the length of pauses between alignments in `span`. """
    return [s.audio_length for s in span.nonalignment_spans().spans[1:-1]]


def span_total_silence(span: Span) -> float:
    return sum(span_pauses(span))


def span_max_silence(span: Span) -> float:
    pauses = span_pauses(span)
    return max(pauses) if len(pauses) > 0 else 0.0


def span_audio_slice(span: Span, second: float, lengths: typing.Tuple[float, float]) -> np.ndarray:
    """ Get the audio at `second`. """
    clamp_ = lambda x: clamp(x, min_=0, max_=span.passage.audio_file.length)
    start = clamp_(span.audio_start + second - lengths[0])
    end = clamp_(span.audio_start + second + lengths[1])
    return read_wave_audio(span.passage.audio_file, start, end - start)


def span_audio_boundary_rms_level(
    span: Span,
    lengths: typing.Tuple[float, float] = (ALIGNMENT_PRECISION / 2, ALIGNMENT_PRECISION / 2),
) -> typing.Tuple[float, float]:
    """ Get the audio RMS level at the boundaries. """
    return (span_audio_left_rms_level(span, lengths), span_audio_right_rms_level(span, lengths))


def span_audio_left_rms_level(
    span: Span,
    lengths: typing.Tuple[float, float] = (ALIGNMENT_PRECISION / 2, ALIGNMENT_PRECISION / 2),
) -> float:
    """ Get the audio RMS level at the left boundary. """
    return _signal_to_db_rms_level(span_audio_slice(span, 0, lengths))


def span_audio_right_rms_level(
    span: Span,
    lengths: typing.Tuple[float, float] = (ALIGNMENT_PRECISION / 2, ALIGNMENT_PRECISION / 2),
) -> float:
    """ Get the audio RMS level at the right boundary. """
    return _signal_to_db_rms_level(span_audio_slice(span, span.audio_length, lengths))


def span_audio_rms_level(span: Span) -> float:
    return _signal_to_db_rms_level(span_audio(span))


def span_audio_loudness(span: Span) -> float:
    return _signal_to_loudness(span_audio(span), span.audio_file.sample_rate)


def span_visualize_signal(span: Span) -> alt.Chart:
    """ Visualize `span` signal as a waveform chart with lines for alignments. """
    signal_chart = make_signal_chart(span_audio(span), span.audio_file.sample_rate)
    intervals = [a.audio for a in span.alignments]
    return (signal_chart + make_interval_chart(intervals)).interactive()


def span_sec_per_char(span: Span):
    """Get the aligned seconds per character.

    TODO: Consider using `non_speech_segments` to remove silences from this calculation.
    """
    return round(sum(a.audio[-1] - a.audio[0] for a in span.alignments) / len(span.script), 2)


def span_sec_per_phon(span: Span):
    """ Get the aligned seconds per character. """
    try:
        phonemes = fast_grapheme_to_phoneme(span.script)
    except AssertionError:
        return math.nan
    return sum(a.audio[-1] - a.audio[0] for a in span.alignments) / len(phonemes.split("|"))
