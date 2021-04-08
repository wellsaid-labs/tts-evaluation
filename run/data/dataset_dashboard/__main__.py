""" Streamlit application for reviewing the dataset.

TODO:
- Add an analysis of `speech_segments`.
  - How long are speech segments? Are any speech segments... too long? Do all speakers have
    good cuts for speech segments?
  - How much of the dataset do they cover? Are they missing parts of the dataset?
- Add an analysis of `non_speech_segments`.
  - Can we use them to filter out bad alignments?
  - Are there any examples with extensive silences? Would those silences cause issues?
  - What's the longest a speaker will go without a `non_speech_segments`?
    Does that vary per speaker?
  - Could we more accurately calculate "speed" by removing `non_speech_segments`? Is this method
    inclusive of speakers that have more background noise which would have less
    `non_speech_segments`? Could we use it for a more accurate filter on alignments that are too
    fast?
- Add an analysis of `lib.utils.is_voiced`.
  - What are the longest nonalignments that are not considered "voiced"? Are they actually, voiced?
  - Is `is_voiced` accurate? Does it miss anything that is voiced?
- Where could errors be hiding that could cause poor performance?
  - Could long alignments with multiple words be suspect?
  - Could alignments with long pauses in them be suspect?
  - Could spans with long pauses in them be suspect? Does it matter how we calculate those pauses?
    We could calculate pauses with `non_speech_segments` or `alignments`?

Usage:
    $ PYTHONPATH=. streamlit run run/data/dataset_dashboard/__main__.py --runner.magicEnabled=false
"""

import collections
import functools
import logging
import math
import typing

import altair as alt
import pandas as pd
import streamlit as st
import tqdm
from torchnlp.random import fork_rng

import lib
import run
from lib.utils import flatten_2d, mazel_tov, round_, seconds_to_str
from run._config import Dataset
from run._streamlit import (
    audio_to_html,
    clear_session_cache,
    get_dataset,
    map_,
    rmtree_streamlit_static_temp_dir,
)
from run.data._loader import DATASETS, Span, has_a_mistranscription
from run.data.dataset_dashboard import _utils as utils

lib.environment.set_basic_logging_config(reset=True)
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


@lib.utils.log_runtime
def _get_spans(dataset: Dataset, num_samples: int) -> typing.List[Span]:
    """Generate spans from our datasets."""
    logger.info("Generating spans...")
    generator = run._utils.SpanGenerator(dataset, include_span=lambda *a: True)
    with fork_rng(123):
        spans = [next(generator) for _ in tqdm.tqdm(range(num_samples), total=num_samples)]
    logger.info(f"Finished generating spans! {mazel_tov()}")
    return spans


_Columns = typing.Dict[str, typing.List[typing.Any]]


def _default_span_columns(spans: typing.List[Span]) -> _Columns:
    """ Get default columns for `_write_span_table`. """
    logger.info("Getting %d generic span columns...", len(spans))
    _round: typing.Callable[[float, Span], float]
    _round = lambda a, s: round_(a, 1 / s.audio_file.sample_rate)
    columns = [
        ("script", [s.script for s in spans]),
        ("speaker", [s.speaker.label for s in spans]),
        ("length", [_round(s.audio_length, s) for s in spans]),
        ("mistranscriptions", [utils.span_mistranscriptions(s) for s in spans]),
        ("seconds", [[round(i.audio_length, 2) for i in s] for s in spans]),
        ("speed", [[utils.span_sec_per_char(i) for i in s] for s in spans]),
    ]
    return collections.OrderedDict(columns)


def _write_span_table(
    spans: typing.List[Span],
    other_columns: typing.Dict[str, typing.List[typing.Any]] = {},
    default_columns: typing.Callable[[typing.List[Span]], _Columns] = _default_span_columns,
    audio_column="audio",
):
    """Visualize spans as a table with a couple metadata columns."""
    assert len(spans) < 250, "Large tables are slow to visualize"
    if len(spans) == 0:
        return "No Data."
    logger.info("Visualizing %d spans..." % len(spans))
    dfs = [pd.DataFrame.from_dict(default_columns(spans)), pd.DataFrame.from_dict(other_columns)]
    df = pd.concat(dfs, axis=1)
    assert audio_column not in df.columns
    lib.utils.call_once(rmtree_streamlit_static_temp_dir)
    get_audio = lambda s: audio_to_html(utils.span_audio(s))
    df.insert(0, audio_column, map_(spans, get_audio))
    df = df.replace({"\n": "<br>"}, regex=True)
    # NOTE: Temporary fix based on this issue / pr: https://github.com/streamlit/streamlit/pull/3038
    html = "<style>tr{background-color: transparent !important;}</style>"
    st.markdown(html, unsafe_allow_html=True)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)


def _span_metric(
    spans: typing.List[Span],
    func: typing.Callable[[Span], float],
    name: str,
    unit_x: str,
    bucket_size: float,
    unit_y: str,
    max_rows: int,
    run_all: bool,
    note: str = "",
):
    """Visualize a span metric."""
    with utils.st_expander(f"Survey of {name} (in {unit_x.lower()})") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            return

        st.write(f"The {unit_y.lower()} count for each bucket:")
        results = map_(spans, func)
        labels = [s.speaker.label for s in spans]
        chart = utils.bucket_and_chart(results, labels, bucket_size, x=unit_x, y=unit_y + " Count")
        st.altair_chart(chart, use_container_width=True)
        if len(note) > 0:
            st.write(note)
        filtered = [(s, r) for s, r in zip(spans, results) if not math.isnan(r)]
        sorted_ = lambda **k: sorted(filtered, key=lambda i: i[1], **k)[:max_rows]
        for label_, data in (("smallest", sorted_()), ("largest", sorted_(reverse=True))):
            text = f"Show the {label_} valued {unit_y.lower()}(s)."
            if st.checkbox(text, key=label + label_, value=run_all):
                other_columns = {"value": [r[1] for r in data]}
                _write_span_table([r[0] for r in data], other_columns=other_columns)
                st.text("")


@lib.utils.log_runtime
def _analyze_dataset(dataset: Dataset, max_rows: int, run_all: bool):
    logger.info("Analyzing dataset...")
    st.header("Raw Dataset Analysis")
    st.markdown("In this section, we analyze the dataset prior to segmentation.")

    st.markdown(
        f"At a high-level, this dataset has:\n"
        f"- **{seconds_to_str(utils.dataset_total_audio(dataset))}** of audio\n"
        f"- **{seconds_to_str(utils.dataset_total_aligned_audio(dataset))}** of aligned audio\n"
        f"- **{utils.dataset_num_alignments(dataset):,}** alignments.\n"
    )

    question = "How many passage(s) do you want to analyze?"
    num_passages: int = st.sidebar.number_input(question, 0, None, 200)
    passages = utils.random_sample(list(utils.dataset_passages(dataset)), num_passages)
    unigrams = list(utils.passages_alignment_ngrams(passages, 1))
    trigrams = list(utils.passages_alignment_ngrams(passages, 3))
    st.markdown(
        f"Below this analyzes a random sample of **{len(passages):,}** passages with "
        f"{len(unigrams):,} alignments..."
    )

    span_metric_ = functools.partial(_span_metric, max_rows=max_rows, run_all=run_all)

    with utils.st_expander("Random Sample of Alignments") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()

        for span in utils.random_sample(trigrams, max_rows):
            cols = st.beta_columns([2, 1, 1])
            cols[0].altair_chart(utils.span_visualize_signal(span), use_container_width=True)
            cols[1].markdown(
                f"- Script: **{span.script}**\n"
                f"- Loudness: **{utils.span_audio_rms_level(span[1])}**\n"
                f"- Edge loudness: **{utils.span_audio_boundary_rms_level(span[1])}**\n"
                f"- Audio length: **{round(span[1].audio_length, 2)}**\n"
                f"- Num characters: **{len(span[1].script)}**\n"
            )
            cols[2].markdown(
                "\n\n".join([audio_to_html(utils.span_audio(s)) for s in [span[1], span]]),
                unsafe_allow_html=True,
            )

    nonalignment_spans = [s for p in passages for s in p.nonalignment_spans().spans[1:-1]]
    ratio = sum([s.audio_length > 0 for s in nonalignment_spans]) / len(nonalignment_spans)
    span_metric_(
        nonalignment_spans,
        lambda s: s.audio_length,
        "Nonalignment Length",
        "Seconds",
        utils.ALIGNMENT_PRECISION,
        "Nonalignment",
        note=f"**{ratio:.2%}** of pauses are longer than zero.",
    )

    segments = [s for p in passages for s in utils.passage_alignment_speech_segments(p)]
    threshold = 10
    max_length = max(s.audio_length for s in segments)
    above_threshold = sum(s.audio_length for s in segments if s.audio_length > threshold)
    total_seconds = sum(s.audio_length for s in segments)
    span_metric_(
        segments,
        lambda s: s.audio_length,
        "Alignment Speech Segments Length",
        "Seconds",
        utils.ALIGNMENT_PRECISION,
        "Speech Segment",
        note=(
            f"The maximum length, without pauses, is **{max_length:.2f}** seconds.\n\n"
            f"The sum of segments without a pause, longer than {threshold} seconds, "
            f"is **{above_threshold:.2f}** out of **{total_seconds:.2f}** seconds."
        ),
    )

    question = "How many alignment(s) do you want to analyze?"
    num_alignments: int = st.sidebar.number_input(question, 0, None, 200)
    if st.sidebar.checkbox("Only single word alignments", key="single-word-alignments", value=True):
        unigrams = [u for u in unigrams if " " not in u.script]
    samples = utils.random_sample(unigrams, num_alignments)
    st.text("")
    st.markdown(
        f"Below this analyzes a random sample of **{len(samples):,}** alignments "
        f"of {len(unigrams):,} alignments..."
    )

    with utils.st_expander("Random Sample of Alignments (Tabular)"):
        _write_span_table(samples[:max_rows])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (lambda s: s.audio_length, "Alignment Length", "Seconds", utils.ALIGNMENT_PRECISION),
        (lambda s: len(s.script), "Alignment Length", "Characters", 1),
        (utils.span_sec_per_char, "Alignment Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Alignment Speed", "Seconds per phoneme", 0.01),
        (utils.span_audio_rms_level, "Alignment Loudness", "Decibels", 1),
        (utils.span_audio_left_rms_level, "Alignment Onset Loudness", "Decibels", 5),
        (utils.span_audio_right_rms_level, "Alignment Outset Loudness", "Decibels", 5),
    ]
    for args in sections:
        span_metric_(samples, *args, unit_y="Alignment")

    with utils.st_expander("Random Sample of Filtered Alignments"):
        is_include: typing.Callable[[Span], bool]
        is_include = lambda s: s.audio_length > 0.1 and utils.span_sec_per_char(s) >= 0.04
        filtered = [s for s in samples if is_include(s)]
        st.write(f"Filtered out {1 - (len(filtered) / len(samples)):.2%} of alignments.")
        _write_span_table(filtered[:max_rows])

    logger.info(f"Finished analyzing dataset! {mazel_tov()}")


@lib.utils.log_runtime
def _analyze_spans(dataset: Dataset, spans: typing.List[Span], max_rows: int, run_all: bool):
    logger.info("Analyzing spans...")
    st.header("Dataset Segmentation Analysis")
    st.markdown("In this section, we analyze the dataset after segmentation via `Span`s. ")

    audio_length = seconds_to_str(sum([s.audio_length for s in spans]))
    st.markdown(
        f"There are **{len(spans)} ({audio_length})** spans to analyze, representing of "
        f"**{utils.dataset_coverage(dataset, spans):.2%}** all alignments."
    )

    with utils.st_expander("Random Sample of Spans"):
        _write_span_table(spans[:max_rows])

    with utils.st_expander("Survey of Span Mistranscriptions") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()
        st.write("The span mistranscription count for each length bucket:")
        mistranscriptions = [utils.span_mistranscriptions(s) for s in spans]
        ratio = len([m for m in mistranscriptions if len(m) > 0]) / len(spans)
        st.write(f"Note that **{ratio:.2%}** spans have one or more mistranscriptions.")
        items = [(s, t) for m, s in zip(mistranscriptions, spans) for t in m if len(t[0]) > 0]
        values, labels = [len(m[0]) for _, m in items], [s.speaker.label for s, _ in items]
        chart = utils.bucket_and_chart(values, labels, x="Chars")
        st.altair_chart(chart, use_container_width=True)
        st.write("A random sample of mistranscriptions:")
        st.table([{"script": m[0], "transcript": m[1]} for m in flatten_2d(mistranscriptions)])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (utils.span_total_silence, "Span Total Silence", "Seconds", utils.ALIGNMENT_PRECISION),
        (utils.span_max_silence, "Span Max Silence", "Seconds", utils.ALIGNMENT_PRECISION),
        (utils.span_sec_per_char, "Span Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Span Speed", "Seconds per phone", 0.01),
    ]
    for args in sections:
        _span_metric(spans, *args, unit_y="Span", max_rows=max_rows, run_all=run_all)

    logger.info(f"Finished analyzing spans! {mazel_tov()}")


@lib.utils.log_runtime
def _analyze_filtered_spans(
    dataset: Dataset, spans: typing.List[Span], max_rows: int, run_all: bool
):
    """Filter out spans that are not ideal for training, and analyze the rest.

    NOTE: It's normal to have many consecutive words being said with no pauses between
    them, and often the final sounds of one word blend smoothly or fuse with the initial sounds of
    the next word. Learn more: https://en.wikipedia.org/wiki/Speech_segmentation.
    """
    logger.info("Analyzing filtered spans...")
    st.header("Dataset Filtered Segmentation Analysis")
    st.markdown(
        "In this section, we analyze the dataset after segmentation and filtering. "
        "This is the final step in our dataset preprocessing."
    )
    st.spinner("Analyzing spans...")

    _is_include: typing.Callable[[Span], bool]
    _is_include = lambda s: s.audio_length > 0.1 and utils.span_sec_per_char(s) >= 0.04
    is_include: typing.Callable[[Span], bool]
    is_include = lambda s: (
        not has_a_mistranscription(s) and (_is_include(s[0]) and _is_include(s[-1]))
    )
    results = map_(spans, is_include)
    excluded = [s for s, i in zip(spans, results) if not i]
    included = [s for s, i in zip(spans, results) if i]
    st.markdown(
        f"The filtered segmentations represent **{len(included) / len(spans):.2%}** of the "
        f"original spans. In total, there are **{len(included)} "
        f"({seconds_to_str(sum([s.audio_length for s in included]))})** spans to analyze, "
        f"representing of **{utils.dataset_coverage(dataset, included):.2%}** all alignments."
    )

    with utils.st_expander("Random Sample of Included Spans"):
        _write_span_table(included[:max_rows])

    with utils.st_expander("Random Sample of Excluded Spans"):
        _write_span_table(excluded[:max_rows])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (utils.span_total_silence, "Filtered Span Total Silence", "Seconds", 0.1),
        (utils.span_max_silence, "Filtered Span Max Silence", "Seconds", 0.1),
        (utils.span_audio_loudness, "Filtered Span Loudness", "LUFS", 1),
        (utils.span_sec_per_char, "Filtered Span Speed", "Seconds per character", 0.01),
    ]
    for args in sections:
        _span_metric(included, *args, unit_y="Span", max_rows=max_rows, run_all=run_all)


def main():
    run._config.configure()

    st.title("Dataset Dashboard")
    st.write("The dataset dashboard is an effort to understand our dataset and dataset processing.")

    if st.sidebar.button("Clear Session Cache"):
        clear_session_cache()

    sidebar = st.sidebar

    speakers: typing.List[str] = [k.label for k in DATASETS.keys()]
    load_all = sidebar.checkbox("Load all dataset(s)")
    run_all = sidebar.checkbox("Run all analyses")
    default_speakers = speakers if load_all else None

    question = "Which dataset(s) do you want to load?"
    speakers = st.sidebar.multiselect(question, speakers, default_speakers)
    if load_all:
        assert default_speakers is not None
        speakers = default_speakers
    if len(speakers) == 0:
        st.stop()

    question = "How many span(s) do you want to generate?"
    num_samples: int = sidebar.number_input(question, 0, None, 100)

    with st.spinner("Loading dataset..."):
        dataset = get_dataset(frozenset(speakers))
    with st.spinner("Generating spans..."):
        spans = _get_spans(dataset, num_samples=num_samples)

    question = "What is the maximum number of rows per table?"
    max_rows: int = sidebar.number_input(question, 0, None, 50)

    with st.spinner("Analyzing dataset..."):
        _analyze_dataset(dataset, max_rows, run_all)
        st.text("")
    with st.spinner("Analyzing spans..."):
        _analyze_spans(dataset, spans, max_rows, run_all)
        st.text("")
    with st.spinner("Analyzing filtered spans..."):
        _analyze_filtered_spans(dataset, spans, max_rows, run_all)

    st.text("")
    st.success(f"Done! {mazel_tov()}")
    logger.info(f"Done! {mazel_tov()}")


if __name__ == "__main__":
    main()
