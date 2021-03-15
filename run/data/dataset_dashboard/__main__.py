""" Streamlit application for reviewing the dataset.

Usage:
    $ PYTHONPATH=. streamlit run run/data/dataset_dashboard/__main__.py --runner.magicEnabled=false
"""

import collections
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
from lib.datasets import DATASETS, Span
from lib.utils import flatten_2d, mazel_tov, seconds_to_str
from run._config import Dataset
from run._streamlit import audio_to_html, get_dataset, get_session_state, map_
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
    """ Get default columns for `_span_table`. """
    logger.info("Getting %d generic span columns...", len(spans))
    columns = [
        ("script", [s.script for s in spans]),
        ("length", [s.audio_length for s in spans]),
        ("mistranscriptions", [utils.span_mistranscriptions(s) for s in spans]),
        ("seconds", [[round(i.audio_length, 2) for i in s] for s in spans]),
        ("speed", [[utils.span_sec_per_char(i) for i in s] for s in spans]),
    ]
    return collections.OrderedDict(columns)


def _span_table(
    spans: typing.List[Span],
    other_columns: typing.Dict[str, typing.List[typing.Any]] = {},
    default_columns: typing.Callable[[typing.List[Span]], _Columns] = _default_span_columns,
    audio_column="audio",
) -> str:
    """Visualize spans as a table with a couple metadata columns."""
    assert len(spans) < 250, "Large tables are slow to visualize"
    if len(spans) == 0:
        return "No Data."
    logger.info("Visualizing %d spans..." % len(spans))
    dfs = [pd.DataFrame.from_dict(default_columns(spans)), pd.DataFrame.from_dict(other_columns)]
    df = pd.concat(dfs, axis=1)
    assert audio_column not in df.columns
    df.insert(0, audio_column, map_(spans, utils.span_audio))
    formatters = {audio_column: audio_to_html}
    html = df.to_html(formatters=formatters, escape=False, justify="left", index=False)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")
    return html


def _span_metric(
    spans: typing.List[Span],
    func: typing.Callable[[Span], float],
    name: str,
    unit: str,
    bucket_size: float,
    max_rows: int,
    run_all: bool,
):
    """Visualize a span metric."""
    with utils.st_expander(f"Survey of {name} (in {unit.lower()})") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            return

        st.write("The alignment count for each bucket:")
        results = map_(spans, func)
        chart = utils.bucket_and_chart(results, bucket_size, x=unit)
        st.altair_chart(chart, use_container_width=True)
        filtered = [(s, r) for s, r in zip(spans, results) if not math.isnan(r)]
        sorted_ = lambda **k: sorted(filtered, key=lambda i: i[1], **k)[:max_rows]
        for label, data in (("smallest", sorted_()), ("largest", sorted_(reverse=True))):
            st.write(f"The {label} valued alignments:")
            other_columns = {"value": [r[1] for r in data]}
            table = _span_table([r[0] for r in data], other_columns=other_columns)
            st.markdown(table, unsafe_allow_html=True)
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
    passages = utils.random_sample(list(utils._dataset_passages(dataset)), num_passages)
    unigrams = list(utils.passages_alignment_ngrams(passages, 1))
    trigrams = list(utils.passages_alignment_ngrams(passages, 3))
    st.markdown(
        f"Below this analyzes a random sample of **{len(passages):,}** passages with "
        f"{len(unigrams):,} alignments..."
    )

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

    with utils.st_expander("Survey of Pause Lengths (in seconds)"):
        st.write("The pause count for each length bucket:")
        pauses = list(utils.dataset_pause_lengths_in_seconds(dataset))
        chart = utils.bucket_and_chart(pauses, utils.ALIGNMENT_PRECISION, x="Seconds")
        st.altair_chart(chart, use_container_width=True)
        ratio = sum([p > 0 for p in pauses]) / len(pauses)
        st.write(f"**{ratio:.2%}** of pauses are longer than zero.")

    question = "How many alignment(s) do you want to analyze?"
    num_alignments: int = st.sidebar.number_input(question, 0, None, 200)
    samples = utils.random_sample(unigrams, num_alignments)
    st.text("")
    st.markdown(
        f"Below this analyzes a random sample of **{len(samples):,}** alignments "
        f"of {len(unigrams):,} alignments..."
    )

    with utils.st_expander("Random Sample of Alignments (Tabular)"):
        st.markdown(_span_table(samples[:max_rows]), unsafe_allow_html=True)

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (lambda s: s.audio_length, "Alignment Length", "Seconds", utils.ALIGNMENT_PRECISION),
        (lambda s: len(s.script), "Alignment Length", "Characters", 1),
        (utils.span_sec_per_char, "Alignment Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Alignment Speed", "Seconds per phoneme", 0.01),
        (utils.span_audio_rms_level, "Alignment Loudness", "Decibels", 1),
        (utils.span_audio_left_rms_level, "Alignment Onset Loudness", "Decibels", 5),
        (utils.span_audio_right_rms_level, "Alignment Outset Loudness", "Decibels", 5),
    ]
    [_span_metric(samples, *args, max_rows=max_rows, run_all=run_all) for args in sections]

    with utils.st_expander("Random Sample of Filtered Alignments"):
        is_include: typing.Callable[[Span], bool]
        is_include = lambda s: s.audio_length > 0.1 and utils.span_sec_per_char(s) >= 0.04
        filtered = [s for s in samples if is_include(s)]
        st.write(f"Filtered out {1 - (len(filtered) / len(samples)):.2%} of alignments.")
        st.markdown(_span_table(filtered[:max_rows]), unsafe_allow_html=True)

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
        st.markdown(_span_table(spans[:max_rows]), unsafe_allow_html=True)

    with utils.st_expander("Survey of Span Mistranscriptions") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()
        st.write("The span mistranscription count for each length bucket:")
        mistranscriptions = [utils.span_mistranscriptions(s) for s in spans]
        flat = flatten_2d(mistranscriptions)
        ratio = len([m for m in mistranscriptions if len(m) > 0]) / len(spans)
        st.write(f"Note that **{ratio:.2%}** spans have one or more mistranscriptions.")
        chart = utils.bucket_and_chart([len(m[0]) for m in flat if len(m[0]) > 0], x="Char(s)")
        st.altair_chart(chart, use_container_width=True)
        st.write("A random sample of mistranscriptions:")
        st.table([{"script": m[0], "transcript": m[1]} for m in flat])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (utils.span_total_silence, "Span Silence", "Seconds", utils.ALIGNMENT_PRECISION),
        (utils.span_sec_per_char, "Span Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Span Speed", "Seconds per phone", 0.01),
    ]
    [_span_metric(spans, *args, max_rows=max_rows, run_all=run_all) for args in sections]

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
        len(utils.span_mistranscriptions(s)) == 0 and (_is_include(s[0]) and _is_include(s[-1]))
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
        st.markdown(_span_table(included[:max_rows]), unsafe_allow_html=True)

    with utils.st_expander("Random Sample of Excluded Spans"):
        st.markdown(_span_table(excluded[:max_rows]), unsafe_allow_html=True)

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (utils.span_audio_loudness, "Filtered Span Loudness", "LUFS", 1),
        (utils.span_sec_per_char, "Filtered Span Speed", "Seconds per character", 0.01),
    ]
    [_span_metric(included, *args, max_rows=max_rows, run_all=run_all) for args in sections]


def main():
    run._config.configure()

    st.title("Dataset Dashboard")
    st.write("The dataset dashboard is an effort to understand our dataset and dataset processing.")

    if st.sidebar.button("Clear Session Cache"):
        logger.info("Clearing cache...")
        [v.cache_clear() for v in get_session_state()["cache"].values()]

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
    max_rows: int = sidebar.number_input(question, 0, None, 25)

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
