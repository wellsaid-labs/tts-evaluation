""" Streamlit application for reviewing the dataset.

TODO:
- Add an analysis of `non_speech_segments`.
  - Can we use them to filter out bad alignments?
  - Are there any examples with extensive silences? Would those silences cause issues?
  - What's the longest a speaker will go without a `non_speech_segments`?
    Does that vary per speaker?
  - Could we more accurately calculate "speed" by removing `non_speech_segments`? Is this method
    inclusive of speakers that have more background noise which would have less
    `non_speech_segments`? Could we use it for a more accurate filter on alignments that are too
    fast?
  - What if we change the parameters for `non_speech_segments`? There are many segments which end
    on a word whose pronunciation depends on the next word. Could that be reduced?
  - What is the length distribution of `non_speech_segments`? What about the length distribution of
    legitimate `non_speech_segments`? Can we set a minimum pause length per speaker? Can we use
    the speakers speed to set a minimum pause length? Can we we the length distribution to set
    a pause length?
- Where could errors be hiding that could cause poor performance?
  - Could long alignments with multiple words be suspect?
  - Could alignments with long pauses in them be suspect?
  - Could spans with long pauses in them be suspect? Does it matter how we calculate those pauses?
    We could calculate pauses with `non_speech_segments` or `alignments`?

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset/dashboard/__main__.py \
          --runner.magicEnabled=false
"""

import collections
import functools
import logging
import math
import typing
import warnings

import altair as alt
import config as cf
import pandas as pd
import streamlit as st
import tqdm
from spacy.language import Language
from torchnlp.random import fork_rng

import lib
import run
from lib.utils import flatten_2d, mazel_tov, round_, seconds_to_str
from run._config import configure, is_voiced
from run._streamlit import audio_to_html, get_dataset, load_en_core_web_md, map_, st_data_frame
from run._utils import Dataset, _passages_len, split_dataset
from run.data._loader import DATASETS, Passage, Span, has_a_mistranscription
from run.review.dataset.dashboard import _utils as utils

lib.environment.set_basic_logging_config(reset=True)
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


@lib.utils.log_runtime
def _get_spans(dataset: Dataset, num_samples: int) -> typing.List[Span]:
    """Generate spans from our datasets."""
    logger.info("Generating spans...")
    generator = cf.partial(run._utils.SpanGenerator)(
        dataset, include_span=lambda *_: True, max_pause=math.inf
    )
    with fork_rng(123):
        try:
            spans = [next(generator) for _ in tqdm.tqdm(range(num_samples), total=num_samples)]
        except ValueError:
            logger.warning("ValueError: All selected datasets are empty or omitted.")
            spans = []
    logger.info(f"Finished generating spans! {mazel_tov()}")
    return spans


_Columns = typing.Dict[str, typing.List[typing.Any]]


def _default_span_columns(spans: typing.List[Span]) -> _Columns:
    """Get default columns for `_write_span_table`."""
    logger.info("Getting %d generic span columns...", len(spans))
    _round: typing.Callable[[float, Span], float]
    _round = lambda a, s: round_(a, 1 / s.audio_file.sample_rate)
    columns = [
        ("script", [s.script for s in spans]),
        ("speaker", [s.speaker.label for s in spans]),
        ("length", [_round(s.audio_length, s) for s in spans]),
        ("seconds", [[round(i.audio_length, 2) for i in s] for s in spans]),
        ("speed", [[utils.span_sec_per_char(i) for i in s] for s in spans]),
    ]
    if all(s.passage_alignments is s.passage.alignments for s in spans):
        columns.append(("mistranscriptions", [utils.span_mistranscriptions(s) for s in spans]))
    else:
        columns.append(("transcript", [s.transcript for s in spans]))
    return collections.OrderedDict(columns)


def _write_span_table(
    spans: typing.List[Span],
    other_columns: typing.Dict[str, typing.List[typing.Any]] = {},
    default_columns: typing.Callable[[typing.List[Span]], _Columns] = _default_span_columns,
    audio_column="audio",
):
    """Visualize spans as a table with a couple metadata columns."""
    assert len(spans) < 1001, "Large tables are slow to visualize"
    if len(spans) == 0:
        return "No Data."
    logger.info("Visualizing %d spans..." % len(spans))
    dfs = [pd.DataFrame.from_dict(default_columns(spans)), pd.DataFrame.from_dict(other_columns)]
    df = pd.concat(dfs, axis=1)
    assert audio_column not in df.columns
    get_audio = lambda s: audio_to_html(utils.span_audio(s))
    df.insert(0, audio_column, map_(spans, get_audio))
    st_data_frame(df)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")


def _span_metric(
    spans: typing.List[Span],
    get_val: typing.Union[typing.Callable[[Span], float], typing.List[float]],
    name: str,
    unit_x: str,
    bucket_size: float,
    unit_y: str,
    max_rows: int,
    run_all: bool,
    note: typing.Union[str, typing.List[str]] = "",
    **kwargs,
):
    """Visualize a span metric.

    Args:
        ...
        get_val: Get a measurement for a `Span`.
        name: The title of the analysis.
        unit_x: The name of the measurement unit for `func`.
        bucket_size: The size of each bin in a bar chart showing the distribution of measurements.
        unit_y: A label for each `Span` in `spans`.
        max_rows: The maximum number of values to show in a table.
        run_all: Iff `run_all`, then run all analyses.
        note: Any additional notes to add to this section.
    """
    with utils.st_expander(f"Survey of {unit_y} {name} (in {unit_x.lower()})") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            return

        st.write(f"The {unit_y.lower()} count for each bucket:")
        results = map_(spans, get_val) if callable(get_val) else get_val
        assert len(results) == len(spans)
        labels = [s.speaker.label for s in spans]
        kwargs = dict(x=unit_x, y=unit_y + " Count", **kwargs)
        chart = utils.bucket_and_chart(results, labels, bucket_size, **kwargs)
        st.altair_chart(chart, use_container_width=True)
        if len(note) > 0:
            st.write("**Note(s):**\n\n")
            st.write(note) if isinstance(note, str) else [st.write(n) for n in note]
            st.write("")
        filtered = [(s, r) for s, r in zip(spans, results) if not math.isnan(r)]
        sorted_ = lambda **k: sorted(filtered, key=lambda i: i[1], **k)[:max_rows]
        for label_, data in (("smallest", sorted_()), ("largest", sorted_(reverse=True))):
            text = f"Show the {label_} valued {unit_y.lower()}(s)."
            if st.checkbox(text, key=label + label_, value=run_all):
                other_columns = {"value": [r[1] for r in data]}
                _write_span_table([r[0] for r in data], other_columns=other_columns)
                st.text("")


def _grouped_bar_chart(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    color_feature: str,
    column_feature: str,
    x_label: str = "",
    y_label: str = "",
    **kwargs,
):
    """Visualize data, such as per-speaker data, in a grouped bar chart.

    Args:
        df: The dataframe to plot.
        x_feature: The feature to be plotted on the x-axis.
        y_feature: The feature to be plotted on the y-axis.
        color_feature: The feature to be encoded by color.
        column_feature: The feature to be encoded by column.
        x_label: The x-axis label.
        y_label: The y-axis label.
    """
    header = alt.Header(labelAngle=-25, labelPadding=-50)  # type: ignore
    grouped_bar_chart = (
        alt.Chart(df)  # type: ignore
        .mark_bar()
        .encode(
            x=alt.X(x_feature, title=x_label),  # type: ignore
            y=alt.Y(y_feature, title=y_label),  # type: ignore
            color=color_feature,
            column=alt.Column(column_feature, header=header),  # type: ignore
        )
    )
    st.altair_chart(grouped_bar_chart, use_container_width=False)


@lib.utils.log_runtime
def _analyze_all_passages(dataset: Dataset, **kwargs):
    st.markdown("### Passage Analysis")
    spans = [p[:] for d in dataset.values() for p in d]
    _span_metric(spans, lambda s: s.audio_length, "Length", "Seconds", 5, "Passage", **kwargs)


@lib.utils.log_runtime
def _analyze_alignment_speech_segments(
    passages: typing.List[Passage], max_rows: int, run_all: bool
):
    """Analyze the distribution of speech segments as defined by alignments
    (i.e. alignments with no pauses in between)."""
    st.markdown("### Alignment Speech Segments Analysis")
    segments = [s for p in passages for s in utils.passage_alignment_speech_segments(p)]
    threshold = 10
    max_length = max(s.audio_length for s in segments)
    above_threshold = sum(s.audio_length for s in segments if s.audio_length > threshold)
    total_seconds = sum(s.audio_length for s in segments)
    _span_metric(
        segments,
        lambda s: s.audio_length,
        "Length",
        "Seconds",
        utils.ALIGNMENT_PRECISION,
        "Alignment Speech Segment",
        note=(
            f"- The maximum length, without pauses, is **{max_length:.2f}** seconds.\n\n"
            f"- The sum of segments without a pause, longer than {threshold} seconds, "
            f"is **{above_threshold:.2f}** out of **{total_seconds:.2f}** seconds "
            f"(**{above_threshold / total_seconds:.1%}**)."
        ),
        max_rows=max_rows,
        run_all=run_all,
    )

    with utils.st_expander("Random Sample of Alignment Speech Segments") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()
        sample = segments[:max_rows]
        edges = []
        for segment in sample:
            start = segment.passage.nonalignments[segment.nonalignments_slice.start].audio
            end = segment.passage.nonalignments[segment.nonalignments_slice.stop - 1].audio
            edges.append((start[-1] - start[0], end[-1] - end[0]))
        other_columns = {"edges": edges, "transcript": [s.transcript for s in sample]}
        _write_span_table(sample, other_columns=other_columns)


@lib.utils.log_runtime
def _analyze_speech_segment_transitions(passages: typing.List[Passage], **kwargs):
    """Analyze the transition periods between speech segments."""
    span_metric = functools.partial(_span_metric, **kwargs)
    unit = "Speech Segment Transition"
    threshold = 1.0
    lengths = []
    spans = []
    total, above_threshold = collections.defaultdict(float), collections.defaultdict(float)
    segments = [(p, p.speech_segments) for p in passages if len(p.speech_segments) != 0]
    pairs = [(p, z) for (p, s) in segments for z in zip(s, s[1:])]
    for passage, (prev, next) in pairs:
        audio_length = next.audio_slice.start - prev.audio_slice.stop
        audio_slice = slice(prev.audio_slice.start, next.audio_slice.stop)
        span = passage.span(slice(prev.slice.start, next.slice.stop), audio_slice)
        total[passage.speaker] += 1
        if not has_a_mistranscription(span):
            spans.append(span)
            lengths.append(audio_length)
            above_threshold[span.speaker] += float(audio_length > threshold)
    info = [(s, above_threshold[s] / t) for s, t in total.items()]
    info = sorted(info, key=lambda k: k[1], reverse=True)
    info = {s.label: f"{t:.3%}" for s, t in info}
    notes = [f"The percentage of span transitions above **{threshold}s**:", info]
    kwargs = dict(normalize=True, note=notes, **kwargs)
    span_metric(spans, lengths, "Length", "Seconds", 0.5, unit, **kwargs)


def _total_pauses(span: Span, threshold: float = 1.0, min_speech_segment: float = 0.1) -> float:
    """Get the sum of pauses longer than `threshold` in `Span`."""
    lengths = []
    intervals = span.passage.non_speech_segments[span.audio_start : span.audio_stop]
    start, stop = max(intervals[0][0], span.audio_start), min(intervals[0][1], span.audio_stop)
    for interval in intervals:
        if interval[0] - stop > min_speech_segment:
            assert stop >= start
            lengths.append(stop - start)
            start = interval[0]
        stop = min(interval[1], span.audio_stop)
    lengths.append(stop - start)
    return 0.0 if len(lengths) == 0 else sum(l for l in lengths if l > threshold)


def _max_nonalignment_length(span: Span) -> float:
    nonalignments = span.passage.nonalignments[span.nonalignments_slice]
    clamp = lambda x: lib.utils.clamp(x, min_=span.audio_start, max_=span.audio_stop)
    return max(clamp(a.audio[-1]) - clamp(a.audio[0]) for a in nonalignments)


@lib.utils.log_runtime
def _analyze_speech_segments(passages: typing.List[Passage], **kwargs):
    st.markdown("### Speech Segments Analysis")

    segments = [s for p in passages for s in p.speech_segments]
    total_seconds = sum(s.audio_length for s in segments)
    audio_length = seconds_to_str(total_seconds)
    num_mistranscription = sum(s.audio_length for s in segments if has_a_mistranscription(s))
    num_slash = sum(s.audio_length for s in segments if "/" in s.script or "\\" in s.script)
    num_digit = sum(s.audio_length for s in segments if lib.text.has_digit(s.script))
    threshold = 15
    above_threshold = sum(s.audio_length for s in segments if s.audio_length > threshold)

    st.markdown(
        f"There are **{len(segments):,} ({audio_length})** segments to analyze, representing of "
        f"**{utils.passages_coverage(passages, segments):.2%}** all alignments in passages. "
        f"At a high-level:\n\n"
        f"- **{sum(s.audio_length for s in segments):.2f}** seconds of speech segments\n\n"
        f"- **{num_mistranscription / total_seconds:.2%}** has a mistranscription\n\n"
        f"- **{num_slash / total_seconds:.2%}** has a slash\n\n"
        f"- **{num_digit / total_seconds:.2%}** has a digit\n\n"
        f"- **{above_threshold / total_seconds:.2%}** is longer than {threshold} seconds\n\n"
        f"- **{max(s.audio_length for s in segments):.2f}** seconds is the longest segment\n\n"
        f"- **{min(s.audio_length for s in segments):.2f}** seconds is the shortest segment\n\n"
        f"- **{sum(_total_pauses(s) for s in segments):.2f}** seconds of long pauses\n\n"
    )

    segments = [s for s in segments if not has_a_mistranscription(s)]
    lambda_: typing.Callable[[Span], float]
    lambda_ = lambda s: s.audio_length
    _span_metric(segments, lambda_, "Length", "Seconds", 0.1, "Speech Segment", **kwargs)

    name, unit_y = "Total Pause Length", "Speech Segment"
    _span_metric(segments, _total_pauses, name, "Seconds", 0.1, unit_y, **kwargs)

    name, unit_y = "Max Non Alignment Length", "Speech Segment"
    _span_metric(segments, _max_nonalignment_length, name, "Seconds", 0.1, unit_y, **kwargs)

    _analyze_speech_segment_transitions(passages, **kwargs)


@lib.utils.log_runtime
def _analyze_nonalignments(passages: typing.List[Passage], max_rows: int, run_all: bool):
    st.markdown("### Nonalignment Analysis")
    nonalignments = [s for p in passages for s in p.nonalignment_spans().spans]

    _span_metric(
        nonalignments,
        lambda s: s.audio_length,
        "Length",
        "Seconds",
        0.1,
        "Nonalignment",
        note=(
            f"- **{sum([s.audio_length > 0 for s in nonalignments]) / len(nonalignments):.2%}** "
            "of pauses are longer than zero."
        ),
        max_rows=max_rows,
        run_all=run_all,
    )

    with utils.st_expander("Survey of Unvoiced Nonalignment Mistranscriptions") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()
        _is_voiced = [
            any(is_voiced(t, s.speaker.language) for t in (s.script, s.transcript))
            for s in nonalignments
        ]
        iterator = list(zip(_is_voiced, nonalignments))
        data = list(set((s.script.strip(), s.transcript.strip()) for i, s in iterator if not i))
        st.write("These mistranscriptions are not considered 'voiced':")
        st.table(pd.DataFrame(data[:max_rows], columns=["script", "transcript"]))


@lib.utils.log_runtime
def _analyze_alignments(passages: typing.List[Passage], max_rows: int, run_all: bool):
    st.markdown("### Alignment Analysis")
    trigrams = list(utils.passages_alignment_ngrams(passages, 3))
    unigrams = list(utils.passages_alignment_ngrams(passages, 1))
    label = "Analyze only single word alignments"
    if st.sidebar.checkbox(label, key="single world alignments", value=True):
        unigrams = [u for u in unigrams if " " not in u.script]

    with utils.st_expander("Random Sample of Alignments") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()

        for span in utils.random_sample(trigrams, max_rows):
            cols = st.columns([2, 1, 1])
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

    with utils.st_expander("Random Sample of Alignments (Tabular)") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()
        _write_span_table(unigrams[:max_rows])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (lambda s: s.audio_length, "Length", "Seconds", utils.ALIGNMENT_PRECISION),
        (lambda s: len(s.script), "Length", "Characters", 1),
        (utils.span_sec_per_char, "Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Speed", "Seconds per phoneme", 0.01),
        (utils.span_audio_rms_level, "Loudness", "Decibels", 1),
        (utils.span_audio_left_rms_level, "Onset Loudness", "Decibels", 5),
        (utils.span_audio_right_rms_level, "Outset Loudness", "Decibels", 5),
    ]
    for args in sections:
        _span_metric(unigrams, *args, unit_y="Alignment", max_rows=max_rows, run_all=run_all)

    with utils.st_expander("Random Sample of Filtered Alignments") as label:
        if not st.checkbox("Analyze", key=label, value=run_all):
            raise GeneratorExit()

        is_include: typing.Callable[[Span], bool]
        is_include = lambda s: s.audio_length > 0.11 and utils.span_sec_per_char(s) >= 0.04
        filtered = [s for s in unigrams if is_include(s)]
        st.write(f"Filtered out {1 - (len(filtered) / len(unigrams)):.2%} of alignments.")
        _write_span_table(filtered[:max_rows])


@lib.utils.log_runtime
def _analyze_non_speech_segments(passages: typing.List[Passage], max_rows: int, run_all: bool):
    st.markdown("### Non Speech Segment Analysis")
    if not st.checkbox("Analyze", key="non-speech-segment-analysis", value=run_all):
        return

    intervals = [(p, i) for p in passages for i in p.non_speech_segments.intervals()]
    segments = [p.span(slice(0, 0), slice(*i)) for p, i in intervals]
    lambda_ = lambda p: p.audio_length
    unit = "Non Speech Segment"
    _span_metric(
        segments,
        lambda_,
        "Length",
        "Seconds",
        0.5,
        unit,
        normalize=True,
        max_rows=max_rows,
        run_all=run_all,
    )


@lib.utils.log_runtime
def _analyze_dataset_split(
    dataset: Dataset, passages: typing.List[Passage], nlp: Language, max_rows: int, run_all: bool
):
    st.markdown("### Dataset Dev/Train Split Analysis")

    if st.checkbox("Analyze", key="dataset-split-analysis-count", value=run_all):
        warnings.filterwarnings("ignore")
        configure(overwrite=True)
        speakers = set([speaker for speaker in dataset])
        train_dataset, dev_dataset = cf.partial(split_dataset)(
            dataset, dev_speakers=speakers, groups=[speakers]
        )
        dev_train_data: typing.List[typing.Tuple[str, str, int, float, int]] = []
        for i, s in enumerate(speakers):
            s_display = f"{s.label} ({s.dialect.name}, {s.style.value})"
            logger.info(f"Analyzing speaker {i+1}/{len(speakers)}: {s_display}...")
            speaker_train: typing.List[Passage] = train_dataset[s] if s in train_dataset else []
            speaker_dev: typing.List[Passage] = dev_dataset[s] if s in dev_dataset else []
            for speaker_data, split_type in [(speaker_dev, "dev"), (speaker_train, "train")]:
                num_p = len(speaker_data)
                len_p = _passages_len(speaker_data)
                word_count = sum(len(nlp(p.transcript)) for p in speaker_data)
                dev_train_data.append((s_display, split_type, num_p, len_p, word_count))
        dataset_split_df = pd.DataFrame(
            dev_train_data,
            columns=["speaker", "split", "num_passages", "len_passages", "word_count"],
        )

        with utils.st_expander("Number of Dev/Train Passages Per Speaker"):
            _grouped_bar_chart(
                dataset_split_df,
                "split:N",
                "num_passages:Q",
                "split:N",
                "speaker:N",
                y_label="num_passages (discrete)",
            )

        with utils.st_expander("Length of Dev/Train Passages Per Speaker"):
            _grouped_bar_chart(
                dataset_split_df,
                "split:N",
                "len_passages:Q",
                "split:N",
                "speaker:N",
                y_label="len_passages (seconds)",
            )

        with utils.st_expander("Word Count of Dev/Train Passages Per Speaker"):
            _grouped_bar_chart(
                dataset_split_df,
                "split:N",
                "word_count:Q",
                "split:N",
                "speaker:N",
                y_label="word_count (discrete)",
            )


@lib.utils.log_runtime
def _analyze_dataset(dataset: Dataset, **kwargs):
    logger.info("Analyzing dataset...")
    st.header("Raw Dataset Analysis")
    st.markdown("In this section, we analyze the dataset prior to segmentation.")
    total_passages = sum(len(d) for d in dataset.values())
    st.markdown(
        f"At a high-level, this dataset has:\n"
        f"- **{total_passages:,}** passages\n"
        f"- **{seconds_to_str(utils.dataset_total_audio(dataset))}** of audio\n"
        f"- **{seconds_to_str(utils.dataset_total_aligned_audio(dataset))}** of aligned audio\n"
        f"- **{utils.dataset_num_alignments(dataset):,}** alignments.\n"
    )

    _analyze_all_passages(dataset, **kwargs)

    question = "How many passage(s) do you want to analyze?"
    sampled = int(st.sidebar.number_input(question, 0, None, 25))  # type: ignore
    passages = list(utils.dataset_passages(dataset))
    passages = utils.random_sample(passages, sampled) if sampled < total_passages else passages
    st.write("")
    st.info(
        f"Below this analyzes a random sample of **{len(passages):,}** passages with "
        f"**{sum(len(p.alignments) for p in passages):,}** alignments..."
    )

    with st.spinner("Loading spaCy..."):
        nlp: Language = load_en_core_web_md(disable=("parser", "ner"))

    _analyze_alignments(passages, **kwargs)
    _analyze_nonalignments(passages, **kwargs)
    _analyze_speech_segments(passages, **kwargs)
    _analyze_alignment_speech_segments(passages, **kwargs)
    _analyze_non_speech_segments(passages, **kwargs)
    _analyze_dataset_split(dataset, passages, nlp, **kwargs)

    logger.info(f"Finished analyzing dataset! {mazel_tov()}")


@lib.utils.log_runtime
def _analyze_spans(dataset: Dataset, spans: typing.List[Span], max_rows: int, run_all: bool):
    logger.info("Analyzing spans...")
    st.header("Dataset Segmentation Analysis")
    st.markdown("In this section, we analyze the dataset after segmentation via `Span`s. ")

    audio_length = seconds_to_str(sum([s.audio_length for s in spans]))
    average_length = sum([s.audio_length for s in spans]) / len(spans)
    coverage = utils.dataset_coverage(dataset, spans)
    st.markdown(
        f"At a high-level:\n\n"
        f"- There are **{len(spans)} ({audio_length})** spans to analyze.\n\n"
        f"- The spans represent **{coverage:.2%}** all alignments.\n\n"
        f"- The average length is **{average_length:.2}** seconds.\n\n"
    )

    with utils.st_expander("Random Sample of Spans") as label:
        if st.checkbox("Analyze", key=label, value=run_all):
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
        (lambda s: s.audio_length, "Length", "Seconds", 1.0),
        (utils.span_total_silence, "Total Silence", "Seconds", utils.ALIGNMENT_PRECISION),
        (utils.span_max_silence, "Max Silence", "Seconds", utils.ALIGNMENT_PRECISION),
        (utils.span_sec_per_char, "Speed", "Seconds per character", 0.01),
        (utils.span_sec_per_phon, "Speed", "Seconds per phone", 0.01),
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
    _is_include = lambda s: s.audio_length > 0.11 and utils.span_sec_per_char(s) >= 0.04
    is_include: typing.Callable[[Span], bool]
    is_include = lambda s: (
        not has_a_mistranscription(s)
        and (_is_include(s[0]) and _is_include(s[-1]))
        and s.audio_length / len(s.script) >= 0.035
        and s.audio_length >= 0.3
    )
    results = map_(spans, is_include)
    excluded = [s for s, i in zip(spans, results) if not i]
    included = [s for s, i in zip(spans, results) if i]
    average_length = sum([s.audio_length for s in included]) / len(included)
    audio_length = seconds_to_str(sum([s.audio_length for s in included]))
    coverage = utils.dataset_coverage(dataset, included)
    st.markdown(
        f"At a high-level:\n\n"
        f"- There are **{len(included)} ({audio_length})** spans to analyze.\n\n"
        f"- The spans represent **{len(included) / len(spans):.2%}** of the original spans.\n\n"
        f"- The spans represent **{coverage:.2%}** all alignments.\n\n"
        f"- The average length is **{average_length:.2}** seconds.\n\n"
    )

    with utils.st_expander("Random Sample of Included Spans") as label:
        if st.checkbox("Analyze", key=label, value=run_all):
            _write_span_table(included[:max_rows])

    with utils.st_expander("Random Sample of Excluded Spans") as label:
        if st.checkbox("Analyze", key=label, value=run_all):
            _write_span_table(excluded[:max_rows])

    sections: typing.List[typing.Tuple[typing.Callable[[Span], float], str, str, float]] = [
        (lambda s: s.audio_length, "Length", "Seconds", 1.0),
        (utils.span_total_silence, "Total Silence", "Seconds", 0.1),
        (utils.span_max_silence, "Max Silence", "Seconds", 0.1),
        (utils.span_audio_loudness, "Loudness", "LUFS", 1),
        (utils.span_sec_per_char, "Speed", "Seconds per character", 0.01),
    ]
    for args in sections:
        _span_metric(included, *args, unit_y="Included Span", max_rows=max_rows, run_all=run_all)


def main():
    run._config.configure(overwrite=True)

    st.title("Dataset Dashboard")
    st.write("The dataset dashboard is an effort to understand our dataset and dataset processing.")

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
    num_samples: int = sidebar.number_input(question, 0, None, 100)  # type: ignore

    dataset = {k: v for s in speakers for k, v in get_dataset(frozenset([s])).items()}
    with st.spinner("Generating spans..."):
        spans = _get_spans(dataset, num_samples=num_samples)

    question = "What is the maximum number of rows per table?"
    max_rows: int = sidebar.number_input(question, 0, None, 50)  # type: ignore

    with st.spinner("Analyzing dataset..."):
        _analyze_dataset(dataset, max_rows=max_rows, run_all=run_all)
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
