""" Streamlit application for reviewing generated annotations.

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset_processing/annotations.py \
        --runner.magicEnabled=false
"""
import logging
import typing

import config as cf
import numpy
import pandas
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from torchnlp.random import fork_rng

import lib
import run
from run._config.labels import _speaker
from run._streamlit import audio_to_url, clip_audio, st_ag_grid, st_tqdm
from run._utils import Dataset, get_datasets
from run.data._loader import Alignment, Span, Speaker
from run.train.spectrogram_model._data import (
    _get_loudness_annotation,
    _get_tempo_annotation,
    _random_nonoverlapping_alignments,
)
from run.train.spectrogram_model._worker import _get_data_generator

lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
logger = logging.getLogger(__name__)


@st.experimental_singleton()
def _get_datasets() -> typing.Tuple[Dataset, Dataset]:
    return get_datasets(False)


@st.experimental_singleton()
def _get_spans(
    _train_dataset: Dataset,
    _dev_dataset: Dataset,
    speaker: typing.Optional[Speaker],
    num_spans: int,
    device_count: int = 4,
) -> typing.Tuple[typing.List[Span], typing.List[numpy.ndarray]]:
    """Get `num_spans` spans from `_train_dataset` for `speaker`. This uses the same code path
    as a training run so it ensures we are analyzing training data directly.

    Args:
        ...
        device_count: The number of devices used during training to set the configuration.
    """
    with st.spinner("Configuring..."):
        datasets = (_train_dataset, _dev_dataset)
        config_ = run._config.make_spectrogram_model_train_config(*datasets, False, device_count)
        cf.add(config_, overwrite=True)

    with st.spinner("Making generators..."):
        if speaker is not None:
            _train_dataset = {speaker: _train_dataset[speaker]}
            _dev_dataset = {speaker: _dev_dataset[speaker]}
        train_gen, _ = cf.partial(_get_data_generator)(_train_dataset, _dev_dataset)

    with st.spinner("Making spans..."):
        spans = [next(train_gen) for _ in st_tqdm(range(num_spans), num_spans)]

    with st.spinner("Loading audio..."):
        signals = [s.audio() for s in st_tqdm(spans)]

    return spans, signals


def _annotate(text: str, alignment: Alignment, prefix: str = "<<<", suffix: str = ">>>") -> str:
    """Mark the `slice_` in the `text` with a `prefix` and `suffix`."""
    text = text[: alignment.script[1]] + suffix + text[alignment.script[1] :]
    return text[: alignment.script[0]] + prefix + text[alignment.script[0] :]


def _gather_data(span_idx: int, span: Span, anno: Alignment, clip: numpy.ndarray):
    """
    Args:
        span_idx: A unique index to identify the span.
        ...
    """
    doc = span.spacy.as_doc()
    script_len = anno.script[1] - anno.script[0]
    text = span.script[anno.script[0] : anno.script[1]]
    spacy_span = doc.char_span(anno.script[0], anno.script[1], alignment_mode="expand")
    assert spacy_span is not None
    assert len(spacy_span.text) <= script_len, "Invalid annotation"

    return {
        "index": span_idx,
        "anno_script": repr(_annotate(span.script, anno)),
        "loudness": cf.partial(_get_loudness_annotation)(clip, span.audio_file.sample_rate, anno),
        "tempo": cf.partial(_get_tempo_annotation)(span, anno),
        "clip": audio_to_url(clip_audio(clip, span, anno)),
        "speaker": repr(span.session[0]),
        "session": span.session[1],
        "num_alignments": len(span.alignments),
        "script": text,
        "transcript": span.transcript[anno.transcript[0] : anno.transcript[1]],
        "num_words": len(text.split()),
        "audio_len": round(anno.audio[1] - anno.audio[0], 2),
        "script_len": script_len,
        "transcript_len": anno.transcript[1] - anno.transcript[0],
    }


def _stats(
    spans: typing.List[Span],
    data: typing.List[typing.Dict],
    annotations: typing.List[typing.Tuple[Alignment, ...]],
):
    """Gather and report useful statistics about the data.

    Args:
        spans: The original spans.
        data: A collection of data points per annotation.
        annotations: A list of annotations per span.
    """
    st.subheader("Stats")
    num_annotated_spans = len(set(r["index"] for r in data if r["num_alignments"] >= 10))
    loudness_vals = [r["loudness"] for r in data if r["loudness"] is not None]
    tempo_vals = [r["tempo"] for r in data]
    total_loudness_script_len = sum(r["script_len"] for r in data if r["loudness"] is not None)
    total_tempo_script_len = sum(r["script_len"] for r in data if r["tempo"])
    stats: typing.Dict[str, float] = {
        "Num `Span`s": len(spans),
        "Num Annotations": len(data),
        "Num `Span`s (< 10 alignments)": sum(1 for s in spans if len(s.alignments) < 10),
        "Num `Span`s (>= 10 alignments)": sum(1 for s in spans if len(s.alignments) >= 10),
        "Num `Span`s (>= 10 alignments, >= 1 annotations)": num_annotated_spans,
        "Num `Span`s (0 annotations)": sum(1 for i in annotations if len(i) == 0),
        "Percent `Span`s (0 annotations)": sum(1 for i in annotations if len(i) == 0),
        "Average Annotations Per Span": round(len(data) / len(spans), 1),
        "Num Loundess Annotations": len(loudness_vals),
        "Max Loundess Annotation": max(loudness_vals),
        "Min Loundess Annotation": min(loudness_vals),
        "Average Loundess Annotation Length": total_loudness_script_len / len(data),
        "Num Tempo Annotations": len(tempo_vals),
        "Max Tempo Annotation": max(tempo_vals),
        "Min Tempo Annotation": min(tempo_vals),
        "Average Tempo Annotation Length": total_tempo_script_len / len(data),
        "Total Seconds Annotated": sum(r["audio_len"] for r in data),
    }

    label = "Average Annotations Per Span (>= 10 alignments)"
    num_anno_enough_alignments = sum(1 for r in data if r["num_alignments"] >= 10)
    stats[label] = round(num_anno_enough_alignments / stats["Num `Span`s (>= 10 alignments)"], 1)

    label = "Average Annotations Per Annotated Span (>= 10 alignments)"
    demon = "Num `Span`s (>= 10 alignments, >= 1 annotations)"
    stats[label] = round(num_anno_enough_alignments / stats[demon], 1)

    label = "Percent `Span`s (0 annotations)"
    stats[label] = stats["Num `Span`s (0 annotations)"] / stats["Num `Span`s"]

    # TODO: Is this really how you average decibels?
    total_loudness = sum(r["audio_len"] * r["loudness"] for r in data if r["loudness"] is not None)
    stats["Average Loundess"] = total_loudness / stats["Total Seconds Annotated"]

    total_tempo = sum(r["audio_len"] * r["tempo"] for r in data)
    stats["Average Tempo"] = total_tempo / stats["Total Seconds Annotated"]

    st.dataframe(
        pandas.DataFrame(data=list(stats.values()), index=list(stats.keys())),
        use_container_width=True,
    )


def _distributions(data: typing.List[typing.Dict], num_cols: int = 3):
    """Plot the distribution of various data points."""
    st.subheader("Distributions")
    distributions = (
        ("Loudness Values", 1, [r["loudness"] for r in data if r["loudness"] is not None]),
        ("Tempo Values", 0.01, [r["tempo"] for r in data]),
        ("Num Annotated Words", 1, [r["num_words"] for r in data]),
        ("Num Annotated Characters", 1, [r["script_len"] for r in data]),
        ("Num Annotated Seconds", 0.1, [r["audio_len"] for r in data]),
    )
    cols: typing.List[DeltaGenerator] = st.columns(num_cols)
    for i, (name, precision, vals) in enumerate(distributions):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, xbins=dict(size=precision)))
        fig.update_layout(
            xaxis_title_text=name,
            yaxis_title_text="Count",
            margin=dict(b=40, l=40, r=0, t=0),
            bargap=0.2,
        )
        col = cols[i % len(cols)]
        col.plotly_chart(fig, use_container_width=True)


def main():
    run._config.configure(overwrite=True)

    st.title("Annotations")
    st.write("The workbook reviews the annotations that are being generated for spans.")

    if st.sidebar.button("Clear Dataset Cache"):
        _get_datasets.clear()
    train_dataset, dev_dataset = _get_datasets()

    form = st.form("settings")
    question = "How many span(s) do you want to generate?"
    # NOTE: Too many spans could cause the `streamlit` to refresh and start over, this has happened
    # around 5000 spans.
    num_spans: int = form.number_input(question, 0, None, 100)  # type: ignore
    format_func = lambda s: "None" if s is None else _speaker(s)
    speakers = [None] + list(train_dataset.keys())
    speaker: typing.Optional[Speaker] = form.selectbox("Speaker", speakers, format_func=format_func)
    random_seed: int = form.number_input("What seed should we use?", value=123)  # type: ignore
    if not form.form_submit_button("Submit"):
        return

    spans, clips = _get_spans(train_dataset, dev_dataset, speaker, num_spans)

    with st.spinner("Generating Annotations..."):
        with fork_rng(seed=random_seed):
            intervals = [
                cf.partial(_random_nonoverlapping_alignments)(s.speech_segments) for s in spans
            ]

    with st.spinner("Assembling data..."):
        data = []
        for idx, (span, annos, clip) in enumerate(zip(spans, intervals, clips)):
            data.extend(_gather_data(idx, span, a, clip) for a in annos)
        df = pandas.DataFrame(data)

    st.subheader("Data")
    st_ag_grid(df, audio_column_name="clip")
    _stats(spans, data, intervals)
    _distributions(data)


if __name__ == "__main__":
    main()
