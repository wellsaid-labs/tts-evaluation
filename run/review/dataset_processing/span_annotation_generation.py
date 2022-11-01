""" Streamlit application for reviewing generated annotations.

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset_processing/span_annotation_generation.py \
        --runner.magicEnabled=false
"""
import collections
import logging
import typing

import config as cf
import numpy
import pandas
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

import lib
import run
from lib.audio import sec_to_sample
from run._streamlit import audio_to_url
from run._utils import Dataset, SpanGenerator, get_datasets
from run.data._loader import Span
from run.data._loader.structures import Alignment
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
def _get_datasets_and_generators() -> typing.Tuple[Dataset, Dataset, SpanGenerator, SpanGenerator]:
    train_dataset, dev_dataset = get_datasets(False)
    cf.add(run._config.make_spectrogram_model_train_config(train_dataset, dev_dataset, False))
    train_gen, dev_gen = cf.partial(_get_data_generator)(train_dataset, dev_dataset)
    return train_dataset, dev_dataset, train_gen, dev_gen


@st.experimental_singleton()
def _get_spans(
    _generator: SpanGenerator, num_spans: int
) -> typing.Tuple[typing.List[Span], typing.List[numpy.ndarray]]:
    """Get a list"""
    with st.spinner("Making spans..."):
        bar = st.progress(0)
        spans = []
        for _ in range(num_spans):
            spans.append(next(_generator))
            bar.progress(len(spans) / num_spans)
        bar.empty()

    with st.spinner("Loading audio..."):
        signals = [s.audio() for s in spans]

    return spans, signals


def _annotate(text: str, alignment: Alignment, prefix: str = "<<<", suffix: str = ">>>") -> str:
    """Mark the `slice_` in the `text` with a `prefix` and `suffix`."""
    text = text[: alignment.script[1]] + suffix + text[alignment.script[1] :]
    return text[: alignment.script[0]] + prefix + text[alignment.script[0] :]


def _bucket_and_visualize(values: typing.List[float]):
    """Bucket `values` and visualize the distribution."""
    buckets = collections.defaultdict(int)
    for val in values:
        buckets[val] += 1
    bucket_items = list(buckets.items())
    data = {
        "buckets": [i[0] for i in bucket_items],
        "counts": [i[1] for i in bucket_items],
    }
    st.bar_chart(pandas.DataFrame(data), x="buckets", y="counts")


def _clip_audio(audio: numpy.ndarray, span: Span, alignment: Alignment):
    """Get a clip of `audio` at `alignment`."""
    sample_rate = span.audio_file.sample_rate
    start_ = sec_to_sample(max(alignment.audio[0], 0), sample_rate)
    stop_ = sec_to_sample(min(alignment.audio[1], span.audio_length), sample_rate)
    return audio[start_:stop_]


def main():
    run._config.configure(overwrite=True)

    st.title("Span Annotation Generation")
    st.write("The workbook reviews our span annotation generation functionality.")

    form = st.form("settings")
    question = "How many span(s) do you want to generate?"
    # NOTE: Too many spans could cause the `streamlit` to refresh and start over, this has happened
    # around 5000 spans.
    num_spans: int = form.number_input(question, 0, None, 100)  # type: ignore
    use_dev = form.checkbox("Analyze development dataset")
    if not form.form_submit_button("Submit"):
        return

    if st.sidebar.button("Clear Dataset Cache"):
        _get_datasets_and_generators.clear()
    _, _, train_gen, dev_gen = _get_datasets_and_generators()
    generator = dev_gen if use_dev else train_gen
    if st.sidebar.button("Clear Span Cache"):
        _get_spans.clear()
    spans, clips = _get_spans(generator, num_spans)

    with st.spinner("Generating Annotations..."):
        intervals = [cf.partial(_random_nonoverlapping_alignments)(s.alignments) for s in spans]

    with st.spinner("Assembling data..."):
        data = [
            {
                "index": j,
                "script": repr(_annotate(s.script, a)),
                "loudness": cf.partial(_get_loudness_annotation)(c, s.audio_file.sample_rate, a),
                "tempo": cf.partial(_get_tempo_annotation)(a),
                "clip": audio_to_url(_clip_audio(c, s, a)),
                "speaker": repr(s.session[0]),
                "session": s.session[1],
                "num_alignments": len(s.alignments),
                "transcript": s.transcript[a.transcript[0] : a.transcript[1]],
                "num_words": len(s.script[a.script[0] : a.script[1]].split()),
                "audio_len": round(a.audio[1] - a.audio[0], 1),
                "script_len": a.script[1] - a.script[0],
                "transcript_len": a.transcript[1] - a.transcript[0],
            }
            for j, (s, i, c) in enumerate(zip(spans, intervals, clips))
            for a in i
        ]
        df = pandas.DataFrame(data)

    options = GridOptionsBuilder.from_dataframe(df)
    options.configure_pagination()
    options.configure_default_column(wrapText=True, autoHeight=True, min_column_width=1)
    # NOTE: This follows the examples highlighted here:
    # https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/main/cell_renderer_class_example.py
    # https://github.com/PablocFonseca/streamlit-aggrid/issues/119
    renderer = JsCode(
        'function(params) {return `<audio controls preload="none" src="${params.value}" />`}'
    )
    options.configure_column("clip", cellRenderer=renderer)
    options = options.build()
    st.subheader("Data")
    AgGrid(
        data=df,
        gridOptions=options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=750,
    )

    st.subheader("Stats")
    st.write(f"- Number of `Span`s:  {len(spans)}")
    st.write(f"- Number of annotations: {len(data)}")

    num_spans_little_alignments = sum(1 for s in spans if len(s.alignments) < 10)
    st.write(f"- Number of `Span`s with less than 10 alignments: {num_spans_little_alignments}")

    avg_annotations = round(len(data) / len(spans), 1)
    st.write(f"- Average annotations per `Span`: {avg_annotations}")

    num_anno_enough_alignments = sum(1 for r in data if r["num_alignments"] >= 10)
    num_spans_enough_alignments = sum(1 for s in spans if len(s.alignments) >= 10)
    avg_annotations = round(num_anno_enough_alignments / num_spans_enough_alignments, 1)
    st.write(f"- Average annotations per `Span` with more than 10 alignments: {avg_annotations}")

    num_anno_enough_alignments = sum(1 for r in data if r["num_alignments"] >= 10)
    num_spans_enough_alignments = len(set(r["index"] for r in data if r["num_alignments"] >= 10))
    avg_annotations = round(num_anno_enough_alignments / num_spans_enough_alignments, 1)
    label = "Average annotations per annotated `Span` with more than 10 alignments"
    st.write(f"- {label}: {avg_annotations}")

    num_no_annotations = sum(1 for i in intervals if len(i) == 0)
    per_no_annotations = round((num_no_annotations / len(spans)) * 100)
    st.write(f"- Percentage of `Span`s that recieved no annotations: {per_no_annotations}%")

    loudness_vals = [r["loudness"] for r in data if r["loudness"] is not None]
    st.write(f"- Number of loudness annotations: {len(loudness_vals)}.")

    st.write(f"- Loudness range: {max(loudness_vals)} to {min(loudness_vals)} db.")
    tempo_vals = [r["tempo"] for r in data]

    st.write(f"- Tempo range: {min(tempo_vals)} to {max(tempo_vals)} seconds per character")

    st.subheader("Loudness Annotation Values Distribution")
    _bucket_and_visualize(loudness_vals)

    st.subheader("Tempo Annotation Values Distribution")
    _bucket_and_visualize(tempo_vals)

    st.subheader("Annotation Length Distribution (words)")
    _bucket_and_visualize([r["num_words"] for r in data])

    st.subheader("Annotation Length Distribution (characters)")
    _bucket_and_visualize([r["script_len"] for r in data])

    st.subheader("Annotation Length Distribution (seconds)")
    _bucket_and_visualize([r["audio_len"] for r in data])


if __name__ == "__main__":
    main()
