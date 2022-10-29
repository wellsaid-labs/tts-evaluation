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
from run._models.spectrogram_model import SpanAnnotations
from run._streamlit import audio_to_url, span_audio
from run._utils import Dataset, SpanGenerator, get_datasets
from run.data._loader import Span
from run.train.spectrogram_model._data import (
    _random_loudness_annotations,
    _random_tempo_annotations,
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


def _annotate(text: str, slice_: slice, prefix: str = "<<<", suffix: str = ">>>"):
    """Mark the `slice_` in the `text` with a `prefix` and `suffix`."""
    text = text[: slice_.stop] + suffix + text[slice_.stop :]
    return text[: slice_.start] + prefix + text[slice_.start :]


def _visualize_annotations(
    spans: typing.List[Span], annotations_batch: typing.List[SpanAnnotations]
):
    data = [
        {
            "script": repr(_annotate(s.script, i[0])),
            "annotation_len": i[0].stop - i[0].start,
            "value": i[1],
            "clip": audio_to_url(span_audio(s)),
        }
        for s, a in zip(spans, annotations_batch)
        for i in a
    ]
    df = pandas.DataFrame(data)
    options = GridOptionsBuilder.from_dataframe(df)
    options.configure_pagination()
    options.configure_default_column(wrapText=True, autoHeight=True)
    # NOTE: This follows the examples highlighted here:
    # https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/main/cell_renderer_class_example.py
    # https://github.com/PablocFonseca/streamlit-aggrid/issues/119
    renderer = JsCode("""function(params) {return `<audio controls src="${params.value}" />`}""")
    options.configure_column("clip", cellRenderer=renderer)
    options.configure_column("script", flex=1)
    options = options.build()
    st.subheader("Data")
    AgGrid(
        data=df,
        gridOptions=options,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=1000,
    )

    st.subheader("Distribution")
    buckets = collections.defaultdict(int)
    for annotations in annotations_batch:
        for annotation in annotations:
            buckets[annotation[1]] += 1
    bucket_items = list(buckets.items())
    data = {
        "buckets": [i[0] for i in bucket_items],
        "counts": [i[1] for i in bucket_items],
    }
    st.bar_chart(pandas.DataFrame(data), x="buckets", y="counts")


def main():
    run._config.configure(overwrite=True)

    st.title("Annotation Generation")
    st.write("The workbook reviews our annotation generation functionality.")

    form = st.form("settings")
    question = "How many span(s) do you want to generate?"
    num_spans: int = form.number_input(question, 0, None, 1)  # type: ignore
    use_dev = form.checkbox("Analyze development dataset")
    if not form.form_submit_button("Submit"):
        return

    if st.sidebar.button("Clear Dataset Cache"):
        _get_datasets_and_generators.clear()
    _, _, train_gen, dev_gen = _get_datasets_and_generators()
    generator = dev_gen if use_dev else train_gen
    if st.sidebar.button("Clear Span Cache"):
        _get_spans.clear()
    spans, signals = _get_spans(generator, num_spans)

    st.header("Loudness Annotations")
    loudness_annotations = [_random_loudness_annotations(s, a) for s, a in zip(spans, signals)]
    _visualize_annotations(spans, loudness_annotations)

    st.header("Tempo Annotations")
    tempo_annotations = [cf.partial(_random_tempo_annotations)(s) for s in spans]
    _visualize_annotations(spans, tempo_annotations)


if __name__ == "__main__":
    main()
