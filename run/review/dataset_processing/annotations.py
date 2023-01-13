""" Streamlit application for reviewing generated annotations.

TODO: It appears some speakers have like Patrick, Wade, Sofia have a relatively small variation
      in loudness while having a much larger variation in tempo. This is unexpected because I'd
      imagine speakers that are speaking consistently would be consistent in both dimensions.
      Let's investigate this, and document our understanding of variation in these datasets
      with examples.
TODO: Tobin has a 3x higher duplication rate and our other speakers. Joe and Garry are more like
      2x higher. While some duplication rate is expected (the data is being randomly sampled), this
      should be relatively even accross the board, unless we are oversampling certain portions
      of the data. Given that we use estimators to weigh each passage and speaker, it's likely
      those are off, and it'd be interesting to investigate Tobin specifically. Given the
      complexity of our pipeline with many different samplers and filtering rules, it might
      be beneficial to implement a dynamic algorithm for recalibrating our sampling weights
      based on the number of duplicates seen. This would be similar to how we dynamically sample
      speakers based on the number of audio seconds we've already generated for each speaker.

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset_processing/annotations.py \
        --runner.magicEnabled=false
"""
import logging
import typing
from collections import defaultdict
from statistics import stdev

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
from run._streamlit import audio_to_url, clip_audio, get_spans, st_ag_grid, st_download_bytes
from run._utils import Dataset, get_datasets
from run.data._loader import Alignment, Session, Span, Speaker
from run.train.spectrogram_model._data import (
    _get_loudness_annotation,
    _get_tempo_annotation,
    _random_nonoverlapping_alignments,
)

lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
logger = logging.getLogger(__name__)


@st.experimental_singleton()
def _get_datasets() -> typing.Tuple[Dataset, Dataset]:
    return get_datasets(False)


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
    text = span.script[anno.script_slice]
    spacy_span = doc.char_span(*anno.script, alignment_mode="expand")
    assert spacy_span is not None
    assert len(spacy_span.text) <= anno.script_len, "Invalid annotation"
    loudness = cf.partial(_get_loudness_annotation)(clip, span.audio_file.sample_rate, anno)
    tempo = cf.partial(_get_tempo_annotation)(span, anno)
    # TODO: Consider adding this to the `Span` implementation for hashing, equality, etc.
    key = (
        span.passage.audio_file.path.name,
        (span.audio_start + anno.audio[0], anno.audio_len),
    )

    return {
        "index": span_idx,
        "key": key,
        "anno_script": repr(_annotate(span.script, anno)),
        "tempo": cf.partial(_get_tempo_annotation)(span, anno),
        "diff_loudness": loudness if loudness is None else loudness - span.session.loudness,
        "audio_len": round(anno.audio_len, 2),
        "clip": audio_to_url(clip_audio(clip, span, anno)),
        "speaker": repr(span.session.spk),
        "session": span.session.lbl,
        "loudness": loudness,
        "diff_tempo": tempo - span.session.spk_tempo,
        "sesh_loudness": span.session.loudness,
        "sesh_tempo": span.session.tempo,
        "num_words": len(text.split()),
        "num_alignments": len(span.alignments),
        "script": text,
        "transcript": span.transcript[anno.transcript_slice],
        "script_len": anno.script_len,
        "transcript_len": anno.transcript_len,
        "anno_script_slice": anno.script,
        "_session": span.session,
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


def _anno_value_distributions(data: typing.List[typing.Dict], num_cols: int = 3):
    """Plot the distribution of various data points."""
    st.subheader("Annotation Value Distributions")
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


def _speakers_variability(data: typing.List[typing.Dict]):
    """Show a table summarizing the speakers variation in loudness and tempo.

    NOTE: We choose to demonstrate range by looking at the 10th or 100th example because we only
          need 10s of examples per speaker to demonstrate range, probably.
    """
    st.subheader("Speaker Variability Statistics")
    filters = (
        ("[0 long]", lambda r: r["audio_len"] >= 1),
        ("[1 no filter]", lambda _: True),
        ("[2 short]", lambda r: r["audio_len"] < 1),
    )
    attrs = (("diff_loudness", "loudness"), ("diff_tempo", "spk_tempo"))

    stats: typing.Dict[Speaker, typing.Dict[str, typing.Any]] = defaultdict(dict)
    for filter_name, filter_ in filters:
        for diff, avg in attrs:
            spk_anno: typing.Dict[Speaker, typing.List[float]] = defaultdict(list)
            spk_dups: typing.Dict[Speaker, typing.Set] = defaultdict(set)
            for row in data:
                if row[diff] is not None and filter_(row):
                    spk_anno[row["_session"].spk].append(row[diff])
                    spk_dups[row["_session"].spk].add(row["key"])

            for spk, vals in spk_anno.items():
                v = sorted(vals)
                prefix = f"[2] {filter_name}"
                percent = int(round(0.01 * len(v)))
                min_max: typing.Dict[str, typing.Optional[typing.Tuple[float, float]]] = {
                    f"{prefix} `{diff}` 10th": (v[10], v[-10]) if len(v) > 10 else None,
                    f"{prefix} `{diff}` 100th": (v[100], v[-100]) if len(v) > 100 else None,
                    f"{prefix} `{diff}` 5%": (v[percent * 5], v[-(percent * 5 + 1)]),
                }
                items = min_max.items()
                plus_minus = {f"{k} ±": v if v is None else (v[1] - v[0]) / 2 for k, v in items}
                num_unique = spk_dups[spk]
                stats[spk] = {
                    f"{prefix} Num `{diff}` Vals": len(v),
                    f"{prefix} Num `{diff}` Dups": len(v) - len(num_unique),
                    f"{prefix} % `{diff}` Dups": (1 - (len(num_unique) / len(v))) * 100,
                    f"{prefix} `{diff}` Stdev": stdev(v) if len(v) > 2 else None,
                    **plus_minus,
                    **min_max,
                    **stats[spk],
                }

    seshs: typing.Set[Session] = set(r["_session"] for r in data)
    for diff, avg in attrs:
        for sesh in seshs:
            s = stats[sesh.spk]

            key = "[1] Sesh"
            s[key] = s[key] if key in s else set()
            s[key].add(sesh.lbl)

            key = f"[1] Sesh `{avg}`"
            val = float(getattr(sesh, avg))  # TODO: Remove
            s[key] = s[key] if key in s else set()
            s[key].add(val)

    indicies = list(sorted(set(k for v in stats.values() for k in v.keys())))
    rows = {repr(s): [v[k] if k in v else None for k in indicies] for s, v in stats.items()}
    df = pandas.DataFrame(rows, index=indicies)
    st.dataframe(df, use_container_width=True)
    name = "speaker_variability.csv"
    st_download_bytes(name, "📁 Download", df.to_csv().encode("utf-8"))
    st.dataframe(df.T, use_container_width=True)
    st_download_bytes(name, "📁 Download", df.T.to_csv().encode("utf-8"))


def main():
    run._config.configure(overwrite=True)

    st.title("Annotations")
    st.write("The workbook reviews the annotations that are being generated for spans.")

    if st.sidebar.button("Clear Dataset Cache"):
        _get_datasets.clear()
    train_dataset, dev_dataset = _get_datasets()

    form: DeltaGenerator = st.form("settings")
    question = "How many span(s) do you want to generate?"
    # NOTE: Too many spans could cause the `streamlit` to refresh and start over, this has happened
    # around 5000 spans.
    num_spans: int = form.number_input(question, 0, None, 100)  # type: ignore
    format_func = lambda s: "None" if s is None else _speaker(s)
    speakers = [None] + list(train_dataset.keys())
    speaker: typing.Optional[Speaker] = form.selectbox("Speaker", speakers, format_func=format_func)
    random_seed: int = form.number_input("What seed should we use?", value=123)  # type: ignore
    load_individual = form.checkbox("Load individual annotations", value=True)
    load_distributions = form.checkbox("Load annotation value distributions", value=False)
    if not form.form_submit_button("Submit"):
        return

    spans, clips = get_spans(train_dataset, dev_dataset, speaker, num_spans, False)

    with st.spinner("Generating Annotations..."):
        with fork_rng(seed=random_seed):
            intervals = [
                cf.partial(_random_nonoverlapping_alignments)(s.speech_segments) for s in spans
            ]

    with st.spinner("Assembling data..."):
        data = []
        for i, (span, annos, clip) in enumerate(zip(spans, intervals, clips)):
            data.extend(_gather_data(i, span, a, clip) for a in annos)
        df = pandas.DataFrame(data)

    if load_individual:
        st.subheader("Individual Annotations")
        st_ag_grid(df, audio_column_name="clip")

    _speakers_variability(data)
    _stats(spans, data, intervals)

    if load_distributions:
        _anno_value_distributions(data)


if __name__ == "__main__":
    main()
