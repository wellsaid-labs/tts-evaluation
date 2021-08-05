""" Streamlit that runs a couple of analyses on opinion scores, including calculating the MOS.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/mos_score.py --runner.magicEnabled=false
"""
import io
import itertools
import pathlib
import typing
from functools import partial

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_datatable import st_datatable

from run._streamlit import path_to_web_path, web_path_to_url

st.set_page_config(layout="wide")


def audio_to_html(row, audio_name_column: str, audio_directory: str):
    audio_path = pathlib.Path(audio_directory) / (str(row[audio_name_column]) + ".wav")
    return f'<audio controls src="{ web_path_to_url(path_to_web_path(audio_path))}"></audio>'


def st_merge(dfs: typing.List[pd.DataFrame]) -> pd.DataFrame:
    """Streamlit interface for merging `DataFrame`s."""
    df = dfs[0]

    merge_options = []
    for cols in itertools.product(*tuple(list(d.columns) for d in dfs)):
        sets = [set(d[col]) for d, col in zip(dfs, cols)]
        biggest = max(sets, key=lambda s: len(s))
        if all(s.issubset(biggest) for s in sets):
            merge_options.append(cols)
    st.info(f"Found only {len(merge_options)} groups of columns which can be merged.")

    merge_columns = st.selectbox("Merge On", merge_options, format_func=lambda a: ", ".join(a))
    for i, other_df in enumerate(dfs[1:]):
        df = df.merge(other_df, left_on=merge_columns[i], right_on=merge_columns[i + 1])

    assert len(df) == min(len(d) for d in dfs)
    st.info(f"Merge discarded {max(len(d) for d in dfs) - len(df)} rows")
    return df


def st_bar_chart(
    df: pd.DataFrame,
    bucket_columns: typing.Union[typing.Tuple[str], typing.Tuple[str, str]],
    metric_column: str,
    min_val: float,
    max_val: float,
):
    """Create a bar chart from `df` with `bucket_columns` as the x-axis and `metric_column` as the
    y-axis.
    """
    scale = alt.Scale(domain=[min_val, max_val])
    aggregated_column = "Mean of " + metric_column
    bar_chart = (
        alt.Chart(df)
        .transform_aggregate(
            [alt.AggregatedFieldDef(field=metric_column, op="mean", **{"as": aggregated_column})],
            groupby=[bucket_columns[0]],
        )
        .mark_bar()
        .encode(
            x=alt.X(field=bucket_columns[0], type="nominal", title=""),
            y=alt.Y(field=aggregated_column, type="quantitative", scale=scale),
            color=alt.X(field=bucket_columns[0], type="nominal"),
            tooltip=[alt.Tooltip(field=aggregated_column, type="quantitative")],
        )
    )
    error_bar_chart = (
        alt.Chart(df)
        .mark_errorbar(extent="ci", clip=True)
        .encode(
            x=alt.X(field=bucket_columns[0], type="nominal"),
            y=alt.Y(field=metric_column, type="quantitative", scale=scale),
        )
    )
    chart = bar_chart + error_bar_chart
    if len(bucket_columns) == 2:
        chart = chart.facet(
            column=alt.Column(
                field=bucket_columns[1],
                type="nominal",
                title="",
                # NOTE: Learn more:
                # https://stackoverflow.com/questions/61134669/rotated-column-headers-in-altair-have-uneven-offset
                header=alt.Header(labelAngle=-90, labelAlign="right"),
            ),
        )
    st.altair_chart(chart)


def main():
    st.markdown("# Opinion Score Evaluation")
    st.write("Analyze opinion scores to calculate the MOS.")
    st.write(
        "Upload CSV file(s) with the various data and metadata "
        "required to complete the analysis."
    )
    files = st.file_uploader("CSV(s)", accept_multiple_files=True)
    if len(files) == 0:
        st.stop()

    dfs = [typing.cast(pd.DataFrame, pd.read_csv(io.BytesIO(f.read()))) for f in files]
    df = dfs[0] if len(dfs) == 1 else st_merge(dfs)

    partial_ = partial(
        audio_to_html,
        audio_name_column=st.selectbox("Audio Name Column", list(df.columns)),
        audio_directory=st.text_input("Audio Directory"),
    )
    df["Audio"] = df.apply(partial_, axis=1)
    st_datatable(df)

    st.markdown("## Aggregate Statistics")
    segment_columns = st.multiselect("Segment On", list(df.columns))
    numeric_columns = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    metric_column = st.selectbox("Metric Column", numeric_columns)
    min_val, max_val = min(df[metric_column]), max(df[metric_column])
    min_val, max_val = st.slider("Chart range", min_val, max_val, (min_val, max_val), step=0.25)
    if len(segment_columns) > 2 or len(segment_columns) == 0:
        st.error("Please select only 1 or 2 columns.")
        st.stop()

    selected = df[[metric_column] + segment_columns]
    grouped = selected.groupby(segment_columns).agg(["mean", "std"]).reset_index()
    grouped.columns = [" | ".join(r for r in c if r) for c in grouped.columns]
    st_datatable(grouped)
    st_bar_chart(selected, segment_columns, metric_column, min_val, max_val)


if __name__ == "__main__":
    main()