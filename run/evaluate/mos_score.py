"""

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/mos_score.py --runner.magicEnabled=false
"""
import io
import typing

import pandas as pd
import streamlit as st


def main():
    st.markdown("# MOS Score Evaluation")
    files = st.file_uploader("CSV(s)", accept_multiple_files=True)
    if len(files) == 0:
        st.stop()

    dfs = [typing.cast(pd.DataFrame, pd.read_csv(io.BytesIO(f.read()))) for f in files]
    columns = [list(df.columns) for df in dfs]
    st.info(columns)
    shared_columns = list(set(columns[0]).intersection(*columns[1:]))
    if len(dfs) > 1 and len(shared_columns) == 0:
        st.error("No common columns to merge on.")
        st.stop()

    st.info(shared_columns)
    merge_on = st.selectbox("Merge On", shared_columns)
    df = dfs[0]
    for other_df in dfs[1:]:
        df = df.merge(other_df, on=merge_on)
    st.write(df)


if __name__ == "__main__":
    main()
