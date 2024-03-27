""" A workbook to load local audio and evaluate test cases.

Usage:
    $ python -m streamlit run evaluate/absolute/manual_eval.py
"""

import os
import tempfile
import zipfile
from functools import partial
from typing import Callable

import pandas as pd
import soundfile as sf
import streamlit as st
from package_utils._streamlit import (
    audio_to_web_path,
    make_temp_web_dir,
    st_download_files,
)

col_widths = [1, 2, 1.5, 1, 1]
issue_options = sorted(
    [
        "Slurring",
        "Gibberish",
        "Word-Skip",
        "Word-Cutoff",
        "Mispronunciation",
        "Unnatural-Intonation",
        "Speaker-Switching",
    ]
)
issue_options.append("Other")


def initialize_state() -> None:
    """Set values in st.session_state for first run"""
    for key in st.session_state.keys():
        if key.startswith("Note") or key.startswith("Issue"):  # type: ignore
            st.session_state[key] = ""
    st.session_state.input_audio = []
    st.session_state.output_metadata = pd.DataFrame()
    st.session_state.input_metadata = pd.DataFrame()
    st.session_state.output_path = ""
    st.session_state.ready_to_download = False
    st.session_state.zip_path = ""
    st.session_state.output_rows = []


def unzip_audio_and_metadata(zip_file: str) -> None:
    """Unzip file containing audio wavs and metadata.csv. Update session state
    with data from zip file.
    Args:
        zip_file (File): The UploadedFile object from st.file_upload
    Returns:
        None: This function only updates st.session_state
    """
    zf = zipfile.ZipFile(zip_file)
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        files = os.listdir(tempdir)
        metadata = "metadata.csv"
        if metadata not in files:
            st.error("MISSING METADATA.CSV")
        df_path = os.path.join(tempdir, metadata)
        df = pd.read_csv(df_path)
        for idx, row in df.iterrows():
            data, sr = sf.read(os.path.join(tempdir, row.Audio))
            audio_path = audio_to_web_path(data, name=row.Audio)
            st.session_state["input_audio"].append(audio_path)
        st.session_state["input_metadata"] = df


def get_widget_values(
    widget_prefix: str, value_func: Callable, sort_func: Callable
) -> list:
    """Helper function to retrieve and format items from st.session_state
    Args:
        widget_prefix: The starting characters of the widgets to fetch
        value_func: Function to format the values of the widgets
        sort_func: Function to sort the values of the widgets
    Returns:
        values: The list of values associated with the widgets
    """
    widgets = {
        k: v for k, v in st.session_state.items() if k.startswith(widget_prefix)
    }
    values = [value_func(v) for k, v in sorted(widgets.items(), key=sort_func)]
    return values


def fetch_user_inputs() -> list:
    """Fetch evaluations from st.session_state
    Returns:
        download_paths: A list of paths to download, including original audio
            files and new user inputs
    """
    issues = get_widget_values(
        widget_prefix="Issue",
        value_func=lambda x: ", ".join(x),
        sort_func=lambda x: int(x[0].split("_")[1]),
    )
    notes = get_widget_values(
        widget_prefix="Notes",
        value_func=lambda x: x,
        sort_func=lambda x: int(x[0].split("_")[1]),
    )
    new_rows = []
    for row, issue, note in zip(st.session_state.output_rows, issues, notes):
        row["Issues"] = issue
        row["Notes"] = note
        new_rows.append(row)
    df = pd.DataFrame(new_rows)
    st.session_state.output_metadata = df
    download_paths = []
    download_paths.extend(st.session_state.input_audio)
    df_path = make_temp_web_dir() / "evaluations.csv"
    df_path.write_text(st.session_state.output_metadata.to_csv())
    download_paths.append(df_path)
    return download_paths


def setup_columns() -> None:
    """Set up column headers for survey()"""
    h1, h2, h3, h4, h5 = st.columns(col_widths)
    h1.subheader("Speaker")
    h2.subheader("Script")
    h3.subheader("Audio")
    h4.subheader("Issues")
    h5.subheader("Notes")
    st.divider()


def set_ready_to_dl(output_rows) -> None:
    """Callback function which indicates the output is ready to be downloaded"""
    st.session_state.output_rows = output_rows
    st.session_state.ready_to_download = True


def survey() -> None:
    """The main survey portion of the evaluation app. Since this is function
    is called within a form, the actual values input by users are stored in
    st.session_state and retrieved later."""
    setup_columns()
    output_rows = []
    audios_and_metadata = zip(
        st.session_state.input_audio, st.session_state.input_metadata.iterrows()
    )
    for idx, (audio_path, (row_idx, row)) in enumerate(audios_and_metadata):
        col1, col2, col3, col4, col5 = st.columns(col_widths)
        with col1:
            st.markdown(f"{row.Speaker}_{row.Style}")
        with col2:
            st.markdown(row.Script)
        with col3:
            st.audio(str(audio_path))
        with col4:
            st.multiselect(
                label="Issues",
                options=issue_options,
                key=f"Issues_{row_idx}",
                label_visibility="collapsed",
            )
        with col5:
            st.text_area(
                label="Notes",
                key=f"Notes_{row_idx}",
                label_visibility="collapsed",
            )
        st.divider()
        output_row = {
            "Speaker": row.Speaker,
            "Style": row.Style,
            "Speaker_ID": row.Speaker_ID,
            "Model_Version": row.Model_Version,
            "Script": row.Script,
            "Audio": str(audio_path).split("/")[-1],
        }
        output_rows.append(output_row)

    st.form_submit_button(
        "⚡️ SAVE CHANGES ⚡️",
        on_click=partial(set_ready_to_dl, output_rows),
    )


def main() -> None:
    st.set_page_config(layout="wide")
    if "input_audio" not in st.session_state:
        initialize_state()
    st.markdown("# Local audio evaluation tool")
    st.markdown("Use this workbook to evaluate test cases.")
    selected_file = st.file_uploader(
        "Upload zipfile", accept_multiple_files=False, type=".zip"
    )
    if not st.session_state.ready_to_download:
        if st.button("⚡️ GO ⚡️") and selected_file:
            initialize_state()
            st.session_state.zip_path = f"eval_{selected_file.name}"
            unzip_audio_and_metadata(selected_file)
            with st.form(key="survey"):
                survey()
    else:
        download_paths = fetch_user_inputs()
        st_download_files(
            st.session_state.zip_path, "💾  DOWNLOAD  💾", download_paths
        )
        st.session_state.clear()


if __name__ == "__main__":
    main()
