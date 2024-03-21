""" A workbook to load local audio and evaluate test cases.

Usage:
    $ python -m streamlit run evaluate/absolute/manual_eval.py
"""

import os
import tempfile
import zipfile
from functools import partial

import pandas as pd
import soundfile as sf
import streamlit as st
from package_utils._streamlit import (
    audio_to_web_path,
    make_temp_web_dir,
    st_download_files,
)

col_widths = [1, 1, 1, 1, 1]
issue_options = [
    "Slurring",
    "Gibberish",
    "Word-Skip",
    "Word-Cutoff",
    "Mispronunciation",
    "Unnatural-Intonation",
    "Speaker-Switching",
    "Other",
]


def unzip_audio_and_metadata(zip_file):
    """Unzip file containing audio .wavs and metadata.csv. Return metadata as
        pd.DataFrame and audio as list of absolute paths.
    Args:
        zip_file (File): The UploadedFile object from st.file_upload
    Returns:
        Tuple(pd.DataFrame, list): Tuple containing the metadata as a dataframe
            and paths to the audio files
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
            audio_path = audio_to_web_path(
                data, name=f"{row.Speaker}_{row.Script}.wav"
            )
            st.session_state["input_audio"].append(audio_path)
        st.session_state["input_metadata"] = df


def initialize_state():
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


def setup_columns():
    h1, h2, h3, h4, h5 = st.columns(col_widths)
    h1.subheader("Speaker")
    h2.subheader("Script")
    h3.subheader("Audio")
    h4.subheader("Issues")
    h5.subheader("Notes")
    st.divider()


def prepare_metadata():
    issues = [
        {
            k.split("_")[1]: v
            for k, v in st.session_state.items()
            if k.startswith("Issue") and k.endswith(str(i))
        }
        for i in range(len(st.session_state.input_audio))
    ]
    notes = [
        st.session_state[i] for i in st.session_state if i.startswith("Note")
    ]
    new_rows = []
    for row, issue, note in zip(st.session_state.output_rows, issues, notes):
        row["Issues"] = ", ".join(i for i in issue if issue[i])
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


def set_ready_to_dl(output_rows):
    st.session_state.output_rows = output_rows
    st.session_state.ready_to_download = True


def survey():
    setup_columns()
    notes, issues, output_rows = [], [], []
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
            opts = [
                st.checkbox(i, key=f"Issue_{i}_{len(issues)}")
                for i in issue_options
            ]
            issues.append(
                [issue_options[idx] for idx, i in enumerate(opts) if i]
            )
        with col5:
            notes.append(
                st.text_area(
                    "Notes",
                    key=f"Note_{idx}",
                    label_visibility="hidden",
                )
            )
        st.divider()
        output_row = {
            "Speaker": row.Speaker,
            "Style": row.Style,
            "Script": row.Script,
            "Audio": str(audio_path).split("/")[-1],
        }
        output_rows.append(output_row)

    st.form_submit_button(
        "⚡️ SAVE CHANGES ⚡️",
        on_click=partial(set_ready_to_dl, output_rows),
    )


def main():
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
        download_paths = prepare_metadata()
        st_download_files(
            st.session_state.zip_path, "💾  DOWNLOAD  💾", download_paths
        )
        st.session_state.clear()


if __name__ == "__main__":
    main()
