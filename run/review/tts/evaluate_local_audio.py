""" A workbook to load local audio and evaluate test cases.
TODO: Use context manager and tempfile.TemporaryDirectory() to unzip files instead of extracting
    them to /tmp-eval and deleting that directory

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/evaluate_local_audio.py --runner.magicEnabled=false
"""
import os
import zipfile

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from dataclasses import dataclass
import run
from run._streamlit import st_download_files, path_to_web_path, make_temp_web_dir, audio_to_web_path

import lib


@dataclass
class Session:
    speaker: str
    style: str
    script: str
    dialect: str
    session: str
    audio_path: str


temp_dir = "/tmp/eval"


def unzip_audios_and_metadata(zip_file):
    """Unzip file containing audio .wavs and metadata.csv. Return metadata as pd.DataFrame and
    audio as list of absolute paths.

    Args:
        zip_file (File): The UploadedFile object from st.file_upload

    Returns:
        Tuple(pd.DataFrame, list): Tuple containing the metadata as a dataframe and paths to the
            audio files
    """
    zf = zipfile.ZipFile(zip_file)
    zf.extractall(temp_dir)
    files = os.listdir(temp_dir)
    metadata = [i for i in files if i.endswith(".csv")][0]
    df_path = os.path.join(temp_dir, metadata)
    df = pd.read_csv(df_path)
    audios = [os.path.join(temp_dir, i) for i in df.Audio.tolist()]
    return df, audios


def cleanup_files():
    """Delete unzipped files"""
    files = os.listdir(temp_dir)
    cleanup_paths = [os.path.join(temp_dir, i) for i in files]
    for i in cleanup_paths:
        os.remove(i)
    os.rmdir(temp_dir)


def initialize_state():
    for key in st.session_state.keys():
        if key.startswith("vote") or key.startswith("note"):  # type: ignore
            st.session_state[key] = ""

    st.session_state.audios = []
    st.session_state.metadata = []
    st.session_state.datatable = pd.DataFrame()
    st.session_state.output_path = ""
    st.session_state.ready_to_download = False


st.set_page_config(layout="wide")

st.markdown(
    """ <style>
            div[role="radiogroup"] >  :first-child{
                display: none !important;
            }
            audio {
                width: 86% !important;
                height: 58px !important;
            }
            div.row-widget.stRadio {
                left-padding: 10px !important;
            }
        </style>
        """,
    unsafe_allow_html=True,
)

col_widths = [1, 2, 2, 1, 2]


def setup_columns():
    h1, h2, h3, h4, h5 = st.columns(col_widths)
    h1.subheader("Speaker")
    h2.subheader("Script")
    h3.subheader("Audio")
    h4.subheader("Vote")
    h5.subheader("Note")
    st.divider()


def prepare_download_paths():
    """Turn audio and evaluations dataframe into WebPath objects, which can then be zipped and
    downloaded.
    """
    download_paths = []
    for i, a in enumerate(st.session_state.audios):
        audio_path = make_temp_web_dir() / f"audio{i}.wav"
        with open(a, "rb") as out_file:
            audio_path.write_bytes(out_file.read())
        download_paths.append(audio_path)
    df_path = make_temp_web_dir() / "evaluations.csv"
    df_path.write_text(st.session_state.datatable.to_csv())
    download_paths.append(df_path)
    return download_paths


def main():
    run._config.configure(overwrite=True)
    st.markdown("# Test Case Audio Generator")
    st.markdown("Use this workbook to evaluate test cases.")
    metadata, audio_paths, zip_path = pd.DataFrame(), [], ""

    if "metadata" not in st.session_state:
        initialize_state()

    form: DeltaGenerator = st.form(key="go")
    with form:
        selected_file = st.file_uploader("Upload zipfile", accept_multiple_files=False, type=".zip")
        if selected_file is not None:
            metadata, audio_paths = unzip_audios_and_metadata(selected_file)
            zip_path = f"eval_{selected_file.name}"

        if st.form_submit_button("⚡️ GO ⚡️") and selected_file:
            with st.spinner("Loading audio..."):
                initialize_state()
                metadata = [
                    Session(
                        speaker=r.Speaker,
                        style=r.Style,
                        script=r.Script,
                        dialect=r.Dialect,
                        session=r.Session,
                        audio_path=r.Audio
                    )
                    for i, r in metadata.iterrows()
                ]
                st.session_state.metadata.extend(metadata)
                st.session_state.audios.extend(audio_paths)

    if st.session_state.audios:
        with st.spinner("Generating survey..."):
            with st.form(key="survey"):
                setup_columns()
                votes, notes, rows = [], [], []
                audios_and_metadata = zip(st.session_state.audios, st.session_state.metadata)
                for i, (wave, session) in enumerate(audios_and_metadata):
                    col1, col2, col3, col4, col5 = st.columns(col_widths)
                    with col1:
                        st.markdown("\n")
                        st.markdown(session.speaker)
                    with col2:
                        st.markdown("\n")
                        st.markdown(session.script)
                    with col3:
                        st.markdown("\n")
                        st.audio(wave)
                    with col4:
                        votes.append(
                            st.radio(
                                "vote",
                                ["", "⭐", "❌"],
                                key=f"vote{i}",
                                horizontal=True,
                                label_visibility="hidden",
                            )
                        )
                    with col5:
                        notes.append(
                            st.text_input("note", key=f"note{i}", label_visibility="hidden")
                        )
                    st.divider()
                    row = {
                        "Speaker": session.speaker,
                        "Style": session.style,
                        "Script": session.script,
                        "Vote": "",
                        "Note": "",
                        "Dialect": session.dialect,
                        "Session": session.session,
                        "Audio": session.audio_path
                    }
                    rows.append(row)

                if rows:
                    saved = st.form_submit_button("⚡️ SAVE CHANGES ⚡️")
                    if saved:
                        for row, vote, note in zip(rows, votes, notes):
                            row["Vote"] = vote
                            row["Note"] = note
                        df = pd.DataFrame(rows)
                        st.session_state.datatable = df
                        st.session_state.ready_to_download = True

    if selected_file and st.session_state.ready_to_download:
        download_paths = prepare_download_paths()
        cleanup_files()
        st_download_files(zip_path, "💾  DOWNLOAD  💾", download_paths)


if __name__ == "__main__":
    main()
