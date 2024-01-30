""" A workbook to load local audio and evaluate test cases.
TODO: Use context manager and tempfile.TemporaryDirectory() to unzip files instead of extracting
    them to /tmp-eval and deleting that directory

Usage:
    $ python -m streamlit run evaluate_audio/manual/evaluate_local_audio.py --runner.magicEnabled=false
"""
import os
import zipfile
from dataclasses import dataclass

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from utils._streamlit import make_temp_web_dir, st_download_files


@dataclass
class Utterance:
    speaker: str
    dialect: str
    session: str
    style: str
    script: str
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
    metadata = "metadata.csv"
    if metadata not in files:
        st.error("MISSING METADATA.CSV")
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
        if key.startswith("Note") or key.startswith("Issue"):  # type: ignore
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
    h4.subheader("Issues")
    h5.subheader("Notes")
    st.divider()


def prepare_download_paths():
    """Turn audio and evaluations dataframe into WebPath objects, which can then be zipped and
    downloaded.
    """
    download_paths = []
    for a in st.session_state.audios:
        audio_path = make_temp_web_dir() / f"{a.split('/')[-1:][0]}"
        with open(a, "rb") as out_file:
            audio_path.write_bytes(out_file.read())
        download_paths.append(audio_path)
    df_path = make_temp_web_dir() / "evaluations.csv"
    df_path.write_text(st.session_state.datatable.to_csv())
    download_paths.append(df_path)
    return download_paths


def main():
    st.markdown("# Local audio evaluation tool")
    st.markdown("Use this workbook to evaluate test cases.")
    metadata, audio_paths, zip_path = pd.DataFrame(), [], ""
    issue_options = [
        "Slurring",
        "Gibberish",
        "Word Skip",
        "Word Cutoff",
        "Mispronunciation",
        "Unnatural Intonation",
        "Speaker Switching",
        "Other"
    ]
    if "metadata" not in st.session_state:
        initialize_state()

    form: DeltaGenerator = st.form(key="go")
    with form:
        selected_file = st.file_uploader(
            "Upload zipfile", accept_multiple_files=False, type=".zip"
        )
        if selected_file is not None:
            metadata, audio_paths = unzip_audios_and_metadata(selected_file)
            zip_path = f"eval_{selected_file.name}"

        if st.form_submit_button("⚡️ GO ⚡️") and selected_file:
            with st.spinner("Loading audio..."):
                initialize_state()
                metadata = [
                    Utterance(
                        speaker=r.Speaker,
                        style=r.Style,
                        script=r.Script,
                        dialect=r.Dialect,
                        session=r.Session,
                        audio_path=r.Audio,
                    )
                    for i, r in metadata.iterrows()
                ]
                st.session_state.metadata.extend(metadata)
                st.session_state.audios.extend(audio_paths)

    if st.session_state.audios:
        with st.spinner("Generating survey..."):
            with st.form(key="survey"):
                setup_columns()
                notes, issues, rows = [], [], []
                audios_and_metadata = zip(
                    st.session_state.audios, st.session_state.metadata
                )
                for i, (wave, utterance) in enumerate(audios_and_metadata):
                    col1, col2, col3, col4, col5 = st.columns(col_widths)
                    with col1:
                        st.markdown("\n")
                        st.markdown(f"{utterance.speaker}_{utterance.style}")
                    with col2:
                        st.markdown("\n")
                        st.markdown(utterance.script)
                    with col3:
                        st.markdown("\n")
                        st.audio(wave)
                    with col4:
                        opts = [st.checkbox(i, key=f"{i}_{len(issues)}") for i in issue_options]
                        issues.append([issue_options[idx] for idx, i in enumerate(opts) if i])
                    with col5:
                        notes.append(
                            st.text_area(
                                "Notes",
                                key=f"Note{i}",
                                label_visibility="hidden",
                            )
                        )

                    st.divider()
                    row = {
                        "Speaker": utterance.speaker,
                        "Style": utterance.style,
                        "Script": utterance.script,
                        "Issues": "",
                        "Notes": "",
                        "Audio": utterance.audio_path,
                    }
                    rows.append(row)

                if rows:
                    saved = st.form_submit_button("⚡️ SAVE CHANGES ⚡️")
                    if saved:
                        for row, note, issue in zip(rows, notes, issues):
                            row["Notes"] = note
                            row["Issues"] = ", ".join(issue)
                        df = pd.DataFrame(rows)
                        st.session_state.datatable = df
                        st.session_state.ready_to_download = True

    if selected_file and st.session_state.ready_to_download:
        download_paths = prepare_download_paths()
        cleanup_files()
        st_download_files(zip_path, "💾  DOWNLOAD  💾", download_paths)


if __name__ == "__main__":
    main()
