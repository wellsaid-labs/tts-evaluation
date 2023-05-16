""" A workbook to load local audio and evaluate test cases.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/evaluate_local_audio.py --runner.magicEnabled=false
"""
import os

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from dataclasses import dataclass

import lib


@dataclass
class Session:
    speaker: str
    style: str
    script: str
    dialect: str
    session: str


def initialize_state():
    for key in st.session_state.keys():
        if key.startswith("vote") or key.startswith("note"):  # type: ignore
            st.session_state[key] = ""

    st.session_state.audios = []
    st.session_state.metadata = []
    st.session_state.datatable = pd.DataFrame()
    st.session_state.output_path = ""


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


def main():
    st.markdown("# Test Case Audio Generator")
    st.markdown("Use this workbook to evaluate test cases.")

    if "metadata" not in st.session_state:
        initialize_state()

    form: DeltaGenerator = st.form(key="go")
    with form:
        metadata_dir = st.text_input(
            f"Absolute path to directory containing metadata csv and audio wavs, "
            f"e.g. `/Users/{os.getenv('USER')}/Downloads/test-data`"
        )
        st.session_state.output_path = f'{metadata_dir.split("/")[-1]}-evaluations.csv'
        if metadata_dir:
            st.markdown(f"Your evaluations will be stored as `{st.session_state.output_path}`")
        if st.form_submit_button("⚡️ GO ⚡️") and metadata_dir:
            with st.spinner("Loading audio..."):
                initialize_state()
                try:
                    metadata_path = [
                        os.path.join(dir_path, fname)
                        for dir_path, _, fnames in os.walk(metadata_dir)
                        for fname in fnames
                        if fname.endswith(".csv")
                    ][0]
                except IndexError:
                    st.write(f"No .csv found in {metadata_dir}")
                    raise IndexError
                metadata = pd.read_csv(metadata_path)
                audio_paths = [os.path.join(metadata_dir, i) for i in metadata.Audio.values.tolist()]
                metadata = [
                    Session(
                        speaker=r.Speaker,
                        style=r.Style,
                        script=r.Script,
                        dialect=r.Dialect,
                        session=r.Session
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
                        "Session": session.session
                    }
                    rows.append(row)

                if rows:
                    if st.form_submit_button("⚡️ SAVE CHANGES ⚡️"):
                        for row, vote, note in zip(rows, votes, notes):
                            row["Vote"] = vote
                            row["Note"] = note
                        df = pd.DataFrame(rows)
                        st.session_state.datatable = df

    if st.download_button(
        "💾  DOWNLOAD  💾",
        st.session_state.datatable.to_csv(),
        file_name=st.session_state.output_path
    ):
        st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
