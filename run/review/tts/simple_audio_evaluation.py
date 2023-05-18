""" A workbook to generate audio and evaluate various test cases.

TODO: Implement `batch_griffin_lim_tts` to support batch generation, speeding up this script.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/simple_audio_evaluation.py --runner.magicEnabled=false
"""
import random
import typing
from functools import partial

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import run
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType
from run._config import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._models.spectrogram_model import SpectrogramModel
from run._streamlit import audio_to_web_path, make_temp_web_dir, st_download_files, st_select_path
from run._tts import griffin_lim_tts
from run.data._loader.structures import Session
from run.deploy.worker import _MARKETPLACE
from run.review.tts.v11_test_cases import V11_TEST_CASES


def generate_test_cases(
    spec_export: SpectrogramModel, test_cases: typing.List[str], seed: int = 123
):
    # with fork_rng(seed):
    spk_sessions: typing.List[Session]
    spk_sessions = [Session(*args) for args in _MARKETPLACE.values()]
    for case in test_cases:
        sesh = random.choice(spk_sessions)
        yield (sesh, case, griffin_lim_tts(spec_export, XMLType(case), sesh))


OPTIONS = {k: partial(generate_test_cases, test_cases=v) for k, v in V11_TEST_CASES.items()}


def initialize_state():
    for key in st.session_state.keys():
        if key.startswith("vote") or key.startswith("note"):  # type: ignore
            st.session_state[key] = ""

    st.session_state.audios = []


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
    st.markdown("Use this workbook to generate batches of audio for evaluating our test cases.")
    run._config.configure(overwrite=True)

    if "audios" not in st.session_state:
        initialize_state()

    form: DeltaGenerator = st.form(key="go")

    with form:
        label = "Spectrogram Checkpoints"
        spec_path = st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
        items = sorted(OPTIONS.items(), reverse=True)
        format_test_case_name = lambda i: i[0].replace("_", " ").title()
        option = st.selectbox("Test Cases", items, format_func=format_test_case_name)
        assert option is not None
        zip_path = f"eval_{option[0]}.zip"
        if st.form_submit_button("⚡️ GO ⚡️"):
            with st.spinner("Generating audio..."):
                initialize_state()
                spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))  # type: ignore
                spec_export = spec_ckpt.export()
                st.session_state.audios.extend(option[1](spec_export))

    if st.session_state.audios != []:
        with st.spinner("Generating survey..."):
            with st.form(key="survey"):
                setup_columns()
                votes, notes, rows, paths = [], [], [], []
                for i, (sesh, script, wave) in enumerate(st.session_state.audios):
                    path = audio_to_web_path(wave, name=(f"audio{i}.wav"))
                    col1, col2, col3, col4, col5 = st.columns(col_widths)
                    with col1:
                        st.markdown("\n")
                        st.markdown(sesh.spkr.label)
                        st.markdown(sesh.spkr.style.value)
                    with col2:
                        st.markdown("\n")
                        st.markdown(script)
                    with col3:
                        st.markdown("\n")
                        st.audio(wave, sample_rate=24000)
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
                        "Speaker": sesh.spkr.label,
                        "Style": sesh.spkr.style.value,
                        "Script": script,
                        "Vote": "",
                        "Note": "",
                        "Session": sesh.label,
                        "Dialect": sesh.spkr.dialect.value[1],
                        "Audio": path.name,
                    }
                    rows.append(row)
                    paths.append(path)

                if rows != []:
                    if st.form_submit_button("⚡️ SAVE CHANGES ⚡️"):
                        for row, vote, note in zip(rows, votes, notes):
                            row["Vote"] = vote
                            row["Note"] = note
                        df = pd.DataFrame(rows)

                        with st.spinner("Making Zipfile..."):
                            st.text("")
                            df_path = make_temp_web_dir() / "evaluations.csv"
                            df_path.write_text(df.to_csv())
                            paths.append(df_path)
                            st_download_files(zip_path, "💾  DOWNLOAD  💾", paths)


if __name__ == "__main__":
    main()
