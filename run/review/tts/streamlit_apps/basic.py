""" A workbook to generate audio for quick evaluations.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/streamlit_apps/basic.py --runner.magicEnabled=false
"""
import typing

import streamlit as st

import lib
import run
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType, natural_keys
from run._config import DEFAULT_SCRIPT, SIG_MODEL_EXP_PATH, SPEC_MODEL_EXP_PATH
from run._models import signal_model, spectrogram_model
from run._streamlit import (
    audio_to_web_path,
    load_tts,
    st_download_files,
    st_html,
    st_select_path,
    st_set_page_config,
    web_path_to_url,
)
from run._tts import CHECKPOINTS_LOADERS, TTSPackage, batch_tts, make_batches
from run.data._loader import Session

st_set_page_config()


def main():
    st.markdown("# Simple Audio Generator")
    st.markdown("Use this workbook to generate audio for quick evaluation.")
    run._config.configure(overwrite=True)

    options = [None] + [k.name for k in CHECKPOINTS_LOADERS.keys()]
    checkpoint = st.selectbox("(Optional) Combined Checkpoints", options=options)
    label = "(Optional) Spectrogram Checkpoint"
    spec_path = st_select_path(label, SPEC_MODEL_EXP_PATH, PT_EXTENSION)
    label = "(Optional) Signal Checkpoint"
    sig_path = st_select_path(label, SIG_MODEL_EXP_PATH, PT_EXTENSION)

    spec_model, sig_model = None, None
    if checkpoint is not None:
        with st.spinner("Loading checkpoint(s)..."):
            tts = load_tts(checkpoint)
            spec_model, sig_model = tts.spec_model, tts.signal_model
    if spec_path is not None:
        spec_model = typing.cast(spectrogram_model.SpectrogramModel, load(spec_path).export())
    if sig_path is not None:
        sig_model = typing.cast(signal_model.SignalModel, load(sig_path).export())
    if spec_model is None or sig_model is None:
        st.error("Both a Spectrogram and Signal model need to be specified.")
        return
    tts = TTSPackage(spec_model, sig_model)

    form = st.form(key="form")

    seshs = tts.session_vocab()
    seshs = sorted([s for s in seshs], key=lambda s: natural_keys(s.spkr.label + s.label))
    format_sesh: typing.Callable[[Session], str] = lambda s: f"{s.spkr.label}/{s.label}"
    sesh = form.selectbox("Session", options=seshs, format_func=format_sesh)
    sesh = typing.cast(Session, sesh)
    assert sesh.spkr.name is not None
    speaker_name = sesh.spkr.name.split()[0].lower()

    script: str = XMLType(form.text_area("Script", value=DEFAULT_SCRIPT, height=150))

    num_clips = form.number_input("Number of Clips", min_value=1, max_value=None, value=1)
    num_clips = typing.cast(int, num_clips)

    if not form.form_submit_button("Submit"):
        return

    paths = []
    with st.spinner("Generating audio..."):
        inputs = [(script, sesh) for _ in range(num_clips)]
        batches = make_batches(inputs)
        for i, generated in enumerate(batch_tts(tts, batches)):
            clip_num = i % num_clips + 1
            sesh = inputs[i][-1][1]
            sesh = sesh[:-4] if (sesh.endswith(".wav") or sesh.endswith(".mp3")) else sesh
            if clip_num == 1:
                st.markdown(f"##### Session: **{sesh}**")
            st.markdown(f"###### Clip: **{clip_num}**")
            name = f"spkr={speaker_name},sesh={sesh},clp={clip_num}.wav"
            audio_web_path = audio_to_web_path(generated.sig_model[0], name)
            st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')
            paths.append(audio_web_path)

    with st.spinner("Making zipfile..."):
        st.text("")
        zip_name = f"{speaker_name}_samples.zip"
        st_download_files(zip_name, f"📁 Download All {len(paths)} (zip)", paths)

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
