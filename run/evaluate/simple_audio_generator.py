""" A workbook to generate audio for quick evaluations.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/simple_audio_generator.py --runner.magicEnabled=false
"""
import typing

import streamlit as st

import lib
import run
from lib.text import natural_keys
from run._streamlit import (
    audio_to_web_path,
    load_tts,
    paths_to_html_download_link,
    st_html,
    web_path_to_url,
)
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, batch_text_to_speech
from run.data._loader import Session, Speaker

DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)


def main():
    st.markdown("# Simple Audio Generator")
    st.markdown("Use this workbook to generate audio for quick evaluation.")
    run._config.configure()

    options = list(CHECKPOINTS_LOADERS.keys())
    format_: typing.Callable[[Checkpoints], str] = lambda s: s.value
    checkpoints_key: Checkpoints = st.selectbox("Checkpoints", options=options, format_func=format_)

    with st.spinner("Loading checkpoint(s)..."):
        tts = load_tts(checkpoints_key)

    frm = st.form(key="form")
    format_speaker: typing.Callable[[Speaker], str] = lambda s: s.label
    speakers = sorted(tts.input_encoder.speaker_encoder.index_to_token)
    speaker: Speaker = frm.selectbox("Speaker", options=speakers, format_func=format_speaker)
    assert speaker.name is not None
    speaker_name = speaker.name.split()[0].lower()

    spk_sesh: typing.List[typing.Tuple[Speaker, Session]]
    spk_sesh = tts.input_encoder.session_encoder.index_to_token
    sessions = sorted([sesh for spk, sesh in spk_sesh if spk == speaker], key=natural_keys)

    all_sessions: bool = frm.checkbox("Sample all %d sessions" % len(sessions))
    session: Session = frm.selectbox("Session", options=["All"] if all_sessions else sessions)
    script: str = frm.text_area("Script", value=DEFAULT_SCRIPT, height=300)

    label = "Number of Clips Per Session"
    num_clips: int = frm.number_input(label, min_value=1, max_value=None, value=5)

    if not frm.form_submit_button("Submit"):
        return

    st.info(f"The script has {len(script):,} character(s).")

    paths = []
    with st.spinner("Generating Audio..."):
        inputs = [(script, speaker, s) for s in (sessions if all_sessions else [session])]
        inputs = [i for i in inputs for _ in range(num_clips)]
        for i, generated in enumerate(batch_text_to_speech(tts, inputs)):
            clip_num = i % num_clips + 1
            st.markdown(f"Session: **{inputs[i][-1]}**\t Clip: **{clip_num}**")
            name = f"{speaker_name}_session{i // num_clips}_{clip_num}.wav"
            audio_web_path = audio_to_web_path(generated.sig_model, name)
            st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')
            paths.append(audio_web_path)

    with st.spinner("Making Zipfile..."):
        st.text("")
        zip_name = f"{speaker_name}_samples.zip"
        st_html(paths_to_html_download_link(zip_name, "Download All (zip)", paths))

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
