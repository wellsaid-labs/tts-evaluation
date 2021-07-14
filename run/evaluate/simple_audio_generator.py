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
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, text_to_speech
from run.data._loader import Speaker

DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)


def _get_speaker_name(speaker):
    return typing.cast(Speaker, speaker).name.split()[0]


def _generate_audio(tts, script, speaker, session, num_clips, session_count):
    paths = []
    speaker_name = _get_speaker_name(speaker)

    for clip_count, audio in enumerate(text_to_speech(tts, script, speaker, session, num_clips)):
        audio_web_path = audio_to_web_path(
            audio, name="%s_session%d_%d.wav" % (speaker_name, session_count, clip_count + 1)
        )
        st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')
        url = web_path_to_url(audio_web_path)
        st_html(f'<a href="{url}" download="{audio_web_path.name}">Download Generated Audio</a>')
        st.text("")
        paths.append(audio_web_path)

    return paths


def main():
    st.markdown("# Simple Audio Generator ")
    st.markdown("Use this workbook to generate audio for quick evaluation.")
    run._config.configure()

    options = list(CHECKPOINTS_LOADERS.keys())
    format_: typing.Callable[[Checkpoints], str] = lambda s: s.value
    checkpoints_key: Checkpoints = st.selectbox("Checkpoints", options=options, format_func=format_)

    with st.spinner("Loading checkpoint(s)..."):
        tts = load_tts(checkpoints_key)

    format_speaker: typing.Callable[[Speaker], str] = lambda s: s.label
    speakers = sorted(tts.input_encoder.speaker_encoder.index_to_token)
    speaker = st.selectbox("Speaker", options=speakers, format_func=format_speaker)

    sessions = tts.input_encoder.session_encoder.index_to_token
    sessions = sorted([sesh for spk, sesh in sessions if spk == speaker], key=natural_keys)

    all_sessions = st.checkbox("Sample all %d sessions" % len(sessions))
    session = st.selectbox("Session", options=["All"] if all_sessions else sessions)
    script = st.text_area("Script", value=DEFAULT_SCRIPT, height=300)
    st.info(f"The script has {len(script):,} character(s).")

    num_clips = st.number_input(
        "Number of Clips",
        min_value=1,
        max_value=1 if all_sessions else None,
        value=1 if all_sessions else 5,
    )

    if not st.button("Generate"):
        st.stop()

    paths = []
    with st.spinner("Generating Audio..."):
        if all_sessions:
            for s, sesh in enumerate(sessions):
                paths.extend(_generate_audio(tts, script, speaker, sesh, num_clips, s + 1))
        else:
            paths.extend(
                _generate_audio(
                    tts, script, speaker, session, num_clips, sessions.index(session) + 1
                )
            )

    with st.spinner("Making Zipfile..."):
        st.text("")
        zip_name = "%s_samples.zip" % _get_speaker_name(speaker)
        st_html(paths_to_html_download_link(zip_name, "Download All (zip)", paths))

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
