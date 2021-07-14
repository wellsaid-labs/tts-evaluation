""" A workbook to stream clip generation.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/simple_audio_generator.py --runner.magicEnabled=false
"""
import logging
import multiprocessing
import threading
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

st.set_page_config(layout="wide")
lib.environment.set_basic_logging_config(reset=True)
logger = logging.getLogger(__name__)
Event = typing.Union[multiprocessing.synchronize.Event, threading.Event]


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

    all_sessions = st.checkbox("Sample all sessions")
    sessions = tts.input_encoder.session_encoder.index_to_token
    sessions = sorted([sesh for spk, sesh in sessions if spk == speaker], key=natural_keys)
    session = st.selectbox("Session", options=["All"] if all_sessions else sessions)
    script = st.text_area("Script", value=DEFAULT_SCRIPT, height=300)
    st.info(f"The script has {len(script):,} character(s).")

    num_clips = st.slider(
        "Number of Clips",
        min_value=0,
        max_value=1 if all_sessions else 10,
        value=1 if all_sessions else 5,
    )

    if not st.button("Generate"):
        st.stop()

    paths = []
    speaker_name = typing.cast(Speaker, speaker).name.split()[0]
    with st.spinner("Generating Audio..."):
        if all_sessions:
            for s, sesh in enumerate(sessions):
                for i, audio in enumerate(text_to_speech(tts, script, speaker, sesh, num_clips)):
                    audio_web_path = audio_to_web_path(
                        audio, name="%s_session%d.wav" % (speaker_name, s + 1)
                    )
                    st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')
                    url = web_path_to_url(audio_web_path)
                    st_html(
                        f'<a href="{url}" download="{audio_web_path.name}">Download Generated Audio</a>'
                    )
                    st.text("")
                    paths.append(audio_web_path)
        else:
            for i, audio in enumerate(text_to_speech(tts, script, speaker, session, num_clips)):
                audio_web_path = audio_to_web_path(audio, name="%s_%d.wav" % (speaker_name, i))
                st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')
                url = web_path_to_url(audio_web_path)
                st_html(
                    f'<a href="{url}" download="{audio_web_path.name}">Download Generated Audio</a>'
                )
                st.text("")
                paths.append(audio_web_path)

    with st.spinner("Making Zipfile..."):
        st.text("")
        zip_name = "%s_samples.zip" % speaker_name
        st_html(paths_to_html_download_link(zip_name, "Download All (zip)", paths))

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
