""" A workbook to generate audio for quick evaluations.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/griffin_lim.py --runner.magicEnabled=false
"""
import typing

import config as cf
import streamlit as st

import lib
import run
from lib.environment import PT_EXTENSION, load
from lib.text import XML_PATTERN, XMLType, natural_keys
from run._config import DEFAULT_SCRIPT, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._config.data import _get_loudness_annotation, _get_tempo_annotation
from run._streamlit import audio_to_web_path, st_html, st_select_path, web_path_to_url
from run._tts import griffin_lim_text_to_speech
from run.data._loader import Session, Speaker


def main():
    st.markdown("# Griffin Lim Audio Generator")
    st.markdown("Use this workbook to generate griffin lim audio for quick evaluation.")
    run._config.configure(overwrite=True)

    label = "Spectrogram Checkpoints"
    spec_path = st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION)
    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))
    spec_export = spec_ckpt.export()

    format_speaker: typing.Callable[[Speaker], str] = lambda s: s.label
    speakers = sorted(set(s[0] for s in spec_export.session_embed.get_vocab()))
    speaker = st.selectbox("Speaker", options=speakers, format_func=format_speaker)  # type: ignore
    speaker = typing.cast(Speaker, speaker)
    assert speaker.name is not None

    spk_sesh = spec_export.session_embed.get_vocab()
    sessions = sorted([s for s in spk_sesh if s[0] == speaker], key=lambda s: natural_keys(s[1]))
    session = st.selectbox("Session", options=sessions, format_func=lambda s: s[1])
    session = typing.cast(Session, session)

    form = st.form(key="form")
    script: str = form.text_area("Script", value=DEFAULT_SCRIPT, height=150)

    if not form.form_submit_button("Submit"):
        return

    with st.spinner("Generating audio..."):
        wave = griffin_lim_text_to_speech(spec_export, XMLType(script), session)
        audio_web_path = audio_to_web_path(wave)
        st_html(f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>')

    # TODO: Add a expected tempo computed based on inner tempo annotations, if we are not marking
    # the entire passage.
    # TODO: Add a loundess computed via spectrogram.
    audio_len = cf.partial(lib.audio.sample_to_sec)(wave.shape[0])
    no_tags_script = XML_PATTERN.sub("", script)
    st.info(f"Generated Tempo: {cf.partial(_get_tempo_annotation)(no_tags_script, audio_len)}")
    st.info(f"Generated Griffin-Lim Loudness: {cf.partial(_get_loudness_annotation)(wave)}")

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
