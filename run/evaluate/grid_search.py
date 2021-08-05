""" A workbook to find the best permutation of a set of model(s), a speaker, a recording session,
and a script.

TODO:
- Speed up this workbook by supporting batch TTS inference.
- Speed up this workbook by caching checkpoints.
- For consistency, add a method for caching spectrogram outputs, if the inputs don't change. This
  would help ensure that the evaluation is consistent.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/grid_search.py --runner.magicEnabled=false
"""
import functools
import itertools
import pathlib
import typing
from typing import cast

import pandas as pd
import streamlit as st
from streamlit_datatable import st_datatable

import lib
import run
from lib.environment import PT_EXTENSION, load
from run import train
from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._streamlit import audio_to_web_path, paths_to_html_download_link, st_html, web_path_to_url
from run._tts import TTSPackage, text_to_speech

st.set_page_config(layout="wide")


DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)


def path_label(path: pathlib.Path) -> str:
    """Get a short label for `path`."""
    return f"{path.parent.name}/{path.name}"


def st_select_paths(label: str, dir: pathlib.Path, suffix: str) -> typing.List[pathlib.Path]:
    """Display a path selector for the directory `dir`."""
    options = [p for p in dir.iterdir() if p.suffix == suffix or p.is_dir()]
    format_: typing.Callable[[pathlib.Path], str] = lambda p: f"{p.name}/" if p.is_dir() else p.name
    path = st.selectbox(label, options=options, format_func=format_)
    if path.is_file():
        assert path.suffix == suffix
        paths = [path]
    else:
        paths = list(path.glob(f"**/*{suffix}"))
    st.info(f"Selected {label}:\n" + "".join(["\n - " + path_label(p) for p in paths]))
    return paths


def get_sample_sessions(
    speakers: typing.List[run.data._loader.Speaker],
    sessions: typing.List[typing.Tuple[run.data._loader.Speaker, run.data._loader.Session]],
    max_sessions: int,
) -> typing.List[typing.Tuple[run.data._loader.Speaker, run.data._loader.Session]]:
    """For each `speaker` randomly sample `max_sessions`."""
    sessions_sample = []
    for speaker in speakers:
        speaker_sessions = [s for s in sessions if s[0] == speaker]
        sessions_sample.extend(lib.utils.random_sample(speaker_sessions, max_sessions))
    return sessions_sample


def main():
    st.markdown("# Grid Search Evaluation")
    st.markdown(
        "Use this workbook to find the best permutation of a "
        "set of model(s), a speaker, a recording session, and a script."
    )
    run._config.configure()

    get_paths = functools.partial(st_select_paths, suffix=PT_EXTENSION)
    sig_paths = get_paths("Signal Checkpoints(s)", SIGNAL_MODEL_EXPERIMENTS_PATH)
    spec_paths = get_paths("Spectrogram Checkpoints(s)", SPECTROGRAM_MODEL_EXPERIMENTS_PATH)
    speakers = list(run._config.DATASETS.keys())
    is_all = st.sidebar.checkbox("Select all speakers by default")
    format_: typing.Callable[[run.data._loader.Speaker], str] = lambda s: s.label
    default = speakers if is_all else speakers[:1]
    speakers = st.multiselect("Speaker(s)", options=speakers, format_func=format_, default=default)
    max_sessions = st.number_input("Maximum Recording Sessions", min_value=1, value=1, step=1)
    scripts = st.text_area("Script(s)", value=DEFAULT_SCRIPT)
    scripts = [s.strip() for s in scripts.split("\n") if len(s.strip()) > 0]

    if not st.button("Generate"):
        st.stop()

    with st.spinner("Loading checkpoints..."):
        spec_ckpts = [cast(train.spectrogram_model._worker.Checkpoint, load(p)) for p in spec_paths]
        spec_export = [(c.export(), p) for c, p in zip(spec_ckpts, spec_paths)]
        sig_ckpts = [cast(train.signal_model._worker.Checkpoint, load(p)) for p in sig_paths]
        sig_export = [(c.export(), p) for c, p in zip(sig_ckpts, sig_paths)]

    rows = []
    paths = []
    bar = st.progress(0)
    sessions: typing.List[typing.Tuple[run.data._loader.Speaker, run.data._loader.Session]]
    sessions = list(set(s for c in spec_ckpts for s in c.input_encoder.session_encoder.vocab))
    sessions_sample = get_sample_sessions(speakers, sessions, max_sessions)
    iter_ = list(itertools.product(spec_export, sig_export, sessions_sample, scripts))
    for i, (spec_items, (sig_model, sig_path), (speaker, session), script) in enumerate(iter_):
        (input_encoder, spec_model), spec_path = spec_items
        package = TTSPackage(input_encoder, spec_model, sig_model)
        audio = text_to_speech(package, script, speaker, session)
        sesh = str(session).replace("/", "__")
        name = f"i={i},spec={spec_path.stem},sig={sig_path.stem},spk={speaker.label},"
        name += f"sesh={sesh},script={id(script)}.wav"
        audio_web_path = audio_to_web_path(audio, name=name)
        row = {
            "Audio": f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>',
            "Spectrogam Model": path_label(spec_path),
            "Signal Model": path_label(sig_path),
            "Speaker": speaker.label,
            "Session": session,
            "Script": f"'{script[:25]}...'",
        }
        rows.append(row)
        paths.append(audio_web_path)
        bar.progress(len(rows) / len(iter_))
    bar.empty()
    st_datatable(pd.DataFrame(rows))

    with st.spinner("Making Zipfile..."):
        st.text("")
        app_name = __file__.rstrip(".py")
        st_html(paths_to_html_download_link(f"{app_name}_audios.zip", "Download Audio(s)", paths))


if __name__ == "__main__":
    main()