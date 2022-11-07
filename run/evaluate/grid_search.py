""" A workbook to find the best permutation of a set of model(s), a speaker, a recording session,
and a script.

TODO:
- Speed up this workbook by supporting batch TTS inference.
- Speed up this workbook by caching checkpoints.
- For consistency, add a method for caching spectrogram outputs, if the inputs don't change. This
  would help ensure that the evaluation is consistent.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/grid_search.py --runner.magicEnabled=false

Use gcloud port-forwarding to interact via your local browser:
```
VM_NAME="name-of-remote-machine"
VM_ZONE="zone-of-remote-machine"
PROJECT_ID=voice-research-255602
LOCAL_PORT=2222
REMOTE_PORT=8501

gcloud compute ssh $VM_NAME \
    --project $PROJECT_ID \
    --zone $VM_ZONE \
    -- -NL $LOCAL_PORT":localhost:"$REMOTE_PORT
```
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
from lib.environment import PT_EXTENSION, ROOT_PATH, load
from lib.text import natural_keys
from run import train
from run._config import (
    DEFAULT_SCRIPT,
    SIGNAL_MODEL_EXPERIMENTS_PATH,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
)
from run._streamlit import audio_to_web_path, paths_to_html_download_link, st_html, web_path_to_url
from run._tts import TTSPackage, get_session_vocab, text_to_speech
from run.data._loader import Session, Speaker

st.set_page_config(layout="wide")


def path_label(path: pathlib.Path) -> str:
    """Get a short label for `path`."""
    return str(path.relative_to(ROOT_PATH)) + "/" if path.is_dir() else str(path.name)


def st_select_paths(label: str, dir: pathlib.Path, suffix: str) -> typing.List[pathlib.Path]:
    """Display a path selector for the directory `dir`."""
    options = sorted(
        [p for p in dir.glob("**/*") if p.suffix == suffix or p.is_dir()] + [dir],
        key=lambda x: natural_keys(str(x)),
        reverse=True,
    )
    paths = st.multiselect(label, options=options, format_func=path_label)  # type: ignore
    paths = cast(typing.List[pathlib.Path], paths)
    paths = [f for p in paths for f in ([p] if p.is_file() else list(p.glob(f"**/*{suffix}")))]
    if len(paths) > 0:
        st.info(f"Selected {label}:\n" + "".join(["\n - " + path_label(p) for p in paths]))
    return paths


def get_sample_sessions(
    speakers: typing.List[Speaker], sessions: typing.Set[Session], max_sessions: int
) -> typing.List[Session]:
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
    run._config.configure(overwrite=True)

    get_paths = functools.partial(st_select_paths, suffix=PT_EXTENSION)
    sig_paths = get_paths("Signal Checkpoints(s)", SIGNAL_MODEL_EXPERIMENTS_PATH)
    spec_paths = get_paths("Spectrogram Checkpoints(s)", SPECTROGRAM_MODEL_EXPERIMENTS_PATH)

    form = st.form("data_form")
    speakers = sorted(list(run._config.DATASETS.keys()))
    is_all = st.sidebar.checkbox("Select all speakers by default")
    format_: typing.Callable[[Speaker], str] = lambda s: f"{s.label} ({s.style.value})"
    default = speakers if is_all else speakers[:1]
    label = "Speaker(s)"
    speakers = form.multiselect(label, options=speakers, format_func=format_, default=default)
    speakers = cast(typing.List[Speaker], speakers)
    max_sessions = form.number_input("Maximum Recording Sessions", min_value=1, value=1, step=1)
    max_sessions = cast(int, max_sessions)
    scripts = form.text_area("Script(s)", value=DEFAULT_SCRIPT)
    scripts = [s.strip() for s in scripts.split("\n") if len(s.strip()) > 0]
    file_name = form.text_input(label="Zipfile Name", value="audio(s)")
    if not form.form_submit_button(label="Generate"):
        return

    with st.spinner("Loading checkpoints..."):
        spec_ckpts = [cast(train.spectrogram_model._worker.Checkpoint, load(p)) for p in spec_paths]
        spec_export = [(c.export(), p) for c, p in zip(spec_ckpts, spec_paths)]
        sig_ckpts = [cast(train.signal_model._worker.Checkpoint, load(p)) for p in sig_paths]
        sig_export = [(c.export(), p) for c, p in zip(sig_ckpts, sig_paths)]

    rows = []
    paths = []
    bar = st.progress(0)
    sessions = get_session_vocab(*tuple(c.model for c in spec_ckpts + sig_ckpts))
    sessions_sample = get_sample_sessions(speakers, sessions, max_sessions)
    iter_ = list(itertools.product(sessions_sample, spec_export, sig_export, scripts))
    for i, (
        session,
        (spec_model, spec_path),
        (sig_model, sig_path),
        script,
    ) in enumerate(iter_):
        package = TTSPackage(spec_model, sig_model)
        audio = text_to_speech(package, script, session)
        sesh = str(session).replace("/", "__")
        name = f"i={i},spec={spec_path.stem},sig={sig_path.stem},spk={session[0].label},"
        name += f"sesh={sesh},script={id(script)}.wav"
        audio_web_path = audio_to_web_path(audio, name=name)
        row = {
            "Audio": f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>',
            "Spectrogam Model": path_label(spec_path),
            "Signal Model": path_label(sig_path),
            "Speaker": session[0].label,
            "Session": session[1],
            "Script": f"'{script[:25]}...'",
        }
        rows.append(row)
        paths.append(audio_web_path)
        bar.progress(len(rows) / len(iter_))
    bar.empty()
    st_datatable(pd.DataFrame(rows))

    with st.spinner("Making Zipfile..."):
        st.text("")
        label = "📁 Download Audio(s) (zip)"
        st_html(paths_to_html_download_link(f"{file_name}.zip", label, paths))


if __name__ == "__main__":
    main()
