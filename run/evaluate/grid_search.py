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

import lib
import run
from lib.environment import PT_EXTENSION, load
from run import train
from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._streamlit import (
    audio_to_web_path,
    paths_to_html_download_link,
    st_data_frame,
    st_html,
    web_path_to_url,
)
from run._tts import TTSPackage, text_to_speech

st.set_page_config(layout="wide")


DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)

ST_FORMAT_FILE: typing.Callable[[pathlib.Path], str] = (
    lambda p: f"{p.name}/" if p.is_dir() else p.name
)


def path_label(path: pathlib.Path) -> str:
    """Get a short label for `path`."""
    return f"{path.parent.name}/{path.name}"


def st_select_paths(label: str, dir: pathlib.Path) -> pathlib.Path:
    """Display a subdirectory selector for the directory `dir`. Then display a checkpoint selector
    for the selected subdirectory `subdir`. Return list of checkpoint paths."""
    options = [p for p in dir.iterdir()]
    options.sort(reverse=True)
    path = typing.cast(
        pathlib.Path,
        st.selectbox("%s Training Path" % label, options=options, format_func=ST_FORMAT_FILE),
    )
    subdirs = [p for p in path.iterdir()]
    subdirs.sort(reverse=True)
    assert subdirs, "Subdirs empty:\t%s" % path
    subdir = typing.cast(
        pathlib.Path,
        st.selectbox("%s Checkpoints Path" % label, options=subdirs, format_func=ST_FORMAT_FILE)
        if subdirs
        else ["None Available"],
    )

    return subdir


def st_select_checkpoints(
    label: str, subdir: pathlib.Path, num_ckpts=3
) -> typing.List[pathlib.Path]:
    ckpt_options = list(subdir.glob(f"**/*{PT_EXTENSION}"))
    ckpt_options.sort(reverse=True)
    ckpts_selected = typing.cast(
        typing.List[pathlib.Path],
        st.multiselect(
            label="%s Checkpoint(s)" % label,
            options=ckpt_options,
            default=ckpt_options[:num_ckpts],
            format_func=ST_FORMAT_FILE,
        ),
    )
    ckpts_selected.sort()
    st.info(
        "Selected %s Checkpoint(s):\n" % label
        + "".join(["\n - " + path_label(p) for p in ckpts_selected])
    )
    return ckpts_selected


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
    get_paths = functools.partial(st_select_paths)
    sig_ckpt_paths = get_paths("Signal", SIGNAL_MODEL_EXPERIMENTS_PATH)
    sig_ckpts_selected = st_select_checkpoints("Signal", sig_ckpt_paths, 1)
    spec_ckpt_paths = get_paths("Spectrogram", SPECTROGRAM_MODEL_EXPERIMENTS_PATH)

    with st.form(key="data_form"):
        spec_ckpts_selected = st_select_checkpoints("Spectrogram", spec_ckpt_paths, 1)

        speakers = list(run._config.DATASETS.keys())
        speakers.sort()
        is_all = st.sidebar.checkbox("Select all speakers by default")
        format_: typing.Callable[[run.data._loader.Speaker], str] = lambda s: s.label
        default = speakers if is_all else speakers[:1]
        speakers = typing.cast(
            typing.List[run.data._loader.Speaker],
            st.multiselect("Speaker(s)", options=speakers, format_func=format_, default=default),
        )
        max_sessions = typing.cast(
            int, st.number_input("Maximum Recording Sessions", min_value=1, value=1, step=1)
        )
        scripts = st.text_area("Script(s)", value=DEFAULT_SCRIPT)
        scripts = [s.strip() for s in scripts.split("\n") if len(s.strip()) > 0]
        file_name = st.text_input(label="Zipfile Name", value="audio(s)")
        submit = st.form_submit_button(label="Generate")

    if submit:
        with st.spinner("Loading checkpoints..."):
            spec_ckpts = [
                cast(train.spectrogram_model._worker.Checkpoint, load(p))
                for p in spec_ckpts_selected
            ]
            spec_export = [
                (c.export(), c.comet_experiment_key, c.step, p)
                for c, p in zip(spec_ckpts, spec_ckpts_selected)
            ]
            sig_ckpts = [
                cast(train.signal_model._worker.Checkpoint, load(p)) for p in sig_ckpts_selected
            ]
            sig_export = [
                (c.export(), c.comet_experiment_key, c.step, p)
                for c, p in zip(sig_ckpts, sig_ckpts_selected)
            ]

        rows = []
        paths = []
        bar = st.progress(0)
        sessions: typing.List[typing.Tuple[run.data._loader.Speaker, run.data._loader.Session]]
        sessions = list(set(s for c in spec_ckpts for s in c.input_encoder.session_encoder.vocab))
        sessions_sample = get_sample_sessions(speakers, sessions, max_sessions)
        iter_ = list(itertools.product(sessions_sample, spec_export, sig_export, scripts))
        for i, (
            (speaker, session),
            spec_items,
            sig_items,
            script,
        ) in enumerate(iter_):
            (input_encoder, spec_model), spec_comet_key, spec_step, spec_path = spec_items
            (sig_model, sig_comet_key, sig_step, sig_path) = sig_items
            package = TTSPackage(
                input_encoder,
                spec_model,
                sig_model,
                spec_comet_key,
                spec_step,
                sig_comet_key,
                sig_step,
            )
            audio = text_to_speech(package, script, speaker, session)
            sesh = str(session).replace("/", "__")
            name = f"spec={spec_path.stem},sig={sig_path.stem},spk={speaker.label},"
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
        st_data_frame(pd.DataFrame(rows))

        with st.spinner("Making Zipfile..."):
            st.text("")
            st_html(paths_to_html_download_link("%s.zip" % file_name, "Download Audio(s)", paths))


if __name__ == "__main__":
    main()
