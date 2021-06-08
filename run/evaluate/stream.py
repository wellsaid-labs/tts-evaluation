""" A workbook to stream clip generation.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/stream.py --runner.magicEnabled=false
"""
import copy
import logging
import multiprocessing
import pathlib
import threading
import time
import typing

import hparams.hparams
import requests
import streamlit as st
from flask import Flask, Response, request

import lib
import run
from lib.audio import get_audio_metadata
from lib.text import natural_keys
from run._streamlit import (
    get_session_state,
    load_en_core_web_md,
    load_tts,
    make_temp_web_dir,
    st_html,
    web_path_to_url,
)
from run._tts import (
    CHECKPOINTS_LOADERS,
    Checkpoints,
    EncodedInput,
    TTSPackage,
    encode_tts_inputs,
    text_to_speech_ffmpeg_generator,
)
from run.data._loader import Speaker

DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)
STREAMING_SERVICE_PORT = 5000
STREAMING_SERVICE_ENDPOINT = f"http://localhost:{STREAMING_SERVICE_PORT}"


st.set_page_config(layout="wide")
lib.environment.set_basic_logging_config(reset=True)
logger = logging.getLogger(__name__)


class _State(typing.TypedDict, total=False):
    service: typing.Union[multiprocessing.Process, threading.Thread]


def _generation_service(
    input: EncodedInput,
    tts: TTSPackage,
    file_path: pathlib.Path,
    stop_event: threading.Event,
):
    """Generate a voice over from `input` using `tts` and store the results in `file_path`."""
    with file_path.open("ab") as file_:
        for bytes_ in text_to_speech_ffmpeg_generator(tts, input):
            if len(bytes_) > 0:
                file_.write(bytes_)
    stop_event.set()


def _stream_file_contents(file_path: pathlib.Path, stop_event: threading.Event):
    """Helper function for streaming bytes from `file_path`."""
    with file_path.open("rb") as file_:
        while True:
            bytes_ = file_.read(1024)
            if not bytes_ and stop_event.is_set():
                break
            elif not bytes_:
                time.sleep(0.1)
            else:
                yield bytes_


def _streaming_service(
    config: typing.Dict, file_path: pathlib.Path, *args, port=STREAMING_SERVICE_PORT
):
    """Run an HTTP service which streams the voice over generation.

    NOTE: This starts a seperate thread for generation, so that, the generator continues to generate
    even if the result isn't being consumed.
    """
    hparams.hparams._configuration = config
    app = Flask(__name__)

    @app.route("/shutdown")
    def shutdown():
        request.environ.get("werkzeug.server.shutdown")()
        return "Server shutting down..."

    @app.route("/stream.mp3")
    def stream():
        request.environ.get("werkzeug.server.shutdown")()
        stop_event = threading.Event()
        _args = (*args, file_path, stop_event)
        thread = threading.Thread(target=_generation_service, args=_args, daemon=True)
        thread.start()
        return Response(_stream_file_contents(file_path, stop_event), mimetype="audio/mpeg")

    app.run(port=port)


def main():
    st.markdown("# Stream Generation ")
    st.markdown("Use this workbook to stream clip generation.")
    run._config.configure()

    state = typing.cast(_State, get_session_state())

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
    session = st.selectbox("Session", options=sessions)
    script = st.text_area("Script", value=DEFAULT_SCRIPT, height=300)
    use_process = st.checkbox("Multiprocessing")
    st.info(f"The script has {len(script):,} character(s).")

    if not st.button("Generate"):
        st.stop()

    with st.spinner("Loading spaCy..."):
        nlp = load_en_core_web_md(disable=("parser", "ner"))

    with st.spinner("Processing inputs..."):
        inputs = encode_tts_inputs(nlp, tts.input_encoder, script, speaker, session)
        st.info(f"{inputs.phonemes.shape[0]:,} token(s) were inputted.")

    if "service" in state and state["service"].is_alive():
        logger.info("Shutting down streaming service...")
        requests.get(f"{STREAMING_SERVICE_ENDPOINT}/shutdown")
        state["service"].join()
        del state["service"]

    web_path = make_temp_web_dir() / "audio.mp3"
    with st.spinner("Starting streaming service..."):
        args = (copy.deepcopy(hparams.hparams.get_config()), web_path, inputs, tts)
        container = multiprocessing.Process if use_process else threading.Thread
        state["service"] = container(target=_streaming_service, args=args, daemon=True)
        state["service"].start()

    # NOTE: `v={time.time()}` is used for cache busting, learn more:
    # https://css-tricks.com/strategies-for-cache-busting-css/
    endpoint = f"{STREAMING_SERVICE_ENDPOINT}/stream.mp3?v={time.time()}"
    st_html(f'<audio controls src="{endpoint}"></audio>')
    url = web_path_to_url(web_path)
    st_html(f'<a href="{url}" download="{web_path.name}">Download Generated Audio</a>')
    st_html(f'<a href="{url}" target="_blank">Link to Generated Audio</a>')
    stats = st.empty()
    start_time = None
    while state["service"].is_alive():
        if web_path.exists() and web_path.stat().st_size > 0:
            if start_time is None:
                start_time = time.time()
            seconds_generated = get_audio_metadata(web_path).length
            elapsed = time.time() - start_time
            stats.info(
                "Stats:\n\n"
                f"- {seconds_generated}s of audio have been generated\n\n"
                f"- {elapsed}s have elapsed since generation started\n\n"
                f"- The real time factor is {elapsed / seconds_generated}x\n\n"
            )
        else:
            stats.info("No audio has been generated, yet...")
        time.sleep(0.05)

    if use_process:
        code = typing.cast(multiprocessing.Process, state["service"]).exitcode
        st.info(f"Process Exit Code: {code}")

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
