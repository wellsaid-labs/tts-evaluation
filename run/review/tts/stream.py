""" A workbook to stream clip generation.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/stream.py --runner.magicEnabled=false
"""
import copy
import logging
import multiprocessing
import multiprocessing.synchronize
import pathlib
import threading
import time
import typing

import config as cf
import requests
import streamlit as st
from flask import Flask, request
from flask.wrappers import Response

import lib
import run
from lib.audio import get_audio_metadata
from lib.text import XMLType, natural_keys
from run._config import DEFAULT_SCRIPT
from run._streamlit import (
    WebPath,
    load_en_core_web_md,
    load_tts,
    make_temp_web_dir,
    st_html,
    web_path_to_url,
)
from run._tts import (
    CHECKPOINTS_LOADERS,
    Inputs,
    PreprocessedInputs,
    TTSPackage,
    process_tts_inputs,
    tts_ffmpeg_generator,
)
from run.data._loader import Session, Speaker

STREAMING_SERVICE_PORT = 5000
STREAMING_SERVICE_ENDPOINT = f"http://localhost:{STREAMING_SERVICE_PORT}"


st.set_page_config(layout="wide")
lib.environment.set_basic_logging_config(reset=True)
logger = logging.getLogger(__name__)
Event = typing.Union[multiprocessing.synchronize.Event, threading.Event]
Container = typing.Union[multiprocessing.Process, threading.Thread]


class _State(typing.TypedDict, total=False):
    service: Container


def _generation_service(
    input: Inputs,
    preprocessed: PreprocessedInputs,
    tts: TTSPackage,
    file_path: pathlib.Path,
    is_streaming: Event,
):
    """Generate a voice over from `input` using `tts` and store the results in `file_path`."""
    with file_path.open("ab") as file_:
        for bytes_ in tts_ffmpeg_generator(tts, input, preprocessed, **cf.get()):
            if len(bytes_) > 0:
                file_.write(bytes_)
    is_streaming.clear()


def _stream_file_contents(file_path: pathlib.Path, is_streaming: Event):
    """Helper function for streaming bytes from `file_path`."""
    with file_path.open("rb") as file_:
        while True:
            bytes_ = file_.read(1024)
            if not bytes_ and not is_streaming.is_set():
                break
            elif not bytes_:
                time.sleep(0.1)
            else:
                yield bytes_


def _streaming_service(
    configuration: cf.Config,
    file_path: pathlib.Path,
    is_streaming: Event,
    *args,
    port=STREAMING_SERVICE_PORT,
):
    """Run an HTTP service which streams the voice over generation.

    NOTE: This starts a seperate thread for generation, so that, the generator continues to generate
    even if the result isn't being consumed.
    """
    cf.add(configuration)
    app = Flask(__name__)

    @app.route("/healthy", methods=["GET"])
    def healthy():
        return "ok"

    @app.route("/shutdown")
    def shutdown():
        shutdown = request.environ.get("werkzeug.server.shutdown")
        assert shutdown is not None
        shutdown()
        return "Server shutting down..."

    @app.route("/stream.mp3")
    def stream():
        is_streaming.set()
        _args = (*args, file_path, is_streaming)
        thread = threading.Thread(target=_generation_service, args=_args, daemon=True)
        thread.start()
        return Response(_stream_file_contents(file_path, is_streaming), mimetype="audio/mpeg")

    app.run(port=port)


def wait_until_healthy():
    """Block until the streaming service is healthy."""
    while True:
        try:
            request = requests.get(f"{STREAMING_SERVICE_ENDPOINT}/healthy")
            if request.status_code == 200:
                break
        except BaseException:
            time.sleep(0.05)


def wait_until_first_byte(service: Container, web_path: WebPath):
    """Block until the first byte is written to `web_path` by `service`."""
    with st.spinner("Waiting for first audio byte..."):
        while service.is_alive() and (not web_path.exists() or web_path.stat().st_size == 0):
            time.sleep(0.05)


def main():
    st.markdown("# Stream Generation")
    st.markdown("Use this workbook to stream clip generation.")
    run._config.configure(overwrite=True)

    options = [k.name for k in CHECKPOINTS_LOADERS.keys()]
    checkpoint = typing.cast(str, st.selectbox("Checkpoints", options=options))

    with st.spinner(f"Loading `{checkpoint}` checkpoint(s)..."):
        tts = load_tts(checkpoint)

    format_speaker: typing.Callable[[Speaker], str] = lambda s: str(s)
    speakers = sorted((s.spkr for s in tts.session_vocab()), key=lambda k: str(k))
    speaker = st.selectbox("Speaker", options=speakers, format_func=format_speaker)  # type: ignore

    sessions = tts.session_vocab()
    format_session: typing.Callable[[Session], str] = lambda s: s.label
    sesh_sort_key: typing.Callable[[Session], typing.List] = lambda s: natural_keys(s.label)
    sessions = sorted((s for s in sessions if s.spkr == speaker), key=sesh_sort_key)
    session = st.selectbox("Session", options=sessions, format_func=format_session)  # type: ignore
    session = typing.cast(Session, session)
    script = XMLType(st.text_area("Script", value=DEFAULT_SCRIPT, height=300))
    use_process = st.checkbox("Multiprocessing")
    st.info(f"The script has {len(script):,} character(s).")

    # TODO: Use `streamlit`s official form submit button, instead. Learn more:
    # https://blog.streamlit.io/introducing-submit-button-and-forms/
    if not st.button("Generate"):
        st.stop()

    with st.spinner("Loading spaCy..."):
        nlp = load_en_core_web_md(disable=("parser", "ner"))

    with st.spinner("Processing inputs..."):
        inputs = process_tts_inputs(tts, nlp, script, session)
        st.info(f"{len(inputs[1].tokens[0]):,} token(s) were inputted.")

    if "service" in st.session_state and st.session_state["service"].is_alive():
        logger.info("Shutting down streaming service...")
        requests.get(f"{STREAMING_SERVICE_ENDPOINT}/shutdown")
        st.session_state["service"].join()
        del st.session_state["service"]

    web_path = make_temp_web_dir() / "audio.mp3"
    with st.spinner("Starting streaming service..."):
        is_streaming = (multiprocessing.Event if use_process else threading.Event)()
        args = (copy.deepcopy(cf.export()), web_path, is_streaming, *inputs, tts)
        container = multiprocessing.Process if use_process else threading.Thread
        st.session_state["service"] = container(target=_streaming_service, args=args, daemon=True)
        st.session_state["service"].start()
        wait_until_healthy()

    # NOTE: `v={time.time()}` is used for cache busting, learn more:
    # https://css-tricks.com/strategies-for-cache-busting-css/
    endpoint = f"{STREAMING_SERVICE_ENDPOINT}/stream.mp3?v={time.time()}"
    st_html(f'<audio controls src="{endpoint}"></audio>')
    start_time = time.time()
    wait_until_first_byte(st.session_state["service"], web_path)
    assert is_streaming.is_set()

    url = web_path_to_url(web_path)
    st_html(f'<a href="{url}" download="{web_path.name}">Download Generated Audio</a>')
    st_html(f'<a href="{url}" target="_blank">Link to Generated Audio</a>')
    ttfb = time.time() - start_time
    start_time = time.time()
    stats = st.empty()
    while st.session_state["service"].is_alive() and web_path.exists() and is_streaming.is_set():
        seconds_generated = get_audio_metadata(web_path).length
        elapsed = time.time() - start_time

        stats.info(
            "Stats:\n\n"
            f"- {ttfb}s till first audio byte was generated\n\n"
            f"- {lib.utils.seconds_to_str(seconds_generated)} of audio have been generated\n\n"
            f"- {lib.utils.seconds_to_str(elapsed)} have elapsed since generation started\n\n"
            f"- The real time factor is {elapsed / seconds_generated}x\n\n"
        )
        time.sleep(0.05)

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
