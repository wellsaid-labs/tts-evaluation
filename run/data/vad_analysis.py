""" Streamlit application for analyzing an audio file with voice activity detection (VAD).

TODO:
- Load a `Passage` and compared VAD to alignments
- Answer: Do we need to use a sophisiticated VAD, or can we use a basic loudness threshold?
- Compare `webrtcvad` to a more modern deep learning based VAD, like:
https://github.com/snakers4/silero-vad
- Refactor `_chart_signal` with the similar function `_visualize_signal`.

Usage:
    $ PYTHONPATH=. streamlit run run/data/vad_analysis.py --runner.magicEnabled=false
"""
import itertools
import logging
import pathlib
import typing

import altair as alt
import librosa.util
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import pandas as pd
import streamlit as st
import webrtcvad

import lib
import run

lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


def _chart_signal(
    signal: np.ndarray,
    sample_rate: int,
    max_sample_rate: int = 2000,
    x: str = "seconds",
    y: typing.Tuple[str, str] = ("y_min", "y_max"),
) -> alt.Chart:
    is_floating = np.issubdtype(signal.dtype, np.floating)
    signal = signal / (1.0 if is_floating else np.abs(np.iinfo(signal.dtype).min))
    ratio = sample_rate // max_sample_rate
    frames = librosa.util.frame(signal, ratio, ratio, axis=0)  # type: ignore
    envelope = np.max(np.abs(frames), axis=-1)
    ticks = np.arange(0, envelope.shape[0] * ratio / sample_rate, ratio / sample_rate)
    scale = alt.Scale(domain=(-1.0, 1.0))
    return (
        alt.Chart(pd.DataFrame({x: ticks, y[0]: -envelope, y[1]: envelope}))
        .mark_area()
        .encode(
            x=alt.X(x, type="quantitative"),
            y=alt.Y(y[0], scale=scale, type="quantitative"),
            y2=alt.Y2(y[1]),
        )
    )


def main():
    run._config.configure()

    st.title("VAD Analysis")
    st.write("Analyze an audio file with VAD.")

    input = st.text_input("Which audio file should we analyze?")
    if len(input) == 0:
        st.stop()

    path = pathlib.Path(input)
    if not path.exists() or not path.is_file():
        st.error("File not found.")
        st.stop()

    question = "What is the frame size in milliseconds?"
    milli_frame_size: int = st.number_input(question, min_value=0, max_value=30, value=20, step=10)

    question = "How sensitive should voice activity detection be?"
    mode: int = st.number_input(question, min_value=0, max_value=3, value=1, step=1)

    sample_rate = 16000
    encoding = "pcm_s16le"
    with st.spinner("Normalizing audio..."):
        kwargs = dict(sample_rate=sample_rate, encoding=encoding)
        normalized_path = run._utils.normalized_audio_path(path, **kwargs)
        normalized_path.parent.mkdir(exist_ok=True, parents=False)
        if not normalized_path.exists():
            lib.audio.normalize_audio(path, normalized_path, **kwargs)

    with st.spinner("Loading audio..."):
        metadata = lib.audio.get_audio_metadata(normalized_path)
        audio = lib.audio.read_wave_audio(metadata, dtype=np.int16)[: 16000 * 60]

    with st.spinner("Detecting Voice Activity..."):
        samples_frame_size = int(round(milli_frame_size / 1000 * sample_rate))
        vad = webrtcvad.Vad(mode)
        bar = st.progress(0)
        is_speech: typing.List[bool] = []
        padded = np.pad(audio, (0, samples_frame_size - 1))
        frames = stride_tricks.sliding_window_view(padded, samples_frame_size)
        assert len(frames) == len(audio)
        for i, frame in enumerate(frames):
            is_speech.append(vad.is_speech(frame.tobytes(), sample_rate))
            bar.progress(i / len(frames))

    with st.spinner("Segmenting audio..."):
        x_min, x_max = [], []
        offset = 0
        for is_speech_, group in itertools.groupby(is_speech):
            group = list(group)
            if not is_speech_:
                x_min.append(offset / sample_rate)
                x_max.append((offset + len(group)) / sample_rate + milli_frame_size / 1000)
            offset += len(group)

    with st.spinner("Visualizing segmented audio..."):
        signal_chart = _chart_signal(audio, sample_rate)
        pausing_chart = (
            alt.Chart(pd.DataFrame({"x_min": x_min, "x_max": x_max}))
            .mark_rect(opacity=0.3, color="#85C5A6")
            .encode(x=alt.X("x_min", type="quantitative"), x2=alt.X2("x_max"))
        )
        st.altair_chart((signal_chart + pausing_chart).interactive(), use_container_width=True)


if __name__ == "__main__":
    main()
