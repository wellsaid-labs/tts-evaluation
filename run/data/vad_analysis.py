""" Streamlit application for analyzing an audio file with voice activity detection (VAD).

TODO:
- Question: Do we need to use a sophisiticated VAD, or can we use a basic loudness threshold?
- Once https://github.com/streamlit/streamlit/issues/2263 is resolved, it'd be nice to be able
  to hear the segmented audio.
- Look into NVIDIA Nemo VAD, here:
  https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels?ncid=partn-99401
  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/zh/latest/voice_activity_detection/tutorial.html
- Add smoothing to `webrtcvad` results (either a tigger based approach or median window)
- Try this simple approach: https://maelfabien.github.io/project/Speech_proj/#rolling-window
- Try Kaldi VAD: https://github.com/pykaldi/pykaldi/tree/master/examples/setups/ltsv-based-vad

Usage:
    $ python -m pip install webrtcvad
    $ python -m pip install torchaudio torch==1.7.1
    $ PYTHONPATH=. streamlit run run/data/vad_analysis.py --runner.magicEnabled=false
"""
import itertools
import logging
import random
import typing

import numpy as np
import streamlit as st
import torch
import torch.hub
from numpy.lib.stride_tricks import sliding_window_view
from third_party import LazyLoader

import lib
import run
from lib.datasets import DATASETS, Passage
from run._streamlit import (
    clear_session_cache,
    get_dataset,
    integer_signal_to_floating,
    make_interval_chart,
    make_signal_chart,
    passage_audio,
)

if typing.TYPE_CHECKING:  # pragma: no cover
    import webrtcvad
else:
    webrtcvad = LazyLoader("webrtcvad", globals(), "webrtcvad")


lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


def _normalize_audio(passage: Passage, sample_rate=16000, encoding="pcm_s16le") -> Passage:
    """Normalize `passage` audio to `sample_rate`."""
    path = passage.audio_file.path
    kwargs = dict(sample_rate=sample_rate, encoding=encoding)
    normalized_path = run._utils.normalized_audio_path(path, **kwargs)
    normalized_path.parent.mkdir(exist_ok=True, parents=False)
    if not normalized_path.exists():
        lib.audio.normalize_audio(path, normalized_path, **kwargs)
    metadata = lib.audio.get_audio_metadata(normalized_path)
    return lib.datasets.update_passage_audio(passage, metadata)


def _has_alnum(s: str):
    return any(c.isalnum() for c in s)


def _stt_alignments_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Google Speech-to-Text (STT) API")

    with st.spinner("Visualizing..."):
        signal_chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        nonalignments = [a for s, t, a in passage.script_nonalignments() if not _has_alnum(s + t)]
        start = passage.alignments[0].audio[0]
        end = passage.alignments[-1].audio[-1]
        x_min = [max(a - start, 0) for a, _ in nonalignments]
        x_max = [min(b - start, end - start) for _, b in nonalignments]
        interval_chart = make_interval_chart(np.array(x_min), np.array(x_max), strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


def _webrtc_vad(audio: np.ndarray, sample_rate: int):
    st.markdown("### Google WebRTC Voice Activity Detection (VAD) module")
    if not st.checkbox("Run", key=_webrtc_vad.__qualname__):
        return

    question = "What is the frame size in milliseconds?"
    milli_frame_size: int = st.slider(question, min_value=0, max_value=30, value=20, step=10)
    sec_frame_size: float = milli_frame_size / 1000
    frame_size: int = int(round(sec_frame_size * sample_rate))

    question = "What is the stide size in samples?"
    stride_size: int = st.slider(
        question,
        min_value=1,
        max_value=int(round(30 / 1000 * sample_rate)),
        value=int(round(1 / 1000 * sample_rate)),
        step=1,
    )

    question = "How sensitive should voice activity detection be?"
    mode: int = st.slider(question, min_value=0, max_value=3, value=1, step=1)
    vad = webrtcvad.Vad(mode)

    with st.spinner("Detecting voice activity..."):
        bar = st.progress(0)
        is_speech: typing.List[bool] = []
        padded = np.pad(audio, (0, frame_size - 1))
        frames = sliding_window_view(padded, frame_size)[::stride_size]
        indicies = sliding_window_view(np.arange(padded.shape[0]), frame_size)[::stride_size]
        for i, frame in enumerate(frames):
            is_speech.append(vad.is_speech(frame.tobytes(), sample_rate))
            bar.progress(i / len(frames))

    with st.spinner("Grouping segments..."):
        x_min, x_max = [], []
        for is_speech_, group in itertools.groupby(zip(is_speech, indicies), key=lambda i: i[0]):
            group = list(group)
            if not is_speech_:
                x_min.append(float(group[0][1][0]) / sample_rate)
                x_max.append(float(group[-1][1][-1]) / sample_rate)

    with st.spinner("Visualizing..."):
        signal_chart = make_signal_chart(audio, sample_rate)
        interval_chart = make_interval_chart(np.array(x_min), np.array(x_max), strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


class _SpeechTs(typing.TypedDict):
    start: float
    end: float


def _silero_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Silero Voice Activity Detection (VAD) model")
    if not st.checkbox("Run", key=_silero_vad.__qualname__):
        return

    assert passage.audio_file.sample_rate == 16000
    repo = "snakers4/silero-vad"
    model, utils = torch.hub.load(repo_or_dir=repo, model="silero_vad_micro", force_reload=True)
    get_speech_ts, _, _, _, _, _ = utils
    speech_ts: typing.List[_SpeechTs]
    tensor = torch.tensor(integer_signal_to_floating(audio)).float()
    torch.set_num_threads(1)
    speech_ts = get_speech_ts(
        tensor,
        model,
        min_speech_samples=160,
        min_silence_samples=160,
    )
    speech_ts = (
        [_SpeechTs(start=0, end=0)] + speech_ts + [_SpeechTs(start=len(audio), end=len(audio))]
    )

    with st.spinner("Visualizing..."):
        signal_chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        intervals = [(a["end"], b["start"]) for a, b in zip(speech_ts, speech_ts[1:])]
        x_min = [a / passage.audio_file.sample_rate for a, _ in intervals]
        x_max = [b / passage.audio_file.sample_rate for _, b in intervals]
        interval_chart = make_interval_chart(np.array(x_min), np.array(x_max), strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


def main():
    run._config.configure()

    st.title("VAD Analysis")
    st.write("Analyze an audio file with voice activity detection (VAD).")

    if st.sidebar.button("Clear Session Cache"):
        clear_session_cache()

    speakers: typing.List[str] = [k.label for k in DATASETS.keys()]
    question = "Which dataset do you want to sample from?"
    speaker = st.selectbox(question, speakers)

    with st.spinner("Loading dataset..."):
        dataset = get_dataset(frozenset([speaker]))

    passage = random.choice(list(dataset.values())[0])
    audio_length = passage.alignments[-1].audio[-1] - passage.alignments[0].audio[0]
    st.info(
        "### Randomly Choosen Passage\n"
        "#### Audio File\n"
        f"`{passage.audio_file.path.relative_to(lib.environment.ROOT_PATH)}`\n\n"
        "#### Audio Length\n"
        f"{lib.utils.seconds_to_str(audio_length)}\n\n"
        "#### Script\n"
        f"{passage.script}\n\n"
    )
    with st.spinner("Normalizing audio..."):
        passage = _normalize_audio(passage)

    with st.spinner("Loading audio..."):
        audio = passage_audio(passage)

    with st.spinner("Visualizing Google Speech-to-Text Alignments..."):
        _stt_alignments_vad(passage, audio)

    with st.spinner("Running Google WebRTC VAD..."):
        _webrtc_vad(audio, passage.audio_file.sample_rate)

    with st.spinner("Running Silero VAD..."):
        _silero_vad(passage, audio)


if __name__ == "__main__":
    main()
