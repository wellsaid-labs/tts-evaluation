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
    $ PYTHONPATH=. streamlit run run/data/vad_workbook.py --runner.magicEnabled=false
"""
import itertools
import logging
import math
import random
import typing

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.hub
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from third_party import LazyLoader
from torchnlp.random import fork_rng

import lib
import run
from lib.datasets import DATASETS, Passage
from lib.utils import Interval, Timeline, seconds_to_str
from run._streamlit import (
    audio_to_html,
    clear_session_cache,
    dataset_passages,
    get_dataset,
    get_session_state,
    has_alnum,
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


def _audio_intervals(
    passage: Passage, audio: typing.List[typing.Tuple[float, float]]
) -> typing.Tuple[typing.List[float], typing.List[float]]:
    """Bound and normalize `audio` intervals at `passage.first` and `passage.last`."""
    start = passage.first.audio[0]
    x_min = [max(a - start, 0.0) for a, _ in audio]
    x_max = [min(b - start, passage.last.audio[-1] - start) for _, b in audio]
    return x_min, x_max


def _stt_alignments_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Google Speech-to-Text (STT) API")
    st.write("Use dataset alignments to detect voice activity.")

    with st.spinner("Visualizing..."):
        pauses = [a for s, t, a in passage.script_nonalignments() if not has_alnum(s + t)]
        mistrascriptions = [a for s, t, a in passage.script_nonalignments() if has_alnum(s + t)]
        chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        interval_chart_ = lambda a, **k: make_interval_chart(*_audio_intervals(passage, a), **k)
        chart += interval_chart_(pauses, strokeWidth=0)
        chart += interval_chart_(mistrascriptions, strokeWidth=0, color="darkred")
        st.altair_chart(chart.interactive(), use_container_width=True)


def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Original:
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], analog=False, btype="band", output="sos")
    return signal.sosfiltfilt(sos, data)


def _median(x: np.ndarray) -> float:
    """ Get the median value of a sorted array. """
    return x[math.floor(len(x) / 2) : math.ceil(len(x) / 2) + 1].mean().item()


def _chart_db_rms(seconds: np.ndarray, rms_level_db: np.ndarray):
    """
    Args:
        ...
        seconds: The second each frame is located at.
        rms_level_db: Frame-level RMS levels.
    """
    st.write("**dB RMS**")
    return (
        alt.Chart(pd.DataFrame({"seconds": seconds, "decibels": rms_level_db}))
        .mark_area()
        .encode(
            x=alt.X("seconds", type="quantitative"),
            y=alt.Y("decibels", scale=alt.Scale(domain=(-200.0, 0)), type="quantitative"),
        )
    )


def _chart_alignments_and_pauses(
    passage: Passage, x_min: typing.List[float], x_max: typing.List[float]
):
    """Chart `passage` alignments versus pauses as defined by `x_min` and `x_max`."""
    st.write("**Alignments and Pauses**")
    span = passage[:]
    alignments_x_min = [a.audio[0] for a in span.alignments]
    alignments_x_max = [a.audio[1] for a in span.alignments]
    data = {"interval": ["pauses"] * len(x_min), "start": x_min, "end": x_max}
    return (
        alt.Chart(pd.DataFrame(data))
        .mark_bar(stroke="#000", strokeWidth=1, strokeOpacity=0.3)
        .encode(x="start", x2="end", y="interval", color="interval")
    ) + make_interval_chart(alignments_x_min, alignments_x_max, color="gray", strokeOpacity=1.0)


def _filter_pauses(passage: Passage, x_min: typing.List[float], x_max: typing.List[float]):
    """Filter pauses at `x_min` and `x_max` based on `passage.alignments`."""
    span = passage[:]
    alignments = list(span.alignments)
    timeline = Timeline([Interval(a.audio, a) for a in alignments])
    _has_mistranscription = lambda v, a: has_alnum(
        getattr(span, a)[min((getattr(o, a)[1] for o in v)) : max((getattr(o, a)[0] for o in v))]
    )
    for min_, max_ in zip(x_min, x_max):
        vals = list(timeline[min_:max_])
        if len(vals) == 0:
            yield min_, max_
        elif (
            len(vals) == 2
            # NOTE: Ensure there is overlap between the pause and both alignments
            and min((o.audio[1] for o in vals)) <= max_
            and max((o.audio[0] for o in vals)) >= min_
            # NOTE: Ensure alignment(s) are not inside the pause
            and min((o.audio[0] for o in vals)) <= min_
            and max((o.audio[1] for o in vals)) >= max_
            # NOTE: Ensure that there is no mistranscription between the alignments
            and not _has_mistranscription(vals, "script")
            and not _has_mistranscription(vals, "transcript")
        ):
            yield min_, max_


def _baseline_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Baseline Voice Activity Detection (VAD)")
    st.write("Use an RMS threshold with a bandpass filter to detect voice activity.")

    sample_rate = passage.audio_file.sample_rate
    col = st.beta_columns([1, 1])
    question = "What is the frame size in milliseconds?"
    milli_frame_size: int = col[0].slider(question, min_value=0, max_value=250, value=50, step=1)
    sec_frame_size: float = milli_frame_size / 1000
    frame_size: int = int(round(sec_frame_size * sample_rate))

    question = "What is the stide size in samples?"
    stride_size: int = col[1].slider(
        question,
        min_value=1,
        max_value=int(round(250 / 1000 * sample_rate)),
        value=int(round(5 / 1000 * sample_rate)),
        step=1,
    )

    question = "What is the threshold for silence in decibels?"
    threshold: int = st.slider(question, min_value=-100, max_value=0, value=-60, step=1)

    col = st.beta_columns([1, 1])
    nyq_freq = int(sample_rate / 2)
    question = "What is the low frequency cutoff in Hz?"
    low_cut: int = col[0].slider(question, min_value=1, max_value=nyq_freq - 1, value=300, step=1)

    question = "What is the high frequency cutoff in Hz?"
    high_cut: int = col[1].slider(
        question, min_value=1, max_value=nyq_freq - 1, value=nyq_freq - 1, step=1
    )

    with st.spinner("Detecting voice activity..."):
        audio = integer_signal_to_floating(audio)
        filtered = _butter_bandpass_filter(audio, low_cut, high_cut, sample_rate)
        frames = sliding_window_view(filtered, frame_size)[::stride_size]
        indicies = sliding_window_view(np.arange(filtered.shape[0]), frame_size)[::stride_size]
        rms_level_power = np.mean(np.abs(frames) ** 2, axis=1)
        rms_level_db = 10.0 * np.log10(np.clip(rms_level_power, 1e-10, None))
        is_pause = (rms_level_db < threshold).tolist()
        seconds = np.array([_median(i) / sample_rate for i in indicies])
        chart = _chart_db_rms(seconds, rms_level_db)
        st.altair_chart(chart.interactive(), use_container_width=True)

    with st.spinner("Grouping segments..."):
        x_min, x_max = [], []
        for is_pause_, group in itertools.groupby(zip(is_pause, indicies), key=lambda i: i[0]):
            group = list(group)
            if is_pause_:
                x_min.append(float(group[0][1][0]) / sample_rate)
                x_max.append(float(group[-1][1][-1]) / sample_rate)
        chart = _chart_alignments_and_pauses(passage, x_min, x_max)
        st.altair_chart(chart.interactive(), use_container_width=True)

    with st.spinner("Filtering segments..."):
        x_min, x_max = zip(*list(_filter_pauses(passage, x_min, x_max)))
        x_min, x_max = list(x_min), list(x_max)

    with st.spinner("Visualizing..."):
        st.write("**Pauses**")
        signal_chart = make_signal_chart(audio, sample_rate)
        pausing_chart = make_interval_chart(x_min, x_max, strokeWidth=0, opacity=0.6)
        st.altair_chart((signal_chart + pausing_chart).interactive(), use_container_width=True)


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
        interval_chart = make_interval_chart(x_min, x_max, strokeWidth=0)
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
        interval_chart = make_interval_chart(x_min, x_max, strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


def _get_challenging_passages(
    dataset: run._config.Dataset, threshold: float = 20
) -> typing.Set[Passage]:
    """Get challenging passages for VAD. So far, we have defined challenging passages as ones
    that have long segments without pausing based on `alignments`."""
    passages = set()
    all_passages = list(dataset_passages(dataset))
    for passage in all_passages:
        start = passage.first.audio[0]
        for script, transcript, audio in passage.script_nonalignments()[1:-1]:
            if not has_alnum(script + transcript) and audio[1] - audio[0] > 0:
                if audio[0] - start > threshold:
                    passages.add(passage)
                    break
                start = audio[1]
        if passage.last.audio[-1] - start > threshold:
            passages.add(passage)
    st.info(
        f"Found **{len(passages)}** out of **{len(all_passages)}** "
        "passages with challenging segments."
    )
    return passages


def _init_random_seed(key="random_seed", default_value=123) -> int:
    """ Create a persistent state for the random seed. """
    state = get_session_state()
    value = st.sidebar.number_input("Random Seed", value=default_value)
    if key not in state or value != default_value:
        state[key] = default_value
    return key


def main():
    run._config.configure()

    st.title("Voice Activity Detection (VAD) Workbook")
    st.write("Analyze an audio file with voice activity detection (VAD).")

    if st.sidebar.button("Clear Session Cache"):
        clear_session_cache()

    state = get_session_state()
    random_seed_key = _init_random_seed()

    speakers: typing.List[str] = [k.label for k in DATASETS.keys()]
    question = "Which dataset do you want to sample from?"
    speaker = st.selectbox(question, speakers)

    with st.spinner("Loading dataset..."):
        dataset = get_dataset(frozenset([speaker]))

    use_challenging = st.checkbox("Find Challenging Passage", value=True)
    passages = _get_challenging_passages(dataset) if use_challenging else list(dataset.values())[0]
    state[random_seed_key] += int(st.button("New Passage"))
    with fork_rng(state[random_seed_key]):
        passage = random.choice(list(passages))
    start = passage.first.audio[0]
    end = passage.last.audio[-1]
    audio_length = end - start
    st.info(
        "### Randomly Choosen Passage\n"
        "#### Random Seed\n"
        f"{state[random_seed_key]}\n"
        "#### Audio File\n"
        f"`{passage.audio_file.path.relative_to(lib.environment.ROOT_PATH)}`\n\n"
        "#### Audio Slice\n"
        f"{seconds_to_str(audio_length)} ({seconds_to_str(start)}, {seconds_to_str(end)})\n\n"
        "#### Script\n"
        f"{passage.script}\n\n"
    )
    with st.spinner("Normalizing audio..."):
        passage = _normalize_audio(passage)
        sample_rate = passage.audio_file.sample_rate

    with st.spinner("Loading audio..."):
        audio = passage_audio(passage)

    st.markdown("### Audio")
    html = audio_to_html(integer_signal_to_floating(audio), sample_rate=sample_rate)
    st.markdown(html, unsafe_allow_html=True)

    with st.spinner("Visualizing Google Speech-to-Text Alignments..."):
        _stt_alignments_vad(passage, audio)

    with st.spinner("Running Baseline VAD..."):
        _baseline_vad(passage, audio)

    with st.spinner("Running Google WebRTC VAD..."):
        _webrtc_vad(audio, sample_rate)

    with st.spinner("Running Silero VAD..."):
        _silero_vad(passage, audio)


if __name__ == "__main__":
    main()
