""" Streamlit application for analyzing an audio file with voice activity detection (VAD).

NOTE: This workbook is optimized for `streamlit`s dark theme.

TODO:
- Once https://github.com/streamlit/streamlit/issues/2263 is resolved, it'd be nice to be able
  to hear the segmented audio.
- Look into NVIDIA Nemo VAD, here:
  https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels?ncid=partn-99401
  https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/zh/latest/voice_activity_detection/tutorial.html
- Add smoothing to `webrtcvad` results (either a tigger based approach or median window)
- Try this simple approach: https://maelfabien.github.io/project/Speech_proj/#rolling-window
- Try Kaldi VAD: https://github.com/pykaldi/pykaldi/tree/master/examples/setups/ltsv-based-vad
- Add audio padding to Baseline VAD.
- Trying this VAD: https://github.com/pyannote/pyannote-audio-hub

Usage:
    $ python -m pip install webrtcvad
    $ python -m pip install torchaudio torch==1.7.1
    $ PYTHONPATH=. streamlit run run/review/audio_processing/voice_activity_detection.py \
          --runner.magicEnabled=false
"""
import logging
import math
import random
import typing
from functools import partial

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.hub
from numpy.lib.stride_tricks import sliding_window_view
from third_party import LazyLoader
from torchnlp.random import fork_rng

import lib
import run
from lib.audio import (
    AudioDataType,
    AudioMetadata,
    _get_non_speech_segments_helper,
    get_audio_metadata,
    group_audio_frames,
    milli_to_sample,
    sample_to_milli,
    sample_to_sec,
)
from lib.utils import Timeline, seconds_to_str
from run._streamlit import (
    audio_to_html,
    clear_session_cache,
    dataset_passages,
    get_dataset,
    make_interval_chart,
    make_signal_chart,
    passage_audio,
    read_wave_audio,
    st_html,
)
from run.data import _loader
from run.data._loader import (
    DATASETS,
    Passage,
    Span,
    maybe_normalize_audio_and_cache,
    voiced_nonalignment_spans,
)

if typing.TYPE_CHECKING:  # pragma: no cover
    import webrtcvad  # type: ignore
else:
    webrtcvad = LazyLoader("webrtcvad", globals(), "webrtcvad")


lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


def _normalize_cache_read_audio(
    passage: Passage, **kwargs
) -> typing.Tuple[AudioMetadata, np.ndarray]:
    """Normalize, cache, and read passage audio."""
    other_audio_path = maybe_normalize_audio_and_cache(passage.audio_file, **kwargs)
    other_audio_metadata = get_audio_metadata(other_audio_path)
    other_audio_length = passage.aligned_audio_length()
    other_audio = read_wave_audio(other_audio_metadata, passage.first.audio[0], other_audio_length)
    return other_audio_metadata, other_audio


def _audio_intervals(
    passage: Passage, spans: typing.List[Span]
) -> typing.List[typing.Tuple[float, float]]:
    """Bound and normalize span audio intervals with `passage.first` and `passage.last`."""
    start = passage.first.audio[0]
    audio = [s.alignment.audio for s in spans]
    return [(max(a - start, 0.0), min(b - start, passage.last.audio[-1] - start)) for a, b in audio]


def _chart_alignments_and_non_speech_segments(
    passage: Passage, non_speech_segments: typing.Sequence[typing.Tuple[float, float]]
):
    """Chart `passage` alignments and nonalignments versus `non_speech_segments`."""
    st.write("**Alignments and Non-Speech Segments**")
    alignment_audio_intervals = [a.audio for a in passage[:].alignments]
    spans, is_voiced = voiced_nonalignment_spans(passage)
    mistranscriptions = [s for s, i in zip(spans.spans, is_voiced) if i]
    nonalignment_audio_intervals = _audio_intervals(passage, mistranscriptions)
    source = {
        "interval": ["non_speech_segments"] * len(non_speech_segments),
        "start": [s[0] for s in non_speech_segments],
        "end": [s[1] for s in non_speech_segments],
    }
    chart = (
        alt.Chart(pd.DataFrame(source))  # type: ignore
        .mark_bar(stroke="#000", strokeWidth=1, strokeOpacity=0.3)  # type: ignore
        .encode(x="start", x2="end", y="interval", color="interval")
    )
    chart += make_interval_chart(nonalignment_audio_intervals, color="darkred", strokeWidth=0)
    chart += make_interval_chart(
        alignment_audio_intervals, color="white", fillOpacity=0.1, strokeOpacity=1.0, opacity=1.0
    )
    return chart


def _median(x: np.ndarray) -> float:
    """Get the median value of a sorted array."""
    return x[math.floor(len(x) / 2) : math.ceil(len(x) / 2) + 1].mean().item()


def _chart_db_rms(seconds: np.ndarray, rms_level_db: np.ndarray):
    """
    Args:
        seconds: The second each frame is located at.
        rms_level_db: Frame-level RMS levels.
    """
    st.write("**dB RMS**")
    return (
        alt.Chart(pd.DataFrame({"seconds": seconds, "decibels": rms_level_db}))  # type: ignore
        .mark_area()
        .encode(
            x=alt.X("seconds", type="quantitative"),  # type: ignore
            y=alt.Y(
                "decibels",  # type: ignore
                scale=alt.Scale(domain=(-200.0, 0)),  # type: ignore
                type="quantitative",  # type: ignore
            ),
        )
    )


def _stt_alignments_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Google Speech-to-Text (STT) API")
    st.write("Use dataset alignments to detect voice activity.")

    interval_chart_: typing.Callable[[typing.List[Span]], None]
    interval_chart_ = lambda a, **k: make_interval_chart(_audio_intervals(passage, a), **k)

    with st.spinner("Visualizing..."):
        spans, is_voiced = voiced_nonalignment_spans(passage)
        mistranscriptions = [s for s, i in zip(spans.spans, is_voiced) if i]
        rest = [s for s, i in zip(spans.spans, is_voiced) if not i]
        chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        chart += interval_chart_(rest, strokeWidth=0)
        chart += interval_chart_(mistranscriptions, strokeWidth=0, color="red")
        st.altair_chart(chart.interactive(), use_container_width=True)


def _current_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Current Voice Activity Detection (VAD)")
    st.write("Visualize the implemented VAD algorithm.")
    nss = passage.non_speech_segments[passage.audio_start : passage.audio_stop]
    nss = np.clip(nss - passage.audio_start, 0.0, passage.audio_stop).tolist()
    st.info(f"The non-speech segments are: {nss}")

    with st.spinner("Visualizing..."):
        st.write("**Non-Speech Segments**")
        non_speech_segments_chart = make_interval_chart(nss, strokeWidth=0)
        chart = make_signal_chart(audio, passage.audio_file.sample_rate) + non_speech_segments_chart
        st.altair_chart(chart.interactive(), use_container_width=True)

    pairs = zip(passage.speech_segments, passage.speech_segments[1:])
    nss = [(a.audio_stop, b.audio_start) for a, b in pairs]
    nss.insert(0, (passage.audio_start, passage.speech_segments[0].audio_start))
    nss.append((passage.speech_segments[-1].audio_stop, passage.audio_stop))
    nss = [(s[0] - passage.audio_start, s[1] - passage.audio_start) for s in nss]
    st.info(f"The filtered non-speech segments are: {list(nss)}")

    with st.spinner("Visualizing..."):
        st.write("**Filtered Non-Speech Segments**")
        non_speech_segments_chart = make_interval_chart(nss, strokeWidth=0)
        chart = make_signal_chart(audio, passage.audio_file.sample_rate) + non_speech_segments_chart
        st.altair_chart(chart.interactive(), use_container_width=True)

    st.markdown("### Speech Segment Clips")
    for span in passage.speech_segments:
        st.markdown(
            f"**Location:** {span.audio_start - passage.audio_start}s, "
            f"{span.audio_stop - passage.audio_start}s\n\n"
            f"**Script:** '{span.script}'"
        )
        st_html(audio_to_html(span.audio(), sample_rate=span.audio_file.sample_rate))


def _baseline_vad(passage: Passage, audio: np.ndarray):
    st.markdown("### Baseline Voice Activity Detection (VAD)")
    st.write("Use an RMS threshold with a bandpass filter to detect voice activity.")

    sample_rate = passage.audio_file.sample_rate
    col = st.columns([1, 1])
    label = "What is the frame size in milliseconds?"
    milli_frame_size: int = col[0].slider(label, min_value=0, max_value=250, value=50, step=1)

    label = "What is the stride size in samples?"
    max_value = milli_to_sample(250, sample_rate)
    value = milli_to_sample(5, sample_rate)
    stride_size: int = col[1].slider(label, min_value=1, max_value=max_value, value=value, step=1)
    milli_stride_size = sample_to_milli(stride_size, sample_rate)

    label = "What is the threshold for silence in decibels?"
    threshold: int = st.slider(label, min_value=-100, max_value=0, value=-60, step=1)

    nyq_freq = int(sample_rate / 2)
    label = "What is the low frequency cutoff in Hz?"
    low_cut: int = st.slider(label, min_value=1, max_value=nyq_freq - 1, value=300, step=1)

    with st.spinner("Measuring RMS..."):
        audio_file = passage.audio_file
        indicies, rms_level_power = _get_non_speech_segments_helper(
            audio,
            audio_file,
            low_cut=low_cut,
            frame_length=milli_frame_size,
            hop_length=milli_stride_size,
        )
        rms_level_db = lib.audio.power_to_db(rms_level_power)
        is_not_speech: typing.Callable[[bool], bool] = lambda is_speech: not is_speech
        is_speech: typing.List[bool] = list(rms_level_db > threshold)
        non_speech_segments = group_audio_frames(sample_rate, is_speech, indicies, is_not_speech)
        non_speech_segments_chart = make_interval_chart(non_speech_segments, strokeWidth=0)
        seconds = np.array([_median(i) / sample_rate for i in indicies])
        chart = _chart_db_rms(seconds, rms_level_db) + non_speech_segments_chart
        st.altair_chart(chart.interactive(), use_container_width=True)

    with st.spinner("Grouping and filtering..."):
        intervals = [a.audio for a in passage[:].alignments]
        length = len(non_speech_segments)
        non_speech_segments = _loader.structures._filter_non_speech_segments(
            intervals, Timeline(intervals), [slice(*s) for s in non_speech_segments]
        )
        non_speech_segments = list([(s.start, s.stop) for s in non_speech_segments])
        num_removed = length - len(non_speech_segments)
        st.info(f"Filtered **{num_removed}** of **{length}** non-speech segments.")
        st.info(f"The choosen non-speech segments are: {non_speech_segments}")
        chart = _chart_alignments_and_non_speech_segments(passage, non_speech_segments)
        st.altair_chart(chart.interactive(), use_container_width=True)

    with st.spinner("Visualizing..."):
        st.write("**Non-Speech Segments**")
        non_speech_segments_chart = make_interval_chart(non_speech_segments, strokeWidth=0)
        chart = make_signal_chart(audio, sample_rate) + non_speech_segments_chart
        st.altair_chart(chart.interactive(), use_container_width=True)


def _webrtc_vad(passage: Passage, audio: np.ndarray, sample_rate: int = 16000):
    st.markdown("### Google WebRTC Voice Activity Detection (VAD) module")
    if not st.checkbox("Run", key=_webrtc_vad.__qualname__):
        return

    question = "What is the frame size in milliseconds?"
    milli_frame_size: int = st.slider(question, min_value=0, max_value=30, value=20, step=10)
    frame_size = milli_to_sample(milli_frame_size, sample_rate)

    question = "What is the stide size in samples?"
    max_value = milli_to_sample(30, sample_rate)
    value = milli_to_sample(1, sample_rate)
    stride_size: int = st.slider(question, min_value=1, max_value=max_value, value=value, step=1)

    question = "How sensitive should voice activity detection be?"
    mode: int = st.slider(question, min_value=0, max_value=3, value=1, step=1)
    vad = webrtcvad.Vad(mode)

    with st.spinner("Normalizing audio..."):
        _, norm_audio = _normalize_cache_read_audio(
            passage,
            data_type=AudioDataType.SIGNED_INTEGER,
            bits=16,
            format_=lib.audio.AudioFormat(
                sample_rate=sample_rate,
                bit_rate="364k",
                precision="16-bit",
                num_channels=1,
                encoding=lib.audio.AudioEncoding.PCM_INT_16_BIT,
            ),
        )

    with st.spinner("Detecting voice activity..."):
        bar = st.progress(0)
        is_speech: typing.List[bool] = []
        padded = np.pad(norm_audio, (0, frame_size - 1))
        frames = sliding_window_view(padded, frame_size)[::stride_size]
        indicies = sliding_window_view(np.arange(padded.shape[0]), frame_size)[::stride_size]
        for i, frame in enumerate(frames):
            is_speech.append(vad.is_speech(frame.tobytes(), sample_rate))
            bar.progress(i / len(frames))
        bar.empty()
        is_not_speech: typing.Callable[[bool], bool] = lambda is_speech: not is_speech
        non_speech_segments = group_audio_frames(sample_rate, is_speech, indicies, is_not_speech)

    with st.spinner("Visualizing..."):
        signal_chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        interval_chart = make_interval_chart(non_speech_segments, strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


class _SpeechTs(typing.TypedDict):
    start: int
    end: int


def _silero_vad(passage: Passage, audio: np.ndarray, sample_rate: int = 16000):
    st.markdown("### Silero Voice Activity Detection (VAD) model")
    if not st.checkbox("Run", key=_silero_vad.__qualname__):
        return

    repo: str = st.text_input("Github Repository", value="snakers4/silero-vad")
    model: str = st.text_input("Model", value="silero_vad_micro")

    with st.spinner("Loading Model..."):
        model, utils = torch.hub.load(repo_or_dir=repo, model=model, force_reload=True)
        get_speech_ts, _, _, _, _, _ = utils

    with st.spinner("Normalizing audio..."):
        torch.set_num_threads(1)
        # TODO: `bit_rate` and `precision` should be set automatically.
        format_ = lib.audio.AudioFormat(
            sample_rate=sample_rate,
            num_channels=1,
            encoding=lib.audio.AudioEncoding.PCM_FLOAT_32_BIT,
            bit_rate="768k",
            precision="25-bit",
        )
        _, norm_audio = _normalize_cache_read_audio(passage, format_=format_)

    with st.spinner("Predicting..."):
        speech_ts: typing.List[_SpeechTs]
        speech_ts = get_speech_ts(
            torch.tensor(norm_audio), model, min_speech_samples=160, min_silence_samples=160
        )
        length = len(norm_audio)
        speech_ts = [_SpeechTs(start=0, end=0)] + speech_ts + [_SpeechTs(start=length, end=length)]
        pairs = zip(speech_ts, speech_ts[1:])
        sample_to_sec_ = partial(sample_to_sec, sample_rate=sample_rate)
        intervals = [(sample_to_sec_(a["end"]), sample_to_sec_(b["start"])) for a, b in pairs]

    with st.spinner("Visualizing..."):
        signal_chart = make_signal_chart(audio, passage.audio_file.sample_rate)
        interval_chart = make_interval_chart(intervals, strokeWidth=0)
        st.altair_chart((signal_chart + interval_chart).interactive(), use_container_width=True)


def _get_hard_passages(dataset: run._utils.Dataset, threshold: float = 20) -> typing.Set[Passage]:
    """Get hards passages for Google STT. So far, we have defined hard passages as ones that have
    long segments without pausing based on `alignments`."""
    passages = set()
    all_passages = list(dataset_passages(dataset))
    for passage in all_passages:
        start = passage.first.audio[0]
        spans, spans_is_voiced = voiced_nonalignment_spans(passage)
        for span, is_voiced in zip(spans.spans, spans_is_voiced):
            if not is_voiced and span.audio_length > 0:
                if span.audio_slice.start - start > threshold:
                    passages.add(passage)
                    break
                start = span.audio_slice.stop
        if passage.last.audio[-1] - start > threshold:
            passages.add(passage)
    st.info(
        f"Found **{len(passages)}** out of **{len(all_passages)}** "
        "passages with challenging segments."
    )
    return passages


def _init_random_seed(key: str = "random_seed", default_value: int = 123) -> str:
    """Create a persistent state for the random seed."""
    value = st.sidebar.number_input("Random Seed", value=default_value)  # type: ignore
    if key not in st.session_state or value != default_value:
        st.session_state[key] = default_value
    return key


def main():
    run._config.configure(overwrite=True)

    st.title("Voice Activity Detection (VAD) Workbook")
    st.write("Analyze an audio file with voice activity detection (VAD).")

    if st.sidebar.button("Clear Session Cache"):
        clear_session_cache()

    label = "Max Passage Seconds"
    max_len = st.sidebar.number_input(label, 0.0, value=60.0, step=1.0)  # type: ignore

    random_seed_key = _init_random_seed()

    speakers: typing.List[str] = [k.label for k in DATASETS.keys()]
    question = "Which dataset do you want to sample from?"
    speaker = st.selectbox(question, speakers)

    with st.spinner("Loading dataset..."):
        dataset = get_dataset(frozenset([speaker]))

    use_hard = st.checkbox("Find Challenging Passage", value=True)
    passages = _get_hard_passages(dataset) if use_hard else list(dataset.values())[0]
    passages = [p for p in passages if p.aligned_audio_length() < max_len]
    st.session_state[random_seed_key] += int(st.button("New Passage"))
    with fork_rng(st.session_state[random_seed_key]):
        passage = random.choice(list(passages))
    start = passage.audio_start
    end = passage.audio_stop
    st.info(
        "### Randomly Choosen Passage\n"
        f"#### Random Seed\n{st.session_state[random_seed_key]}\n"
        f"#### Audio File\n`{passage.audio_file.path.relative_to(lib.environment.ROOT_PATH)}`\n\n"
        "#### Audio Slice\n"
        f"{seconds_to_str(end - start)} ({seconds_to_str(start)}, {seconds_to_str(end)})\n\n"
        f"#### Script\n{passage.script}\n\n"
    )

    with st.spinner("Loading audio..."):
        audio = passage_audio(passage)

    st.markdown("### Audio")
    st_html(audio_to_html(audio, sample_rate=passage.audio_file.sample_rate))

    _current_vad(passage, audio)
    _stt_alignments_vad(passage, audio)
    _baseline_vad(passage, audio)
    _webrtc_vad(passage, audio)
    _silero_vad(passage, audio)


if __name__ == "__main__":
    main()
