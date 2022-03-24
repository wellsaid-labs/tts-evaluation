""" Analyze an audio file by it's features.

Usage:
    $ PYTHONPATH=. streamlit run run/data/audio_features_workbook.py --runner.magicEnabled=false
"""
import functools
import pathlib
import tempfile
import typing

import altair as alt
import config as cf
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib import pyplot

import lib
import run
from lib.audio import SignalTodBMelSpectrogram, amp_to_db, framed_rms_to_rms, sample_to_sec
from run._streamlit import audio_to_html, st_html


def _chart_framed_rms_level(framed_rms_level: np.ndarray, frame_hop: int):
    rms_level_db = amp_to_db(framed_rms_level)
    samples = [i * frame_hop + frame_hop // 2 for i in range(len(rms_level_db))]
    seconds = [sample_to_sec(s, **cf.get()) for s in samples]
    return (
        alt.Chart(pd.DataFrame({"Seconds": seconds, "Decibels": rms_level_db}))
        .mark_area()
        .encode(
            x=alt.X("Seconds", type="quantitative"),
            y=alt.Y("Decibels", scale=alt.Scale(domain=(-100.0, 0)), type="quantitative"),
        )
    )


def main():
    pyplot.style.use("dark_background")
    st.markdown("# Audio Features Workbook")
    run._config.configure()

    uploaded_file = st.file_uploader("Audio File", "wav")
    assert not isinstance(uploaded_file, list)
    ndigits = st.sidebar.slider("Precision", min_value=1, max_value=10, value=5)
    round_ = functools.partial(round, ndigits=ndigits)
    get_weighting_callables = (
        lib.audio.identity_weighting,
        lib.audio.a_weighting,
        lib.audio.iso226_weighting,
        lib.audio.k_weighting,
    )
    format_func: typing.Callable[[typing.Callable], str] = lambda f: f.__name__
    get_weighting: typing.Callable = st.sidebar.selectbox(
        "Weighting",
        get_weighting_callables,
        format_func=format_func,  # type: ignore
        index=2,
    )
    signal_to_spectrogram = SignalTodBMelSpectrogram(get_weighting=get_weighting, **cf.get())

    if uploaded_file is None:
        st.stop()

    temp = tempfile.NamedTemporaryFile(suffix=".wav")
    temp.write(uploaded_file.getbuffer())
    path = pathlib.Path(temp.name)
    metadata = lib.audio.get_audio_metadata(path)
    run._config.audio.configure(metadata.sample_rate)
    audio = lib.audio.read_wave_audio(metadata)
    audio = lib.audio.pad_remainder(audio, **cf.get())

    meter = lib.audio.get_pyloudnorm_meter(**cf.get())
    specs = signal_to_spectrogram(torch.tensor(audio), intermediate=True, aligned=True)
    rms_level = round_(lib.audio.signal_to_rms(audio).item())
    rms_db_level = round_(amp_to_db(rms_level))
    peek_level = round_(torch.tensor(audio).abs().max().item())
    peek_db_level = round_(amp_to_db(peek_level))
    power_spec = lib.audio.db_to_power(specs.db_mel)
    framed_rms_level = lib.audio.power_spectrogram_to_framed_rms(power_spec, **cf.get())
    frame_hop = signal_to_spectrogram.frame_hop
    signal_length = signal_to_spectrogram.frame_hop * power_spec.shape[-2]
    spec_rms_level = round_(framed_rms_to_rms(framed_rms_level, frame_hop, signal_length).item())
    spec_rms_db_level = round_(amp_to_db(spec_rms_level))
    lufs = round_(meter.integrated_loudness(audio))
    weighting = signal_to_spectrogram.get_weighting.__name__

    st.write("### Metadata")
    st.write({k: str(v) for k, v in lib.utils.dataclass_as_dict(metadata).items()})

    # NOTE: Learn more about dBov, here: https://en.wikipedia.org/wiki/DBFS
    st.write("### Loudness")
    st.write(
        f"- **Peek Level:** {peek_level} ({peek_db_level} dBov)\n\n"
        f"- **RMS Level:** {rms_level} ({rms_db_level} dBov)\n\n"
        f"- **RMS `{weighting}` Level (Spec):** {spec_rms_level} ({spec_rms_db_level} dBov)\n\n"
        f"- **LUFS:** {lufs} dB\n\n"
    )

    st.markdown("### Loudness Over Time")
    chart = _chart_framed_rms_level(framed_rms_level.numpy(), frame_hop)
    st.altair_chart(chart.interactive(), use_container_width=True)

    st.markdown("### dB Mel Spectrogram")
    st.pyplot(lib.visualize.plot_mel_spectrogram(specs.db_mel, **cf.get()), transparent=True)
    st.markdown("### dB Spectrogram")
    st.pyplot(lib.visualize.plot_spectrogram(specs.db, **cf.get()), transparent=True)
    st.markdown("### Spectrogram")
    st.pyplot(lib.visualize.plot_spectrogram(specs.amp, **cf.get()), transparent=True)

    st.markdown("### Original Audio")
    st_html(audio_to_html(audio))
    st.markdown("### Griffin-Lim Audio")
    griffin_lim = lib.audio.griffin_lim(specs.db_mel.numpy(), **cf.get())
    st_html(audio_to_html(griffin_lim))


if __name__ == "__main__":
    main()
