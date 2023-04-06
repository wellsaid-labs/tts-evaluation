""" A workbook to generate a batch of predictions for MOS scoring.

TODO: In addition to measuring the current metrics, add support for running speech-to-text on the
      generated audio. This should help uncover egregious errors like gibbirsh, word skipping, etc.

      For example, speech-to-text will predict for a clip that contains gibbirish...
      "That aids in solving communal challenges at Ellenton."
      "That aids in solving communal, challenges Island and Hanna."
      "That aids in solving communal challenges, Allen Honda."
      when the original script was...
      "that aids in solving communal challenges as".

      In this example, speech-to-text predicted for a clip that contains gibbirish...
      "because of lack of available, liquidity, tivity witted to"
      "because of lack of available liquidity to edited edited."
      "because of lack of available liquidity. Tim to it is it is it, is it at"
      when the original script was...
      "because of lack of available liquidity to"

      Given that there is a large enough difference, I think we should be able to detect and measure
      gibbirsh by comparing the speech-to-text output with the original output.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/mos_clips.py --runner.magicEnabled=false
"""
import pathlib
import random
import typing

import config as cf
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from streamlit_datatable import st_datatable
from tqdm import tqdm

import lib
import run
from run._streamlit import (
    audio_to_web_path,
    figure_to_url,
    get_dev_dataset,
    load_tts,
    st_download_files,
    web_path_to_url,
)
from run._tts import CHECKPOINTS_LOADERS, batch_span_to_speech
from run.data._loader import DICTIONARY_DATASETS, Span
from run.train.spectrogram_model._metrics import (
    get_alignment_norm,
    get_alignment_std,
    get_num_pause_frames,
)

st.set_page_config(layout="wide")


def make_row(span: Span, audio: typing.Optional[np.ndarray] = None) -> typing.Dict[str, str]:
    audio_web_path = audio_to_web_path(span.audio() if audio is None else audio)
    return {
        "Transcript": span.transcript,
        "Script": span.script,
        "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
        "Language": span.speaker.dialect.value[1],
        "Speaker": span.speaker.label,
        "Session": str(span.session),
        "Audio": f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>',
        "Audio Path": str(audio_web_path.relative_to(lib.environment.ROOT_PATH)),
    }


def main():
    st.markdown("# Batch Generation ")
    st.markdown("Use this workbook to generate a batch of clips and export them for MOS.")
    run._config.configure(overwrite=True)

    form: DeltaGenerator = st.form("form")
    options = [k.name for k in CHECKPOINTS_LOADERS.keys()]
    checkpoints_keys = form.multiselect("Checkpoints", options=options)
    checkpoints_keys = typing.cast(typing.List[str], checkpoints_keys)
    num_fake = int(form.number_input("Number of Generated Clips", min_value=0, value=16, step=1))
    num_real = int(form.number_input("Number of Real Clips", min_value=0, value=16, step=1))
    shuffle = form.checkbox("Shuffle Clips", value=True)
    include_dic = form.checkbox("Include Dictionary Dataset", value=False)
    if not form.form_submit_button("Generate"):
        return

    with st.spinner("Loading and exporting model(s)..."):
        packages = [load_tts(c) for c in checkpoints_keys]
    session_vocab = set.intersection(*tuple(p.session_vocab() for p in packages))

    get_weight = run.train.spectrogram_model._data.dev_get_weight
    generator = cf.partial(run._utils.SpanGenerator)(get_dev_dataset(), get_weight=get_weight)
    include_span: typing.Callable[[Span], bool] = lambda s: s.session in session_vocab and (
        include_dic or s.speaker not in DICTIONARY_DATASETS
    )
    generator = (s for s in generator if include_span(s))
    rows = [{"Checkpoints": "original", **make_row(next(generator))} for _ in range(num_real)]

    for package, checkpoints_ in zip(packages, checkpoints_keys):
        spans = [next(generator) for _ in tqdm(range(num_fake), total=num_fake)]
        with st.spinner(f"Generating clips with `{checkpoints_}` checkpoints..."):
            results = batch_span_to_speech(package, spans)

        for span, (_, pred, audio) in tqdm(zip(spans, results), total=len(spans)):
            figure_url = figure_to_url(lib.visualize.plot_alignments(pred.alignments[:, 0]))
            num_frames = pred.frames.shape[0]
            num_pause_frames = cf.partial(get_num_pause_frames)(pred.frames, None)
            row = {
                "Checkpoints": checkpoints_,
                "Frames Per Token": num_frames / pred.num_tokens[0].item(),
                "Num Pause Frames": num_pause_frames[0],
                "Alignment Norm": (get_alignment_norm(pred)[0] / num_frames).item(),
                "Alignment STD": (get_alignment_std(pred)[0] / num_frames).item(),
                "Alignment": f'<img src="{figure_url}" />',
                **make_row(span, audio[0]),
            }
            rows.append(row)

    if shuffle:
        random.shuffle(rows)

    for index, row in enumerate(rows):
        row["Id"] = str(index)

    data_frame = pd.DataFrame(rows)
    with st.spinner("Visualizing data..."):
        st_datatable(data_frame, title="Clips")

    with st.spinner("Making Zipfile..."):
        paths = [pathlib.Path(r["Audio Path"]) for r in rows]
        archive_paths = [pathlib.Path(str(r["Id"]) + p.suffix) for r, p in zip(rows, paths)]
        st_download_files("audios.zip", "Download Audio(s)", paths, archive_paths)


if __name__ == "__main__":
    main()