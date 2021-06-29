""" A workbook to generate a batch of predictions.

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
    $ PYTHONPATH=. streamlit run run/evaluate/batch_generate.py --runner.magicEnabled=false
"""
import pathlib
import random
import typing

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_datatable import st_datatable
from tqdm import tqdm

import lib
import run
from run._streamlit import (
    audio_to_web_path,
    get_dev_dataset,
    load_tts,
    make_temp_web_dir,
    paths_to_html_download_link,
    st_html,
    web_path_to_url,
)
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, batch_span_to_speech
from run.data._loader import Span
from run.train.spectrogram_model._metrics import (
    get_alignment_norm,
    get_alignment_std,
    get_num_pause_frames,
    get_num_skipped,
)

st.set_page_config(layout="wide")


def make_result(span: Span, audio: np.ndarray) -> typing.Dict[str, str]:
    audio_web_path = audio_to_web_path(audio)
    return {
        "Transcript": span.transcript,
        "Script": span.script,
        "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
        "Speaker": span.speaker.label,
        "Session": str(span.session),
        "Audio": f'<audio controls src="{web_path_to_url(audio_web_path)}"></audio>',
        "Audio Path": str(audio_web_path.relative_to(lib.environment.ROOT_PATH)),
    }


def main():
    st.markdown("# Batch Generation ")
    st.markdown(
        "Use this workbook to generate a batch of clips and export them for further evaluation."
    )
    run._config.configure()

    options = list(CHECKPOINTS_LOADERS.keys())
    format_: typing.Callable[[Checkpoints], str] = lambda s: s.value
    checkpoints_keys: typing.List[Checkpoints]
    checkpoints_keys = st.multiselect("Checkpoints", options=options, format_func=format_)
    num_fake_clips = st.number_input("Number of Generated Clips", min_value=1, value=16, step=1)
    num_real_clips = st.number_input("Number of Real Clips", min_value=1, value=16, step=1)
    shuffle = st.checkbox("Shuffle Clips", value=True)

    if not st.button("Generate"):
        st.stop()

    results = []
    generator = run._utils.SpanGenerator(get_dev_dataset(), balanced=True)
    for _ in range(num_real_clips):
        span = next(generator)
        results.append({"Checkpoints": "original", **make_result(span, span.audio())})

    with st.spinner("Loading and exporting model(s)..."):
        packages = [load_tts(c) for c in checkpoints_keys]

    for package, checkpoints_ in zip(packages, checkpoints_keys):
        spans = [next(generator) for _ in tqdm(range(num_fake_clips), total=num_fake_clips)]
        with st.spinner(f"Generating clips with `{checkpoints_.name}` checkpoints..."):
            in_outs = batch_span_to_speech(package, spans)

        for span, (params, pred, audio) in zip(spans, in_outs):
            image_web_path = make_temp_web_dir() / "alignments.png"
            lib.visualize.plot_alignments(pred.alignments).savefig(image_web_path)
            num_frames = pred.frames.shape[0]
            alignments = pred.alignments.unsqueeze(1)
            alignment_norm = (get_alignment_norm(alignments, None, None)[0] / num_frames).item()
            result = {
                "Checkpoints": checkpoints_.name,
                "Frames Per Token": num_frames / params.tokens.shape[0],
                "Num Pause Frames": get_num_pause_frames(pred.frames.unsqueeze(1), None)[0],
                "Alignment Norm": alignment_norm,
                "Alignment STD": (get_alignment_std(alignments, None, None)[0] / num_frames).item(),
                "Alignment Skips": get_num_skipped(alignments, None, None)[0].item(),
                "Alignment": f'<img src="{web_path_to_url(image_web_path)}" />',
                **make_result(span, audio),
            }
            results.append(result)

    if shuffle:
        random.shuffle(results)

    for index, result in enumerate(results):
        result["Id"] = index

    data_frame = pd.DataFrame(results)
    with st.spinner("Visualizing data..."):
        st_datatable(data_frame, title="Clips")

    with st.spinner("Making Zipfile..."):
        paths = [pathlib.Path(r["Audio Path"]) for r in results]
        archive_paths = [pathlib.Path(str(r["Id"]) + p.suffix) for r, p in zip(results, paths)]
        label = "Download Audio(s)"
        st_html(paths_to_html_download_link("audios.zip", label, paths, archive_paths))


if __name__ == "__main__":
    main()
