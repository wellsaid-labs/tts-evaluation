""" A workbook to generate a batch of predictions.

Usage:
    $ PYTHONPATH=. streamlit run run/evaluate/batch_generate.py --runner.magicEnabled=false
"""
import pathlib
import random
import typing

import pandas as pd
import streamlit as st
from streamlit_datatable import st_datatable
from tqdm import tqdm

import lib
import run
from run._streamlit import (
    audio_temp_path_to_html,
    audio_to_static_temp_path,
    get_dev_dataset,
    get_static_temp_path,
    image_temp_path_to_html,
    zip_to_html,
)
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, batch_span_to_speech, package_tts
from run.data._loader import Span
from run.train.spectrogram_model._metrics import (
    get_alignment_norm,
    get_alignment_std,
    get_num_pause_frames,
    get_num_skipped,
)

st.set_page_config(layout="wide")


def make_result(span: Span) -> typing.Dict[str, str]:
    return {
        "Transcript": span.transcript,
        "Script": span.script,
        "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
        "Speaker": span.speaker.label,
        "Session": str(span.session),
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
        audio_path = audio_to_static_temp_path(span.audio())
        result = {
            "Checkpoints": "original",
            "Audio": audio_temp_path_to_html(audio_path),
            "Audio Path": str(audio_path.relative_to(lib.environment.ROOT_PATH)),
            **make_result(span),
        }
        results.append(result)

    with st.spinner("Loading and exporting model(s)..."):
        checkpoints = [CHECKPOINTS_LOADERS[c]() for c in checkpoints_keys]
        packages = [package_tts(spec, sig) for spec, sig in checkpoints]

    for package, checkpoints_ in zip(packages, checkpoints_keys):
        spans = [next(generator) for _ in tqdm(range(num_fake_clips), total=num_fake_clips)]
        with st.spinner(f"Generating clips with `{checkpoints_.name}` checkpoints..."):
            in_outs = batch_span_to_speech(package, spans)

        for span, (params, pred, audio) in zip(spans, in_outs):
            audio_path = audio_to_static_temp_path(audio)
            image_path = get_static_temp_path("alignments.png")
            lib.visualize.plot_alignments(pred.alignments).savefig(image_path)
            num_frames = pred.frames.shape[0]
            alignments = pred.alignments.unsqueeze(1)
            alignment_norm = (get_alignment_norm(alignments, None, None)[0] / num_frames).item()
            result = {
                "Checkpoints": checkpoints_.name,
                "Audio": audio_temp_path_to_html(audio_path),
                "Audio Path": str(audio_path.relative_to(lib.environment.ROOT_PATH)),
                "Frames Per Token": num_frames / params.tokens.shape[0],
                "Num Pause Frames": get_num_pause_frames(pred.frames.unsqueeze(1), None)[0],
                "Alignment Norm": alignment_norm,
                "Alignment STD": (get_alignment_std(alignments, None, None)[0] / num_frames).item(),
                "Alignment Skips": get_num_skipped(alignments, None, None)[0].item(),
                "Alignment": image_temp_path_to_html(image_path),
                **make_result(span),
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
        html = zip_to_html("audios.zip", "Download Audio(s)", paths, archive_paths)
        st.markdown(html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
