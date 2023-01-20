""" A workbook to generate a batch of predictions with griffin lim.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/batch_griffin_lim.py --runner.magicEnabled=false
"""
import typing

import config as cf
import pandas as pd
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator
from tqdm import tqdm

import lib
import run
from lib.environment import PT_EXTENSION, load
from run import _streamlit as _st
from run._config import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._tts import batch_griffin_lim_tts, make_training_batches
from run.train.spectrogram_model import _metrics
from run.train.spectrogram_model._worker import Checkpoint

st.set_page_config(layout="wide")


def main():
    st.markdown("# Batch Generation ")
    st.markdown("Use this workbook to generate a batch of clips and export them.")
    run._config.configure(overwrite=True)

    form: DeltaGenerator = st.form("form")

    label = "Spectrogram Checkpoints"
    spec_path = _st.st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION)
    num_samples = int(form.number_input("Number of Samples", min_value=0, value=512, step=1))
    include_dic = form.checkbox("Include Dictionary Dataset", value=False)
    if not form.form_submit_button("Generate"):
        return

    with st.spinner("Loading and exporting model(s)..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec_ckpt = typing.cast(Checkpoint, load(spec_path, device))
        spec_export = spec_ckpt.export()

    # TODO: Use `_get_spans` which I've verified to be similar to the training pipeline
    datasets = _st.get_datasets()
    spans = _st.get_spans(*datasets, num_samples, is_train_spans=False, include_dic=include_dic)

    with st.spinner("Generating clips..."):
        batches = make_training_batches(spans)
        results = batch_griffin_lim_tts(spec_export, batches)

    rows = []
    for span, (batch, pred, audio) in tqdm(zip(spans, results), total=len(spans)):
        image_web_path = _st.make_temp_web_dir() / "alignments.png"
        lib.visualize.plot_alignments(pred.alignments[:, 0]).savefig(str(image_web_path))
        num_frames = pred.frames.shape[0]
        gold_num_frames = batch.spectrogram.lengths[0]
        loudness = _metrics.get_power_rms_level_sum(pred.frames, pred.frames_mask)
        loudness = (loudness[0] / num_frames).item()
        spec_mask = batch.spectrogram_mask.tensor.transpose(0, 1)
        gold_loudness = _metrics.get_power_rms_level_sum(batch.spectrogram.tensor, spec_mask)
        gold_loudness = (gold_loudness[0] / gold_num_frames).item()
        num_pause_frames = cf.partial(_metrics.get_num_pause_frames)(pred.frames, None)
        row = {
            "Frames Per Token": num_frames / pred.num_tokens[0].item(),
            "Num Pause Frames": num_pause_frames[0],
            "Loudness": loudness,
            "Gold Loudness": gold_loudness,
            "Loudness Diff": gold_loudness - loudness,
            "Audio Length Diff": gold_num_frames / num_frames,
            "Alignment Norm": (_metrics.get_alignment_norm(pred)[0] / num_frames).item(),
            "Alignment STD": (_metrics.get_alignment_std(pred)[0] / num_frames).item(),
            "Alignment Hang Time": _metrics.get_alignment_hang_time(pred)[0].item(),
            "Alignment Reached": _metrics.get_alignment_was_aligned(pred)[0].item(),
            "Alignment": f'<img src="{_st.web_path_to_url(image_web_path)}" />',
            "Transcript": span.transcript,
            "Script": span.script,
            "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
            "Language": span.speaker.dialect.value[1],
            "Speaker": span.speaker.label,
            "Session": str(span.session),
            "Audio": _st.audio_to_url(audio),
            "Gold Audio": _st.audio_to_url(span.audio()),
        }
        rows.append(row)

    _st.st_ag_grid(pd.DataFrame(results), ["Predicted Audio", "Original Audio"])


if __name__ == "__main__":
    main()
