"""A workbook to generate a batch of predictions with griffin lim.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/batch_griffin_lim.py --runner.magicEnabled=false
"""
import typing

import config as cf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from streamlit.delta_generator import DeltaGenerator

import lib
import run
from lib.audio import amp_to_db, db_to_power, power_spectrogram_to_framed_rms, power_to_db
from lib.environment import PT_EXTENSION, load
from run import _streamlit as _st
from run._config import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._config.train import _config_spec_model_training
from run._models.spectrogram_model import Preds
from run._tts import batch_griffin_lim_tts, make_training_batches
from run.data._loader import Span
from run.train.spectrogram_model._data import Batch
from run.train.spectrogram_model._metrics import (
    get_alignment_norm,
    get_alignment_std,
    get_hang_time,
    get_num_pause_frames,
    get_power_rms_level_sum,
)
from run.train.spectrogram_model._worker import Checkpoint

st.set_page_config(layout="wide")


@st.cache_resource()
def _make_examples(spec_path, num_samples, include_dic):
    """Make examples and cache results."""
    with st.spinner("Loading and exporting model(s)..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spec_ckpt = typing.cast(Checkpoint, load(spec_path, device))
        spec_export = spec_ckpt.export()
        spec_export.allow_unk_on_eval(True)

    datasets = _st.get_datasets()
    spans = _st.get_spans(*datasets, num_samples, is_train_spans=False, include_dic=include_dic)

    with st.spinner("Generating clips..."):
        batches = make_training_batches(spans, batch_size=32 if torch.cuda.is_available() else 8)
        results = batch_griffin_lim_tts(spec_export, batches, iterations=5)

    return spans, results


def get_loudness(db_spectrogram: torch.Tensor, mask: torch.Tensor):
    """Count the number of silent frames at the end of `db_spectrogram`.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])

    Returns:
        (torch.FloatTensor [batch_size, num_frames])
    """
    # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
    power_frames = db_to_power(db_spectrogram).transpose(0, 1)
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    frame_rms_level = cf.partial(power_spectrogram_to_framed_rms)(power_frames)
    return amp_to_db(frame_rms_level) * mask


def get_num_hang_sil_frames(
    db_spectrogram: torch.Tensor, mask: torch.Tensor, min_loudness: float
) -> torch.Tensor:
    """Count the number of silent frames at the end of `db_spectrogram`.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])
        min_loudness: The minimum loudness speech can be.
    """
    is_speech = (get_loudness(db_spectrogram, mask) >= min_loudness) * mask
    idx_last_speech_frame = mask.shape[1] - is_speech.flip(dims=[1]).long().argmax(dim=1)
    return mask.sum(dim=1) - idx_last_speech_frame


def get_avg_loudness(frames: torch.Tensor, mask: torch.Tensor, num_frames: torch.Tensor):
    """Get the average loudness for `frames`.

    Args:
        frames (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])
        num_frames (torch.FloatTensor [batch_size])

    Returns:
        (torch.FloatTensor [batch_size])
    """
    avg_loudness = get_power_rms_level_sum(frames, mask)
    return power_to_db(avg_loudness / num_frames)


def _gather(
    span: Span, batch: Batch, pred: Preds, audio: np.ndarray
) -> typing.Dict[str, typing.Any]:
    """Gather various statistics, and report them."""
    num_frames = pred.frames.shape[0]
    gold_num_frames = batch.spectrogram.lengths[0].item()
    spec_mask = batch.spectrogram_mask.tensor.transpose(0, 1)

    # TODO: Use the `sesh.loudness` to create a dynamic silence threshold. The issue with this
    # is that our average loudness per session is in LUFS whilst our spectrograms which we
    # can use to compute loudness are ISO 226 weighted with a Hann window.
    sil_thresh = -50
    num_pause_frames = cf.call(
        get_num_pause_frames,
        pred.frames,
        pred.frames_mask,
        max_loudness=sil_thresh,
        _overwrite=True,
    )
    num_hang_sil = get_num_hang_sil_frames(pred.frames, pred.frames_mask, sil_thresh)[0].item()
    num_hang_stop = cf.partial(get_hang_time)(pred.stop_tokens, pred.frames_mask, pred.num_frames)
    num_hang_stop = num_hang_stop[0].item()
    loudness = get_loudness(pred.frames, pred.frames_mask)[0][:num_frames]
    gold_loudness = get_loudness(batch.spectrogram.tensor, spec_mask)[0][:gold_num_frames]
    num_sil_frames = (loudness < sil_thresh).sum().item()
    num_gold_sil_frames = (gold_loudness < sil_thresh).sum().item()
    num_trimmed_frames = num_frames - num_hang_stop
    num_non_sil_frames = num_frames - num_sil_frames
    num_gold_non_sil_frames = gold_num_frames - num_gold_sil_frames

    # TODO: Let's consider summing and reporting some average metrics, also, that would be
    # in-line with the comet metrics we report.
    row = {
        "Xml": batch.xmls[0],
        "Audio": _st.audio_to_url(audio),
        "Gold Audio": _st.audio_to_url(span.audio()),
        "Alignment": _st.figure_to_url(lib.visualize.plot_alignments(pred.alignments[:, 0])),
        "Stop Token": _st.figure_to_url(lib.visualize.plot_logits(pred.stop_tokens[:, 0])),
        "Loudness": _st.figure_to_url(lib.visualize.plot_loudness(loudness)),
        "Loudness [Gold]": _st.figure_to_url(lib.visualize.plot_loudness(gold_loudness)),
        "Session Avg Loudness": span.session.loudness,
        "Session Avg Tempo": span.session.tempo,
        "Num Pause Frames": num_pause_frames[0],
    }

    avg_loudness = get_avg_loudness(pred.frames, pred.frames_mask, pred.num_frames)
    gold_avg_loudness = get_avg_loudness(batch.spec.tensor, spec_mask, batch.spec_mask.lengths)
    avg_loudness, gold_avg_loudness = avg_loudness[0].item(), gold_avg_loudness[0].item()
    num_sil_frames_ratio = (
        None if num_gold_sil_frames == 0 else num_sil_frames / num_gold_sil_frames
    )

    row = {
        **row,
        "Alignment Norm": (get_alignment_norm(pred)[0] / num_frames).item(),
        "Alignment STD": (get_alignment_std(pred)[0] / num_frames).item(),
        "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
        "Average Loudness [Gold]": gold_avg_loudness,
        "Average Loudness Delta": gold_avg_loudness - avg_loudness,
        "Average Loudness": avg_loudness,
        "Frames Per Token": num_frames / pred.num_tokens[0].item(),
        "Num Frames [Gold]": gold_num_frames,
        "Num Frames Ratio": num_frames / gold_num_frames,
        "Num Frames": num_frames,
        "Num Hanging Frames [Loudness]": num_hang_sil,
        "Num Hanging Frames [Stop Token]": num_hang_stop,
        "Num Hanging Frames Delta": num_hang_sil - num_hang_stop,
        "Num Non-Silent Frames [Gold]": num_gold_non_sil_frames,
        "Num Non-Silent Frames Ratio": num_non_sil_frames / num_gold_non_sil_frames,
        "Num Non-Silent Frames": num_non_sil_frames,
        "Num Silent Frames [Gold]": num_gold_sil_frames,
        "Num Silent Frames Delta": num_sil_frames - num_gold_sil_frames,
        "Num Silent Frames Ratio": num_sil_frames_ratio,
        "Num Silent Frames": num_sil_frames,
        "Num Trimmed Frames Ratio": num_trimmed_frames / gold_num_frames,
        "Num Trimmed Frames": num_trimmed_frames,
    }

    row = {
        **row,
        "Language": span.speaker.dialect.value[1],
        "Script": span.script,
        "Session": span.session.label,
        "Speaker": span.speaker.label,
        "Transcript": span.transcript,
    }

    return row


def main():
    st.markdown("# Batch Griffin-Lim Generation")
    st.markdown("Use this workbook to generate a batch of clips and measure them.")
    run._config.configure(overwrite=True)
    # NOTE: The various parameters map to configurations that are not relevant for this workbook.
    _config_spec_model_training(0, 0, 0, 0, 0, False, overwrite=True)

    form: DeltaGenerator = st.form("form")
    label = "Spectrogram Checkpoint(s)"
    spec_path = _st.st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
    num_samples = int(form.number_input("Num of Sample(s)", min_value=0, value=512, step=1))
    include_dic = form.checkbox("Include Dictionary Dataset(s)", value=False)
    if not form.form_submit_button("Generate"):
        return

    spans, results = _make_examples(spec_path, num_samples, include_dic)

    rows = []
    for span, (batch, pred, audio) in _st.st_tqdm(zip(spans, results), total=len(spans)):
        # TODO: Use more generic typing so that this assertion isn't needed.
        assert isinstance(batch, Batch)
        rows.append(_gather(span, batch, pred, audio))

    # TODO: Automatically find columns with PNG or WAV file paths, and display them.
    img_cols = ["Alignment", "Stop Token", "Loudness", "Loudness [Gold]"]
    _st.st_ag_grid(pd.DataFrame(rows), ["Audio", "Gold Audio"], img_cols)

    st.header("Average Relative Speed Distribution")
    total_num_gold_frames = sum([r["Num Frames [Gold]"] for r in rows])
    total_num_frames = sum([r["Num Frames"] for r in rows])
    total_num_trimmed_frames = sum([r["Num Trimmed Frames"] for r in rows])
    total_num_sil_frames = sum([r["Num Silent Frames"] for r in rows])
    total_num_gold_sil_frames = sum([r["Num Silent Frames [Gold]"] for r in rows])
    total_num_non_sil_frames = sum([r["Num Non-Silent Frames"] for r in rows])
    total_num_gold_non_sil_frames = sum([r["Num Non-Silent Frames [Gold]"] for r in rows])
    st.info(
        f"Num Frames Ratio: {total_num_frames / total_num_gold_frames}\n\n"
        f"Num Frames Delta: {total_num_frames - total_num_gold_frames}\n\n"
        f"Num Trimmed Frames Ratio: {total_num_trimmed_frames / total_num_gold_frames}\n\n"
        f"Num Silent Frames Ratio: {total_num_sil_frames / total_num_frames}\n\n"
        f"Num Gold Silent Frames Ratio: {total_num_gold_sil_frames / total_num_gold_frames}\n\n"
        f"Num Fake vs Gold Silent Frames Ratio: "
        f"{total_num_sil_frames / total_num_gold_sil_frames}\n\n"
        f"Num Fake vs Gold Silent Frames Delta: "
        f"{total_num_sil_frames - total_num_gold_sil_frames}\n\n"
        f"Num Non-Silent Frames Ratio: "
        f"{total_num_non_sil_frames / total_num_gold_non_sil_frames}\n\n"
    )
    for feat in (
        "Num Frames Ratio",
        "Num Trimmed Frames Ratio",
        "Num Non-Silent Frames Ratio",
        "Num Silent Frames Ratio",
    ):
        vals = [r[feat] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, xbins=dict(size=0.02)))
        fig.update_layout(
            xaxis_title_text=feat,
            yaxis_title_text="Count",
            margin=dict(b=0, l=0, r=0, t=0),
            bargap=0.2,
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
