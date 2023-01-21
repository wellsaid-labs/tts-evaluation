"""A workbook to generate a batch of predictions with griffin lim.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/batch_griffin_lim.py --runner.magicEnabled=false
"""
import typing

import config as cf
import pandas as pd
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
from run._tts import batch_griffin_lim_tts, make_training_batches
from run.train.spectrogram_model import _metrics
from run.train.spectrogram_model._worker import Checkpoint

st.set_page_config(layout="wide")


@st.experimental_singleton()
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
        results = batch_griffin_lim_tts(spec_export, batches, iterations=15)

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


def get_num_hanging_silent_frames(
    db_spectrogram: torch.Tensor, mask: torch.Tensor, min_loudness: float, **kwargs
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


def main():
    st.markdown("# Batch Griffin-Lim Generation")
    st.markdown("Use this workbook to generate a batch of clips and measure them.")
    run._config.configure(overwrite=True)
    # NOTE: The various parameters map to configurations that are not relevant for this workbook.
    _config_spec_model_training(0, 0, 0, 0, 0, False, overwrite=True)

    form: DeltaGenerator = st.form("form")
    label = "Spectrogram Checkpoint(s)"
    spec_path = _st.st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
    num_samples = int(form.number_input("Number of Sample(s)", min_value=0, value=512, step=1))
    include_dic = form.checkbox("Include Dictionary Dataset(s)", value=False)
    if not form.form_submit_button("Generate"):
        return

    spans, results = _make_examples(spec_path, num_samples, include_dic)

    rows = []
    for span, (batch, pred, audio) in _st.st_tqdm(zip(spans, results), total=len(spans)):
        num_frames = pred.frames.shape[0]
        gold_num_frames = batch.spectrogram.lengths[0].item()
        spec_mask = batch.spectrogram_mask.tensor.transpose(0, 1)

        avg_loudness = _metrics.get_power_rms_level_sum(pred.frames, pred.frames_mask)
        avg_loudness = power_to_db((avg_loudness[0] / num_frames).item())
        gold_avg_loudness = _metrics.get_power_rms_level_sum(batch.spectrogram.tensor, spec_mask)
        gold_avg_loudness = power_to_db((gold_avg_loudness[0] / gold_num_frames).item())

        # TODO: Incorperate a dynamic silence threshold into our metrics, since our speakers,
        # have such variable loudnesses.
        silence_threshold = span.session.loudness - 30
        num_pause_frames = cf.call(
            _metrics.get_num_pause_frames,
            pred.frames,
            pred.frames_mask,
            max_loudness=silence_threshold,
        )
        num_hanging_silent = get_num_hanging_silent_frames(
            pred.frames, pred.frames_mask, silence_threshold
        )
        loudness = get_loudness(pred.frames, pred.frames_mask)

        # TODO: Incorperate "stop token hangtime" which is how long the model keeps generating after
        # the probability of stopping is greater than `stop_prob_threshold`.
        stop_prob_threshold = 0.03
        first_stop_option = torch.sigmoid(pred.stop_tokens[:, 0]) >= stop_prob_threshold
        first_stop_option = first_stop_option.long().argmax().item()

        # TODO: Let's consider summing and reporting some average metrics, also, that would be
        # in-line with the comet metrics we report.
        row = {
            "Xml": batch.xmls[0],
            "Audio": _st.audio_to_url(audio),
            "Gold Audio": _st.audio_to_url(span.audio()),
            "Alignment": _st.figure_to_url(lib.visualize.plot_alignments(pred.alignments[:, 0])),
            "Stop Token": _st.figure_to_url(lib.visualize.plot_logits(pred.stop_tokens[:, 0])),
            "Loudness": _st.figure_to_url(lib.visualize.plot_loudness(loudness[0])),
            "Session Avg Loudness": span.session.loudness,
            "Session Avg Tempo": span.session.tempo,
            "Num Pause Frames": num_pause_frames[0],
            "Num Hanging Silent Frames": num_hanging_silent[0].item(),
            "Stop Token Hang Time": num_frames - first_stop_option,
            "Hang Time Diff": abs(num_hanging_silent[0].item() - (num_frames - first_stop_option)),
            "Alignment Hang Time": _metrics.get_alignment_hang_time(pred)[0].item(),
            "Loudness Diff": gold_avg_loudness - avg_loudness,
            "Audio Length Diff": gold_num_frames / num_frames,
            "Audio Length Diff (minus hang time)": gold_num_frames / first_stop_option,
            "Frames Per Token": num_frames / pred.num_tokens[0].item(),
            "Average Loudness": avg_loudness,
            "Gold Average Loudness": gold_avg_loudness,
            "Alignment Norm": (_metrics.get_alignment_norm(pred)[0] / num_frames).item(),
            "Alignment STD": (_metrics.get_alignment_std(pred)[0] / num_frames).item(),
            "Alignment Reached": _metrics.get_alignment_was_aligned(pred).long()[0].item(),
            "Transcript": span.transcript,
            "Script": span.script,
            "Audio File": str(span.audio_file.path.relative_to(lib.environment.ROOT_PATH)),
            "Language": span.speaker.dialect.value[1],
            "Speaker": span.speaker.label,
            "Session": span.session.label,
        }
        rows.append(row)

    # TODO: Automatically find columns with PNG or WAV file paths, and display them.
    img_cols = ["Alignment", "Stop Token", "Loudness"]
    _st.st_ag_grid(pd.DataFrame(rows), ["Audio", "Gold Audio"], img_cols)


if __name__ == "__main__":
    main()
