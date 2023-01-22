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
from run._models.spectrogram_model import Preds
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


def _get_token_idx_frames(preds: Preds, token_idx: int = 2, threshold: float = 0.2) -> torch.Tensor:
    """Given `alignments` from frames to tokens, get a bool tensor where it's true if frame was
    aligned to `preds.num_tokens - token_idx` determined by reaching or exceeding a `threshold`
    of focus.

    Returns:
        torch.BoolTensor [batch_size, num_frames]
    """
    num_frames = preds.alignments.shape[0]
    batch_size = preds.alignments.shape[1]
    if preds.alignments.numel() == 0:
        return torch.zeros(batch_size, num_frames, device=preds.alignments.device)

    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    # [batch_size] → [batch_size, 1]
    token_idx_ = (preds.num_tokens - token_idx).view(1, -1, 1).expand(num_frames, batch_size, 1)
    # [num_frames, batch_size, num_tokens] → [num_frames, batch_size]
    frames = torch.gather(alignments, dim=2, index=token_idx_).squeeze(2) > threshold
    # [num_frames, batch_size] → [batch_size, num_frames]
    frames = frames.transpose(0, 1)
    return frames.masked_fill(~preds.frames_mask, 0)


def get_alignment_was_aligned(preds: Preds, **kwargs) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this gets the number of sequences where a frame
    was aligned with `preds.num_tokens - token_idx`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    token_idx_frames = _get_token_idx_frames(preds, **kwargs)
    return token_idx_frames.sum(dim=1) != 0


def get_alignment_hang_time(preds: Preds, **kwargs) -> torch.Tensor:
    """Given `alignments` from frames to tokens, the gets the number of frames after
    `preds.num_tokens - token_idx` has been reached.

    TODO: This metric could be adjusted to match our stop token metric, which is based off anytime
    the model is focused on `token_idx` or any token higher than that.

    Returns:
        torch.FloatTensor [batch_size]
    """
    # [batch_size, num_frames]
    token_idx_frames = _get_token_idx_frames(preds, **kwargs)
    if token_idx_frames.numel() == 0:
        return torch.empty(0, device=token_idx_frames.device)

    # NOTE: Get the first frame where frames is focused on `token_idx`.
    # [batch_size, num_frames] → [batch_size]
    frame_idx = token_idx_frames.long().argmax(dim=1)
    aligned = token_idx_frames.sum(dim=1) != 0
    return (preds.num_frames - frame_idx - 1).float() * aligned


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
            "Alignment Hang Time": get_alignment_hang_time(pred)[0].item(),
            "Loudness Diff": gold_avg_loudness - avg_loudness,
            "Audio Length Diff": gold_num_frames / num_frames,
            "Audio Length Diff (minus hang time)": gold_num_frames / first_stop_option,
            "Frames Per Token": num_frames / pred.num_tokens[0].item(),
            "Average Loudness": avg_loudness,
            "Gold Average Loudness": gold_avg_loudness,
            "Alignment Norm": (_metrics.get_alignment_norm(pred)[0] / num_frames).item(),
            "Alignment STD": (_metrics.get_alignment_std(pred)[0] / num_frames).item(),
            "Alignment Reached": get_alignment_was_aligned(pred).long()[0].item(),
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
