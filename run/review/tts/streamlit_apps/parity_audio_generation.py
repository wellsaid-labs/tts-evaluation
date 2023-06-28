""" A workbook to generate audio for quick evaluations.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/parity_audio_generation.py --runner.magicEnabled=false
"""
import random
import typing
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import lib
import run
from lib.environment import PT_EXTENSION, ROOT_PATH, load
from lib.text import natural_keys
from lib.text.utils import XMLType
from run._models.spectrogram_model import SpectrogramModel
from run._streamlit import audio_to_web_path, make_temp_web_dir, st_download_files
from run._tts import griffin_lim_tts
from run.data._loader import Session
from run.deploy.worker import _MARKETPLACE
from run.review.tts.parity_test_cases import PARITY_TEST_CASES

SPEC_MODEL_EXP_PATH = ROOT_PATH / "disk" / "experiments" / "spectrogram_model"

SPEAKER_IDS = [
    3,
    4,
    5,
    7,
    8,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    # 33, JUDE D is inexplicably missing from the V10 checkpoint
    34,
    35,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    71,
    72,
]


def path_label(path: Path) -> str:
    """Get a short label for `path`."""
    return str(path.relative_to(ROOT_PATH)) + "/" if path.is_dir() else str(path.name)


def st_select_path(
    label: str,
    dir: Path,
    suffix: str,
    st: DeltaGenerator = typing.cast(DeltaGenerator, st),
) -> typing.Optional[Path]:
    """Display a path selector for the directory `dir`."""
    options = [p for p in dir.glob("**/*") if p.suffix == suffix]
    options = sorted(options, key=lambda x: natural_keys(str(x)), reverse=True)
    selected = st.selectbox(label, options=options, format_func=path_label)
    return typing.cast(typing.Optional[Path], selected)


def generate_test_cases(
    spec_export: SpectrogramModel, test_cases: typing.List[str], seed: int = 123
):
    # with fork_rng(seed):
    spk_sessions: typing.List[Session]
    session_vocab = list(set(spec_export.session_embed.vocab.keys()))
    spk_sessions = [Session(*args) for i, args in _MARKETPLACE.items() if i in SPEAKER_IDS]
    for sp_s in spk_sessions:
        if sp_s not in session_vocab:
            st.write(f"Speaker missing: {sp_s}")

    for case in test_cases:
        sesh = random.choice(spk_sessions)
        yield (sesh, case, griffin_lim_tts(spec_export, XMLType(case), sesh))


OPTIONS = {k: partial(generate_test_cases, test_cases=v) for k, v in PARITY_TEST_CASES.items()}


def main():
    st.markdown("# Griffin Lim Audio Generator")
    st.markdown("Use this workbook to generate griffin lim audio for quick evaluation.")
    run._config.configure(overwrite=True)

    label = "Spectrogram Checkpoints"
    spec_path = st_select_path(label, SPEC_MODEL_EXP_PATH, PT_EXTENSION)
    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))
    spec_export = spec_ckpt.export()

    test_cases = sorted(OPTIONS.items())

    form = st.form(key="form")

    if not form.form_submit_button("Submit"):
        return

    teammates = ["Julia", "Jordan", "Alison", "Vic", "Sara", "Rhyan", "Alecia", "Michael"]
    for mate in teammates:
        with st.spinner(f"Generating surveys for {mate}..."):
            for option in test_cases:
                case_name = option[0]
                audios, paths, rows = [], [], []

                audios.extend(option[1](spec_export))

                for i, (sesh, script, wave) in enumerate(audios):
                    path = audio_to_web_path(wave, name=(f"{case_name}_audio{i}.wav"))
                    paths.append(path)

                    speaker = sesh[0]
                    row = {
                        "Speaker": speaker.label,
                        "Style": speaker.style.value,
                        "Script": script,
                        "Session": sesh[1],
                        "Dialect": speaker.dialect.value[1],
                        "Audio": path.name,
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)
                metadata_path = make_temp_web_dir() / "metadata.csv"
                df.to_csv(metadata_path, index=False)
                paths.append(metadata_path)

                with st.spinner("Making Zipfile..."):
                    st.text("")
                    label = f"📁 Download {mate} ({case_name})"
                    v11_name = case_name.replace("10", "11")
                    st_download_files(f"{v11_name}_{mate}.zip", label, paths)

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
