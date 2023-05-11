""" A workbook to generate audio for evaluating various test cases.

TODO: Create a script that downloads multiple checkpoints at various points, generates scripts
      with them, and produces a zip file. We can use `disk/envs` to get the information I need
      to generating something like this.
TODO: Instead of using random speakers and sessions, let's consider using the choosen session
      and speakers in `deploy.sh`. Those will be deployed, anyways.
TODO: Implement `batch_griffin_lim_tts` to support batch generation, speeding up this script.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/test_cases.py --runner.magicEnabled=false
"""

import random
import string
import typing
from functools import partial

import config as cf
import numpy as np
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from torchnlp.random import fork_rng

import lib
import run
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType
from run._config import DEFAULT_SCRIPT, SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._config.data import _get_loudness_annotation, _get_tempo_annotation
from run._models.spectrogram_model import SpectrogramModel
from run._streamlit import (
    audio_to_web_path,
    st_download_files,
    st_html,
    st_select_path,
    web_path_to_url,
)
from run._tts import griffin_lim_tts
from run.data._loader import Speaker
from run.data._loader.english import wsl
from run.review.tts.test_cases import TEST_CASES


def generate_test_cases(
    spec_export: SpectrogramModel, test_cases: typing.List[str], seed: int = 123
):
    with fork_rng(seed):
        vocab = sorted(list(spec_export.session_embed.get_vocab()))
        seshs = [random.choice(vocab) for case in test_cases]

    for sesh, case in zip(seshs, test_cases):
        st.info(f"Seshion: {sesh}\n\nScript: {case}")
        yield griffin_lim_tts(spec_export, XMLType(case), sesh)


Generator = typing.Callable[[SpectrogramModel], typing.Generator[np.ndarray, None, None]]
OPTIONS: typing.Dict[str, Generator]
OPTIONS = {k: partial(generate_test_cases, test_cases=v) for k, v in TEST_CASES.items()}

# TODO: Create a notebook where we go through all the speakers, on various tempos/loudnesses, and
# then review the accuracy.


def generate_annos(
    spec_export: SpectrogramModel,
    anno: typing.Tuple[str, typing.Sequence],
    speakers: typing.Sequence[Speaker] = [
        wsl.TRISTAN_F,  # NOTE: They have a low annotation range around ±2db & ±25%.
        wsl.GIA_V,  # NOTE: They have a low annotation range around ±2db & ±25%.
        wsl.DIARMID_C,  # NOTE: They have a high annotation range around ±4db & ±45%.
        wsl.GARRY_J__STORY,  # NOTE: They have a high annotation range around ±4db & ±45%.
        # NOTE: They have an average range around ±3db & ±30% and have been difficult to work with.
        wsl.JUDE_D__EN_GB,
    ],
):
    for speaker in speakers:
        sesh_vocab = spec_export.session_embed.get_vocab()
        sesh = random.choice([s for s in sesh_vocab if s.spkr == speaker])
        st.info(f"Session: {sesh}")
        tag, range = anno
        for val in range:
            if tag == "tempo":
                val = sesh.spkr_tempo + val
            xml = XMLType(f"<{tag} value='{val}'>{DEFAULT_SCRIPT}</{tag}>")
            wave = griffin_lim_tts(spec_export, xml, sesh)
            audio_len = cf.partial(lib.audio.sample_to_sec)(wave.shape[0])
            tempo = cf.partial(_get_tempo_annotation)(DEFAULT_SCRIPT, audio_len)
            loudness = cf.partial(_get_loudness_annotation)(wave)
            st.info(
                f"- Tag: {tag}={val}\n"
                f"- Tempo: {tempo}\n"
                f"- Generated Griffin-Lim Loudness: {loudness}\n"
            )
            # TODO: Add a loundess computed via spectrogram.
            # TODO: Use a some signal model, and then measure the loudness based on that.
            yield wave


OPTIONS = {
    "LOUDNESS": partial(generate_annos, anno=("loudness", list(lib.utils.arange(6, -7, -3)))),
    "TEMPO": partial(generate_annos, anno=("tempo", list(lib.utils.arange(-0.5, 0.6, 0.25)))),
    **OPTIONS,
}


def main():
    st.markdown("# Test Case Audio Generator")
    st.markdown("Use this workbook to generate batches of audio for evaluating our test cases.")
    run._config.configure(overwrite=True)

    form: DeltaGenerator = st.form(key="form")

    label = "Spectrogram Checkpoints"
    spec_path = st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
    items = sorted(OPTIONS.items())
    format_test_case_name = lambda i: i[0].replace("_", " ").title()
    option = form.selectbox("Test Cases", items, format_func=format_test_case_name)
    assert option is not None

    if not form.form_submit_button("Submit"):
        return

    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))
    spec_export = spec_ckpt.export()

    with st.spinner("Generating audio..."):
        paths = []
        for wave in option[1](spec_export):
            paths.append(audio_to_web_path(wave))
            st_html(f'<audio controls src="{web_path_to_url(paths[-1])}"></audio>')

    with st.spinner("Making Zipfile..."):
        st.text("")
        st_download_files("Audios.zip", "📁 Download Audio(s) (zip)", paths)

    st.success(f"Finished! {lib.utils.mazel_tov()}")


if __name__ == "__main__":
    main()
