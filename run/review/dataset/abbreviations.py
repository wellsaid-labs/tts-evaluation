""" Streamlit application for reviewing abbreviations in the script and transcript.

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset/abbreviations.py \
        --runner.magicEnabled=false
"""
import logging
import random
import typing

import config as cf
import numpy
import pandas
import regex as re
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from torchnlp.random import fork_rng

import lib
import run
from lib.audio import AudioMetadata
from run._streamlit import audio_to_url, metadata_alignment_audio, st_ag_grid, st_tqdm
from run._utils import UnprocessedDataset, get_unprocessed_dataset
from run.data._loader.structures import (
    Alignment,
    Language,
    UnprocessedPassage,
    _is_stand_abbrev_consistent,
)

lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
logger = logging.getLogger(__name__)


@st.experimental_singleton()
def _get_unprocessed_dataset():
    return cf.partial(get_unprocessed_dataset)()


def _no_filter(*_):
    return True


def _is_different(script: str, transcript: str, punc: str = ".?,:!-\"'"):
    """Check iff the script and transcript token are not the same
    (aside from punc at the start or end of a token, and the casing of the first letter)."""
    script = script.strip(punc)
    transcript = transcript.strip(punc)
    if len(script) > 2 and len(transcript) > 2 and script[1:] != transcript[1:]:
        return True
    return script.lower() != transcript.lower()


TWO_UPPER_CHAR = re.compile(r"[A-Z]{2}")


def _is_stand_casing(phrase: str):
    """Check if `phrase` casing is standard.

    The casing is standard if...
    - There are no consecutive uppercase letters.
    - It's not an initialism or acronym with periods or spaces between letters.
    """
    split = phrase.split()
    if len(split) > 1 and all(len(w) == 1 for w in split):
        return False
    return all(TWO_UPPER_CHAR.search(w) is None for w in split)


def _is_casing_ambiguous(script: str, transcript: str):
    """Determine if casing is ambiguous and it needs to be filtered out
    (This was the Dec 2022 approach for validating data)."""
    script = run._config.replace_punc(script, " ", Language.ENGLISH)
    transcript = run._config.replace_punc(transcript, " ", Language.ENGLISH)
    script, transcript = script.strip(), transcript.strip()
    if len(script) == 1 and len(transcript) == 1:
        return False

    return _is_stand_casing(script) != _is_stand_casing(transcript)


@st.experimental_singleton()
def _get_alignments(
    _dataset: UnprocessedDataset, num_alignments: int, picker: str, negate: bool
) -> typing.List[typing.Tuple[UnprocessedPassage, Alignment, numpy.ndarray, AudioMetadata]]:
    """Get all alignments in dataset.

    Args:
        ...
        num_alignments: The number of alignments to sample.
        picker: The callable used to filter alignments.
        negate: Iff then the picker results are negated.

    Returns: A list of alignments and related objects.
    """
    alignments: typing.List[typing.Tuple[UnprocessedPassage, Alignment]] = []
    for _, documents in _dataset.items():
        for document in documents:
            for passage in document:
                if passage.alignments is None:
                    continue

                for alignment in passage.alignments:
                    script = passage.script[alignment.script[0] : alignment.script[1]]
                    transcript = passage.transcript[
                        alignment.transcript[0] : alignment.transcript[1]
                    ]
                    if globals()[picker](script, transcript) is not negate:
                        alignments.append((passage, alignment))

    with fork_rng():
        alignments = random.sample(alignments, min(num_alignments, len(alignments)))

    # NOTE: Sort by audio path to maximize caching when reading audio.
    key = lambda k: (
        k[0].audio_path,
        0 if k[0].alignments is None else k[0].alignments[0].audio[0],
    )
    alignments = sorted(alignments, key=key)

    with st.spinner("Reading audio metadata..."):
        audio_paths = list(set(passage.audio_path for passage, _ in alignments))
        metadatas = {a: m for a, m in zip(audio_paths, lib.audio.get_audio_metadata(audio_paths))}

    with st.spinner("Reading audio..."):
        iter_ = st_tqdm(alignments)
        return [
            (p, a, metadata_alignment_audio(metadatas[p.audio_path], a), metadatas[p.audio_path])
            for p, a in iter_
        ]


def _gather(
    passage: UnprocessedPassage, alignment: Alignment, clip: numpy.ndarray, meta: AudioMetadata
):
    """Gather data on `alignment`."""
    script = passage.script[alignment.script[0] : alignment.script[1]]
    transcript = passage.transcript[alignment.transcript[0] : alignment.transcript[1]]
    return {
        "script": script,
        "transcript": transcript,
        "clip": audio_to_url(clip, sample_rate=meta.sample_rate),
        "is_casing_ambiguous": _is_casing_ambiguous(script, transcript),
        "is_stand_abbrev_consistent": _is_stand_abbrev_consistent(script, transcript),
        "num_upper": sum(c.isupper() for c in script + transcript),
        "per_upper_transcript": sum(c.isupper() for c in transcript) / len(transcript),
        "num_punc": sum(not c.isalnum() for c in script + transcript),
        "num_numeric": sum(c.isnumeric() for c in script + transcript),
        "num_words": len(script.split()) + len(transcript.split()),
    }


def main():
    run._config.configure(overwrite=True)

    st.title("Script & Transcript Abbreviations")
    st.write("The workbook reviews the abbreviations in the script and transcript.")

    form: DeltaGenerator = st.form("settings")
    question = "How many alignments do you want to analyze?"
    # NOTE: Too many alignments could cause the `streamlit` to refresh and start over.
    num_alignments = int(form.number_input(question, 0, 10000, 3000))
    pickers_ = [_no_filter, _is_different, _is_casing_ambiguous, _is_stand_abbrev_consistent]
    pickers: typing.List[str] = [p.__name__ for p in pickers_]
    picker = typing.cast(str, form.selectbox("Picker", pickers))
    negate = form.checkbox("Negate Picker")
    if not form.form_submit_button("Submit"):
        return

    dataset = _get_unprocessed_dataset()
    alignments = _get_alignments(dataset, num_alignments, picker, negate)
    rows = [_gather(*a) for a in alignments]
    st_ag_grid(pandas.DataFrame(rows), audio_column_name="clip")


if __name__ == "__main__":
    main()
