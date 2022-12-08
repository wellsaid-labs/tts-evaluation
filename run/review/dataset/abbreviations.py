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
from run.data._loader.structures import Alignment, Language, UnprocessedPassage, _is_stand_casing

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


def _is_casing_ambiguous(script: str, transcript: str):
    """Determine if casing is ambiguous and it needs to be filtered out
    (Dec 2022 approach for validating data)."""
    script = run._config.replace_punc(script, " ", Language.ENGLISH)
    transcript = run._config.replace_punc(transcript, " ", Language.ENGLISH)
    script, transcript = script.strip(), transcript.strip()
    if len(script) == 1 and len(transcript) == 1:
        return False

    return _is_stand_casing(script) != _is_stand_casing(transcript)


# NOTE: There are some abbreviations we consider non-standard like "t-shirt", "PhD", or "Big C".
# This makes no attempt at detecting these.
STANDARD_ABBRE = re.compile(
    r"("
    # GROUP 2: Abbr seperated with dots like "a.m.".
    r"(?:\b)"  # NON-GROUP: Abbreviation starting
    r"([A-Za-z]\.){2,}"
    r"(?:\B)"  # NON-GROUP: Abbreviation ending
    r"|"
    # GROUP 3: Upper-case abbr maybe seperated other punctuation that starts on a word break
    # like "PCI-DSS", "U. S." or "W-USA".
    r"(?:\b)"
    r"((?:[A-Z]\s?[&\-\.\s*]?\s?)+(?:[A-Z]-?)*[A-Z])"
    r"(?=\b|[0-9])"
    r"|"
    # GROUP 4: Upper-case abbr like "MiniUSA.com", "fMRI" or "DirecTV".
    r"([A-Z0-9]{2,})"
    r"(?=\b|[a-z0-9])"
    r")"
)


def _get_abbr_letters(text: str):
    """Get all letters for the abbreviations in `text`."""
    return tuple(c.lower() for m in STANDARD_ABBRE.findall(text) for c in m[0] if c.isalpha())


def _is_abbrs_valid(script: str, transcript: str):
    """Check that the abbreviations in the script are in fact abbreviations in the transcript, also.

    NOTE: This ensures that if there is a standard abbreviation, both the script and transcript
          agree that it is. It is possible for non-standard abbreviations to make it through if
          both the script and transcript agree, for example "PhD" or "t-shirt".
    TODO: We want to ensure all upper-case sequences are initialisms. To do so, we'd need to have
          a list of acronyms to filter out.
    """
    return _get_abbr_letters(script) == _get_abbr_letters(transcript)


assert not _is_abbrs_valid("ABC", "abc")
assert not _is_abbrs_valid("NDAs", "nda's")
assert not _is_abbrs_valid("HAND-CUT", "hand cut")
assert not _is_abbrs_valid("NOVA/national", "Nova National")
assert not _is_abbrs_valid("I V As?", "ivas")
assert not _is_abbrs_valid("I.V.A.", "iva")
assert not _is_abbrs_valid("information...ELEVEN", "information 11")
assert not _is_abbrs_valid("(JNA)", "JN a")
assert not _is_abbrs_valid("PwC", "PWC")
assert not _is_abbrs_valid("JC PENNEY", "JCPenney")
assert not _is_abbrs_valid("DIRECTV", "DirecTV")
assert not _is_abbrs_valid("M*A*C", "Mac")
assert not _is_abbrs_valid("fMRI", "fmri")
assert not _is_abbrs_valid("RuBP.", "rubp")
assert not _is_abbrs_valid("MiniUSA.com,", "mini usa.com.")
assert not _is_abbrs_valid("7UP", "7-Up")
assert _is_abbrs_valid("NDT", "ND T")
assert _is_abbrs_valid("L.V.N,", "LVN")
assert _is_abbrs_valid("I", "i")
assert _is_abbrs_valid("PM", "p.m.")
assert _is_abbrs_valid("place...where", "place where")
assert _is_abbrs_valid("Smucker's.", "Smuckers")
assert _is_abbrs_valid("DVD-Players", "DVD players")
assert _is_abbrs_valid("PCI-DSS,", "PCI DSS.")
assert _is_abbrs_valid("UFOs", "UFO's,")
assert _is_abbrs_valid("most[JT5]", "most JT 5")
assert _is_abbrs_valid("NJ--at", "NJ. At")
assert _is_abbrs_valid("U. S.", "u.s.")
assert _is_abbrs_valid("ADHD.Some", "ADHD some")
assert _is_abbrs_valid("W-USA", "WUSA.")
assert _is_abbrs_valid("P-S-E-C-U", "PSECU")
assert _is_abbrs_valid("J. V.", "JV")
assert _is_abbrs_valid("PM", "P.m.")
assert _is_abbrs_valid("Big-C", "Big C.")
assert _is_abbrs_valid("U-Boats", "U-boats")
assert _is_abbrs_valid("Me...obsessive?...I", "me obsessive. I")
assert _is_abbrs_valid("apiece,", "A piece")
assert _is_abbrs_valid("well.I'll,", "well. I'll")
assert _is_abbrs_valid("Rain-x-car", "Rain-X car")
assert _is_abbrs_valid("L.L.Bean", "LL Bean")
assert _is_abbrs_valid("WBGP -", "W BG P.")
assert _is_abbrs_valid("KRCK", "K RC K")
assert _is_abbrs_valid("DVD-L10", "DVD L10")
assert _is_abbrs_valid("DVD-L10", "DVD, L10")
assert _is_abbrs_valid("t-shirt", "T-shirt")


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
        "casing_ambiguous": _is_casing_ambiguous(script, transcript),
        "abbrs_valid": _is_abbrs_valid(script, transcript),
        "num_upper": sum(c.isupper() for c in script + transcript),
        "per_upper_transcript": sum(c.isupper() for c in transcript) / len(transcript),
        "num_punc": sum(not c.isalnum() for c in script + transcript),
        "num_numeric": sum(c.isnumeric() for c in script + transcript),
        "num_words": len(script.split()) + len(transcript.split()),
    }


def main():
    run._config.configure(overwrite=True)

    st.title("Casing Consistency")
    st.write("The workbook reviews the casing consistency between the script and transcript.")

    form: DeltaGenerator = st.form("settings")
    question = "How many alignments do you want to analyze?"
    # NOTE: Too many alignments could cause the `streamlit` to refresh and start over.
    num_alignments = int(form.number_input(question, 0, 10000, 3000))
    pickers_ = [_no_filter, _is_different, _is_casing_ambiguous, _is_abbrs_valid]
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
