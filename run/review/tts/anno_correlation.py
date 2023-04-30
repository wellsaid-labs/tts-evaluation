""" A workbook that correlates annotations with the output.

TODO: Make sure the correlation has no bias.

Usage:
    $ PYTHONPATH=. streamlit run run/review/tts/anno_correlation.py --runner.magicEnabled=false
"""
import random
import typing

import config as cf
import pandas
import plotly.express as px
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import lib
import run
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType, xml_to_text
from run._config import SPECTROGRAM_MODEL_EXPERIMENTS_PATH
from run._config.data import _get_loudness_annotation, _get_tempo_annotation
from run._models.spectrogram_model import SpectrogramModel
from run._streamlit import audio_to_url, st_ag_grid, st_select_path, st_set_page_config
from run._tts import batch_griffin_lim_tts, make_batches

st_set_page_config()

TEST_SCRIPTS = [
    # NOTE: These questions each have a different expected inflection.
    "If you can instantly become an expert in something, what would it be?",
    "What led to the two of you having a disagreement?",
    "If you could rid the world of one thing, what would it be?",
    "How common are HSPs?",
    "There may be times where you have to RDP to a node and manually collect logs for some "
    "reason. So, another question you may have is, exactly where on disk are all these logs?",
    # NOTE: All these questions should have an upward inflection at the end.
    "Can our AI say AI?",
    "When deciding between organic and non-organic coffees, is the price premium worth it?",
    "Can you make yourself disappear?",
    "Have you ever been stalked by an animal that later became your pet?",
    "If you have made it this far, do you relate to any of these signs? Are you a highly "
    "sensitive person?",
    # NOTE: These have various initialisms.
    "Each line will have GA Type as Payment, Paid Amount along with PAC, and GA Code.",
    "Properly use and maintain air-line breathing systems and establish a uniform procedure "
    "for all employees, for both LACC and LCLA contractors, to follow when working jobs that "
    "require the use of fresh air.",
    "QCBS is a method of selecting transaction advisors based on both the quality of their "
    "technical proposals and the costs shown in their financial proposals.",
    "HSPs account for fifteen to twenty percent of the population.",
    "We used to have difficulty with AOA and AMA, but now we are A-okay.",
    "As far as AIs go, ours is pretty great!",
    # NOTE: These statements have a mix of heteronyms, initialisms, hard words (locations,
    # medical terms, technical terms), etc for testing pronunciation.
    "For more updates on covid nineteen, please contact us via the URL at the bottom of the "
    "screen, or visit our office in Seattle at the address shown here.",
    "I've listed INTJ on my resume because it's important for me that you understand how I "
    "conduct myself in stressful situations.",
    "The website is live and you can access your records via the various APIs slash URLs or use "
    "the Studio as an alternate avenue.",
    "The nurses will resume the triage conduct around the oropharyngeal and test for "
    "tachydysrhythmia to ensure the patient lives another day.",
    "Access your clusters using the Kubernetes API. You can alternate between the CLI and the "
    "web interface.",
    "Live from Seattle, it's AIQTV, with the governor's special address on the coronavirus. Don't "
    "forget to record this broadcast for viewing later.",
    "Let's add a row on our assay tracking sheet so we can build out the proper egress "
    "measurements.",
    "Hello! Can you put this contractor into a supervisory role?",
]


def _correlate(name: str, values: typing.List[float], other: str, other_values: typing.List[float]):
    """Correlate two sets of values and visualize it."""
    df = pandas.DataFrame({name: values, other: other_values})
    trendline = dict(
        trendline="ols",
        trendline_options={"add_constant": False},
        trendline_color_override=px.colors.qualitative.Plotly[8],
    )
    fig = px.scatter(df, x=name, y=other, **trendline)
    fig.update_layout(margin=dict(b=0, l=0, r=0, t=0))
    results = px.get_trendline_results(fig)
    st.markdown(f"#### Correlation `{name}` vs `{other}`")
    st.info(
        f"Stats:\n"
        f"- Slope: **{results.px_fit_results.iloc[0].params[0]}**\n"
        f"- R-Squared: **{results.px_fit_results.iloc[0].rsquared}**\n"
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource()
def _generate(
    _spec_exp: SpectrogramModel,
    experiment_key: str,  # NOTE: This is largely only used for cache invalidation.
    exp_step: int,  # NOTE: This is largely only used for cache invalidation.
    num_samples: int,
    min_loudness: float,
    max_loudness: float,
    min_tempo: float,
    max_tempo: float,
):
    sesh_vocab = _spec_exp.session_embed.get_vocab()
    inputs = []
    expected = []
    for _ in range(num_samples):
        script = random.choice(TEST_SCRIPTS)
        spkr = random.choice(list(run._config.data.DEV_SPEAKERS))
        seshs = [s for s in sesh_vocab if s.spkr == spkr]
        if len(seshs) == 0:
            st.warning(f"Skipping `{spkr}` not found in model vocab.")
            continue
        sesh = random.choice(seshs)
        rel_loudness = random.uniform(min_loudness, max_loudness)
        expected_loudness = sesh.loudness + rel_loudness
        rel_tempo = random.uniform(min_tempo, max_tempo)
        expected_tempo = rel_tempo * sesh.tempo
        xml = f"<loudness value='{rel_loudness}'><tempo value='{rel_tempo}'>"
        xml += f"{script}</tempo></loudness>"
        xml = XMLType(xml)
        inputs.append((xml, sesh))
        expected.append((expected_loudness, expected_tempo))

    return expected, list(batch_griffin_lim_tts(_spec_exp, make_batches(inputs), iterations=15))


def main():
    st.markdown("# Annotation Correlation")
    st.markdown(
        "Use this workbook to determine how well a model's annotations correlate "
        "with the output."
    )
    run._config.configure(overwrite=True)

    form: DeltaGenerator = st.form(key="form")

    label = "Spectrogram Checkpoints"
    spec_path = st_select_path(label, SPECTROGRAM_MODEL_EXPERIMENTS_PATH, PT_EXTENSION, form)
    num_samples = int(form.number_input("Number of Samples", min_value=1, value=100))
    min_loudness = float(form.number_input("Minimum Loudness", value=-4))
    max_loudness = float(form.number_input("Maximum Loudness", value=3))
    min_tempo = float(form.number_input("Minimum Tempo", value=0.82))
    max_tempo = float(form.number_input("Maximum Tempo", value=1.52))
    if not form.form_submit_button("Submit"):
        return

    assert spec_path is not None
    spec_ckpt = typing.cast(run.train.spectrogram_model._worker.Checkpoint, load(spec_path))
    spec_exp = spec_ckpt.export()
    expected, batches = _generate(
        spec_exp,
        spec_ckpt.comet_experiment_key,
        spec_ckpt.step,
        num_samples,
        min_loudness,
        max_loudness,
        min_tempo,
        max_tempo,
    )
    results = []
    for result in batches:
        audio_len = cf.partial(lib.audio.sample_to_sec)(result.sig_model.shape[0])
        tempo = cf.partial(_get_tempo_annotation)(xml_to_text(result.inputs.xmls[0]), audio_len)
        loudness = cf.partial(_get_loudness_annotation)(result.sig_model)
        results.append((loudness, tempo))

    data = [
        {
            "Speaker": repr(b.inputs.inputs.session[0].spkr),  # type: ignore
            "Speaker Tempo": b.inputs.inputs.session[0].spkr_tempo,  # type: ignore
            "Session": b.inputs.inputs.session[0].label,  # type: ignore
            "Session Tempo": b.inputs.inputs.session[0].tempo,  # type: ignore
            "Session Loudness": b.inputs.inputs.session[0].loudness,  # type: ignore
            "Script": b.inputs.xmls[0],
            "Audio": audio_to_url(b.sig_model),
            "Expected Loudness": e[0],
            "Outputed Loudness": r[0],
            "Expected Tempo": e[1],
            "Outputed Tempo": r[1],
        }
        for b, e, r, in zip(batches, expected, results)
    ]
    st_ag_grid(pandas.DataFrame(data), ["Audio"])
    _correlate("loudness", [e[0] for e in expected], "out_loudness", [r[0] for r in results])
    _correlate("tempo", [e[1] for e in expected], "out_tempo", [r[1] for r in results])


if __name__ == "__main__":
    main()
