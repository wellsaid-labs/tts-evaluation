""" Streamlit application for reviewing the correlation between text length and audio length.

TODO:
- Update the synthetic GCP dataset to be more aligned with other datasets. While our other datasets
  are at most 0.9x faster to 1.32x slower than average. The GCP dataset is a wooping 2x slower
  than average. This makes it difficult to create a reasonable maximum audio length formula. One
  thing that might be skewing this, is that it seems to have 200 milliseconds of pausing added to
  the end of each clip. Even if we fully accounted for that, the GCP dataset would still be an
  outlier, so there is more going on.

Usage:
    $ PYTHONPATH=. streamlit run run/review/dataset_processing/text_audio_length_correlation.py \
        --runner.magicEnabled=false
"""
import collections
import logging
import typing

import numpy
import pandas
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn import linear_model
from streamlit.delta_generator import DeltaGenerator

import lib
import run
from lib.text.utils import Pronunciation, get_pronunciation, get_pronunciations, load_cmudict_syl
from run._config.labels import _speaker
from run._config.lang import _get_long_abbrevs, get_avg_audio_length, get_max_audio_length
from run._config.train import _config_spec_model_training
from run._streamlit import audio_to_url, clip_audio, get_datasets, get_spans, st_ag_grid, st_tqdm
from run.data._loader import Span, Speaker

lib.environment.set_basic_logging_config(reset=True)
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
logger = logging.getLogger(__name__)


def _get_letter_feats(text: str, common_punct: typing.Set[str]) -> typing.Dict[str, int]:
    """Analyze `text` and extract various features.

    Args:
        ...
        common_punct: A list of common punctuation marks that could be in `span.script`.
    """
    punct_feats = {repr(l): text.count(l) for l in common_punct}
    num_alpha = sum(c.isalpha() for c in text)
    num_vowels = sum(text.lower().count(x) for x in "aeiou")
    abbreviations = "".join(_get_long_abbrevs(text))
    num_upper_initials = sum(c.isupper() for c in abbreviations)
    num_lower_initials = sum(c.islower() for c in abbreviations)
    num_initial_dots = sum(c == "." for c in abbreviations)
    feats = {
        "num_chars": len(text),
        "num_alpha": num_alpha,
        "num_vowels": num_vowels,
        "num_constants": num_alpha - num_vowels,
        "num_initials": num_upper_initials + num_lower_initials,
        "num_upper": sum(c.isupper() for c in text) - num_upper_initials,
        "num_lower": sum(c.islower() for c in text) - num_lower_initials,
        "num_uncommon_punc": len(text) - num_alpha - sum(punct_feats.values()),
        **punct_feats,
    }
    feats[repr(".")] -= num_initial_dots
    assert feats["num_upper"] + feats["num_lower"] + feats["num_initials"] == num_alpha
    assert feats["num_constants"] + feats["num_vowels"] == num_alpha
    assert sum(c.isnumeric() for c in text) == 0
    return feats


FEATS_PREFIX = "feat"


def _assemble(
    span: Span,
    clip: numpy.ndarray,
    feats: typing.Dict[str, int],
    pronuns: typing.Optional[typing.List[Pronunciation]],
    analyze_pronun: bool,
) -> typing.Dict[str, typing.Any]:
    """Assemble data into a dictionary for visualization.

    Args:
        span: The `Span` to analyze.
        clip: The `span` audio clip.
        feats: The computed features.
        pronuns: If available, the pronunciation for `span.script`.
        analyze_pronun: If available, add pronunciation columns to data.
    """
    result, expected_audio_len, phone_per_char = {}, None, None
    if analyze_pronun and pronuns and feats["num_phonemes"]:
        phone_per_char = span.audio_length / feats["num_phonemes"]
        pronun_str = " ".join("-".join("|".join(p) for p in s) for s in pronuns)
        result = {"pronun": pronun_str, "phone_per_char": phone_per_char}

    expected_audio_len = get_avg_audio_length(span.script)
    max_audio_len = get_max_audio_length(span.script)
    pronun_str = " ".join("-".join("|".join(p) for p in s) for s in pronuns) if pronuns else None
    result = {
        "script": span.script,
        "clip": audio_to_url(clip),
        "speaker": repr(span.session[0]),
        "session": span.session.label,
        "transcript": span.transcript,
        "audio_len": round(span.audio_length, 1),
        f"{FEATS_PREFIX}_expected_audio_len": expected_audio_len,
        "max_audio_len": max_audio_len,
        "actual_expected_ratio": span.audio_length / expected_audio_len,
        "actual_max_ratio": span.audio_length / max_audio_len,
        "script_len": len(span.script),
        "transcript_len": len(span.transcript),
        "sec_per_char": span.audio_length / len(span.script),
        **{f"{FEATS_PREFIX}_{k}": v for k, v in feats.items()},
        **result,
    }

    return result


def _get_pronun_feats(
    pronuns: typing.Optional[typing.List[Pronunciation]],
) -> typing.Dict[str, typing.Optional[int]]:
    """Analyze `pronuns` and extract various features."""
    return {
        "num_syllables": sum(len(p) for p in pronuns) if pronuns else None,
        "num_phonemes": sum(sum(len(s) for s in p) for p in pronuns) if pronuns else None,
    }


def _gather_span_data(
    span: Span,
    clip: numpy.ndarray,
    common_punct: typing.Set[str],
    pick_first: bool,
    analyze_pronun: bool,
):
    """Gather data on `span`.

    Args:
        span: The `Span` to analyze.
        clip: The audio clip for `context`.
        common_punct: A list of common punctuation marks that could be in `span.script`.
        pick_first: Use the first pronunciation if there are multiple available.
        analyze_pronun: Add pronunciation features to data.
    """
    letter_feats = _get_letter_feats(span.script, common_punct)

    word_tokens = [t for t in span.spacy if any(c.isalnum() for c in t.text)]
    num_word_chars = sum(len(w.text) for w in word_tokens)
    word_feats = {
        "num_words": len(word_tokens),
        "num_word_chars": num_word_chars,
        "num_non_word_chars": len(span.script) - num_word_chars,
    }

    pronuns = None
    if analyze_pronun:
        pronuns = []
        for token in word_tokens:
            pronun = get_pronunciations(token.text, load_cmudict_syl())
            pronun = pronun[0] if len(pronun) == 1 or (pick_first and len(pronun) > 0) else None
            pronuns.append(pronun)
            if pronuns[-1] is None:
                logger.info(f"Unable to find or guess pronunciation for: {token.text}")
        pronuns = None if any(p is None for p in pronuns) else pronuns
        pronuns = typing.cast(typing.Optional[typing.List[Pronunciation]], pronuns)
        word_feats = {**_get_pronun_feats(pronuns), **word_feats}

    feats = {**letter_feats, **word_feats}
    return _assemble(span, clip, feats, pronuns, analyze_pronun)


def _gather_alignment_data(
    context: Span, i: int, clip: numpy.ndarray, common_punct: typing.Set[str], analyze_pronun: bool
):
    """Gather data on `context[i]`.

    Args:
        context: The parent `Span`.
        int: The alignment in `context` to analyze.
        clip: The audio clip for `context`.
        common_punct: A list of common punctuation marks that could be in `span.script`.
    """
    span = context[i]
    feats = _get_letter_feats(span.script, common_punct)

    pronun = None
    if analyze_pronun:
        pronun = get_pronunciation(span.script, load_cmudict_syl())
        pronun = None if pronun is None else [pronun]
        feats = {**_get_pronun_feats(pronun), **feats}

    # TODO: This should be factored out in a function similar to `_include_span`
    # (i.e. `include_alignment`).
    # NOTE: This approach to filtering our bad alignment data is taken from `_include_span` in
    # our configuration in Dec 2022. The idea is that audio clips that are too short or too fast
    # tend to be error prone.
    sec_per_char = span.audio_length / len(span.script)
    if span.audio_length < 0.2 or sec_per_char < 0.04:
        return None

    clip = clip_audio(clip, context, context.alignments[i])
    return _assemble(span, clip, feats, pronun, analyze_pronun)


def _summarize(spans: typing.List[Span], df: pandas.DataFrame):
    """Gather useful statistics for data."""
    st.header("Statistics")
    stats = {
        "Num Spans": len(spans),
        "Num Rows": df.shape[0],
        "Num Rows w/ Pronunciations": df["pronun"].notnull().sum() if "pronun" in df else 0,
    }
    st.dataframe(pandas.DataFrame(data=stats.values(), index=list(stats.keys())))


def _distributions(
    data: typing.List[typing.Dict],
    feats: typing.List[str],
    precisions: typing.List[float],
    num_cols: int = 3,
):
    """Create a `num_cols` column grid with distributions for every feature in `feats`."""
    st.header("Feature Distributions")
    assert len(feats) == len(precisions)
    cols: typing.List[DeltaGenerator] = st.columns(num_cols)
    for i, (feat, precision) in enumerate(zip(feats, precisions)):
        vals = [r[feat] for r in data if r[feat] is not None]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, xbins=dict(size=precision)))
        fig.update_layout(
            xaxis_title_text=feat,
            yaxis_title_text="Count",
            margin=dict(b=0, l=0, r=0, t=0),
            bargap=0.2,
        )
        col = cols[i % len(cols)]
        col.plotly_chart(fig, use_container_width=True)


def _speaker_distribution(data: typing.List[typing.Dict]):
    """Plot the distribution of speaker data."""
    st.header("Speaker Distribution")
    counter = collections.defaultdict(float)
    for row in data:
        counter[row["speaker"]] += row["audio_len"]
    fig = px.bar(x=counter.keys(), y=counter.values(), labels={"x": "speaker", "y": "seconds"})
    st.plotly_chart(fig, use_container_width=True)


def _find_max_audio_len_weight_and_bias(data: typing.List[typing.Dict], slowest_pace: float):
    """Plot the distribution of possible biases and weights to predict maximum audio length."""
    st.header("Slowest Pace Of Speech")
    expected = f"{FEATS_PREFIX}_expected_audio_len"
    speakers = set(r["speaker"] for r in data)
    pacing = []
    for speaker in speakers:
        selection = [r for r in data if r["speaker"] == speaker]
        total_expected = sum(r[expected] for r in selection)
        total = sum(r["audio_len"] for r in selection)
        row = {
            "speaker": speaker,
            "audio_len": total,
            "expected_audio_len": total_expected,
            "num_examples": len(selection),
            "pace": total / total_expected,
        }
        pacing.append(row)
    df = pandas.DataFrame(pacing).sort_values(by=["pace"], ascending=False)
    st.markdown("This chart determines the slowest speakers compared to the expected audio length.")
    st.dataframe(df, use_container_width=True)
    data = sorted(data, key=lambda r: r["audio_len"] - r[expected] * slowest_pace, reverse=True)
    rows = [
        {
            "speaker": r["speaker"],
            "script": r["script"],
            "clip": r["clip"],
            "offset": r["audio_len"] - (r[expected] * slowest_pace),
        }
        for r in data[:50]
    ]
    df = pandas.DataFrame(rows).sort_values(by=["offset"], ascending=False)
    st.markdown(f"This chart determines the maximum offset from a pace of {slowest_pace}.")
    st_ag_grid(df, audio_cols=["clip"])
    offset = data[0]["audio_len"] - (data[0][expected] * slowest_pace)
    st.markdown(
        f"The maximum audio length is `average_audio_len * {slowest_pace} + {offset}`.\n"
        "*Keep in mind, this formula is based on outliers. It might be helpful to dig in "
        "and see if the outliers are valid.*"
    )


def _correlate(name: str, values: typing.List[float], other: str, other_values: typing.List[float]):
    """Correlate two sets of values and visualize it.

    TODO: We shoulder consider coloring the points based on speaker.
    """
    df = pandas.DataFrame({name: values, other: other_values})
    fig = px.scatter(
        df,
        x=other,
        y=name,
        trendline="ols",
        trendline_color_override=px.colors.qualitative.Plotly[8],
    )
    fig.update_layout(margin=dict(b=0, l=0, r=0, t=0))
    results = px.get_trendline_results(fig)
    results.px_fit_results.iloc[0].rsquared
    return fig, results.px_fit_results.iloc[0].rsquared


def _correlations(
    data: typing.List[typing.Dict],
    y: str,
    feats: typing.List[str],
    num_cols: int = 3,
):
    """Create a `num_cols` column grid correlating every feature in `data` to the `name` column."""
    st.header("Correlations")
    data = [r for r in data if all(r[f] is not None for f in feats)]
    st.info(f"Analyzing {len(data)} rows.")
    vals = [r[y] for r in data]
    correlations = [_correlate(y, vals, k, [r[k] for r in data]) for k in feats]
    figs, rsquareds = tuple(zip(*tuple(correlations)))
    meta = pandas.DataFrame(data=rsquareds, index=feats, columns=["rSquared"])
    meta = meta.sort_values(by=["rSquared"], ascending=False)
    st.dataframe(meta, use_container_width=True)
    cols = st.columns(num_cols)
    for i, (_, fig) in enumerate(zip(feats, figs)):
        col = cols[i % num_cols]
        col.plotly_chart(fig, use_container_width=True)


def _multivariate_regression(
    data: typing.List[typing.Dict], y: str, feats_name: str, feats: typing.List[str]
):
    """Regress with `feats` trying to model `y`."""
    regression = linear_model.LinearRegression(positive=True)
    Y = [r[y] for r in data]
    X = [[r[f] for f in feats] for r in data]
    regression.fit(X, Y)
    df = pandas.DataFrame(data={"Coef": regression.coef_}, index=feats)
    df = df.sort_values(by=["Coef"], ascending=False)
    fig, rsquared = _correlate(y, Y, feats_name, regression.predict(X))
    return df, regression.intercept_, rsquared, fig


def _multivariate_regressions(
    data: typing.List[typing.Dict],
    feats: typing.List[str],
    common_punct: typing.Set[str],
    analyze_pronun: bool,
    num_cols: int = 2,
):
    """Try and visualize a number of multivariate regressions."""
    st.header("Multivariate Linear Regressions")
    data = [r for r in data if all(r[f] is not None for f in feats)]
    st.info(f"Analyzing {len(data)} rows.")

    puncs = [repr(l) for l in common_punct] + ["num_uncommon_punc"]
    meta: typing.List[typing.Tuple[str, typing.List[str]]] = [
        ("All", feats),
        ("Number of Characters", ["num_chars"]),
        ("Word & Non-Word Characters", ["num_word_chars", "num_non_word_chars"]),
        ("Vowels & Constants", ["num_vowels", "num_constants"] + puncs),
        (
            "Upper & Lower Cases",
            ["num_upper", "num_lower", "num_initials"] + puncs,
        ),
        ("Alpha", ["num_alpha"] + puncs),
        ("Word Characters", ["num_word_chars"] + puncs),
    ]
    if analyze_pronun:
        meta.append(("Phonemes", ["num_phonemes"] + puncs))
        meta.append(("Syllables", ["num_syllables"] + puncs))

    names = [f"{n} ({len(c)})" for n, c in meta]
    combos = [c for _, c in meta]
    results = []
    for name, combo in zip(names, combos):
        prefixed = [c if c.startswith(FEATS_PREFIX) else f"{FEATS_PREFIX}_{c}" for c in combo]
        results.append(_multivariate_regression(data, "audio_len", name, prefixed))

    df = pandas.DataFrame(data=[r[2] for r in results], index=names, columns=["rSquared"])
    st.dataframe(df.sort_values(by=["rSquared"], ascending=False), use_container_width=True)

    cols = st.columns(num_cols)
    for i, (name, (coefs, intercept, rsquared, fig)) in enumerate(zip(names, results)):
        col = cols[i % num_cols]
        col.subheader(name)
        col.dataframe(coefs.transpose(), use_container_width=True)
        col.info(f"The intercept is **{intercept}**, and R-Squared is **{rsquared}**.")
        col.plotly_chart(fig, use_container_width=True)


def main():
    run._config.configure(overwrite=True)
    # NOTE: The various parameters map to configurations that are not relevant for this workbook.
    _config_spec_model_training(0, 0, 0, 0, 0, False, overwrite=True)

    st.title("Text Audio Correlation")
    with st.expander("ℹ️", expanded=True):
        st.markdown(
            "This app analyzes the correlation between the text length and audio length. In order "
            "to faciliate that analysis, this provides a couple options:"
            "\n"
            "- This analyzes the correlation between character counts and audio length. This has "
            "  an option for filtering out uncommon characters.\n"
            "- This analyzes training data spans; however, if you wanted to analyze "
            "  individual words you can review alignments instead. Unfortunately, these are "
            "  also less accurate because speech doesn't have clear word boundaries.\n"
            "- This doesn't analyze pronunciation; however, you can analyze it. Keep in "
            "  mind that we don't have the pronunciation for every word, so we will need to "
            "  filter out some data. Also, there are some words with multiple pronunciations, so "
            "  we need to pick a strategy for resolving that ambiguity. By default, this picks "
            "  the first word in CMUDict, if there are multiple."
        )

    train_dataset, dev_dataset = get_datasets()

    form: DeltaGenerator = st.form("settings")
    question = "How many span(s) do you want to generate?"
    num_spans: int = int(form.number_input(question, 0, 50000, 500))
    format_func = lambda s: "None" if s is None else _speaker(s)
    speakers = [None] + list(train_dataset.keys())
    speaker: typing.Optional[Speaker] = form.selectbox("Speaker", speakers, format_func=format_func)
    question = "How often do characters need to show up, in percent, for them to be analyzed?"
    char_threshold: float = int(form.number_input(question, 0, 100, 1, 1)) / 100
    question = "What is the slowest authentic pace of speech? (See charts to determine this value)"
    slowest_pace = float(form.number_input(question, 0.0, None, 1.6))
    analyze_alignments: bool = form.checkbox("Analyze Individual Alignments")
    # NOTE: Generally, we have found that phonemes and syllables are not helpful. They are also
    # restrictive because we cannot always find pronunciation data, so this is turned off by
    # default.
    analyze_pronun: bool = form.checkbox("Analyze Pronunciation", False)
    # TODO: Warn the user if pronunciations have different counts.
    # TODO: Add additional strategies for picking pronunciations like by part of speech.
    pick_first: bool = form.checkbox("Pick First Pronunciation (if there are multiple)", True)
    dev_speakers: bool = form.checkbox("Analyze Only Dev Speakers", True)
    if not form.form_submit_button("Submit"):
        return

    spans = get_spans(train_dataset, dev_dataset, num_spans, speaker, dev_speakers)

    with st.spinner("Loading audio..."):
        clips = [s.audio() for s in st_tqdm(spans)]

    with st.spinner("Finding common punctuation marks..."):
        common_punct = collections.Counter()
        common_threshold = char_threshold * len(spans)
        for span in spans:
            common_punct.update(span.script)
        common_punct = {k: count for k, count in common_punct.items() if not k.isalnum()}
        uncommon_punct = {k for k, count in common_punct.items() if count < common_threshold}
        common_punct = {k for k, count in common_punct.items() if count >= common_threshold}
        message = "Excluded these less common punctuation marks from analysis:"
        st.info(f"{message} {repr(''.join(sorted(list(uncommon_punct))))[1:-1]}")

    with st.spinner("Assembling data..."):
        data: typing.List[typing.Dict[str, typing.Any]] = []
        for span, clip in zip(spans, clips):
            if analyze_alignments:
                rows = [
                    _gather_alignment_data(span, i, clip, common_punct, analyze_pronun)
                    for i in range(len(span))
                ]
            else:
                rows = [_gather_span_data(span, clip, common_punct, pick_first, analyze_pronun)]
            data.extend(r for r in rows if r is not None)

    df = pandas.DataFrame(data)
    features = [k for k in data[0].keys() if k.startswith(FEATS_PREFIX)]
    _summarize(spans, df)
    st.header("Data")
    st_ag_grid(df, audio_cols=["clip"])
    _find_max_audio_len_weight_and_bias(data, slowest_pace=slowest_pace)
    _speaker_distribution(data)
    _distributions(data, ["audio_len"] + features, [0.1] + [1.0] * len(features))
    _correlations(data, "audio_len", features)
    _multivariate_regressions(data, features, common_punct, analyze_pronun=analyze_pronun)


if __name__ == "__main__":
    main()
