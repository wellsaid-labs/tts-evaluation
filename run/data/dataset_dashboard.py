""" Streamlit application for reviewing the dataset.

Usage:
    $ PYTHONPATH=. streamlit run run/data/dataset_dashboard.py
"""
from __future__ import annotations

import base64
import collections
import contextlib
import dataclasses
import functools
import io
import logging
import math
import multiprocessing.pool
import os
import pathlib
import random
import typing

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import tqdm
from librosa.util import utils as librosa_utils
from third_party import session_state
from torchnlp.random import fork_rng

import lib
import run
from lib.audio import amplitude_to_db, signal_to_rms
from lib.datasets import Alignment, Passage
from lib.utils import clamp, flatten, mazel_tov, seconds_to_string
from run._config import Dataset

lib.environment.set_basic_logging_config(reset=True)
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)

ALIGNMENT_PRECISION = 0.1
AUDIO_COLUMN = "audio"
DEFAULT_COLUMNS = [AUDIO_COLUMN, "script", "transcript"]
HASH_FUNCS = {Passage: lambda p: p.key}

# TODO: Visualize a passage, and the alignments that would be valid "anchor" alignments. This
# would put in context what the characteristics are for an "anchor" alignment, and it'll help
# us better understand what the coverage of those alignments are.
# TODO: Print the phoneme coverage of the dataset, we're using. How much of the CMUDict
# does it cover? What's the distribution of that coverage? How many words in the dataset
# are missed by CMUDict and what are they?


_SessionCacheInputType = typing.TypeVar(
    "_SessionCacheInputType", bound=typing.Callable[..., typing.Any]
)


def _session_cache(
    func: typing.Optional[_SessionCacheInputType] = None, **kwargs
) -> _SessionCacheInputType:
    """ `lru_cache` wrapper for `streamlit` that caches accross reruns. """
    if not func:
        return functools.partial(_session_cache, **kwargs)

    session = session_state.get(cache={})

    use_session_cache = st.sidebar.checkbox(f"Cache `{func.__qualname__}`", value=True)
    if func.__qualname__ not in session["cache"] or not use_session_cache:
        logger.info("Creating `%s` cache.", func.__qualname__)
        session["cache"][func.__qualname__] = functools.lru_cache(**kwargs)(func)

    return session["cache"][func.__qualname__]


@_session_cache(maxsize=None)
def _read_audio_slice(*args, **kwargs):
    return lib.audio.read_audio_slice(*args, **kwargs)


def _round(x, bucket_size):
    return bucket_size * round(x / bucket_size)


assert _round(0.3, 1) == 0
assert _round(0.4, 0.25) == 0.5


def _static_symlink(target: pathlib.Path) -> pathlib.Path:
    """System link `target` to `root / static`, and return the linked location.

    Learn more:
    https://github.com/st/st/issues/400#issuecomment-648580840
    https://github.com/st/st/issues/1567
    """
    root = pathlib.Path(st.__file__).parent / "static"
    static = pathlib.Path("static") / "_private"
    assert root.exists()
    (root / static).mkdir(exist_ok=True)
    target = target.relative_to(lib.environment.ROOT_PATH)
    if not (root / static / target).exists():
        (root / static / target.parent).mkdir(exist_ok=True, parents=True)
        (root / static / target).symlink_to(lib.environment.ROOT_PATH / target)
    return static / target


_MapInputType = typing.TypeVar("_MapInputType")
_MapReturnType = typing.TypeVar("_MapReturnType")


def _map(
    list_: typing.List[_MapInputType],
    func: typing.Callable[[_MapInputType], _MapReturnType],
    chunk_size=8,
    max_parallel=os.cpu_count() * 3,
) -> typing.List[_MapReturnType]:
    """ Apply `func` to `list_` in parallel. """
    with multiprocessing.pool.ThreadPool(processes=max_parallel) as pool:
        iterator = pool.imap(func, list_, chunksize=chunk_size)
        return list(tqdm.tqdm(iterator, total=len(list_)))


def _signal_to_db_rms(signal: np.ndarray) -> float:
    return typing.cast(float, amplitude_to_db(torch.tensor(signal_to_rms(signal))).item())


@dataclasses.dataclass(frozen=True)
class Span(lib.datasets.Span):
    """`lib.datasets.Span` with additional attributes.

    Attributes:
        mistranscriptions: List of unaligned alphanumeric `script` and `transcript` text.
        seconds_per_character: The average speed seconds per character.
    """

    mistranscriptions: typing.List[typing.Tuple[str, str]] = dataclasses.field(init=False)
    seconds_per_character: float = dataclasses.field(init=False)

    @staticmethod
    def _isalnum(s: str):
        return any(c.isalnum() for c in s)

    def __post_init__(self):
        super().__post_init__()
        set = object.__setattr__
        mistranscriptions = [(a, b) for a, b, _ in self.unaligned if self._isalnum(a + b)]
        set(self, "mistranscriptions", [(a.strip(), b.strip()) for a, b in mistranscriptions])
        set(self, "seconds_per_character", self.audio_length / len(self.script))

    @property
    def audio(self):
        start = self.passage.alignments[self.span][0].audio[0]
        return _read_audio_slice(self.passage.audio_file.path, start, self.audio_length)

    def _audio(self, second, interval):
        index = lambda i: clamp(round(i * self.audio_file.sample_rate), 0, self.audio.shape[0])
        return self.audio[index(second - interval[0]) : index(second + interval[1])]

    @property
    def rms(self) -> float:
        return round(_signal_to_db_rms(self.audio), 1)

    def min_rms(
        self,
        second: float,
        interval: typing.Tuple[float, float] = (ALIGNMENT_PRECISION / 2, ALIGNMENT_PRECISION / 2),
        frame_length: float = ALIGNMENT_PRECISION / 10,
        hop_length: float = ALIGNMENT_PRECISION / 40,
    ):
        """ Get the minimum RMS within `interval`. """
        audio = self._audio(second, interval)
        frames = librosa_utils.frame(
            audio,
            int(frame_length * self.audio_file.sample_rate),
            int(hop_length * self.audio_file.sample_rate),
            axis=0,
        )
        return round(min([_signal_to_db_rms(f) for f in frames]), 1)

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        """ Get a non-circular `dict`. """
        fields = dataclasses.fields(self)
        return {f.name: getattr(self, f.name) for f in fields if f.type != Passage}


@contextlib.contextmanager
def beta_expander(label):
    with st.beta_expander(label) as expander:
        logger.info("Visualizing '%s'...", label)
        yield expander


def _visualize_signal(
    signal: np.ndarray,
    rules: typing.List[float] = [],
    max_sample_rate: int = 4096,
    sample_rate: int = 24000,
) -> alt.Chart:
    """Visualize a signal envelope similar to `librosa.display.waveplot`.

    Learn more about envelopes: https://en.wikipedia.org/wiki/Envelope_detector
    """
    ratio = sample_rate // max_sample_rate
    frames = librosa_utils.frame(signal, ratio, ratio, axis=0)  # type: ignore
    assert frames.shape[1] == ratio
    envelope = np.max(np.abs(frames), axis=-1)
    assert envelope.shape[0] == frames.shape[0]
    seconds = np.arange(0, signal.shape[0] / sample_rate, ratio / sample_rate)
    waveform = alt.Chart(pd.DataFrame({"seconds": seconds, "y_max": envelope, "y_min": -envelope}))
    waveform = waveform.mark_area().encode(x="seconds:Q", y="y_min:Q", y2="y_max:Q")
    line = alt.Chart(pd.DataFrame({"x": rules})).mark_rule(color="darkred").encode(x="x")
    return (line + waveform).interactive()


def _bucket_and_visualize(
    iterable: typing.Iterable[typing.Union[float, int]],
    bucket_size: float = 1,
    ndigits: int = 7,
    x: str = "Buckets",
    y: str = "Count",
):
    """ Bucket `iterable` and display a bar chart. """
    buckets = collections.defaultdict(int)
    nan_count = 0
    for item in iterable:
        if math.isnan(item):
            nan_count += 1
        else:
            buckets[round(_round(item, bucket_size), ndigits)] += 1
    if nan_count > 0:
        logger.warning("Ignoring %d NaNs...", nan_count)
    df = pd.DataFrame({x: buckets.keys(), y: buckets.values()})
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(x=x, y=y, tooltip=[x, y]).interactive(),
        use_container_width=True,
    )


def _get_pause_lengths_in_seconds(dataset: Dataset) -> typing.Iterator[float]:
    for _, passages in dataset.items():
        for passage in passages:
            for prev, next in zip(passage.alignments, passage.alignments[1:]):
                yield next.audio[0] - prev.audio[1]


def _get_alignments(dataset: Dataset) -> typing.Iterator[typing.Tuple[Passage, Alignment]]:
    for _, passages in dataset.items():
        for passage in passages:
            yield from [(passage, a) for a in passage.alignments]


def _ngrams(list_: typing.Sequence, n: int) -> typing.Iterator[slice]:
    yield from (slice(i, i + n) for i in range(len(list_) - n + 1))


_get_ngrams = lambda l, n: [l[s] for s in _ngrams(l, n)]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=1) == [[1], [2], [3], [4], [5], [6]]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]


@_session_cache(maxsize=None)
def _get_dataset(speaker_names: typing.FrozenSet[str]) -> Dataset:
    """Load dataset."""
    logger.info("Loading dataset...")
    datasets = {k: v for k, v in run._config.DATASETS.items() if k.name in speaker_names}
    dataset = run._config.get_dataset(datasets)
    logger.info(f"Finished loading dataset! {mazel_tov()}")
    return dataset


def _get_passages(dataset: Dataset) -> typing.Iterator[Passage]:
    for _, passages in dataset.items():
        yield from passages


def _get_alignment_ngrams(
    dataset: Dataset,
    n: int = 1,
    max_passages=128,
) -> typing.Iterator[Span]:
    """ Get ngram `Span`s with `n` alignments. """
    passages = list(_get_passages(dataset))
    with fork_rng(123):
        sample = random.sample(passages, min(len(passages), max_passages))
    for passage in tqdm.tqdm(sample):
            yield from (Span(passage, s) for s in _ngrams(passage.alignments, n=n))


def _get_spans(dataset: Dataset, num_samples: int, slice_: bool = True) -> typing.List[Span]:
    """Generate spans from our datasets."""
    logger.info("Generating spans...")
    kwargs = {} if slice_ else {"max_seconds": math.inf}
    generator = run._config.span_generator(dataset, **kwargs)
    spans = [next(generator) for _ in tqdm.tqdm(range(num_samples), total=num_samples)]
    return_ = [Span(s.passage, s.span) for s in tqdm.tqdm(spans)]
    logger.info(f"Finished generating spans! {mazel_tov()}")
    return return_


def _span_coverage(dataset, spans) -> float:
    """ Get the percentage of the `dataset` these `spans` cover. """
    alignments = set([(p.key, a) for (p, a) in _get_alignments(dataset)])
    total = len(alignments)
    for span in spans:
        for i in range(span.span.start, span.span.stop):
            key = (span.passage.key, span.passage.alignments[i])
            if key in alignments:
                alignments.remove(key)
    return 1 - (len(alignments) / total)


def _audio_to_base64(audio: np.ndarray) -> str:
    """Encode audio into a `base64` string."""
    in_memory_file = io.BytesIO()
    lib.audio.write_audio(in_memory_file, audio)
    return base64.b64encode(in_memory_file.read()).decode("utf-8")


def _audio_to_html(audio: typing.Union[np.ndarray, pathlib.Path]) -> str:
    """Create an `audio` HTML element."""
    if isinstance(audio, pathlib.Path):
        return f'<audio controls="" src="/{_static_symlink(audio)}"></audio>'
    return f'<audio controls="" src="data:audio/wav;base64,{_audio_to_base64(audio)}"></audio>'


def _visualize_spans(
    spans: typing.List[Span],
    columns: typing.List[str] = DEFAULT_COLUMNS,
    other_columns: typing.Dict[str, typing.Union[typing.List, typing.Tuple]] = {},
    slice_: bool = True,
    max_spans: int = 50,
):
    """Visualize spans as a table."""
    spans = spans[:max_spans]
    logger.info("Visualizing %d spans..." % len(spans))
    df = pd.DataFrame([s.as_dict() for s in spans])
    with multiprocessing.pool.ThreadPool() as pool:
        assert AUDIO_COLUMN not in df.columns
        df[AUDIO_COLUMN] = pool.map(lambda s: s.audio if slice_ else s.audio_file.path, spans)
    df = df[columns]
    for key, values in other_columns.items():
        df[key] = values[:max_spans]
    html = df.to_html(formatters={AUDIO_COLUMN: _audio_to_html}, escape=False, justify="left")
    st.markdown(html, unsafe_allow_html=True)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")


def _maybe_analyze_dataset(dataset: Dataset):
    logger.info("Analyzing dataset...")
    st.header("Raw Dataset Analysis")
    st.markdown("In this section, we analyze the dataset prior to segmentation.")
    if not st.checkbox("Analyze", key=_maybe_analyze_dataset.__name__):
        return

    unigrams = list(_get_alignment_ngrams(dataset, n=1))
    trigrams = list(_get_alignment_ngrams(dataset, n=3))
    aligned_seconds = sum(flatten([[p[:].audio_length for p in v] for v in dataset.values()]))
    files = set(flatten([[p.audio_file for p in v] for v in dataset.values()]))
    total_seconds = sum([f.length for f in files])

    st.markdown(
        f"At a high-level, this dataset has:\n"
        f"- **{seconds_to_string(total_seconds)}** of audio\n"
        f"- **{seconds_to_string(aligned_seconds)}** of aligned audio\n"
        f"- **{sum([len(p.alignments) for p in _get_passages(dataset)]):,}** alignments.\n"
    )
    st.markdown(f"Analyzing a random sample of **{len(unigrams):,}** alignments...")

    with beta_expander("Random Sample of Alignments"):
        for span in random.sample(trigrams, 25):
            cols = st.beta_columns([2, 1])
            rules = list(span.alignments[1].audio)
            cols[0].altair_chart(_visualize_signal(span.audio, rules), use_container_width=True)
            loudness = f"({span[1].min_rms(0)} dB, {span[1].min_rms(span[1].audio_length)} dB)"
            cols[1].markdown(
                f"- Edge Loudness: **({loudness})**\n"
                f"- **{round(span[1].seconds_per_character, 2)}** Seconds per character\n"
            )
            long = _audio_to_html(span.audio)
            markdown = f"**Listen:** '{span.script}'\n\n{_audio_to_html(span[1].audio)}\n\n{long}"
            cols[1].markdown(markdown, unsafe_allow_html=True)

    with beta_expander("Survey of Pause Lengths (in seconds)"):
        st.write("The pause count for each length bucket:")
        iterator = _get_pause_lengths_in_seconds(dataset)
        _bucket_and_visualize(iterator, ALIGNMENT_PRECISION, x="Seconds")

    with beta_expander("Survey of Alignment Lengths (in seconds)"):
        st.write("The alignment count for each length bucket:")
        iterator = [s.audio_length for s in unigrams]
        _bucket_and_visualize(iterator, ALIGNMENT_PRECISION, x="Seconds")

    with beta_expander("Survey of Alignment Lengths (in characters)"):
        st.write("The alignment count for each length bucket:")
        iterator = [len(s.script) for s in unigrams]
        _bucket_and_visualize(iterator, ALIGNMENT_PRECISION, x="Characters")
        st.write("The longest alignments: ")
        sample = sorted(unigrams, key=lambda s: len(s.script), reverse=True)[:50]
        st.table([{"script": s.script, "transcript": s.transcript} for s in sample])

    for attr in ["seconds_per_character"]:
        with beta_expander(f"Survey of Alignment Speeds (`{attr}`)"):
            st.write("The alignment count for each speed bucket:")
            _bucket_and_visualize([getattr(s, attr) for s in unigrams], 0.01, x=attr)
            st.write("The fastest alignments:")
            sample = sorted(unigrams, key=lambda s: getattr(s, attr))[:50]
            _visualize_spans(sample, DEFAULT_COLUMNS + ["audio_length"] + [attr])

    with beta_expander("Survey of Alignment Loudness (in dB)"):
        st.write("The alignment count for each dB bucket:")
        with fork_rng(123):
            sample = random.sample(unigrams, min(len(unigrams), 4096))
        _bucket_and_visualize(_map(sample, lambda s: s.rms), ALIGNMENT_PRECISION, x="dB")
        st.write("The quietest alignments: ")
        sample = [s for s in sample if not math.isnan(s.rms)]
        display = sorted(sample, key=lambda s: s.rms)[:50]
        _visualize_spans(display, other_columns={"rms": [s.rms for s in display]})

    logger.info(f"Finished analyzing dataset! {mazel_tov()}")


def _maybe_analyze_spans(dataset: Dataset, spans: typing.List[Span]):
    logger.info("Analyzing spans...")
    st.header("Dataset Segmentation Analysis")
    audio_length = sum([s.audio_length for s in spans])
    st.markdown("In this section, we analyze the dataset after segmentation via `Span`s. ")
    if not st.checkbox("Analyze", key=_maybe_analyze_spans.__name__):
        return
    st.markdown(
        f"There are **{len(spans)} ({seconds_to_string(audio_length)})** spans to analyze, "
        f"representing of **{_span_coverage(dataset, spans):.2%}** all alignments."
    )

    with beta_expander("Survey of Span Onset RMS dB"):
        st.write("The span count for each onset RMS dB bucket:")
        iterator = _map(spans, lambda s: typing.cast(Span, s).min_rms(0))
        _bucket_and_visualize(iterator, x="RMS dB")

    with beta_expander("Survey of Span Mistranscriptions"):
        st.write("The span mistranscription count for each length bucket:")
        mistranscriptions = [s.mistranscriptions for s in spans if len(s.mistranscriptions) > 0]
        mistranscriptions = flatten([s.mistranscriptions for s in spans])
        iterator = [len(m[0]) for m in mistranscriptions if len(m[0]) > 0]
        _bucket_and_visualize(iterator, x="Characters")
        st.write(
            f"Note that **{len(mistranscriptions)}** spans have one or more mistranscriptions."
        )
        st.write("A random sample of mistranscriptions:")
        unaligned = [{"script": m[0], "transcript": m[1]} for m in mistranscriptions]
        st.table(unaligned)

    logger.info(f"Finished analyzing spans! {mazel_tov()}")


def _maybe_analyze_filtered_spans(dataset: Dataset, spans: typing.List[Span]):
    """Filter out spans that are not ideal for training."""
    logger.info("Analyzing filtered spans...")
    st.header("Dataset Filtered Segmentation Analysis")
    st.markdown(
        "In this section, we analyze the dataset after segmentation and filtering. "
        "This is the final step in our dataset preprocessing. "
    )
    if not st.checkbox("Analyze", key=_maybe_analyze_filtered_spans.__name__):
        return

    lambda_ = lambda s: (s.min_rms(0), s.min_rms(s.audio_length))
    onset_rms, outset_rms = tuple(zip(*_map(spans, lambda_)))
    total = len(spans)
    is_include = (
        lambda s, r0, r1: r0 < -30
        and r1 < -30
        and len(s.mistranscriptions) == 0
        and s[0].seconds_per_character >= 0.03
        and s[0].seconds_per_character <= 0.2
        and s[-1].seconds_per_character >= 0.03
        and s[-1].seconds_per_character <= 0.2
    )
    filtered = [a for a in zip(spans, onset_rms, outset_rms) if is_include(*a)]
    spans, onset_rms, outset_rms = tuple(zip(*filtered))  # type: ignore
    audio_length = sum([s.audio_length for s in spans])
    st.markdown(
        f"The filtered segmentations represent **{len(filtered) / total:.2%}** of the "
        f"original spans. In total, there are **{len(spans)} "
        f"({seconds_to_string(audio_length)})**"
        " spans to analyze, representing of "
        f"**{_span_coverage(dataset, spans):.2%}** all alignments."
    )

    with beta_expander("Random Sample of Filtered Spans"):
        speed = lambda s: str(round(s.seconds_per_character, 2))
        more_columns = {
            "onset_rms": onset_rms,
            "outset_rms": outset_rms,
            "speed_of_alignments": [[speed(s[i]) for i in range(len(s.alignments))] for s in spans],
            "aligned_words": [[s[i].script for i in range(len(s.alignments))] for s in spans],
        }
        _visualize_spans(spans, other_columns=more_columns)

    for label, lambda_, bucket_size in [
        ("loudness", lambda s: round(_signal_to_db_rms(s.audio), 1), 1),
        ("speed", lambda s: s.seconds_per_character, 0.01),
    ]:
        with beta_expander(f"Survey of Span {label.title()}"):
            st.write(f"The span count for each {label} bucket:")
            values = [lambda_(s) for s in spans]
            _bucket_and_visualize(values, bucket_size)
            sorted_ = sorted(zip(spans, values), key=lambda i: i[1])
            spans, values = tuple(zip(*sorted_))  # type: ignore
            st.write("The smallest valued spans:")
            _visualize_spans(spans[:50], other_columns={label: values[:50]})  # type: ignore
            st.write("The largest valued spans:")
            _visualize_spans(spans[-50:], other_columns={label: values[-50:]})  # type: ignore


def main():
    run._config.configure()
    st.title("Dasaset Dashboard")
    st.write("The dataset dashboard is an effort to understand our dataset and dataset processing.")

    sidebar = st.sidebar
    question = "Which dataset(s) do you want to load?"
    speaker_names = [k.name for k in run._config.DATASETS.keys()]
    speakers: typing.FrozenSet[str] = frozenset(st.sidebar.multiselect(question, speaker_names))
    question = "How many spans(s) do you want to generate?"
    num_samples = sidebar.number_input(question, 0, None, 100)

    if len(speakers) == 0:
        st.stop()

    dataset = _get_dataset(speakers)
    with fork_rng(123):
        spans = _get_spans(dataset, num_samples=num_samples)

    _maybe_analyze_dataset(dataset)
    _maybe_analyze_spans(dataset, spans)
    _maybe_analyze_filtered_spans(dataset, spans)

    logger.info(f"Done! {mazel_tov()}")


if __name__ == "__main__":
    main()
