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
import itertools
import logging
import math
import multiprocessing.pool
import os
import pathlib
import random
import typing
import warnings

import altair as alt
import numpy as np
import pandas as pd
import pytest
import streamlit as st
import torch
import tqdm
from librosa.util import utils as librosa_utils
from third_party import session_state
from torchnlp.random import fork_rng

import lib
import run
from lib.audio import amplitude_to_db, signal_to_rms
from lib.datasets import DATASETS, Passage
from lib.utils import clamp, flatten, mazel_tov, round_, seconds_to_string
from run._config import Dataset

lib.environment.set_basic_logging_config(reset=True)
alt.data_transformers.disable_max_rows()
st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)

ALIGNMENT_PRECISION = 0.1
AUDIO_COLUMN = "audio"
DEFAULT_COLUMNS = [AUDIO_COLUMN, "script", "audio_length"]
HASH_FUNCS = {Passage: lambda p: p.key}


_RandomSampleReturnType = typing.TypeVar("_RandomSampleReturnType")


def _random_sample(
    list_: typing.List[_RandomSampleReturnType], max_samples: int, seed: int = 123
) -> typing.List[_RandomSampleReturnType]:
    """ Deterministic random sample. """
    with fork_rng(seed):
        return random.sample(list_, min(len(list_), max_samples))


_MapInputType = typing.TypeVar("_MapInputType")
_MapReturnType = typing.TypeVar("_MapReturnType")


def _map(
    list_: typing.List[_MapInputType],
    func: typing.Callable[[_MapInputType], _MapReturnType],
    chunk_size: int = 8,
    max_parallel: int = os.cpu_count() * 3,
    progress_bar: bool = True,
) -> typing.List[_MapReturnType]:
    """ Apply `func` to `list_` in parallel. """
    with multiprocessing.pool.ThreadPool(processes=max_parallel) as pool:
        iterator = pool.imap(func, list_, chunksize=chunk_size)
        if progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(list_))
        return list(iterator)


def _ngrams(list_: typing.Sequence, n: int) -> typing.Iterator[slice]:
    """ Learn more: https://en.wikipedia.org/wiki/N-gram. """
    yield from (slice(i, i + n) for i in range(len(list_) - n + 1))


_get_ngrams = lambda l, n: [l[s] for s in _ngrams(l, n)]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=1) == [[1], [2], [3], [4], [5], [6]]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]


_SessionCacheInputType = typing.TypeVar(
    "_SessionCacheInputType", bound=typing.Callable[..., typing.Any]
)


def _session_cache(
    func: typing.Optional[_SessionCacheInputType] = None, **kwargs
) -> _SessionCacheInputType:
    """`lru_cache` wrapper for `streamlit` that caches accross reruns.

    Learn more: https://github.com/streamlit/streamlit/issues/2382
    """
    if not func:
        return functools.partial(_session_cache, **kwargs)

    session = session_state.get(cache={})

    use_session_cache = st.sidebar.checkbox(f"Cache `{func.__qualname__}`", value=True)
    if func.__qualname__ not in session["cache"] or not use_session_cache:
        logger.info("Creating `%s` cache.", func.__qualname__)
        session["cache"][func.__qualname__] = functools.lru_cache(**kwargs)(func)

    return session["cache"][func.__qualname__]


@contextlib.contextmanager
def beta_expander(label):
    with st.beta_expander(label) as expander:
        logger.info("Visualizing '%s'...", label)
        yield expander


@_session_cache(maxsize=None)
def _read_audio_slice(*args, **kwargs) -> np.ndarray:
    return lib.audio.read_audio_slice(*args, **kwargs)


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


def _audio_to_base64(audio: np.ndarray) -> str:
    """Encode audio into a `base64` string."""
    in_memory_file = io.BytesIO()
    lib.audio.write_audio(in_memory_file, audio)
    return base64.b64encode(in_memory_file.read()).decode("utf-8")


def _audio_to_html(audio: typing.Union[np.ndarray, pathlib.Path]) -> str:
    """Create an `audio` HTML element."""
    if isinstance(audio, pathlib.Path):
        return f'<audio controls src="/{_static_symlink(audio)}"></audio>'
    return f'<audio controls src="data:audio/wav;base64,{_audio_to_base64(audio)}"></audio>'


def _signal_to_db_rms(signal: np.ndarray) -> float:
    """ Get the dB RMS level of `signal`."""
    if signal.shape[0] == 0:
        return math.nan
    return typing.cast(float, amplitude_to_db(torch.tensor(signal_to_rms(signal))).item())


assert _signal_to_db_rms(lib.audio.full_scale_sine_wave()) == pytest.approx(-3.0103001594543457)


def _signal_to_loudness(signal: np.ndarray, sample_rate: int, block_size: float = 0.4) -> float:
    """Get the loudness in LUFS of `signal`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meter = lib.audio.get_pyloudnorm_meter(block_size=block_size, sample_rate=sample_rate)
        if signal.shape[0] >= lib.audio.seconds_to_samples(block_size, sample_rate):
            return round(meter.integrated_loudness(signal), 1)
        return math.nan


def _frame(array: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    return librosa_utils.frame(array, frame_length, hop_length, axis=0)


np.testing.assert_array_equal(
    _frame(np.array([1, 2, 3, 4, 5, 6]), 3, 2),
    np.array([[1, 2, 3], [3, 4, 5]]),
)
np.testing.assert_array_equal(
    _frame(np.array([1, 2, 3, 4, 5, 6]), 4, 2),
    np.array([[1, 2, 3, 4], [3, 4, 5, 6]]),
)


def _visualize_signal(
    signal: np.ndarray,
    sample_rate: int,
    rules: typing.List[float] = [],
    labels: typing.List[str] = [],
    max_sample_rate: int = 2000,
) -> alt.Chart:
    """Visualize a signal envelope similar to `librosa.display.waveplot`.

    Learn more about envelopes: https://en.wikipedia.org/wiki/Envelope_detector

    Args:
        ...
        rules: Add a rule, for every point in this list.
        labels: Labels for each rule in rules.
        ...
    """
    assert len(labels) == len(rules)
    ratio = sample_rate // max_sample_rate
    frames = librosa_utils.frame(signal, ratio, ratio, axis=0)  # type: ignore
    assert frames.shape[1] == ratio
    envelope = np.max(np.abs(frames), axis=-1)
    assert envelope.shape[0] == frames.shape[0]
    seconds = np.arange(0, envelope.shape[0] * ratio / sample_rate, ratio / sample_rate)
    waveform = alt.Chart(pd.DataFrame({"seconds": seconds, "y_max": envelope, "y_min": -envelope}))
    y = alt.Y("y_min:Q", scale=alt.Scale(domain=(-1.0, 1.0)))
    waveform = waveform.mark_area().encode(x="seconds:Q", y=y, y2="y_max:Q")
    line = alt.Chart(pd.DataFrame({"seconds": rules, "label": labels}))
    line = line.mark_rule().encode(x="seconds", color="label")
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
            buckets[round(round_(item, bucket_size), ndigits)] += 1
    if nan_count > 0:
        logger.warning("Ignoring %d NaNs...", nan_count)
    df = pd.DataFrame({x: buckets.keys(), y: buckets.values()})
    st.altair_chart(
        alt.Chart(df).mark_bar().encode(x=x, y=y, tooltip=[x, y]).interactive(),
        use_container_width=True,
    )


def _get_passages(dataset: Dataset) -> typing.Iterator[Passage]:
    """ Get all passages in `dataset`. """
    for _, passages in dataset.items():
        yield from passages


def _get_pause_lengths_in_seconds(dataset: Dataset) -> typing.Iterator[float]:
    """ Get every pause in `dataset` between alignments. """
    for passage in _get_passages(dataset):
        for prev, next in zip(passage.alignments, passage.alignments[1:]):
            yield next.audio[0] - prev.audio[1]


def _get_alignment_ngrams(passages: typing.List[Passage], n: int = 1) -> typing.Iterator[Span]:
    """ Get ngram `Span`s with `n` alignments. """
    for passage in tqdm.tqdm(passages):
        yield from (Span(passage, s) for s in _ngrams(passage.alignments, n=n))


def _get_num_alignments(dataset: Dataset) -> int:
    """ Get number of `Alignment`s in `dataset`. """
    return sum([len(p.alignments) for p in _get_passages(dataset)])


@_session_cache(maxsize=None)
def _get_dataset(speaker_names: typing.FrozenSet[str]) -> Dataset:
    """Load dataset."""
    logger.info("Loading dataset...")
    datasets = {k: v for k, v in DATASETS.items() if k.label in speaker_names}
    dataset = run._utils.get_dataset(datasets)
    logger.info(f"Finished loading dataset! {mazel_tov()}")
    return dataset


@dataclasses.dataclass(frozen=True)
class Span(lib.datasets.Span):
    """`lib.datasets.Span` with additional attributes.

    Attributes:
        mistranscriptions: List of unaligned alphanumeric `script` and `transcript` text.
    """

    mistranscriptions: typing.List[typing.Tuple[str, str]] = dataclasses.field(init=False)

    @staticmethod
    def _is_alnum(s: str):
        return any(c.isalnum() for c in s)

    def __post_init__(self):
        super().__post_init__()

        set = object.__setattr__
        mistranscriptions = [
            (a, b) for a, b, _ in self.script_nonalignments() if self._is_alnum(a + b)
        ]
        set(self, "mistranscriptions", [(a.strip(), b.strip()) for a, b in mistranscriptions])

        self._test_implementation()

    def _test_implementation(self):
        """ Test `Span` implementation. """
        assert self._samples_to_seconds(self._seconds_to_samples(0.5)) == 0.5
        assert self._samples_to_seconds(self._seconds_to_samples(0.0)) == 0.0
        assert self._hop_length * 4 == self._frame_length

    def prev(self) -> typing.Optional[Span]:
        """Get the previous alignment before `self`.

        NOTE: Unless the `Passages` are completely connected, it doesn't make sense to
        get the next alignment in the next passage.
        """
        if self.slice.start != 0:
            return Span(self.passage, slice(self.slice.start - 1, self.slice.start))
        return None

    def next(self) -> typing.Optional[Span]:
        """ Get the next alignment after `self`. """
        if self.slice.stop != len(self.passage.alignments):
            return Span(self.passage, slice(self.slice.stop, self.slice.stop + 1))
        return None

    def audio(self) -> np.ndarray:
        start = self.passage.alignments[self.slice][0].audio[0]
        return _read_audio_slice(self.passage.audio_file.path, start, self.audio_length)

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        """Get a `dict` with properties, class variables, and static variables.

        Learn more:
        https://stackoverflow.com/questions/1251692/how-to-enumerate-an-objects-properties-in-python
        """
        return {
            k: getattr(self, k, "")
            for k in self.__dir__()
            if k[:2] != "__" and type(getattr(self, k, "")).__name__ != "method"
        }

    def rms(self) -> float:
        return round(_signal_to_db_rms(self.audio()), 1)

    @property
    def _frame_length(self) -> int:
        return self._seconds_to_samples(ALIGNMENT_PRECISION / 10)

    @property
    def _hop_length(self) -> int:
        assert self._frame_length % 4 == 0
        return self._frame_length // 4

    def _seconds_to_samples(self, seconds: float) -> int:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return lib.audio.seconds_to_samples(seconds, self.audio_file.sample_rate)

    def _samples_to_seconds(self, samples: int) -> float:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return lib.audio.samples_to_seconds(samples, self.audio_file.sample_rate)

    def _frame(self, signal: np.ndarray) -> np.ndarray:
        return _frame(signal, self._frame_length, self._hop_length)

    def audio_interval(self, second: float, interval: typing.Tuple[float, float]) -> np.ndarray:
        """ Get the audio surrounding `second`. """
        clamp_ = lambda x: clamp(x, min_=0, max_=self.passage.audio_file.length)
        second = self.passage.alignments[self.slice][0].audio[0] + second
        start = clamp_(second - interval[0])
        end = clamp_(second + interval[1])
        return _read_audio_slice(self.passage.audio_file.path, start, end - start)

    def rms_(self, second: float, interval: typing.Tuple[float, float]) -> float:
        """ Get the RMS level at `second`.  """
        audio = self.audio_interval(second, interval)
        return round(_signal_to_db_rms(audio), 1)

    def rms_edges(self) -> typing.Tuple[float, float]:
        """ Get the RMS level surrounding to the edges of `self`.  """
        interval = (ALIGNMENT_PRECISION / 2, ALIGNMENT_PRECISION / 2)
        return (self.rms_(0, interval), self.rms_(self.audio_length, interval))

    def silence(self) -> float:
        """ Get the length of silence in `self`. """
        return sum((b - a) for _, _, (a, b) in self.script_nonalignments()[1:-1])

    def longest_silence(self) -> float:
        """ Get the length of the longest silence, excluding the edges. """
        if len(self.alignments) <= 1:
            return 0.0
        return max((b - a) for _, _, (a, b) in self.script_nonalignments()[1:-1])

    def num_silences(self) -> int:
        """ Get the number of silences. """
        return sum((b - a) > 0 for _, _, (a, b) in self.script_nonalignments()[1:-1])

    def seconds_per_character(self) -> float:
        return sum(a.audio[-1] - a.audio[0] for a in self.alignments) / len(self.script)

    def _rms_is_silent(self, threshold: int = -50) -> typing.List[bool]:
        """ For an evenly spaced list of audio frames, determine if they are silent or not. """
        padding = self._frame_length - self._hop_length
        padded = np.pad(self.audio(), (padding, padding))
        frames = list(self._frame(padded))  # type: ignore
        return [_signal_to_db_rms(f) < threshold for f in frames]

    def rms_silence(self) -> float:
        """ Get the length of silence in `self`. """
        _is_silent = self._rms_is_silent()
        return (sum(_is_silent) / len(_is_silent)) * self.audio_length

    def longest_rms_silence(self) -> float:
        """ Get the length of the longest silence, excluding the edges. """
        frames = self._rms_is_silent()
        groups = [(k, list(g)) for k, g in itertools.groupby(frames)][1:-1]
        groups_len = [len(g) for k, g in groups if k]
        if len(groups_len) == 0:
            return 0.0
        max_group = max(groups_len)
        return (max_group / len(frames)) * self.audio_length

    def num_rms_silences(self) -> int:
        """ Get the number of continuous silences. """
        return sum([k for k, _ in itertools.groupby(self._rms_is_silent())])

    def rms_seconds_per_character(self) -> float:
        if self.audio_length == 0:
            return 0
        return (self.audio_length - self.rms_silence()) / len(self.script)

    def rms_seconds_per_phoneme(self) -> float:
        if self.audio_length == 0:
            return 0
        try:
            return (self.audio_length - self.rms_silence()) / len(
                lib.text._line_grapheme_to_phoneme([self.script], separator="|")[0].split("|")
            )
        except Exception:
            logger.exception("`rms_seconds_per_phoneme` encountered an issue.")
            return np.nan


def _get_spans(dataset: Dataset, num_samples: int, slice_: bool = True) -> typing.List[Span]:
    """Generate spans from our datasets."""
    logger.info("Generating spans...")
    kwargs = {} if slice_ else {"max_seconds": math.inf}
    generator = run._utils.SpanGenerator(dataset, **kwargs)
    with fork_rng(123):
        spans = [next(generator) for _ in tqdm.tqdm(range(num_samples), total=num_samples)]
    return_ = [Span(s.passage, s.slice) for s in tqdm.tqdm(spans)]
    logger.info(f"Finished generating spans! {mazel_tov()}")
    return return_


def _span_coverage(dataset: Dataset, spans: typing.List[Span]) -> float:
    """ Get the percentage of the `dataset` these `spans` cover. """
    logger.info("Getting span coverage of dataset...")
    alignments = set()
    for span in spans:
        alignments.update((span.passage, i) for i in range(span.slice.start, span.slice.stop))
    return len(alignments) / _get_num_alignments(dataset)


def _span_columns(spans: typing.List[Span]) -> typing.Dict[str, typing.List[typing.Any]]:
    """ Get generic statistics about `spans`. """
    logger.info("Getting %d generic span columns...", len(spans))
    iter_ = lambda s: range(len(s.alignments))
    return {
        "mistranscriptions": [s.mistranscriptions for s in spans],
        "seconds": [[round(s[i].audio_length, 2) for i in iter_(s)] for s in spans],
        "speed": [[round(s[i].seconds_per_character(), 2) for i in iter_(s)] for s in spans],
    }


def _visualize_spans(
    spans: typing.List[Span],
    columns: typing.List[str] = DEFAULT_COLUMNS,
    other_columns: typing.Dict[str, typing.List[typing.Any]] = {},
    get_audio: typing.Callable[[Span], np.ndarray] = lambda s: s.audio(),
    max_spans: int = 50,
):
    """Visualize spans as a table."""
    spans = spans[:max_spans]
    if len(spans) == 0:
        return
    logger.info("Visualizing %d spans..." % len(spans))
    df = pd.DataFrame([s.as_dict() for s in spans])
    assert AUDIO_COLUMN not in df.columns
    df[AUDIO_COLUMN] = _map(spans, get_audio)
    df = df[columns]
    for key, values in other_columns.items():
        df[key] = [str(v) for v in values[:max_spans]]
    formatters = {AUDIO_COLUMN: _audio_to_html}
    html = df.to_html(formatters=formatters, escape=False, justify="left", index=False)
    st.markdown(html, unsafe_allow_html=True)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")


def _maybe_analyze_dataset(dataset: Dataset):
    logger.info("Analyzing dataset...")
    st.header("Raw Dataset Analysis")
    st.markdown("In this section, we analyze the dataset prior to segmentation.")
    if not st.checkbox("Analyze", key=_maybe_analyze_dataset.__name__):
        return

    files = set(flatten([[p.audio_file for p in v] for v in dataset.values()]))
    aligned_seconds = sum(
        flatten([[p.aligned_audio_length() for p in v] for v in dataset.values()])
    )
    st.markdown(
        f"At a high-level, this dataset has:\n"
        f"- **{seconds_to_string(sum([f.length for f in files]))}** of audio\n"
        f"- **{seconds_to_string(aligned_seconds)}** of aligned audio\n"
        f"- **{_get_num_alignments(dataset):,}** alignments.\n"
    )

    passages = _random_sample(list(_get_passages(dataset)), 128)
    unigrams = list(_get_alignment_ngrams(passages, n=1))
    trigrams = list(_get_alignment_ngrams(passages, n=3))
    st.markdown(
        f"Below this analyzes a random sample of **{len(passages):,}** passages with "
        f"**{len(unigrams):,}** alignments..."
    )

    with beta_expander("Random Sample of Alignments"):
        for span in _random_sample(trigrams, 25):
            cols = st.beta_columns([2, 1, 1])
            rules = list(span.alignments[1].audio)
            labels = ["alignment", "alignment"]
            chart = _visualize_signal(span.audio(), span.audio_file.sample_rate, rules, labels)
            cols[0].altair_chart(chart, use_container_width=True)
            cols[1].markdown(
                f"- Script: **{span.script}**\n"
                f"- Loudness: **{span[1].rms()}**\n"
                f"- Edge loudness: **{span[1].rms_edges()}**\n"
                f"- Audio length: **{round(span[1].audio_length, 2)}**\n"
                f"- Num characters: **{len(span[1].script)}**\n"
                f"- **{round(span[1].rms_silence(), 2)}** Seconds of silence\n"
                f"- **{round(span[1].longest_rms_silence(), 2)}** Longest silence\n"
                f"- **{round(span[1].seconds_per_character(), 2)}** Seconds per character\n"
            )
            playlist = [span[1].audio(), span.audio()]
            html = "\n\n".join([_audio_to_html(a) for a in playlist])
            cols[2].markdown(html, unsafe_allow_html=True)

    with beta_expander("Survey of Pause Lengths (in seconds)"):
        st.write("The pause count for each length bucket:")
        iterator = list(_get_pause_lengths_in_seconds(dataset))
        _bucket_and_visualize(iterator, ALIGNMENT_PRECISION, x="Seconds")
        st.write(
            f"**{sum([p > 0 for p in iterator])/len(iterator):.2%}** of pauses are "
            "longer than zero."
        )

    samples = _random_sample(unigrams, 512)
    st.markdown(f"Below this analyzes a random sample of **{len(samples):,}** alignments...")

    with beta_expander("Random Sample of Alignments (Tabular)"):
        _visualize_spans(samples[:50], other_columns=_span_columns(samples[:50]))

    for func, title, unit, bucket_size in [
        (lambda s: s.audio_length, "Alignment Lengths", "Seconds", ALIGNMENT_PRECISION),
        (lambda s: len(s.script), "Alignment Lengths", "Characters", 1),
        (lambda s: s.seconds_per_character(), "Alignment Speeds", "Seconds per character", 0.01),
        (
            lambda s: s.rms_seconds_per_character(),
            "Alignment Speeds (rms)",
            "Seconds per character",
            0.01,
        ),
        (
            lambda s: s.rms_seconds_per_phoneme(),
            "Alignment Speeds (rms)",
            "Seconds per phoneme",
            0.01,
        ),
        (lambda s: s.rms(), "Loudness", "dB", 1),
        (lambda s: s.rms_edges()[0], "Onset Loudness", "dB", 5),
        (lambda s: s.rms_edges()[1], "Outset Loudness", "dB", 5),
        (lambda s: s.longest_rms_silence(), "Long Silences (rms)", "Seconds", 0.1),
    ]:
        with beta_expander(f"Survey of {title} (in {unit.lower()})"):
            st.write("The alignment count for each bucket:")
            _bucket_and_visualize(_map(samples, func), bucket_size, x=unit)
            filtered = [s for s in samples if not math.isnan(func(s))]
            for message, data in (
                ("The smallest valued alignments:", sorted(filtered, key=func)[:25]),
                ("The largest valued alignments:", sorted(filtered, key=func, reverse=True)[:25]),
            ):
                st.write(message)
                other_columns = _span_columns(data)
                other_columns["value"] = [func(s) for s in data]
                _visualize_spans(data, other_columns=other_columns)

    with beta_expander("Random Sample of Filtered Alignments"):
        is_include = lambda s: s.audio_length > 0.1 and s.seconds_per_character() >= 0.04
        display = [s for s in samples if is_include(s)]
        st.write(f"Filtered out {1 - (len(display) / len(samples)):.2%} of alignments.")
        _visualize_spans(display[:50], other_columns=_span_columns(display[:50]))

    logger.info(f"Finished analyzing dataset! {mazel_tov()}")


def _maybe_analyze_spans(dataset: Dataset, spans: typing.List[Span]):
    logger.info("Analyzing spans...")
    st.header("Dataset Segmentation Analysis")
    st.markdown("In this section, we analyze the dataset after segmentation via `Span`s. ")
    if not st.checkbox("Analyze", key=_maybe_analyze_spans.__name__):
        return

    audio_length = sum([s.audio_length for s in spans])
    st.markdown(
        f"There are **{len(spans)} ({seconds_to_string(audio_length)})** spans to analyze, "
        f"representing of **{_span_coverage(dataset, spans):.2%}** all alignments."
    )

    with beta_expander("Random Sample of Spans"):
        _visualize_spans(spans[:50], other_columns=_span_columns(spans[:50]))

    with beta_expander("Survey of Span Mistranscriptions"):
        st.write("The span mistranscription count for each length bucket:")
        mistranscriptions = [s.mistranscriptions for s in spans if len(s.mistranscriptions) > 0]
        st.write(
            f"Note that **{len(mistranscriptions) / len(spans):.2%}** spans have one or more "
            "mistranscriptions."
        )
        mistranscriptions = flatten([s.mistranscriptions for s in spans])
        iterator = [len(m[0]) for m in mistranscriptions if len(m[0]) > 0]
        _bucket_and_visualize(iterator, x="Characters")
        st.write("A random sample of mistranscriptions:")
        unaligned = [{"script": m[0], "transcript": m[1]} for m in mistranscriptions]
        st.table(unaligned)

    for func, title, unit, bucket_size in [
        (lambda s: s.silence(), "Silence", "Seconds", ALIGNMENT_PRECISION),
        (lambda s: s.num_silences(), "Number of Silences", "Silences", 1),
        (lambda s: s.longest_silence(), "Long Silences", "Seconds", ALIGNMENT_PRECISION),
        (lambda s: s.seconds_per_character(), "Speed", "Seconds per character", 0.01),
    ]:
        with beta_expander(f"Survey of {title} (in {unit.lower()})"):
            st.write("The alignment count for each bucket:")
            _bucket_and_visualize(_map(spans, func), bucket_size, x=unit)
            filtered = [s for s in spans if not math.isnan(func(s))]
            for message, data in (
                ("The smallest valued alignments:", sorted(filtered, key=func)[:25]),
                ("The largest valued alignments:", sorted(filtered, key=func, reverse=True)[:25]),
            ):
                st.write(message)
                other_columns = _span_columns(data)
                other_columns["value"] = [func(s) for s in data]
                _visualize_spans(data, other_columns=other_columns)

    logger.info(f"Finished analyzing spans! {mazel_tov()}")


def _maybe_analyze_filtered_spans(dataset: Dataset, spans: typing.List[Span]):
    """Filter out spans that are not ideal for training, and analyze the rest.

    NOTE: It's normal to have many consecutive words being said with no pauses between
    them, and often the final sounds of one word blend smoothly or fuse with the initial sounds of
    the next word. Learn more: https://en.wikipedia.org/wiki/Speech_segmentation.
    """
    logger.info("Analyzing filtered spans...")
    st.header("Dataset Filtered Segmentation Analysis")
    st.markdown(
        "In this section, we analyze the dataset after segmentation and filtering. "
        "This is the final step in our dataset preprocessing. "
    )
    if not st.checkbox("Analyze", key=_maybe_analyze_filtered_spans.__name__):
        return

    total = len(spans)
    _is_include = lambda a: a.audio_length > 0.1 and a.seconds_per_character() >= 0.04
    is_include = lambda s: (
        len(s.mistranscriptions) == 0 and (_is_include(s[0]) and _is_include(s[-1]))
    )

    excluded = [s for s, i in zip(spans, _map(spans, is_include)) if not i]
    spans = [s for s, i in zip(spans, _map(spans, is_include)) if i]
    st.markdown(
        f"The filtered segmentations represent **{len(spans) / total:.2%}** of the "
        f"original spans. In total, there are **{len(spans)} "
        f"({seconds_to_string(sum([s.audio_length for s in spans]))})** spans to analyze, "
        f"representing of **{_span_coverage(dataset, spans):.2%}** all alignments."
    )

    with beta_expander("Random Sample of Included Spans"):
        _visualize_spans(spans[:50], other_columns=_span_columns(spans[:50]))

    with beta_expander("Random Sample of Excluded Spans"):
        _visualize_spans(excluded[:50], other_columns=_span_columns(excluded[:50]))

    for label, func, bucket_size in [
        ("loudness", lambda s: _signal_to_loudness(s.audio(), s.audio_file.sample_rate), 1),
        ("speed", lambda s: s.seconds_per_character(), 0.01),
    ]:
        with beta_expander(f"Survey of Span {label.title()}"):
            st.write("The span count for each bucket:")
            _bucket_and_visualize(_map(spans, func), bucket_size)
            spans = [s for s in spans if not math.isnan(func(s))]
            display = sorted(spans, key=func)
            values = [func(s) for s in display]
            st.write("The smallest valued spans:")
            _visualize_spans(display[:50], other_columns={label: values[:50]})  # type: ignore
            st.write("The largest valued spans:")
            _visualize_spans(display[-50:], other_columns={label: values[-50:]})  # type: ignore


if __name__ == "__main__":
    run._config.configure()
    st.title("Dataset Dashboard")
    st.write("The dataset dashboard is an effort to understand our dataset and dataset processing.")

    sidebar = st.sidebar
    load_all = sidebar.checkbox("Load all dataset(s) by default")
    question = "Which dataset(s) do you want to load?"
    speaker_labels = [k.label for k in DATASETS.keys()]
    args = (question, speaker_labels, speaker_labels if load_all else None)
    speakers: typing.FrozenSet[str] = frozenset(st.sidebar.multiselect(*args))
    question = "How many spans(s) do you want to generate?"
    num_samples: int = sidebar.number_input(question, 0, None, 100)
    if len(speakers) == 0:
        st.stop()

    dataset = _get_dataset(speakers)
    spans = _get_spans(dataset, num_samples=num_samples)

    _maybe_analyze_dataset(dataset)
    _maybe_analyze_spans(dataset, spans)
    _maybe_analyze_filtered_spans(dataset, spans)

    logger.info(f"Done! {mazel_tov()}")
