""" Streamlit application for reviewing the dataset.

Usage:
    $ PYTHONPATH=. streamlit run run/data/review_dataset.py
"""
from __future__ import annotations

import base64
import collections
import contextlib
import copy
import dataclasses
import functools
import io
import logging
import math
import multiprocessing.pool
import pathlib
import random
import typing

import altair as alt
import librosa
import numpy as np
import pandas as pd
import streamlit as st
import torch
import tqdm

import lib
import run
from lib.utils import clamp, mazel_tov
from run._config import Dataset

lib.environment.set_basic_logging_config(reset=True)
alt.data_transformers.disable_max_rows()
logger = logging.getLogger(__name__)

ALIGNMENT_PRECISION = 0.1
AUDIO_COLUMN = "audio"
DEFAULT_COLUMNS = [AUDIO_COLUMN, "script", "transcript"]
HASH_FUNCS = {lib.datasets.Passage: lambda p: p.key}

read_audio_slice = functools.lru_cache(maxsize=None)(lib.audio.read_audio_slice)
read_audio = functools.lru_cache(maxsize=None)(lib.audio.read_audio)


def _static_symlink(
    target: pathlib.Path,
    root: pathlib.Path = pathlib.Path(st.__file__).parent / "static",
    static: pathlib.Path = pathlib.Path("static") / "_private",
) -> pathlib.Path:
    """System link `target` to `root / static`, and return the linked location.

    Learn more:
    https://github.com/st/st/issues/400#issuecomment-648580840
    https://github.com/st/st/issues/1567
    """
    assert root.exists()
    (root / static).mkdir(exist_ok=True)
    target = target.relative_to(lib.environment.ROOT_PATH)
    if not (root / static / target).exists():
        (root / static / target.parent).mkdir(exist_ok=True, parents=True)
        (root / static / target).symlink_to(lib.environment.ROOT_PATH / target)
    return static / target


@dataclasses.dataclass(frozen=True)
class Span(lib.datasets.Span):
    """ `lib.datasets.Span` with additional attributes. """

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

    def insta_rms(self, second: float, interval: float = 0.005) -> float:
        """ Get the instantaneous loudness at `second`. """
        sample_rate = self.audio_file.sample_rate
        max_ = self.audio.shape[0]
        slice_ = (
            round((second - interval) * sample_rate),
            round((second + interval) * sample_rate),
        )
        audio = self.audio[clamp(slice_[0], min_=0) : clamp(slice_[1], max_=max_)]
        db = lib.audio.amplitude_to_db(torch.tensor(lib.audio.signal_to_rms(audio))).item()
        return round(db, 1)

    def __getitem__(self, key) -> Span:
        span = super().__getitem__(key)
        return Span(span.passage, span.span)

    def as_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.type != lib.datasets.Passage
        }


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
    """Visualize a waveform, similar to:
    https://librosa.org/doc/main/generated/librosa.display.waveplot.html"""
    ratio = sample_rate // max_sample_rate
    frames = librosa.util.frame(signal, frame_length=ratio, hop_length=ratio)  # type: ignore
    downsampled = np.max(np.abs(frames), axis=0)
    data = {
        "seconds": np.arange(0, signal.shape[0] / sample_rate, ratio / sample_rate),
        "y_max": downsampled,
        "y_min": -downsampled,
    }
    waveform = alt.Chart(pd.DataFrame(data)).mark_area()
    waveform = waveform.encode(x="seconds:Q", y="y_min:Q", y2="y_max:Q")
    line = alt.Chart(pd.DataFrame({"x": rules})).mark_rule(color="darkred").encode(x="x")
    return (line + waveform).interactive()


def _bucket_and_visualize(
    iterator: typing.Iterable[typing.Union[float, int]],
    bucket_size: float = 1,
    ndigits: int = 7,
):
    """ Bucket `iterator` and display a bar chart. """
    buckets = collections.defaultdict(int)
    nan_count = 0
    for item in iterator:
        if math.isnan(item):
            nan_count += 1
        else:
            buckets[round(bucket_size * round(item / bucket_size), ndigits)] += 1
    logger.warning("Ignoring %d NaNs...", nan_count)
    df = pd.DataFrame({"buckets": buckets.keys(), "count": buckets.values()})
    df = df.set_index("buckets")
    st.bar_chart(df)


def _get_pause_lengths_in_seconds(dataset: Dataset) -> typing.Iterator[float]:
    for _, passages in dataset.items():
        for passage in passages:
            for curr, next in zip(passage.alignments, passage.alignments[1:]):
                yield next.audio[0] - curr.audio[1]


def _ngrams(list_: typing.Sequence, n: int = 1) -> typing.Iterator[slice]:
    left = math.floor((n - 1) / 2)
    right = math.ceil((n + 1) / 2)
    iterator = range(left, len(list_) - right + 1)
    yield from (slice(i - left, i + right) for i in iterator)


_get_ngrams = lambda l, n: [l[s] for s in _ngrams(l, n)]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=1) == [[1], [2], [3], [4], [5], [6]]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=2) == [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
assert _get_ngrams([1, 2, 3, 4, 5, 6], n=3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]
assert _get_ngrams([1, 2, 3, 4, 5], n=3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5]]


def _get_alignment_ngrams(dataset: Dataset, n: int = 1) -> typing.Iterator[Span]:
    """ Get ngram `Span`s with `n` alignments. """
    for _, passages in dataset.items():
        for passage in tqdm.tqdm(passages):
            yield from (Span(passage, s) for s in _ngrams(passage.alignments, n=n))


@st.cache(allow_output_mutation=True, show_spinner=False)
def _get_dataset(speaker_names: typing.List[str]) -> Dataset:
    """Load dataset."""
    logger.info("Loading dataset...")
    datasets = {k: v for k, v in run._config.DATASETS.items() if k.name in speaker_names}
    dataset = run._config.get_dataset(datasets)
    logger.info(f"Finished loading dataset! {mazel_tov()}")
    return dataset


@st.cache(allow_output_mutation=True, show_spinner=False, hash_funcs=HASH_FUNCS)
def _get_spans(dataset: Dataset, slice_: bool, num_samples: int) -> typing.List[Span]:
    """Generate spans from our datasets."""
    logger.info("Generating spans...")
    kwargs = {} if slice_ else {"max_seconds": math.inf}
    generator = run._config.span_generator(dataset, **kwargs)
    spans = [next(generator) for _ in tqdm.tqdm(range(num_samples), total=num_samples)]
    return_ = [Span(s.passage, s.span) for s in tqdm.tqdm(spans)]
    logger.info(f"Finished generating spans! {mazel_tov()}")
    return return_


@st.cache(allow_output_mutation=True, show_spinner=False)
def _load_amepd():
    return lib.text._load_amepd()


def _audio_to_base64(audio: np.ndarray) -> str:
    """Encode audio into a `base64` string."""
    in_memory_file = io.BytesIO()
    lib.audio.write_audio(in_memory_file, audio)
    return base64.b64encode(in_memory_file.read()).decode("utf-8")


def _audio_formatter(audio: typing.Union[np.ndarray, pathlib.Path]) -> str:
    """ Create an `audio` HTML element. """
    if isinstance(audio, pathlib.Path):
        return f'<audio controls="" src="/{_static_symlink(audio)}"></audio>'
    return f'<audio controls="" src="data:audio/wav;base64,{_audio_to_base64(audio)}"></audio>'


def _visualize_spans(
    spans: typing.List[Span],
    columns: typing.List[str] = DEFAULT_COLUMNS,
    additional_data: typing.Dict[str, typing.List] = {},
    slice_: bool = True,
    max_spans: int = 50,
):
    """ Visualize spans as a table. """
    spans = spans[:max_spans]
    logger.info("Visualizing %d spans..." % len(spans))
    df = pd.DataFrame([s.as_dict() for s in spans])
    with multiprocessing.pool.ThreadPool() as pool:
        lambda_ = lambda s: s.audio if slice_ else s.audio_file.path
        df[AUDIO_COLUMN] = pool.map(lambda_, spans)
    df = df[columns]
    for key, values in additional_data.items():
        df[key] = values[:max_spans]
    html = df.to_html(formatters={AUDIO_COLUMN: _audio_formatter}, escape=False, justify="left")
    st.markdown(html, unsafe_allow_html=True)
    logger.info(f"Finished visualizing spans! {mazel_tov()}")


class _SidebarConfig(typing.NamedTuple):
    speakers: typing.Set[str]
    num_samples: int
    slice_: bool


def _get_sidebar_config() -> _SidebarConfig:
    sidebar = st.sidebar
    question = "Which dataset(s) do you want to load?"
    speaker_names = [k.name for k in run._config.DATASETS.keys()]
    speakers: typing.Set[str] = set(st.sidebar.multiselect(question, speaker_names))

    question = "How many spans(s) do you want to generate?"
    num_samples = sidebar.number_input(question, 0, None, 2500)

    slice_ = sidebar.checkbox("Generate Spans", value=True)
    return _SidebarConfig(speakers, num_samples, slice_)


def _get_seconds_per_phoneme(word: str, audio_length: float) -> float:
    words = [w.strip("., \t\n\f").upper() for w in word.split("-")]
    if all(w in _load_amepd() for w in words):
        num_phonemes = sum([len(_load_amepd()[w][0].pronunciation) for w in words])
        return round(audio_length / num_phonemes, 2)
    return math.nan


def _analyze_dataset(dataset: Dataset):
    logger.info("Analyzing dataset...")

    unigrams = list(_get_alignment_ngrams(dataset, n=1))
    trigrams = list(_get_alignment_ngrams(dataset, n=3))

    with beta_expander("Alignment Random Sample Analysis"):
        st.write("A sample of the alignments.")
        for span in random.sample(trigrams, 25):
            cols = st.beta_columns([2, 1])
            rules = list(span.alignments[1].audio)
            cols[0].altair_chart(_visualize_signal(span.audio, rules), use_container_width=True)

            start_rms = span[1].insta_rms(0)
            end_rms = span[1].insta_rms(span[1].audio_length)
            seconds_per_phoneme = _get_seconds_per_phoneme(span[1].script, span[1].audio_length)
            cols[1].markdown(
                f"- Instantaneous Loudness: ({start_rms}, {end_rms})\n"
                f"- {round(span[1].audio_length / len(span[1].script), 2)} Seconds per character\n"
                f"- {seconds_per_phoneme} Seconds per phoneme\n"
            )

            if span.audio_length > 0:
                short = _audio_formatter(span[1].audio)
                long = _audio_formatter(span.audio)
                markdown = f"**Listen:** '{span.script}'\n\n{short}\n\n{long}"
                cols[1].markdown(markdown, unsafe_allow_html=True)

    with beta_expander("Pause Length Analysis"):
        st.write("The pause count for each length, in seconds, bucket.")
        _bucket_and_visualize(_get_pause_lengths_in_seconds(dataset), ALIGNMENT_PRECISION)

    with beta_expander("Alignment Length (in seconds) Analysis"):
        st.write("The alignment count for each length, in seconds, bucket.")
        _bucket_and_visualize([s.audio_length for s in unigrams], ALIGNMENT_PRECISION)

    with beta_expander("Alignment Length (in characters) Analysis"):
        st.write("The alignment count for each length, in characters, bucket.")
        _bucket_and_visualize([len(s.script) for s in unigrams], ALIGNMENT_PRECISION)
        st.write("Longest alignments: ")
        columns = lambda s: {"script": s.script, "transcript": s.transcript}
        key = lambda d: len(d["script"])
        st.table(sorted([columns(s) for s in unigrams if len(s.script) > 25], key=key))

    with beta_expander("Alignment Speed Analysis"):
        st.write("The alignment count for each speed, in seconds per character, bucket.")
        _bucket_and_visualize([s.seconds_per_character for s in unigrams], 0.01)
        filtered = [s for s in unigrams if s.seconds_per_character > 0.1]
        filtered = sorted(filtered, key=lambda s: s.seconds_per_character)
        _visualize_spans(filtered, DEFAULT_COLUMNS + ["seconds_per_character"])

    logger.info(f"Finished analyzing dataset! {mazel_tov()}")


def _analyze_spans(spans: typing.List[Span]):
    logger.info("Analyzing spans...")

    with multiprocessing.pool.ThreadPool() as pool:
        iterator = pool.imap_unordered(lambda s: s.insta_rms(0), spans, chunksize=16)
        values = list(tqdm.tqdm(iterator, total=len(spans)))
        with beta_expander("Span Onset RMS dB Analysis"):
            st.write("The span onset RMS dB count for each loudness, in dB, bucket.")
            _bucket_and_visualize(values)

    with beta_expander("Span Mistranscription Analysis"):
        st.write("The span mistranscription count for each length, in characters, bucket.")
        mistranscriptions = [s.mistranscriptions for s in spans if len(s.mistranscriptions) > 0]
        st.write(f"{len(mistranscriptions)} spans have one or more mistranscriptions.")
        mistranscriptions = lib.utils.flatten([s.mistranscriptions for s in spans])
        _bucket_and_visualize([len(m[0]) for m in mistranscriptions if len(m[0]) > 0])
        st.write("Mistranscriptions: ")
        unaligned = [{"script": m[0], "transcript": m[1]} for m in mistranscriptions]
        st.table(random.sample(unaligned, 50))

    logger.info(f"Finished analyzing spans! {mazel_tov()}")


def _analyze_filtered_spans(spans: typing.List[Span], sidebar_config: _SidebarConfig):
    """Filter out spans that are not ideal for training."""
    with beta_expander("Filtered Span Analysis"):
        with multiprocessing.pool.ThreadPool() as pool:
            lambda_ = lambda s: (s.insta_rms(0), s.insta_rms(s.audio_length))
            iterator = pool.imap_unordered(lambda_, spans, chunksize=16)
            rms = list(tqdm.tqdm(iterator, total=len(spans)))

        total = len(spans)
        is_include = lambda s, r: r[0] < -40 and r[1] < -40 and len(s.mistranscriptions) == 0
        res = [(s, r) for s, r in zip(spans, rms) if is_include(s, r)]
        st.write(
            f"Filtered out **{round(((total - len(res)) / total) * 100)}%** of spans, and the "
            "remaining spans should be ideal for training..."
        )
        additional_columns = {
            "onset_rms": [r[0] for (_, r) in res],
            "outset_rms": [r[1] for (_, r) in res],
            "alignments_seconds_per_character": [
                [str(round(s[i].seconds_per_character, 2)) for i, _ in enumerate(s.alignments)]
                for (s, _) in res
            ],
            "aligned_words": [[s[i].script for i, _ in enumerate(s.alignments)] for (s, _) in res],
        }
        _visualize_spans(
            [s for (s, _) in res],
            DEFAULT_COLUMNS,
            additional_columns,
            sidebar_config.slice_,
            max_spans=25,
        )


def main():
    run._config.configure()
    st.set_page_config(layout="wide")
    st.title("Training Data Review")

    sidebar_config = _get_sidebar_config()
    if len(sidebar_config.speakers) == 0:
        st.stop()

    dataset: Dataset = copy.deepcopy(_get_dataset(sidebar_config.speakers))
    st.header("Dataset Analysis")
    _analyze_dataset(dataset)
    spans: typing.List[Span]
    spans = copy.deepcopy(_get_spans(dataset, sidebar_config.slice_, sidebar_config.num_samples))
    st.header("Span Analysis")
    _analyze_spans(spans)
    _analyze_filtered_spans(spans, sidebar_config)
    logger.info(f"Done! {mazel_tov()}")


if __name__ == "__main__":
    main()
