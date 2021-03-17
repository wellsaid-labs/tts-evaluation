import base64
import functools
import io
import logging
import multiprocessing
import os
import pathlib
import typing

import altair as alt
import librosa.util
import numpy as np
import pandas as pd
import streamlit as st
import tqdm
from streamlit.server.server import Server
from third_party import session_state

import lib
import run
from lib.datasets import Passage
from run._config import Dataset

logger = logging.getLogger(__name__)


def is_streamlit_running() -> bool:
    try:
        Server.get_current()
        return True
    except RuntimeError:
        logger.warning("Streamlit isn't running!")
        return False


def get_session_state():
    return session_state.get(cache={})


_SessionCacheVar = typing.TypeVar("_SessionCacheVar", bound=typing.Callable[..., typing.Any])


class SessionCache(typing.Protocol):
    def __call__(self, func: _SessionCacheVar, **kwargs) -> _SessionCacheVar:
        ...


@typing.overload
def session_cache(func: _SessionCacheVar, **kwargs) -> _SessionCacheVar:
    ...


@typing.overload
def session_cache(func: None = None, **kwargs) -> SessionCache:
    ...


def session_cache(func: typing.Optional[typing.Callable] = None, **kwargs):
    """`lru_cache` wrapper for `streamlit` that caches accross reruns.

    Learn more: https://github.com/streamlit/streamlit/issues/2382
    """
    if not func:
        return functools.partial(session_cache, **kwargs)

    if not is_streamlit_running():
        return func

    session_state = get_session_state()
    if func.__qualname__ not in session_state["cache"]:
        logger.info("Creating `%s` cache.", func.__qualname__)
        session_state["cache"][func.__qualname__] = functools.lru_cache(**kwargs)(func)

    return session_state["cache"][func.__qualname__]


def clear_session_cache():
    logger.info("Clearing cache...")
    [v.cache_clear() for v in get_session_state()["cache"].values()]


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


def audio_to_html(audio: typing.Union[np.ndarray, pathlib.Path]) -> str:
    """Create an `audio` HTML element."""
    if isinstance(audio, pathlib.Path):
        return f'<audio controls src="/{_static_symlink(audio)}"></audio>'
    return f'<audio controls src="data:audio/wav;base64,{_audio_to_base64(audio)}"></audio>'


_MapInputVar = typing.TypeVar("_MapInputVar")
_MapReturnVar = typing.TypeVar("_MapReturnVar")


def map_(
    list_: typing.List[_MapInputVar],
    func: typing.Callable[[_MapInputVar], _MapReturnVar],
    chunk_size: int = 8,
    max_parallel: int = os.cpu_count() * 3,
    progress_bar: bool = True,
) -> typing.List[_MapReturnVar]:
    """ Apply `func` to `list_` in parallel. """
    with multiprocessing.pool.ThreadPool(processes=max_parallel) as pool:
        iterator = pool.imap(func, list_, chunksize=chunk_size)
        if progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(list_))
        return list(iterator)


@session_cache(maxsize=None)
def read_wave_audio(*args, **kwargs) -> np.ndarray:
    """ Read audio slice, and cache. """
    return lib.audio.read_wave_audio(*args, **kwargs)


def span_audio(span: lib.datasets.Span) -> np.ndarray:
    """Get `span` audio using cached `read_audio_slice`."""
    start = span.passage.alignments[span.slice][0].audio[0]
    return read_wave_audio(span.passage.audio_file, start, span.audio_length)


def passage_audio(passage: lib.datasets.Passage) -> np.ndarray:
    """Get `span` audio using cached `read_audio_slice`."""
    start = passage.alignments[0].audio[0]
    end = passage.alignments[-1].audio[-1]
    return read_wave_audio(passage.audio_file, start, end - start)


@lib.utils.log_runtime
@session_cache(maxsize=None)
def get_dataset(speaker_labels: typing.FrozenSet[str]) -> run._config.Dataset:
    """Load dataset subset, and cache. """
    logger.info("Loading dataset...")
    datasets = {k: v for k, v in run._config.DATASETS.items() if k.label in speaker_labels}
    dataset = run._utils.get_dataset(datasets)
    logger.info(f"Finished loading dataset! {lib.utils.mazel_tov()}")
    return dataset


@session_cache(maxsize=None)
def fast_grapheme_to_phoneme(text: str):
    """Fast grapheme to phoneme, cached."""
    return lib.text._line_grapheme_to_phoneme([text], separator="|")[0]


def integer_signal_to_floating(signal: np.ndarray) -> np.ndarray:
    """Transform `signal` from an integer data type to floating data type."""
    is_floating = np.issubdtype(signal.dtype, np.floating)
    return signal / (1.0 if is_floating else np.abs(np.iinfo(signal.dtype).min))


def make_signal_chart(
    signal: np.ndarray,
    sample_rate: int,
    max_sample_rate: int = 2000,
    x: str = "seconds",
    y: typing.Tuple[str, str] = ("y_min", "y_max"),
) -> alt.Chart:
    """Make `altair.Chart` for `signal` similar to `librosa.display.waveplot`.

    Learn more about envelopes: https://en.wikipedia.org/wiki/Envelope_detector
    """
    signal = integer_signal_to_floating(signal)
    ratio = sample_rate // max_sample_rate
    frames = librosa.util.frame(signal, ratio, ratio, axis=0)  # type: ignore
    envelope = np.max(np.abs(frames), axis=-1)
    assert frames.shape[1] == ratio
    assert frames.shape[0] == envelope.shape[0]
    ticks = np.arange(0, envelope.shape[0] * ratio / sample_rate, ratio / sample_rate)
    return (
        alt.Chart(pd.DataFrame({x: ticks, y[0]: -envelope, y[1]: envelope}))
        .mark_area()
        .encode(
            x=alt.X(x, type="quantitative"),
            y=alt.Y(y[0], scale=alt.Scale(domain=(-1.0, 1.0)), type="quantitative"),
            y2=alt.Y2(y[1]),
        )
    )


def make_interval_chart(
    x_min: np.ndarray,
    x_max: np.ndarray,
    opacity=0.3,
    color="#85C5A6",
    stroke="#000",
    strokeWidth=1,
    strokeOpacity=0.3,
    **kwargs,
):
    """Make `altair.Chart` for the intervals between `x_min` and `x_max`."""
    return (
        alt.Chart(pd.DataFrame({"x_min": x_min, "x_max": x_max}))
        .mark_rect(
            opacity=opacity,
            color=color,
            stroke=stroke,
            strokeWidth=strokeWidth,
            strokeOpacity=strokeOpacity,
            **kwargs,
        )
        .encode(x=alt.X("x_min", type="quantitative"), x2=alt.X2("x_max"))
    )


def has_alnum(s: str):
    return any(c.isalnum() for c in s)


def dataset_passages(dataset: Dataset) -> typing.Iterator[Passage]:
    """ Get all passages in `dataset`. """
    for _, passages in dataset.items():
        yield from passages
