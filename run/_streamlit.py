import base64
import functools
import io
import logging
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import typing
import zipfile

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
from run._config import Dataset
from run.data._loader import Passage

logger = logging.getLogger(__name__)

# Learn more:
# https://github.com/streamlit/streamlit/issues/400#issuecomment-648580840
# https://github.com/streamlit/streamlit/issues/1567
STREAMLIT_WEB_ROOT_PATH = pathlib.Path(st.__file__).parent / "static"
STREAMLIT_STATIC_PATH = STREAMLIT_WEB_ROOT_PATH / "static"

# NOTE: These are the WSL TTS directories served by Streamlit.
STREAMLIT_STATIC_PRIVATE_PATH = STREAMLIT_STATIC_PATH / "_wsl_tts"
STREAMLIT_STATIC_TEMP_PATH = STREAMLIT_STATIC_PRIVATE_PATH / "temp"
STREAMLIT_STATIC_SYMLINK_PATH = STREAMLIT_STATIC_PRIVATE_PATH / "symlink"


def is_streamlit_running() -> bool:
    """ Check if `streamlit` server has been initialized. """
    try:
        Server.get_current()
        return True
    except RuntimeError:
        logger.warning("Streamlit isn't running!")
        return False


def get_session_state() -> dict:
    """Get a reference to a session state represented as a `dict`. """
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
    """Clear the cache for `session_cache`."""
    logger.info("Clearing cache...")
    [v.cache_clear() for v in get_session_state()["cache"].values()]


def make_symlink(target: pathlib.Path) -> pathlib.Path:
    """System link `target` to `STREAMLIT_STATIC_SYMLINK_PATH` and return the linked location."""
    STREAMLIT_STATIC_SYMLINK_PATH.mkdir(exist_ok=True, parents=True)
    path = STREAMLIT_STATIC_SYMLINK_PATH / target.relative_to(lib.environment.ROOT_PATH)
    if not path.exists():
        path.parent.mkdir(exist_ok=True, parents=True)
        path.symlink_to(target)
    return path.relative_to(STREAMLIT_WEB_ROOT_PATH)


def _audio_to_base64(audio: np.ndarray, **kwargs) -> str:
    """Encode audio into a `base64` string."""
    in_memory_file = io.BytesIO()
    lib.audio.write_audio(in_memory_file, audio, **kwargs)
    return base64.b64encode(in_memory_file.read()).decode("utf-8")


def rmtree_streamlit_static_temp_dir():
    """Destroy the our Streamlit temporary directory."""
    assert pathlib.Path(st.__file__).parent in STREAMLIT_STATIC_TEMP_PATH.parents
    if STREAMLIT_STATIC_TEMP_PATH.exists():
        shutil.rmtree(STREAMLIT_STATIC_TEMP_PATH)


def audio_path_to_html(audio: pathlib.Path, attrs="controls") -> str:
    """Create an audio HTML element from an audio path."""
    return f'<audio {attrs} src="/{make_symlink(audio)}"></audio>'


def audio_to_base64_html(audio: np.ndarray, attrs="controls", **kwargs) -> str:
    """Create an audio HTML element from a numpy array."""
    data = _audio_to_base64(audio, **kwargs)
    return f'<audio {attrs} src="data:audio/wav;base64,{data}"></audio>'


def get_static_temp_path(name) -> pathlib.Path:
    """Get a temp path in a temp directory within `STREAMLIT_STATIC_TEMP_PATH`."""
    STREAMLIT_STATIC_TEMP_PATH.mkdir(exist_ok=True, parents=True)
    temp_dir = pathlib.Path(tempfile.mkdtemp(dir=STREAMLIT_STATIC_TEMP_PATH))
    return temp_dir / name


def audio_to_static_temp_path(audio: np.ndarray, name: str = "audio.wav", **kwargs) -> pathlib.Path:
    """Create an audio file in `STREAMLIT_STATIC_TEMP_PATH`. """
    temp_path = get_static_temp_path(name)
    lib.audio.write_audio(temp_path, audio, **kwargs)
    return temp_path


def audio_temp_path_to_html(temp_path: pathlib.Path, attrs="controls"):
    """Create an audio HTML element for the audio file at `temp_path."""
    return f'<audio {attrs} src="/{temp_path.relative_to(STREAMLIT_WEB_ROOT_PATH)}"></audio>'


def audio_to_html(audio: np.ndarray, attrs="controls", **kwargs) -> str:
    """Create an audio HTML element from a numpy array."""
    temp_path = audio_to_static_temp_path(audio, **kwargs)
    return audio_temp_path_to_html(temp_path, attrs=attrs)


def zip_to_html(name: str, label: str, paths: typing.List[pathlib.Path]) -> str:
    """ Make a zipfile named `name` that can be downloaded with a button called `label`."""
    temp_path = get_static_temp_path(name)
    with zipfile.ZipFile(temp_path, "w") as zip:
        for path in paths:
            zip.write(path, arcname=path.name)
    temp_path = temp_path.relative_to(STREAMLIT_WEB_ROOT_PATH)
    return f'<a href="/{temp_path}" download="{name}">{label}</a>'


def write_audio(*args, **kwargs):
    st.markdown(audio_to_html(*args, **kwargs), unsafe_allow_html=True)


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


def span_audio(span: run.data._loader.Span) -> np.ndarray:
    """Get `span` audio using cached `read_wave_audio`."""
    return read_wave_audio(span.passage.audio_file, span.audio_start, span.audio_length)


def passage_audio(passage: run.data._loader.Passage) -> np.ndarray:
    """Get `span` audio using cached `read_wave_audio`."""
    start = passage.first.audio[0]
    return read_wave_audio(passage.audio_file, start, passage.aligned_audio_length())


@session_cache(maxsize=None)
def get_dataset(speaker_labels: typing.FrozenSet[str]) -> run._config.Dataset:
    """Load dataset subset, and cache. """
    logger.info("Loading dataset...")
    with st.spinner(f"Loading dataset(s): {','.join(list(speaker_labels))}"):
        datasets = {k: v for k, v in run._config.DATASETS.items() if k.label in speaker_labels}
        dataset = run._utils.get_dataset(datasets)
        logger.info(f"Finished loading {set(speaker_labels)} dataset(s)! {lib.utils.mazel_tov()}")
    return dataset


@session_cache(maxsize=None)
def fast_grapheme_to_phoneme(text: str):
    """Fast grapheme to phoneme, cached."""
    return lib.text._line_grapheme_to_phoneme([text], separator="|")[0]


def make_signal_chart(
    signal: np.ndarray,
    sample_rate: int,
    max_sample_rate: int = 1000,
    x: str = "seconds",
    y: typing.Tuple[str, str] = ("y_min", "y_max"),
) -> alt.Chart:
    """Make `altair.Chart` for `signal` similar to `librosa.display.waveplot`.

    Learn more about envelopes: https://en.wikipedia.org/wiki/Envelope_detector
    """
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
    intervals: typing.List[typing.Tuple[float, float]],
    fillOpacity=0.3,
    color="white",
    stroke="white",
    strokeWidth=3,
    strokeOpacity=0.8,
    **kwargs,
):
    """Make `altair.Chart` for the `intervals`."""
    source = {"x_min": [i[0] for i in intervals], "x_max": [i[1] for i in intervals]}
    return (
        alt.Chart(pd.DataFrame(source))
        .mark_rect(
            fillOpacity=fillOpacity,
            color=color,
            stroke=stroke,
            strokeWidth=strokeWidth,
            strokeOpacity=strokeOpacity,
            **kwargs,
        )
        .encode(x=alt.X("x_min", type="quantitative"), x2=alt.X2("x_max"))
    )


def dataset_passages(dataset: Dataset) -> typing.Iterator[Passage]:
    """ Get all passages in `dataset`. """
    for _, passages in dataset.items():
        yield from passages


def st_data_frame(df: pd.DataFrame):
    """Display the `DataFrame` in the `streamlit` app."""
    df = df.replace({"\n": "<br>"}, regex=True)
    # NOTE: Temporary fix based on this issue / pr: https://github.com/streamlit/streamlit/pull/3038
    html = "<style>tr{background-color: transparent !important;}</style>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown(df.to_markdown(index=False), unsafe_allow_html=True)
