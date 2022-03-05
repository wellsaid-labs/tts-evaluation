import functools
import logging
import multiprocessing
import os
import pathlib
import pickle
import shutil
import tempfile
import typing
import zipfile

import numpy as np
import tqdm
from streamlit.server.server import Server
from third_party import LazyLoader, session_state

import lib
import run
from run._config import Dataset
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, package_tts
from run.data._loader import Passage

if typing.TYPE_CHECKING:  # pragma: no cover
    import altair as alt
    import librosa
    import librosa.util
    import pandas as pd
    import streamlit as st
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    alt = LazyLoader("alt", globals(), "altair")
    pd = LazyLoader("pd", globals(), "pandas")
    st = LazyLoader("st", globals(), "streamlit")


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

# NOTE: This is a default script that can be used with Streamlit apps, if need be.
DEFAULT_SCRIPT = (
    "Your creative life will evolve in ways that you can’t possibly imagine. Trust"
    " your gut. Don’t overthink it. And allow yourself a little room to play."
)


def is_streamlit_running() -> bool:
    """Check if `streamlit` server has been initialized."""
    try:
        Server.get_current()
        return True
    except RuntimeError:
        return False


def get_session_state() -> dict:
    """Get a reference to a session state represented as a `dict`.

    TODO: Upgrade to `streamlit`s official `session_state` implementation, learn more:
    https://blog.streamlit.io/session-state-for-streamlit/
    """
    return session_state.get(cache={})


_WrappedFunction = typing.TypeVar("_WrappedFunction", bound=typing.Callable[..., typing.Any])


def pickle_cache(func: _WrappedFunction = None, **kwargs) -> _WrappedFunction:
    """Cache the inputs and outputs of `func` in a `pickle` format.

    NOTE: Due this the below bug, it's better to cache using `pickle`, so that old objects
    don't stick around. Learn more:
    https://github.com/streamlit/streamlit/issues/2379
    """
    if not func:
        return functools.partial(pickle_cache, **kwargs)  # type: ignore

    cache = {}

    @functools.wraps(func)
    def decorator(*args, **kwargs):
        key = (pickle.dumps(args), pickle.dumps(kwargs))
        if key in cache:
            return pickle.loads(cache[key])
        result = func(*args, **kwargs)
        cache[key] = pickle.dumps(result)
        return result

    def cache_clear():
        cache.clear()

    decorator.cache_clear = cache_clear

    return typing.cast(_WrappedFunction, decorator)


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
        session_state["cache"][func.__qualname__] = pickle_cache(**kwargs)(func)

    return session_state["cache"][func.__qualname__]


def clear_session_cache():
    """Clear the cache for `session_cache`."""
    logger.info("Clearing cache...")
    [v.cache_clear() for v in get_session_state()["cache"].values()]


@session_cache(maxsize=None)
def load_tts(checkpoints_key: Checkpoints):
    return package_tts(*CHECKPOINTS_LOADERS[checkpoints_key]())


@session_cache(maxsize=None)
def rmtree_streamlit_static_temp_dir():
    """Destroy the our Streamlit temporary directory.

    NOTE: With `session_cache`, this function runs once per session.
    """
    if is_streamlit_running():
        assert pathlib.Path(st.__file__).parent in STREAMLIT_STATIC_TEMP_PATH.parents
        if STREAMLIT_STATIC_TEMP_PATH.exists():
            message = "Clearing temporary files at %s..."
            logger.info(message, STREAMLIT_STATIC_TEMP_PATH.relative_to(lib.environment.ROOT_PATH))
            shutil.rmtree(STREAMLIT_STATIC_TEMP_PATH)


rmtree_streamlit_static_temp_dir()


# A `Path` to a file or directory accessible via HTTP in the streamlit app.
WebPath = typing.NewType("WebPath", pathlib.Path)
RelativeUrl = typing.NewType("RelativeUrl", str)


def make_temp_web_dir() -> WebPath:
    """Make a temporary directory accessible via HTTP in the streamlit app."""
    STREAMLIT_STATIC_TEMP_PATH.mkdir(exist_ok=True, parents=True)
    return WebPath(pathlib.Path(tempfile.mkdtemp(dir=STREAMLIT_STATIC_TEMP_PATH)))


def web_path_to_url(path: WebPath) -> RelativeUrl:
    """Get the related URL given a `WebPath`."""
    return RelativeUrl(f"/{path.relative_to(STREAMLIT_WEB_ROOT_PATH)}")


def path_to_web_path(path: pathlib.Path) -> WebPath:
    """Get a system linked web path given a `path`."""
    STREAMLIT_STATIC_SYMLINK_PATH.mkdir(exist_ok=True, parents=True)
    web_path = STREAMLIT_STATIC_SYMLINK_PATH / str(path.resolve()).strip("/")
    if not web_path.exists():
        web_path.parent.mkdir(exist_ok=True, parents=True)
        web_path.symlink_to(path)
    return WebPath(web_path)


def audio_to_web_path(audio: np.ndarray, name: str = "audio.wav", **kwargs) -> WebPath:
    web_path = make_temp_web_dir() / name
    lib.audio.write_audio(web_path, audio, **kwargs)
    return web_path


def audio_to_html(
    audio: np.ndarray, name: str = "audio.wav", attrs: str = "controls", **kwargs
) -> str:
    """Create an audio HTML element from a numpy array."""
    web_path = audio_to_web_path(audio, name, **kwargs)
    return f'<audio {attrs} src="{web_path_to_url(web_path)}"></audio>'


def paths_to_html_download_link(
    name: str,
    label: str,
    paths: typing.List[pathlib.Path],
    archive_paths: typing.Optional[typing.List[pathlib.Path]] = None,
) -> str:
    """Make a zipfile named `name` that can be downloaded with a button called `label`."""
    web_path = make_temp_web_dir() / name
    archive_paths_ = [p.name for p in paths] if archive_paths is None else archive_paths
    with zipfile.ZipFile(web_path, "w") as file_:
        for path, archive_path in zip(paths, archive_paths_):
            file_.write(path, arcname=archive_path)
    return f'<a href="{web_path_to_url(web_path)}" download="{name}">{label}</a>'


_MapInputVar = typing.TypeVar("_MapInputVar")
_MapReturnVar = typing.TypeVar("_MapReturnVar")


def map_(
    list_: typing.List[_MapInputVar],
    func: typing.Callable[[_MapInputVar], _MapReturnVar],
    chunk_size: int = 8,
    max_parallel: int = os.cpu_count() * 3,
    progress_bar: bool = True,
) -> typing.List[_MapReturnVar]:
    """Apply `func` to `list_` in parallel."""
    with multiprocessing.pool.ThreadPool(processes=max_parallel) as pool:
        iterator = pool.imap(func, list_, chunksize=chunk_size)
        if progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(list_))
        return list(iterator)


@session_cache(maxsize=None)
def read_wave_audio(*args, **kwargs) -> np.ndarray:
    """Read audio slice, and cache."""
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
    """Load dataset subset, and cache."""
    logger.info("Loading dataset...")
    with st.spinner(f"Loading dataset(s): {','.join(list(speaker_labels))}"):
        datasets = {k: v for k, v in run._config.DATASETS.items() if k.label in speaker_labels}
        dataset = run._utils.get_dataset(datasets)
        logger.info(f"Finished loading {set(speaker_labels)} dataset(s)! {lib.utils.mazel_tov()}")
    return dataset


@session_cache(maxsize=None)
def get_dev_dataset() -> run._config.Dataset:
    """Load dev dataset, and cache."""
    with st.spinner("Loading dataset..."):
        _, dev_dataset = run._utils.split_dataset(run._utils.get_dataset())
    return dev_dataset


@session_cache(maxsize=None)
def fast_grapheme_to_phoneme(text: str):
    """Fast grapheme to phoneme, cached."""
    return lib.text.grapheme_to_phoneme([text], separator="|")[0]


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
    """Get all passages in `dataset`."""
    for _, passages in dataset.items():
        yield from passages


@session_cache(maxsize=None)
def load_en_core_web_md(*args, **kwargs):
    return lib.text.load_en_core_web_md(*args, **kwargs)


def st_data_frame(df: pd.DataFrame):
    """Display the `DataFrame` in the `streamlit` app."""
    df = df.replace({"\n": "<br>"}, regex=True)
    # NOTE: Temporary fix based on this issue / pr: https://github.com/streamlit/streamlit/pull/3038
    html = "<style>tr{background-color: transparent !important;}</style>"
    st_html(html)
    st_html(df.to_markdown(index=False))


def st_html(html: str):
    """Write HTML to streamlit app."""
    return st.markdown(html, unsafe_allow_html=True)
