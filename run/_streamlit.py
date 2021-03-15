import base64
import functools
import io
import logging
import multiprocessing
import os
import pathlib
import typing

import numpy as np
import streamlit as st
import tqdm
from streamlit.server.server import Server
from third_party import session_state

import lib
import run

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
def read_wave_audio_slice(*args, **kwargs) -> np.ndarray:
    """ Read audio slice, and cache. """
    return lib.audio.read_wave_audio_slice(*args, **kwargs)


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
