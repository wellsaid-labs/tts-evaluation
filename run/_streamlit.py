import logging
import multiprocessing
import multiprocessing.pool
import os
import pathlib
import random
import shutil
import socket
import tempfile
import typing
import zipfile

import config as cf
import numpy as np
import tqdm
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
from streamlit.commands.page_config import get_random_emoji
from streamlit.delta_generator import DeltaGenerator
from third_party import LazyLoader

import lib
import run
from lib.audio import AudioMetadata, sec_to_sample
from lib.environment import ROOT_PATH
from lib.text import natural_keys
from run._config.data import _include_span
from run._tts import CHECKPOINTS_LOADERS, Checkpoints, package_tts
from run._utils import Dataset
from run.data._loader import Alignment, Passage, Span, Speaker
from run.data._loader import structures as struc
from run.train.spectrogram_model._worker import _get_data_generator

if typing.TYPE_CHECKING:  # pragma: no cover
    import altair as alt
    import librosa
    import librosa.util
    import matplotlib.figure
    import pandas as pd
    import streamlit as st
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    alt = LazyLoader("alt", globals(), "altair")
    pd = LazyLoader("pd", globals(), "pandas")
    st = LazyLoader("st", globals(), "streamlit")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")


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


@st.cache_data()
def load_tts(checkpoints_key: str, **kwargs):
    return package_tts(*CHECKPOINTS_LOADERS[Checkpoints[checkpoints_key]](**kwargs))


# A `Path` to a file or directory accessible via HTTP in the streamlit app.
WebPath = typing.NewType("WebPath", pathlib.Path)
RelativeUrl = typing.NewType("RelativeUrl", str)


@st.cache_resource()
def make_temp_root_dir():
    """Make a temporary directory accessible via HTTP in the streamlit app.

    NOTE: With `cache_resource`, this function runs once per session.
    """
    assert pathlib.Path(st.__file__).parent in STREAMLIT_STATIC_TEMP_PATH.parents
    if STREAMLIT_STATIC_TEMP_PATH.exists():
        path = STREAMLIT_STATIC_TEMP_PATH.relative_to(lib.environment.ROOT_PATH)
        logger.info(f"Clearing temporary files at {path}...")
        shutil.rmtree(STREAMLIT_STATIC_TEMP_PATH)
    STREAMLIT_STATIC_TEMP_PATH.mkdir(exist_ok=True, parents=True)


def make_temp_web_dir() -> WebPath:
    """Make a temporary directory accessible via HTTP in the streamlit app."""
    make_temp_root_dir()
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
    cf.partial(lib.audio.write_audio)(web_path, audio, **kwargs)
    return web_path


def figure_to_url(figure: matplotlib.figure.Figure, name: str = "fig.png", **kwargs):
    """Create a URL that can be loaded from `streamlit`."""
    image_web_path = make_temp_web_dir() / name
    figure.savefig(str(image_web_path), **kwargs)
    return web_path_to_url(image_web_path)


def audio_to_url(audio: np.ndarray, name: str = "audio.wav", **kwargs):
    """Create a URL that can be loaded from `streamlit`."""
    return web_path_to_url(audio_to_web_path(audio, name, **kwargs))


def audio_to_html(
    audio: np.ndarray, name: str = "audio.wav", attrs: str = "controls", **kwargs
) -> str:
    """Create an audio HTML element from a numpy array."""
    return f'<audio {attrs} src="{audio_to_url(audio, name, **kwargs)}"></audio>'


# NOTE: While streamlit does provide `st.download`, it'll force the entire app to reload when it's
# used. Here is an issue thread: https://github.com/streamlit/streamlit/issues/4382. The below
# options don't have that backdraw.


def _bytes_to_html_download_link(name: str, label: str, bytes_: bytes) -> str:
    """Make a zipfile named `name` that can be downloaded with a button called `label`."""
    web_path = make_temp_web_dir() / name
    web_path.write_bytes(bytes_)
    return f'<a href="{web_path_to_url(web_path)}" download="{name}">{label}</a>'


def st_download_bytes(*args, **kwargs):
    """Download a text file."""
    return st_html(_bytes_to_html_download_link(*args, **kwargs))


def _paths_to_html_download_link(
    name: str,
    label: str,
    paths: typing.List[pathlib.Path],
    archive_paths: typing.Optional[typing.List[pathlib.Path]] = None,
) -> str:
    """Make a zipfile named `name` that can be downloaded with a button called `label`.

    TODO: Implement a utility like `st_download_bytes` for `paths_to_html_download_link`.
    """
    web_path = make_temp_web_dir() / name
    archive_paths_ = [p.name for p in paths] if archive_paths is None else archive_paths
    with zipfile.ZipFile(web_path, "w") as file_:
        for path, archive_path in zip(paths, archive_paths_):
            file_.write(path, arcname=archive_path)
    return f'<a href="{web_path_to_url(web_path)}" download="{name}">{label}</a>'


def st_download_files(*args, **kwargs):
    """Create a link to download a zip file with `paths`."""
    st_html(_paths_to_html_download_link(*args, **kwargs))


_MapInputVar = typing.TypeVar("_MapInputVar")
_MapReturnVar = typing.TypeVar("_MapReturnVar")
_cpu_count = os.cpu_count()
assert _cpu_count is not None


def map_(
    list_: typing.List[_MapInputVar],
    func: typing.Callable[[_MapInputVar], _MapReturnVar],
    chunk_size: int = 8,
    max_parallel: int = _cpu_count * 3,
    progress_bar: bool = True,
) -> typing.List[_MapReturnVar]:
    """Apply `func` to `list_` in parallel."""
    with multiprocessing.pool.ThreadPool(processes=max_parallel) as pool:
        iterator = pool.imap(func, list_, chunksize=chunk_size)
        if progress_bar:
            iterator = tqdm.tqdm(iterator, total=len(list_))
        return list(iterator)


@st.cache_resource()
def read_wave_audio(*args, **kwargs) -> np.ndarray:
    """Read audio slice, and cache."""
    return lib.audio.read_wave_audio(*args, **kwargs)


def span_audio(span: run.data._loader.Span) -> np.ndarray:
    """Get `span` audio using cached `read_wave_audio`."""
    return read_wave_audio(span.audio_file, span.audio_start, span.audio_length)


def passage_audio(passage: run.data._loader.Passage) -> np.ndarray:
    """Get `span` audio using cached `read_wave_audio`."""
    length = passage.segmented_audio_length()
    return read_wave_audio(passage.audio_file, passage.audio_start, length)


def alignment_audio(
    span: typing.Union[run.data._loader.Span, run.data._loader.Passage], alignment: Alignment
) -> np.ndarray:
    """Get `span` or `Passage` audio using cached `read_wave_audio`."""
    return read_wave_audio(
        span.audio_file,
        span.audio_start + alignment.audio[0],
        span.audio_start + alignment.audio[1],
    )


def metadata_alignment_audio(metadata: AudioMetadata, alignment: Alignment) -> np.ndarray:
    """Get `alignment` audio using cached `read_wave_audio`."""
    return read_wave_audio(metadata, alignment.audio[0], alignment.audio_len)


def clip_audio(audio: np.ndarray, span: Span, alignment: Alignment):
    """Get a clip of `audio` at `alignment`."""
    sample_rate = span.audio_file.sample_rate
    start_ = sec_to_sample(max(alignment.audio[0], 0), sample_rate)
    stop_ = sec_to_sample(min(alignment.audio[1], span.audio_length), sample_rate)
    return audio[start_:stop_]


@st.cache_resource()
def get_dataset(speaker_labels: typing.FrozenSet[str]) -> Dataset:
    """Load dataset subset, and cache."""
    logger.info("Loading dataset...")
    with st.spinner(f"Loading dataset(s): {','.join(list(speaker_labels))}"):
        datasets = {k: v for k, v in run._config.DATASETS.items() if k.label in speaker_labels}
        dataset = cf.call(run._utils.get_dataset, datasets=datasets, _overwrite=True)
        logger.info(f"Finished loading {set(speaker_labels)} dataset(s)! {lib.utils.mazel_tov()}")
    return dataset


@st.cache_resource()
def get_datasets() -> typing.Tuple[Dataset, Dataset]:
    """Load train and dev dataset, and cache."""
    return run._utils.get_datasets(False)


def get_dev_dataset() -> Dataset:
    """Load dev dataset and cache."""
    return get_datasets()[1]


@st.cache_resource()
def get_spans(
    _train_dataset: Dataset,
    _dev_dataset: Dataset,
    num_spans: int,
    speaker: typing.Optional[Speaker] = None,
    is_dev_speakers: bool = True,
    is_train_spans: bool = True,
    include_dic: bool = True,
    device_count: int = 4,
) -> typing.List[Span]:
    """Get `num_spans` spans from `_train_dataset` for `speaker`. This uses the same code path
    as a training run so it ensures we are analyzing training data directly.

    Args:
        ...
        speaker: Pick a speaker to generate spans for.
        num_spans: The number of spans to generate.
        is_dev_speakers: Get spans only for development speakers.
        is_train_spans: Get spans from the training dataset.
        include_dic: Include dictionary datasets that are sometimes used in development; however,
            may not be as applicable for applied use cases.
        device_count: The number of devices used during training to set the configuration.
    """
    with st.spinner("Making generators..."):
        if is_dev_speakers:
            _train_dataset = {s: _train_dataset[s] for s in _dev_dataset.keys()}

        if speaker is not None:
            _train_dataset = {speaker: _train_dataset[speaker]}
            _dev_dataset = {speaker: _dev_dataset[speaker]}

        train_gen, dev_gen = cf.partial(_get_data_generator)(_train_dataset, _dev_dataset)
        gen = train_gen if is_train_spans else dev_gen

    with st.spinner("Making spans..."):
        gen = (s for s in gen if include_dic or s.speaker.style is not struc.Style.DICT)
        spans = [next(gen) for _ in st_tqdm(range(num_spans), num_spans)]

    return spans


def _random_speech_segments(_train_dataset: Dataset):
    """Get random speech segments while balancing each of the speakers.

    NOTE: This implementation of sampling speech segments is simpler than others. This is because
          in other implementations we are trying to sample an interval of data. To ensure we
          sample proportionally, we need to oversample longer passages. In this case, we are
          directly sampling from a fixed pool of non-overlapping spans. With that in mind, we
          can just sample uniformly, and ensure that every alignment has an equal chance of
          getting sampled.
    """
    counter = {s: 0.0 for s in _train_dataset.keys()}
    data = {k: [s for p in v for s in p.speech_segments] for k, v in _train_dataset.items()}
    while True:
        speaker = lib.utils.corrected_random_choice(counter)
        span = random.choice(data[speaker])
        if _include_span(span):
            counter[span.speaker] += span.audio_length
            yield span


@st.cache_resource()
def get_speech_segments(
    _train_dataset: Dataset, speaker: typing.Optional[Speaker], num_spans: int
) -> typing.Tuple[typing.List[Span], typing.List[np.ndarray]]:
    """Get `num_spans` speech segments from `_train_dataset` for `speaker`."""
    with st.spinner("Making spans..."):
        _train_dataset = _train_dataset if speaker is None else {speaker: _train_dataset[speaker]}
        train_gen = _random_speech_segments(_train_dataset)
        spans = [next(train_gen) for _ in st_tqdm(range(num_spans), num_spans)]

    with st.spinner("Loading audio..."):
        signals = [s.audio() for s in st_tqdm(spans)]

    return spans, signals


@st.cache_resource()
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
        alt.Chart(pd.DataFrame({x: ticks, y[0]: -envelope, y[1]: envelope}))  # type: ignore
        .mark_area()
        .encode(
            x=alt.X(x, type="quantitative"),  # type: ignore
            y=alt.Y(y[0], scale=alt.Scale(domain=(-1.0, 1.0)), type="quantitative"),  # type: ignore
            y2=alt.Y2(y[1]),  # type: ignore
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
        alt.Chart(pd.DataFrame(source))  # type: ignore
        .mark_rect(
            fillOpacity=fillOpacity,  # type: ignore
            color=color,  # type: ignore
            stroke=stroke,  # type: ignore
            strokeWidth=strokeWidth,  # type: ignore
            strokeOpacity=strokeOpacity,  # type: ignore
            **kwargs,
        )
        .encode(x=alt.X("x_min", type="quantitative"), x2=alt.X2("x_max"))  # type: ignore
    )


def dataset_passages(dataset: Dataset) -> typing.Iterator[Passage]:
    """Get all passages in `dataset`."""
    for _, passages in dataset.items():
        yield from passages


@st.cache_resource()
def load_en_core_web_md(*args, **kwargs):
    return lib.text.load_spacy_nlp("en_core_web_md", *args, **kwargs)


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


def path_label(path: typing.Optional[pathlib.Path]) -> str:
    """Get a short label for `path`."""
    if path is None:
        return "None"

    return (
        str(path.relative_to(ROOT_PATH)) + "/"
        if path.is_dir()
        else f"{path.parent.name}/{path.name}"
    )


# TODO: Offer options to override `st` throughout the utility functions or consider integrating with
# their `DeltaGenerator`.


def st_select_path(
    label: str,
    dir: pathlib.Path,
    suffix: str,
    st: DeltaGenerator = typing.cast(DeltaGenerator, st),
) -> typing.Optional[pathlib.Path]:
    """Display a path selector for the directory `dir`."""
    options = [p for p in dir.glob("**/*") if p.suffix == suffix]
    options = [None] + sorted(options, key=lambda x: natural_keys(str(x)), reverse=True)
    selected = st.selectbox(label, options=options, format_func=path_label)
    return typing.cast(typing.Optional[pathlib.Path], selected)


def st_select_paths(label: str, dir: pathlib.Path, suffix: str) -> typing.List[pathlib.Path]:
    """Display a file and directory selector for the directory `dir`."""
    options = [p for p in dir.glob("**/*") if p.suffix == suffix or p.is_dir()] + [dir]
    options = sorted(options, key=lambda x: natural_keys(str(x)), reverse=True)
    paths = [st.selectbox(label, options=options, format_func=path_label)]
    paths = typing.cast(typing.List[pathlib.Path], paths)
    paths = [f for p in paths for f in ([p] if p.is_file() else list(p.glob(f"**/*{suffix}")))]
    if len(paths) > 0:
        st.info(f"Selected {label}:\n" + "".join(["\n - " + path_label(p) for p in paths]))
    return paths


_StTqdmVar = typing.TypeVar("_StTqdmVar")


def st_tqdm(
    iterable: typing.Iterable[_StTqdmVar], total: typing.Optional[int] = None
) -> typing.Generator[_StTqdmVar, None, None]:
    """Display a progress bar while iterating through `iterable`."""
    bar = st.progress(0)
    for i, item in enumerate(iterable):
        yield item
        bar.progress(i / (len(iterable) if total is None else total))  # type: ignore
    bar.empty()


# NOTE: This follows the examples highlighted here:
# https://github.com/PablocFonseca/streamlit-aggrid-examples/blob/main/cell_renderer_class_example.py
# https://github.com/PablocFonseca/streamlit-aggrid/issues/119
# https://github.com/PablocFonseca/streamlit-aggrid/issues/198

ImgCellRenderer = JsCode(
    """
        class ImgCellRenderer {
          init(params) {
            this.eGui = document.createElement('img');
            this.eGui.setAttribute('src', params.value);
          }

          getGui() {
            return this.eGui;
          }
        }
    """
)

AudioCellRenderer = JsCode(
    """
        class AudioCellRenderer {
          init(params) {
            this.eGui = document.createElement('audio');
            this.eGui.setAttribute('src', params.value);
            this.eGui.setAttribute('preload', 'none');
            this.eGui.setAttribute('controls', 'controls');
          }

          getGui() {
            return this.eGui;
          }
        }
    """
)


def st_ag_grid(
    df: pd.DataFrame,
    audio_cols: typing.List[str] = [],
    img_cols: typing.List[str] = [],
    height: int = 850,
    page_size: int = 10,
):
    """Display a table to preview `data`."""
    options = GridOptionsBuilder.from_dataframe(df)
    options.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    options.configure_default_column(wrapText=True, autoHeight=True, min_column_width=1)
    [options.configure_column(name, cellRenderer=AudioCellRenderer) for name in audio_cols]
    [options.configure_column(name, cellRenderer=ImgCellRenderer) for name in img_cols]
    return AgGrid(
        data=df,
        gridOptions=options.build(),
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        height=height,
    )


def st_set_page_config(initial_sidebar_state: str = "collapsed", layout: str = "wide", **kwargs):
    """`set_page_config` with some pre-set defaults."""
    title = socket.gethostname() + " • " + os.path.basename(__file__)[:-3] + " • Streamlit"
    random.seed(title)
    emoji = get_random_emoji()
    st.set_page_config(
        initial_sidebar_state=initial_sidebar_state,  # type: ignore
        layout=layout,  # type: ignore
        page_title=title,
        page_icon=emoji,
        **kwargs,
    )
