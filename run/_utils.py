import collections
import functools
import logging
import math
import multiprocessing
import multiprocessing.pool
import pathlib
import random
import typing
from functools import partial

import config as cf
import torch
import torch.nn
from third_party import LazyLoader
from torchnlp.random import fork_rng
from tqdm import tqdm

import lib
from lib.utils import split
from run.data import _loader

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    import scipy
    import scipy.signal
    from google.cloud import storage
    from Levenshtein import StringMatcher
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    scipy = LazyLoader("scipy", globals(), "scipy")
    storage = LazyLoader("storage", globals(), "google.cloud.storage")
    StringMatcher = LazyLoader("StringMatcher", globals(), "Levenshtein.StringMatcher")

logger = logging.getLogger(__name__)

Dataset = typing.Dict[_loader.Speaker, typing.List[_loader.Passage]]


def get_dataset(
    datasets: typing.Dict[_loader.Speaker, _loader.DataLoader],
    path: pathlib.Path,
    include_psge: typing.Callable[[_loader.Passage], bool],
    handle_psge: typing.Callable[[_loader.Passage], _loader.Passage],
    max_workers: int = 0,
    language: typing.Optional[_loader.Language] = None,
) -> Dataset:
    """Define a TTS dataset.

    TODO: `apply_audio_filters` could be used replicate datasets with different audio processing.

    Args:
        datasets: Dictionary of datasets to load.
        path: Directory to cache the dataset.
        ...
    """
    logger.info("Loading dataset...")
    prepared = {s: f for s, f in datasets.items() if language is None or s.language == language}

    load = lambda s, d, **k: (s, [handle_psge(p) for p in d(path, **k) if include_psge(p)])
    if max_workers > 0:
        with multiprocessing.pool.ThreadPool(processes=min(max_workers, len(prepared))) as pool:
            items = list(pool.starmap(load, prepared.items()))
    else:
        items = [load(s, d, add_tqdm=True) for s, d in prepared.items()]

    prepared = {k: v for k, v in items if len(v) > 0}
    _omitted = datasets.keys() - prepared.keys()
    if len(_omitted) > 0:
        logger.info("Omitted %d Speakers: %s", len(_omitted), _omitted)

    return prepared


@functools.lru_cache(maxsize=None)
def _is_duplicate(a: str, b: str, min_sim: float) -> bool:
    """Helper function for `split_dataset` used to judge string similarity."""
    matcher = StringMatcher.StringMatcher(seq1=a, seq2=b)
    return (
        matcher.real_quick_ratio() > min_sim
        and matcher.quick_ratio() > min_sim
        and matcher.ratio() > min_sim
    )


def _find_duplicate_passages(
    dev_scripts: typing.Union[typing.Set[str], typing.Tuple[str]],
    passages: typing.List[_loader.Passage],
    min_sim: float,
) -> typing.Tuple[typing.List[_loader.Passage], typing.List[_loader.Passage]]:
    """Find passages in `passages` that are a duplicate of a passage in `dev_scripts`.

    Args:
        dev_scripts: Set of unique scripts.
        passages: Passages that may have a `script` thats already included in `dev_scripts`.
        min_sim: From 0 - 1, this is the minimum similarity two scripts must have to be considered
            duplicates.
    """
    duplicates, rest = [], []
    for passage in passages:
        if passage.script in dev_scripts:
            duplicates.append(passage)
            continue

        length = len(duplicates)
        for dev_script in dev_scripts:
            if _is_duplicate(dev_script, passage.script, min_sim):
                duplicates.append(passage)
                break

        if length == len(duplicates):
            rest.append(passage)

    return duplicates, rest


def _passages_len(passages: typing.List[_loader.Passage]):
    """Get the cumulative length of all `passages`."""
    return sum(p.segmented_audio_length() for p in passages)


def _len_of_dups(
    item: typing.Tuple[int, _loader.Passage], passages: typing.List[_loader.Passage], min_sim: float
) -> float:
    """Get the length of the duplicate passages to `item`."""
    assert passages[item[0]] is item[1]
    dups = _find_duplicate_passages((item[1].script,), passages[item[0] :], min_sim)[0]
    return _passages_len(dups)


TrainDev = typing.Tuple[Dataset, Dataset]


def _split_dataset(dataset: Dataset, dev_len: int, min_sim: float) -> TrainDev:
    """Split `dataset` into `train` and `dev` such that they contain no duplicates.

    Args
        ...
        dev_len: The approximate number of seconds to include for each speaker split.
        min_sim: The minimum similarity between two passages for them to be considered duplicates.
    """
    dev: Dataset = collections.defaultdict(list)
    train: Dataset = collections.defaultdict(list)
    dev_scripts: typing.Set[str] = set()
    items = sorted(dataset.items(), key=lambda i: i[0].label)
    random.shuffle(items)
    logger.info("Creating initial split...")
    for speaker, passages in tqdm(items):
        duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_sim)
        random.shuffle(rest)
        split_lens = [max(dev_len - _passages_len(duplicates), 0), math.inf]
        val = partial(_len_of_dups, passages=rest, min_sim=min_sim)
        splits = [[p for _, p in s] for s in split(list(enumerate(rest)), split_lens, val)]
        dev[speaker].extend(duplicates + splits[0])
        train[speaker].extend(splits[1])
        dev_scripts.update(d.script for d in dev[speaker])

    length = None
    while length is None or length != len(dev_scripts):
        length = len(dev_scripts)
        logger.info("Filtering out leftover duplicates...")
        for speaker, _ in tqdm(items):
            duplicates, rest = _find_duplicate_passages(dev_scripts, train[speaker], min_sim)
            dev[speaker].extend(duplicates)
            train[speaker] = rest
            dev_scripts.update(d.script for d in duplicates)

    return dict(train), dict(dev)


@lib.utils.log_runtime
def split_dataset(
    dataset: Dataset,
    dev_speakers: typing.Set[_loader.Speaker],
    approx_dev_len: int,
    min_sim: float,
    groups: typing.List[typing.Set[_loader.Speaker]],
    seed: int = 123,
) -> TrainDev:
    """Split `dataset` into a train and development dataset. Ensures that `dev` and `train` share
    no duplicate passages.

    NOTE: This assumes that the amount of data in each passage can be estimated with
    `segmented_audio_length` (via `_passages_len`). For example, if there was a long pause within a
    passage, this metric would include the long pause into the calculation of the dataset size,
    even though the long pause doesn't typically have useful data.
    NOTE: Any duplicate data for a speaker, not in `dev_speakers` will be discarded.
    NOTE: The duplicate cache is cleared after this function is run assuming it's not relevant
    any more.
    NOTE: The RNG state should never change; otherwise, the training and dev datasets may be
    different from process to process.
    TODO: `_split_dataset` can be rerun many times, in order to reduce, the size of the `dev_split`,
    so that it better matches `approx_dev_len`; however, it's not obvious how that would affect
    the distribution of the dev set as compared to the train set.

    Args:
        ...
        dev_speakers: Speakers to include in the development set.
        approx_dev_len: Number of seconds per speaker in the development dataset. The
            deduping algorithm may add extra items above the `approx_dev_length`.
        min_sim: The minimum percentage similarity for two passages to be considered duplicates.
        groups: These are groups of speakers that may share similar scripts.
        ...
    """
    logger.info("Splitting dataset...")
    speakers = {s for g in groups for s in g}
    message = f"Groups have overlapping speakers: {groups}"
    assert len(speakers) == sum(len(g) for g in groups), message
    assert len(set(dataset.keys()) - speakers) == 0, "Dataset speakers not found in groups."
    train: Dataset = {}
    dev: Dataset = {}
    subsets = ({k: v for k, v in dataset.items() if k in g and k in dev_speakers} for g in groups)
    for subset in (s for s in subsets if len(s) > 0):
        with fork_rng(seed=seed):
            train_split, dev_split = _split_dataset(subset, approx_dev_len, min_sim)
        train, dev = {**train, **train_split}, {**dev, **dev_split}

    # NOTE: For non-dev speakers, discard training passages which are already in `dev`.
    for group in groups:
        dev_scripts = {p.script for s, d in dev.items() for p in d if s in group}
        items = ((s, p) for s, p in dataset.items() if s in group and s not in dev_speakers)
        for speaker, passages in items:
            duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_sim)
            if len(duplicates) > 0:
                logger.warning("Discarded %d `%s` duplicates.", len(duplicates), speaker)
            train[speaker] = rest

    for spkr in dataset.keys():
        assert _passages_len(train.get(spkr, [])) > 0, "The `train` dataset has no data."
        if spkr in dev_speakers:
            message = "The `dev` dataset is larger than the `train` dataset."
            assert _passages_len(train.get(spkr, [])) >= _passages_len(dev.get(spkr, [])), message
            assert _passages_len(dev.get(spkr, [])) > 0, "The `dev` dataset has no data."

    _is_duplicate.cache_clear()
    return train, dev


SpanGeneratorGetWeight = typing.Callable[[_loader.Speaker, float], float]


class SpanGenerator(typing.Iterator[_loader.Span]):
    """Define the dataset generator to train and evaluate the TTS models on.

    NOTE: For datasets that are conventional with only one alignment per passage, `SpanGenerator`
    samples directly from that distribution.

    TODO: Create a generic dataset size estimiator that takes into account
    `run._config._include_span` instead of just using
    `float(sum(p.segmented_audio_length() for p in d))`. Based on statistics, the current estimator
    is pretty accurate.

    Args:
        ...
        dataset
        max_seconds: The maximum seconds delimited by an `Span`.
        ...
        get_weight: Given the `Speaker` and the size of it's dataset, get it's weight relative
            to other speakers. The weight is used to determine how many spans to generate from
            it's dataset. If all the speakers have the same weight, then they'll be sampled
            from equally.
    """

    @lib.utils.log_runtime
    def __init__(
        self,
        dataset: Dataset,
        max_seconds: int,
        include_span: typing.Callable[[_loader.Span], bool],
        get_weight: SpanGeneratorGetWeight = lambda *_: 1.0,
        **kwargs,
    ):
        self.max_seconds = max_seconds
        self.dataset = dataset
        self.generators: typing.Dict[_loader.Speaker, _loader.SpanGenerator] = {}
        for speaker, passages in dataset.items():
            is_singles = all([len(p.alignments) == 1 for p in passages])
            max_seconds_ = math.inf if is_singles else max_seconds
            self.generators[speaker] = cf.partial(_loader.SpanGenerator)(
                passages, max_seconds_, **kwargs
            )
        self.counter = {s: 0.0 for s in dataset.keys()}
        self.expected = {
            s: get_weight(s, float(sum(p.segmented_audio_length() for p in d)))
            for s, d in dataset.items()
        }
        self.include_span = include_span

    def __iter__(self) -> typing.Iterator[_loader.Span]:
        return self

    def __next__(self) -> _loader.Span:
        """Sample spans with a uniform speaker distribution based on `span.audio_length`."""
        while True:
            speaker = lib.utils.corrected_random_choice(self.counter, self.expected)
            span = next(self.generators[speaker])
            if span.audio_length <= self.max_seconds and self.include_span(span):
                self.counter[span.speaker] += span.audio_length
                return span


def get_window(window: str, window_length: int, window_hop: int) -> torch.Tensor:
    """Get a `torch.Tensor` window that passes `scipy.signal.check_COLA`.

    NOTE: `torch.hann_window` does not pass `scipy.signal.check_COLA`, for example. Learn more:
    https://github.com/pytorch/audio/issues/452
    """
    numpy_window = librosa.filters.get_window(window, window_length)
    assert scipy.signal.check_COLA(numpy_window, window_length, window_length - window_hop)
    return torch.tensor(numpy_window).float()


@functools.lru_cache(maxsize=1)
def get_storage_client() -> "storage.Client":
    return storage.Client()


def gcs_uri_to_blob(gcs_uri: str) -> "storage.Blob":
    """Parse GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") and return a `Blob`.

    NOTE: This function requires GCS authorization.
    """
    assert len(gcs_uri) > 5, "The URI must be longer than 5 characters to be a valid GCS link."
    assert gcs_uri[:5] == "gs://", "The URI provided is not a valid GCS link."
    path_segments = gcs_uri[5:].split("/")
    bucket = get_storage_client().bucket(path_segments[0])
    name = "/".join(path_segments[1:])
    return bucket.blob(name)


def blob_to_gcs_uri(blob: "storage.Blob") -> str:
    """Get GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") from `blob`."""
    return "gs://" + blob.bucket.name + "/" + blob.name
