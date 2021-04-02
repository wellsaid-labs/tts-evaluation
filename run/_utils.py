import collections
import functools
import itertools
import logging
import math
import multiprocessing
import multiprocessing.pool
import pathlib
import random
import typing

import torch
import torch.nn
import tqdm
from google.cloud import storage
from hparams import HParam, configurable
from Levenshtein.StringMatcher import StringMatcher
from third_party import LazyLoader
from torchnlp.random import fork_rng

import lib
from run._config import Dataset
from run.data import _loader

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    import scipy
    import scipy.signal
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    scipy = LazyLoader("scipy", globals(), "scipy")

logger = logging.getLogger(__name__)


@configurable
def get_dataset(
    datasets: typing.Dict[_loader.Speaker, _loader.DataLoader] = HParam(),
    path: pathlib.Path = HParam(),
    include_passage: typing.Callable[[_loader.Passage], bool] = HParam(),
    handle_passage: typing.Callable[[_loader.Passage], _loader.Passage] = HParam(),
    max_workers: int = 0,
) -> Dataset:
    """Define a TTS dataset.

    TODO: `apply_audio_filters` could be used replicate datasets with different audio processing.

    Args:
        datasets: Dictionary of datasets to load.
        path: Directory to cache the dataset.
        ...
    """
    logger.info("Loading dataset...")
    load = lambda s, d, **k: (s, [handle_passage(p) for p in d(path, **k) if include_passage(p)])
    if max_workers > 0:
        with multiprocessing.pool.ThreadPool(processes=min(max_workers, len(datasets))) as pool:
            items = list(pool.starmap(load, datasets.items()))
    else:
        items = [load(s, d, add_tqdm=True) for s, d in datasets.items()]
    return {k: v for k, v in items}


@functools.lru_cache(maxsize=None)
def _is_duplicate(a: str, b: str, min_similarity: float) -> bool:
    """Helper function for `split_dataset` used to judge string similarity."""
    matcher = StringMatcher(seq1=a, seq2=b)
    return (
        matcher.real_quick_ratio() > min_similarity
        and matcher.quick_ratio() > min_similarity
        and matcher.ratio() > min_similarity
    )


def _find_duplicate_passages(
    dev_scripts: typing.Set[str],
    passages: typing.List[_loader.Passage],
    min_similarity: float,
) -> typing.Tuple[typing.List[_loader.Passage], typing.List[_loader.Passage]]:
    """Find passages in `passages` that are a duplicate of a passage in `dev_scripts`.

    Args:
        dev_scripts: Set of unique scripts.
        passages: Passages that may have a `script` thats already included in `dev_scripts`.
        minimum_similarity: From 0 - 1, this is the minimum similarity two scripts must have to be
          considered duplicates.
    """
    duplicates, rest = [], []
    for passage in passages:
        if passage.script in dev_scripts:
            duplicates.append(passage)
            continue

        length = len(duplicates)
        for dev_script in dev_scripts:
            if _is_duplicate(dev_script, passage.script, min_similarity):
                duplicates.append(passage)
                break

        if length == len(duplicates):
            rest.append(passage)

    return duplicates, rest


def _split_dataset(
    dataset: Dataset,
    dev_speakers: typing.Set[_loader.Speaker],
    approx_dev_length: int,
    min_similarity: float,
    seed: int,
) -> typing.Tuple[Dataset, Dataset]:
    logger.info("Splitting `dataset`...")
    dev: typing.Dict[_loader.Speaker, list] = collections.defaultdict(list)
    train: typing.Dict[_loader.Speaker, list] = collections.defaultdict(list)
    dev_scripts: typing.Set[str] = set()
    len_ = lambda _passage: _passage.aligned_audio_length()
    sum_ = lambda _passages: sum([len_(p) for p in _passages])
    with fork_rng(seed=seed):
        iterator = list(sorted(dataset.items(), key=lambda i: (len(i[1]), i[0])))
        for speaker, passages in tqdm.tqdm(iterator):
            if speaker not in dev_speakers:
                train[speaker] = passages
                continue

            duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_similarity)
            dev[speaker].extend(duplicates)

            random.shuffle(rest)
            seconds = max(approx_dev_length - sum_(dev[speaker]), 0)
            splits = tuple(lib.utils.split(rest, [seconds, math.inf], len_))
            dev[speaker].extend(splits[0])
            train[speaker].extend(splits[1])
            dev_scripts.update(d.script for d in dev[speaker])

            message = "The `dev` dataset is larger than the `train` dataset."
            assert sum_(train[speaker]) >= sum_(dev[speaker]), message
            assert sum_(train[speaker]) > 0, "The train dataset has no aligned audio data."
            assert sum_(dev[speaker]) > 0, "The train dataset has no aligned audio data."

        # NOTE: Run the deduping algorithm until there are no more duplicates.
        length = None
        while length is None or length != len(dev_scripts):
            logger.info("Rerunning until there are no more duplicates...")
            length = len(dev_scripts)
            for speaker, _ in iterator:
                duplicates, rest = _find_duplicate_passages(
                    dev_scripts, train[speaker], min_similarity
                )
                if speaker not in dev_speakers and len(duplicates) > 0:
                    message = "Discarded %d duplicates for non-dev speaker %s. "
                    logger.warning(message, len(duplicates), speaker)
                elif speaker in dev_speakers:
                    dev[speaker].extend(duplicates)
                train[speaker] = rest
                dev_scripts.update(d.script for d in duplicates)

    _is_duplicate.cache_clear()

    return dict(train), dict(dev)


@lib.utils.log_runtime
@configurable
def split_dataset(
    dataset: Dataset,
    dev_speakers: typing.Set[_loader.Speaker] = HParam(),
    approx_dev_length: int = HParam(),
    min_similarity: float = HParam(),
    groups: typing.Optional[typing.List[typing.Set[_loader.Speaker]]] = None,
    seed: int = 123,
) -> typing.Tuple[Dataset, Dataset]:
    """Split the dataset into a train set and development set.

    NOTE: The RNG state should never change; otherwise, the training and dev datasets may be
    different from experiment to experiment.

    NOTE: `len_` assumes that the amount of data in each passage can be estimated with
    `aligned_audio_length`. For example, if there was a long pause within a passage, this estimate
    wouldn't make sense.

    NOTE: Passages are split between the train and development set in groups. The groups are
    dictated by textual similarity. The result of this is that the text in the train and
    development sets is distinct.

    NOTE: Any duplicate data for a speaker, not in `dev_speakers` will be discarded.

    NOTE: The duplicate cache is cleared after this function is run assuming it's not relevant
    any more.

    Args:
        ...
        dev_speakers: Speakers to include in the development set.
        approx_dev_length: Number of seconds per speaker in the development dataset. The
            deduping algorithm may add extra items above the `approx_dev_length`.
        groups: Speakers to be deduplicated, together. Otherwise, the speakers are considered to
            have independent datasets.
        ...
    """
    remaining = set(dataset.keys())
    groups = [] if groups is None else groups
    message = "Groups must not have overlapping speakers."
    assert len(set(itertools.chain(*tuple(groups)))) == sum([len(g) for g in groups]), message
    [remaining.remove(s) for group in groups for s in group if s in remaining]

    train: Dataset = {}
    dev: Dataset = {}
    for group in groups + [set([s]) for s in remaining]:
        subset = {k: v for k, v in dataset.items() if k in group}
        _train, _dev = _split_dataset(subset, dev_speakers, approx_dev_length, min_similarity, seed)
        train.update(_train)
        dev.update(_dev)
    return train, dev


class SpanGenerator(typing.Iterator[_loader.Span]):
    """Define the dataset generator to train and evaluate the TTS models on.

    NOTE: For datasets that are conventional with only one alignment per passage, `SpanGenerator`
    samples directly from that distribution.

    Args:
        dataset
        max_seconds: The maximum seconds delimited by an `Span`.
    """

    @lib.utils.log_runtime
    @configurable
    def __init__(
        self,
        dataset: Dataset,
        max_seconds: int = HParam(),
        include_span: typing.Callable[[_loader.Span], bool] = HParam(),
    ):
        self.max_seconds = max_seconds
        self.dataset = dataset
        self.generators: typing.Dict[_loader.Speaker, typing.Iterator[_loader.Span]] = {}
        for speaker, passages in dataset.items():
            is_singles = all([len(p.alignments) == 1 for p in passages])
            max_seconds_ = math.inf if is_singles else max_seconds
            self.generators[speaker] = _loader.SpanGenerator(passages, max_seconds_)
        self.counter = {s: 0.0 for s in list(dataset.keys())}
        self.include_span = include_span

    def __iter__(self) -> typing.Iterator[_loader.Span]:
        return self

    def __next__(self) -> _loader.Span:
        """ Sample spans with a uniform speaker distribution based on `span.audio_length`. """
        while True:
            speaker = lib.utils.corrected_random_choice(self.counter)
            span = next(self.generators[speaker])
            if span.audio_length < self.max_seconds and self.include_span(span):
                self.counter[span.speaker] += span.audio_length
                return span


def get_window(window: str, window_length: int, window_hop: int) -> torch.Tensor:
    """Get a `torch.Tensor` window that passes `scipy.signal.check_COLA`.

    NOTE: `torch.hann_window` does not pass `scipy.signal.check_COLA`, for example. Learn more:
    https://github.com/pytorch/audio/issues/452
    """
    window = librosa.filters.get_window(window, window_length)
    assert scipy.signal.check_COLA(window, window_length, window_length - window_hop)
    return torch.tensor(window).float()


@functools.lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    return storage.Client()


def gcs_uri_to_blob(gcs_uri: str) -> storage.Blob:
    """Parse GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") and return a `Blob`.

    NOTE: This function requires GCS authorization.
    """
    assert len(gcs_uri) > 5, "The URI must be longer than 5 characters to be a valid GCS link."
    assert gcs_uri[:5] == "gs://", "The URI provided is not a valid GCS link."
    path_segments = gcs_uri[5:].split("/")
    bucket = get_storage_client().bucket(path_segments[0])
    name = "/".join(path_segments[1:])
    return bucket.blob(name)


def blob_to_gcs_uri(blob: storage.Blob) -> str:
    """ Get GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") from `blob`. """
    return "gs://" + blob.bucket.name + "/" + blob.name
