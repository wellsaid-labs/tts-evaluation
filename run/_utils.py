import collections
import functools
import logging
import math
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
from lib.utils import disk_cache, split
from run import _config
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

UnprocessedDataset = typing.Dict[_loader.Speaker, _loader.UnprocessedDataset]
Dataset = typing.Dict[_loader.Speaker, typing.List[_loader.Passage]]


def get_unprocess_data(
    datasets: typing.Dict[_loader.Speaker, _loader.DataLoader],
    path: pathlib.Path,
    language: typing.Optional[_loader.Language] = None,
) -> UnprocessedDataset:
    """Get a raw unprocessed TTS dataset.

    Args:
        datasets: Dictionary of datasets to load.
        path: Directory to cache the dataset.
        ...
    """
    return {s: d(path) for s, d in datasets.items() if language is None or s.language == language}


@lib.utils.log_runtime
def get_dataset(
    datasets: typing.Dict[_loader.Speaker, _loader.DataLoader],
    path: pathlib.Path,
    include_passage: typing.Callable[[_loader.Passage], bool],
    language: typing.Optional[_loader.Language] = None,
) -> Dataset:
    """Get a TTS dataset.

    TODO: `apply_audio_filters` could be used replicate datasets with different audio processing.

    Args:
        datasets: Dictionary of datasets to load.
        path: Directory to cache the dataset.
        ...
    """
    logger.info("Loading dataset...")

    loaders = {s: f for s, f in datasets.items() if language is None or s.language == language}
    processed = [_loader.make_passages(s.label, l(path), add_tqdm=True) for s, l in loaders.items()]
    processed = [[p for p in d if include_passage(p)] for d in processed]
    prepared = {k: v for k, v in zip(loaders.keys(), processed) if len(v) > 0}

    kept = prepared.keys()
    omitted = datasets.keys() - kept
    logger.info(f"Kept {len(kept)} Speakers: {kept}")
    logger.warning(f"Omitted {len(omitted)} Speakers: {omitted}")

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
        passages: Passages that may have a `script` that is already included in `dev_scripts`.
        min_sim: From 0 - 1, this is the minimum similarity two scripts must have to be considered
            duplicates.
    """
    duplicates, rest = [], []
    for passage in passages:
        # NOTE: First, check whether the `passage.script` exactly matches any `dev_script`.
        if passage.script in dev_scripts:
            duplicates.append(passage)
            continue

        # NOTE: Second, check whether `passage.script` has `min_sim` similarity to any `dev_script`.
        is_passage_unique: bool = True
        for dev_script in dev_scripts:
            if _is_duplicate(dev_script, passage.script, min_sim):
                duplicates.append(passage)
                is_passage_unique = False
                break

        if is_passage_unique:
            rest.append(passage)

    return duplicates, rest


def _passages_len(passages: typing.List[_loader.Passage]) -> float:
    """Get the cumulative length of all `passages`."""
    return sum(p.segmented_audio_length() for p in passages)


def _len_of_dups(
    item: typing.Tuple[int, _loader.Passage], passages: typing.List[_loader.Passage], min_sim: float
) -> float:
    """Get the cumulative length of all passages which are duplicates of `item`, including the
    `item` itself.

    This function is called numerous times in sequence, with each passage in `passages` serving as
    `item`, so it only searches for duplicates starting at the index of the current `item` in the
    `passages` list. If `item` had duplicates earlier in the list, they would already be included in
    the dev set during a previous invocation, with the earlier duplicate serving as `item`.

    Args
        item: A tuple consisting of an index and its corresponding passage in the `passages` list.
        passages: A list of passages to compare to `item`, which includes the `item` itself.
        min_sim: The minimum similarity between two passages for them to be considered duplicates.
    """
    item_index, item_passage = item
    assert passages[item_index] is item_passage
    dups, _ = _find_duplicate_passages((item_passage.script,), passages[item_index:], min_sim)
    return _passages_len(dups)


TrainDev = typing.Tuple[Dataset, Dataset]


def _split_dataset(dataset: Dataset, approx_dev_len: int, min_sim: float) -> TrainDev:
    """Split `dataset` into `train` and `dev` such that they contain no duplicates.

    TODO: Refactor this function to be more generic, and easier to test, without needing to load
    an entire dataset.
    NOTE: If there is only one dataset, this function makes no attempt to deduplicate inside
    a single dataset. This assumes there is no meaningful duplicate content within the same dataset.
    TODO: In dataset preprocessing, ensure there are no duplicates within a single dataset.

    Args
        ...
        approx_dev_len: The approximate number of seconds to include in each speaker's dev set.
        min_sim: The minimum similarity between two passages for them to be considered duplicates.
    """
    dev: Dataset = collections.defaultdict(list)
    train: Dataset = collections.defaultdict(list)
    dev_scripts: typing.Set[str] = set()
    logger.debug(f"Dataset length is {len(dataset)} speaker(s).")

    # NOTE: Given a dataset of 1 speaker, assume no duplicates and split naively.
    if len(dataset) == 1:
        ((speaker, passages),) = tuple(dataset.items())
        logger.info(f"Splitting single speaker '{speaker.label}' without deduplication.")
        random.shuffle(passages)
        d, t = list(split(passages, [approx_dev_len, math.inf], lambda p: _passages_len([p])))
        logger.info(f"Split into {len(t)} train, {len(d)} dev passage(s).")
        dev[speaker].extend(d)
        train[speaker].extend(t)
        return dict(train), dict(dev)

    # NOTE: Given a dataset of 2+ speakers, deduplicate for each speaker.
    logger.info("Creating initial split...")
    items = sorted(dataset.items(), key=lambda i: i[0].label)
    random.shuffle(items)
    for spkr, passages in tqdm(items):
        num_p = len(passages)
        logger.info(f"\n\nProcessing speaker {spkr.label} ({spkr.name}) with {num_p} passages.")
        duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_sim)
        num_dup = len(duplicates)
        num_rest = len(rest)
        logger.debug(
            f"Speaker {spkr.label} ({spkr.name}): found {num_dup} duplicate passages, "
            f"{num_rest} remaining passages in the rest of their dataset."
        )
        random.shuffle(rest)

        # NOTE: `_len_of_dups()` gets the cumulative length of all passages which are duplicates of
        # a passage, including the passage itself. Then `split()` incorporates the length determined
        # by`_len_of_dups()` to split out at most `approx_dev_len` passages, via `split_lens`, after
        # duplicates are removed.
        val = partial(_len_of_dups, passages=rest, min_sim=min_sim)
        split_lens = [max(approx_dev_len - _passages_len(duplicates), 0), math.inf]
        d, t = [[p for _, p in s] for s in split(list(enumerate(rest)), split_lens, val)]

        logger.debug(f"Extending {spkr.label} dev set with {len(duplicates)} duplicates.")
        dev[spkr].extend(duplicates + d)
        train[spkr].extend(t)
        dev_scripts.update(d.script for d in dev[spkr])
        num_t = len(train[spkr])
        num_d = len(dev[spkr])
        logger.info(f"Split {num_p} passages for {spkr.label} into {num_t} train, {num_d} dev.")
    logger.debug("Finished initial split.\n")

    # NOTE: Continue deduping for each speaker until all duplicates are filtered between train/dev.
    num_dev_scripts = len(dev_scripts)
    finished_splitting: bool = False
    logger.info("Filtering out leftover duplicates for each speaker...")
    while not finished_splitting:
        num_dev_scripts = len(dev_scripts)
        for speaker, _ in tqdm(items):
            logger.debug(f"\n\nFiltering out leftover duplicates for {speaker.label}.")
            duplicates, rest = _find_duplicate_passages(dev_scripts, train[speaker], min_sim)
            num_dup = len(duplicates)
            num_rest = len(rest)
            logger.debug(
                f"Speaker {speaker.label} ({speaker.name}): {num_dup} duplicates, {num_rest} rest."
            )
            logger.debug(f"Extending {speaker.label} dev set with {len(duplicates)} duplicates.")
            dev[speaker].extend(duplicates)
            train[speaker] = rest
            dev_scripts.update(d.script for d in duplicates)
        finished_splitting = num_dev_scripts == len(dev_scripts)

    return dict(train), dict(dev)


@lib.utils.log_runtime
def split_dataset(
    dataset: Dataset,
    dev_speakers: typing.Set[_loader.Speaker],
    approx_dev_len: int,
    min_sim: float,
    groups: typing.List[typing.Set[_loader.Speaker]],
    min_split_passages: int,
    seed: int = 123,
) -> TrainDev:
    """Split `dataset` into a train and development dataset. Ensures that `dev` and `train` share
    no duplicate passages.

    NOTE: This assumes that the amount of data in each passage can be estimated with
    `segmented_audio_length` (via `_passages_len`). For example, if there was a long pause within a
    passage, this metric would include the long pause into the calculation of the dataset size,
    even though the long pause doesn't typically have useful data.
    NOTE: Any duplicate data for a speaker who is not in `dev_speakers` will be discarded.
    NOTE: The duplicate cache is cleared after this function is run, assuming it's not relevant
    anymore.
    NOTE: The RNG state should never change; otherwise, the training and dev datasets may be
    different from process to process.
    TODO: `_split_dataset` can be rerun many times in order to reduce the size of the `dev_split`,
    so that it better matches `approx_dev_len`; however, it's not obvious how that would affect
    the distribution of the dev set as compared to the train set.

    Args:
        ...
        dev_speakers: Speakers to include in the development set.
        approx_dev_len: Approximate number of seconds per speaker in the development dataset. The
            deduping algorithm may add extra items above the `approx_dev_length`. The dev set may be
            smaller than `approx_dev_len` if a speaker has no duplicate passages.
        min_sim: The minimum percentage similarity for two passages to be considered duplicates.
        groups: Non-overlapping sets of speakers, such that two groups may share similar scripts,
            but duplication is eliminated within each group across train/dev datasets. While ideally
            we would dedupe across the entire dataset, deduping within smaller groups improves
            performance by limiting the scope to require fewer comparisons.
        min_split_passages: The minimum number of passages that a dev speaker must have in each of
            their dev/train sets after splitting.
        ...
    """
    # NOTE: Raise an error for datasets from `dev_speakers` which are shorter than `approx_dev_len`.
    for spkr in dev_speakers:
        len_spkr_p = _passages_len(dataset[spkr])
        if len_spkr_p < approx_dev_len:
            raise ValueError(
                f"Speaker {spkr.label} has a total dataset length of {round(len_spkr_p, 1)}s, "
                f"shorter than `approx_dev_len` of {approx_dev_len}s. Recommended to delete speaker"
                f" from `dev_speakers`."
            )

    # NOTE: Ensure that each speaker is in exactly one group.
    logger.info("Splitting dataset...")
    speakers = {s for g in groups for s in g}
    message = f"Groups have overlapping speakers: {groups}"
    assert len(speakers) == sum(len(g) for g in groups), message
    assert len(set(dataset.keys()) - speakers) == 0, "Dataset speakers not found in groups."

    # NOTE: Create an initial dev/train split for each `subset`, where one `subset` is comprised of
    # the passages of all `dev_speakers` in a group.
    train: Dataset = {}
    dev: Dataset = {}
    subsets = ({s: p for s, p in dataset.items() if s in g and s in dev_speakers} for g in groups)
    for subset in (subset for subset in subsets if len(subset) > 0):
        logger.debug(f"Creating initial split of subset {[spkr.label for spkr in subset.keys()]}.")
        with fork_rng(seed=seed):
            train_split, dev_split = _split_dataset(subset, approx_dev_len, min_sim)
        train, dev = {**train, **train_split}, {**dev, **dev_split}

    # NOTE: For non-dev speakers, discard training passages which are already in `dev`.
    for group in groups:
        logger.debug(
            f"Discarding any non-dev duplicates for group: {', '.join([s.label for s in group])}."
        )
        dev_scripts = {p.script for s, d in dev.items() for p in d if s in group}
        items = ((s, p) for s, p in dataset.items() if s in group and s not in dev_speakers)
        for spkr, passages in items:
            logger.debug(f"Processing non-dev speaker {spkr.label}, {len(passages)} passages.")
            duplicates, rest = _find_duplicate_passages(dev_scripts, passages, min_sim)
            if len(duplicates) > 0:
                logger.debug(f"Discarded {len(duplicates)} `{spkr.label}` duplicates.")
            train[spkr] = rest
        logger.debug("Finished any non-dev discarding for group.")

    # NOTE: Ensure that each speaker's `train` dataset is not empty. For dev speakers, ensure that
    # their `dev` dataset is smaller than their `train` dataset, that `dev` is not empty, and that
    # their data splits so each of their dev/train sets has a minimum size of `min_split_passages`.
    for spkr in dataset.keys():
        train_len = _passages_len(train.get(spkr, []))
        dev_len = _passages_len(dev.get(spkr, []))
        num_train = len(train.get(spkr, []))
        num_dev = len(dev.get(spkr, []))
        assert train_len > 0, f"{spkr} `train` dataset has no data."
        if spkr in dev_speakers:
            msg = f"For {spkr.label}, `dev` is longer than `train` ({dev_len}s > {train_len})s."
            assert train_len >= dev_len, msg
            assert dev_len > 0, f"{spkr} `dev` dataset has no data."
        logger.info(f"Split {spkr} into {num_train} train, {num_dev} dev passages.")
        if num_dev < min_split_passages:
            logger.warning(
                f"For {spkr.label}, dev set has only {num_dev} passage(s), fewer than "
                f"{min_split_passages}."
            )

    _is_duplicate.cache_clear()
    return train, dev


@disk_cache(_config.DATASET_CACHE_PATH)
def _get_datasets():
    """Get a `train` and `dev` dataset."""
    dataset = cf.partial(get_dataset)()
    return cf.partial(split_dataset)(dataset)


def _get_debug_datasets(speakers: typing.Set[_loader.Speaker]):
    """Get small `train` and `dev` datasets for debugging."""
    speaker = next(iter(speakers), None)
    assert speaker is not None
    kwargs = {"datasets": {speaker: _config.DATASETS[speaker]}}
    dataset = cf.call(get_dataset, **kwargs, _overwrite=True)
    return cf.partial(split_dataset)(dataset)


def get_datasets(debug: bool):
    """Get a `train` and `dev` dataset."""
    if debug:
        return cf.partial(_get_debug_datasets)()
    return _get_datasets()


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
