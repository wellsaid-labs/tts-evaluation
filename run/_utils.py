import contextlib
import dataclasses
import enum
import functools
import io
import itertools
import logging
import math
import multiprocessing.pool
import os
import pathlib
import random
import time
import typing
from datetime import timedelta

import numpy
import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import tqdm
from google.cloud import storage
from third_party import LazyLoader

import lib
import run
from lib.utils import flatten, seconds_to_string
from run._config import Cadence, Dataset, DatasetType, get_dataset_label, get_model_label

if typing.TYPE_CHECKING:  # pragma: no cover
    import comet_ml
    import matplotlib.figure
else:
    comet_ml = LazyLoader("comet_ml", globals(), "comet_ml")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Checkpoint:

    checkpoints_directory: pathlib.Path
    comet_experiment_key: str
    step: int
    num_examples: int


def maybe_make_experiment_directories_from_checkpoint(
    checkpoint: Checkpoint, *args, **kwargs
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """For checkpoints saved in the `maybe_make_experiment_directories` directory structure,
    this creates another "run" under the original experiment.
    """
    return maybe_make_experiment_directories(
        checkpoint.checkpoints_directory.parent.parent, *args, **kwargs
    )


def maybe_make_experiment_directories(
    experiment_root: pathlib.Path,
    recorder: lib.environment.RecordStandardStreams,
    run_name: str = "RUN_" + lib.environment.bash_time_label(add_pid=False),
    checkpoints_directory_name: str = "checkpoints",
    run_log_filename: str = "run.log",
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """Create a directory structure to store an experiment run, like so:

      {experiment_root}/
      └── {run_name}/
          ├── run.log
          └── {checkpoints_directory_name}/

    TODO: Could this structure be encoded in some data structure? For example, we could return an
    object called an `ExperimentDirectory` that has `children` called `RunsDirectory`.

    Args:
        experiment_root: Top-level directory to store an experiment, unless a
          checkpoint is provided.
        recorder: This records the standard streams, and saves it.
        run_name: The name of this run.
        checkpoints_directory_name: The name of the directory that stores checkpoints.
        run_log_filename: The run log filename.

    Return:
        run_root: The root directory to store run files.
        checkpoints_directory: The directory to store checkpoints.
    """
    logger.info("Updating directory structure...")
    experiment_root.mkdir(exist_ok=True)
    run_root = experiment_root / run_name
    run_root.mkdir()
    checkpoints_directory = run_root / checkpoints_directory_name
    checkpoints_directory.mkdir()
    recorder.update(run_root, log_filename=run_log_filename)
    return run_root, checkpoints_directory


def _normalize_audio(
    args: typing.Tuple[pathlib.Path, pathlib.Path], callable_: typing.Callable[..., None]
):
    """ Helper function for `normalize_audio`. """
    source, destination = args
    destination.parent.mkdir(exist_ok=True, parents=True)
    callable_(source, destination)


def _normalize_path(path: pathlib.Path) -> pathlib.Path:
    """ Helper function for `normalize_audio`. """
    return path.parent / run._config.TTS_DISK_CACHE_NAME / f"ffmpeg({path.stem}).wav"


@lib.utils.log_runtime
def normalize_audio(
    dataset: Dataset, num_processes: int = typing.cast(int, os.cpu_count()), **kwargs
) -> Dataset:
    """Normalize audio with ffmpeg in `dataset`.

    TODO: Consider using the ffmpeg SoX resampler, instead.
    TODO: In order to better estimate the performance, we could measure progress based on audio
    file length.
    TODO: In order to encourage parallelism, the longest files should be normalized first.
    """
    logger.info("Normalizing dataset audio...")
    audio_paths_ = [[p.audio_file.path for p in v] for v in dataset.values()]
    audio_paths: typing.Set[pathlib.Path] = set(flatten(audio_paths_))
    partial = functools.partial(lib.audio.normalize_audio, **kwargs)
    partial = functools.partial(_normalize_audio, callable_=partial)
    args = [(p, _normalize_path(p)) for p in audio_paths if not _normalize_path(p).exists()]
    with multiprocessing.pool.ThreadPool(num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(partial, args), total=len(args)))

    metadatas = lib.audio.get_audio_metadata([_normalize_path(p) for p in audio_paths])
    lookup = {p: m for p, m in zip(audio_paths, metadatas)}
    logger.info("Updating dataset to use normalized audio files...")
    return_ = {
        s: [lib.datasets.update_passage_audio(p, lookup[p.audio_file.path]) for p in d]
        for s, d in dataset.items()
    }
    logger.info("Normalized dataset audio %s", lib.utils.mazel_tov())
    return return_


def init_distributed(
    rank: int,
    timeout: timedelta = timedelta(minutes=30),
    backend: str = "nccl",
    init_method: str = "tcp://127.0.0.1:29500",
    world_size: int = torch.cuda.device_count(),
) -> torch.device:
    """Initiate distributed for training.

    Learn more about distributed environments here:
    https://pytorch.org/tutorials/intermediate/dist_tuto.htm
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    torch.distributed.init_process_group(backend, init_method, timeout, world_size, rank)
    logger.info("Worker %d started.", torch.distributed.get_rank())
    logger.info("%d GPUs found.", world_size)
    device = torch.device("cuda", rank)

    # NOTE: Unless this is run, PyTorch may use a different GPU for some operations. Learn more:
    # https://github.com/pytorch/pytorch/issues/3477#issuecomment-342294955
    # https://github.com/pytorch/pytorch/issues/7071#issuecomment-437469653
    torch.cuda.set_device(device)
    # TODO: Instead of returning and passing around `torch.device`, rely on `torch.cuda.set_device`
    # or `torch.cuda.device` to set context.
    return device


@lib.utils.log_runtime
def split_passages(
    passages: typing.List[lib.datasets.Passage], dev_size: float
) -> typing.Tuple[typing.List[lib.datasets.Passage], typing.List[lib.datasets.Passage]]:
    """Split a dataset into a development and train set.

    TODO: Since we have multiples copies of the same dataset, with only adjustments in the speaker
    or even the speaker audio preprocessing, our dev dataset is not completely isolated from
    our training data. In order to solve this issue, we need to split the WSL datasets together,
    to ensure that the passages in the dev dataset haven't been seen in any form. We could solve
    this by picking passages for one speaker, and then ensuring that no other speaker has the
    same passages (and if they do, they should be included in the dev dataset).

    Args:
        passages
        dev_size: Number of seconds of audio data in the development set.

    Return:
        train: The rest of the data.
        dev: Dataset with `dev_size` of data.
    """
    passages = passages.copy()
    random.shuffle(passages)
    # NOTE: `len_` assumes that a negligible amount of data is unusable in each passage.
    len_ = lambda p: p.aligned_audio_length()
    dev, train = tuple(lib.utils.split(passages, [dev_size, math.inf], len_))
    dev_size = sum([len_(p) for p in dev])
    train_size = sum([len_(p) for p in train])
    assert train_size >= dev_size, "The `dev` dataset is larger than the `train` dataset."
    assert len(dev) > 0, "The dev dataset has no passages."
    assert len(train) > 0, "The train dataset has no passages."
    return train, dev


def get_dataset_stats(
    train: Dataset, dev: Dataset
) -> typing.Dict[run._config.Label, typing.Union[str, int, float]]:
    """Get `train` and `dev` dataset statistics."""
    stats: typing.Dict[run._config.Label, typing.Union[int, str, float]] = {}
    data: Dataset
    for data, type_ in [(train, DatasetType.TRAIN), (dev, DatasetType.DEV)]:
        label_ = functools.partial(get_dataset_label, cadence=Cadence.STATIC, type_=type_)
        passages_ = flatten([[p for p in v] for v in data.values()])
        for speaker, passages in itertools.chain(list(data.items()), [(None, passages_)]):
            label = label_ if speaker is None else functools.partial(label_, speaker=speaker)
            stats[label("num_audio_files")] = len(set(p.audio_file for p in passages))
            stats[label("num_passages")] = len(passages)
            stats[label("num_characters")] = sum(len(p.script) for p in passages)
            num_seconds = seconds_to_string(sum(p.aligned_audio_length() for p in passages))
            stats[label("num_seconds")] = num_seconds
    return stats


class Context(enum.Enum):
    """ Constants and labels for contextualizing the use-case. """

    TRAIN: typing.Final = "train"
    EVALUATE: typing.Final = "evaluate"
    EVALUATE_INFERENCE: typing.Final = "evaluate_inference"


class CometMLExperiment:
    """Create a `comet_ml.Experiment` or `comet_ml.ExistingExperiment` object with several
    adjustments.

    Args:
        project_name
        experiment_key: Existing experiment identifier.
        workspace
        **kwargs: Other kwargs to pass to comet `Experiment` and / or `ExistingExperiment`
    """

    _BASE_HTML_STYLING = """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.css"
      type="text/css">
<style>
  body {
    background-color: #f4f4f5;
  }

  p {
    font-family: 'Roboto', system-ui, sans-serif;
    margin-bottom: .5em;
  }

  b {
    font-weight: bold
  }

  section {
    padding: 1.5em;
    border-bottom: 2px solid #E8E8E8;
    background: white;
  }
</style>
    """

    def __init__(
        self,
        project_name: typing.Optional[str] = None,
        experiment_key: typing.Optional[str] = None,
        workspace: typing.Optional[str] = None,
        **kwargs,
    ):
        if lib.environment.has_untracked_files():
            raise ValueError(
                "Experiment is not reproducible, Comet does not track untracked files. "
                f"Please track these files via `git`:\n{lib.environment.get_untracked_files()}"
            )

        kwargs.update({"project_name": project_name, "workspace": workspace})
        if experiment_key is None:
            self._experiment = comet_ml.Experiment(**kwargs)
            self._experiment.log_html(self._BASE_HTML_STYLING)
        else:
            self._experiment = comet_ml.ExistingExperiment(
                previous_experiment=experiment_key, **kwargs
            )

        self.log_asset = self._experiment.log_asset
        self.log_html = self._experiment.log_html
        self.get_key = self._experiment.get_key
        self.set_model_graph = self._experiment.set_model_graph

        self._last_step_time: typing.Optional[float] = None
        self._last_step: typing.Optional[int] = None
        self._last_epoch_time: typing.Optional[float] = None
        self._last_epoch_step: typing.Optional[int] = None
        self._first_epoch_time: typing.Optional[float] = None
        self._first_epoch_step: typing.Optional[int] = None

        self._log_environment()

    @property
    def curr_step(self) -> typing.Optional[int]:
        return typing.cast(typing.Optional[int], self._experiment.curr_step)

    @property
    def context(self) -> typing.Optional[str]:
        return typing.cast(typing.Optional[str], self._experiment.context)

    def _log_environment(self):
        # TODO: Collect additional environment details like CUDA, CUDANN, NVIDIA Driver versions
        # with this script:
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
        log_other = lambda k, v: self.log_other(run._config.get_environment_label(k), v)

        log_other("last_git_commit_date", lib.environment.get_last_git_commit_date())
        log_other("git_branch", lib.environment.get_git_branch_name())
        log_other("has_git_patch", str(lib.environment.has_tracked_changes()))
        log_other("gpus", lib.environment.get_cuda_gpus())
        log_other("num_gpus", lib.environment.get_num_cuda_gpus())
        log_other("disks", lib.environment.get_disks())
        log_other("unique_cpus", lib.environment.get_unique_cpus())
        log_other("num_cpus", os.cpu_count())
        log_other("total_physical_memory", lib.environment.get_total_physical_memory())

    def set_step(self, step: typing.Optional[int]):
        self._experiment.set_step(step)
        if self.curr_step is not None:
            seconds_per_step = (
                (time.time() - self._last_step_time) / (self.curr_step - self._last_step)
                if self._last_step is not None
                and self._last_step_time is not None
                and self.curr_step > self._last_step
                else None
            )
            self._last_step_time = time.time()
            # NOTE: Ensure that the variable `last_step` is updated before `log_metric` is called.
            # This prevents infinite recursion via `curr_step > last_step`.
            self._last_step = self.curr_step
            if seconds_per_step is not None:
                label = get_model_label("seconds_per_step", Cadence.STEP)
                self.log_metric(label, seconds_per_step)

    @contextlib.contextmanager
    def context_manager(self, context: Context):
        with self._experiment.context_manager(str(context)):
            yield self

    def log_current_epoch(self, epoch: int):
        self._last_epoch_step = self.curr_step
        self._last_epoch_time = time.time()
        if self._first_epoch_time is None and self._first_epoch_step is None:
            assert self.curr_step is not None
            self._first_epoch_step = self.curr_step
            self._first_epoch_time = time.time()
        self._experiment.log_current_epoch(epoch)

    def log_epoch_end(self, epoch: int):
        # NOTE: Logs an average `steps_per_second` for each epoch.
        if (
            self._last_epoch_step is not None
            and self._last_epoch_time is not None
            and self.curr_step is not None
        ):
            label = get_model_label("steps_per_second", Cadence.MULTI_STEP)
            metric = (self.curr_step - self._last_epoch_step) / (
                time.time() - self._last_epoch_time
            )
            self.log_metric(label, metric)

        # NOTE: Logs an average `steps_per_second` since the training started.
        if (
            self._first_epoch_time is not None
            and self._first_epoch_step is not None
            and self.curr_step is not None
        ):
            with self.context_manager(None):
                label = get_model_label("steps_per_second", Cadence.RUN)
                metric = (self.curr_step - self._first_epoch_step) / (
                    time.time() - self._first_epoch_time
                )
                self.log_metric(label, metric)

        self._experiment.log_epoch_end(epoch)

    def _upload_audio(
        self, file_name: str, data: typing.Union[numpy.ndarray, torch.Tensor]
    ) -> typing.Optional[str]:
        """Upload the audio and return the URL."""
        file_ = io.BytesIO()
        lib.audio.write_audio(file_, data)
        asset = self.log_asset(file_, file_name=file_name)
        return asset["web"] if asset is not None else asset

    def log_html_audio(
        self,
        audio: typing.Dict[str, typing.Union[numpy.ndarray, torch.Tensor]] = {},
        **kwargs,
    ):
        """Audio with related metadata to Comet in the HTML tab.

        Args:
            audio
            **kwargs: Additional metadata to include.
        """
        items = [f"<p><b>Step:</b> {self.curr_step}</p>"]
        param_to_label = lambda s: s.title().replace("_", " ")
        items.extend([f"<p><b>{param_to_label(k)}:</b> {v}</p>" for k, v in kwargs.items()])
        for key, data in audio.items():
            name = param_to_label(key)
            file_name = f"step={self.curr_step},name={name},experiment={self.get_key()}"
            url = self._upload_audio(file_name, data)
            items.append(f"<p><b>{name}:</b></p>")
            items.append(f'<audio controls preload="metadata" src="{url}"></audio>')
        self.log_html("<section>{}</section>".format("\n".join(items)))

    def log_parameter(self, key: run._config.Label, value: typing.Union[str, int, float]):
        self._experiment.log_parameter(key, value)

    def log_parameters(self, dict_: typing.Dict[run._config.Label, typing.Union[str, int, float]]):
        self._experiment.log_parameters(dict_)

    def log_other(self, key: run._config.Label, value: typing.Union[str, int, float]):
        self._experiment.log_other(key, value)

    def log_metrics(self, dict_: typing.Dict[run._config.Label, float]):
        [self.log_metric(k, v) for k, v in dict_.items()]

    def log_metric(self, name: run._config.Label, value: typing.Union[int, float]):
        self._experiment.log_metric(name, value)

    def log_figure(self, name: run._config.Label, figure: matplotlib.figure.Figure):
        self._experiment.log_figure(str(name), figure)

    def log_figures(self, dict_: typing.Dict[run._config.Label, matplotlib.figure.Figure]):
        """ Log multiple figures from `dict_` via `experiment.log_figure`. """
        [self.log_figure(k, v) for k, v in dict_.items()]

    def set_name(self, name: str):
        logger.info('Experiment name set to "%s"', name)
        self._experiment.set_name(name)

    def add_tags(self, tags: typing.List[str]):
        logger.info("Added tags to experiment: %s", tags)
        self._experiment.add_tags(tags)


@contextlib.contextmanager
def set_context(context: Context, model: torch.nn.Module, comet: CometMLExperiment):
    with comet.context_manager(context.value):
        logger.info("Setting context to '%s'.", context.value)
        mode = model.training
        model.train(mode=(context == Context.TRAIN))
        with torch.set_grad_enabled(mode=(context == Context.TRAIN)):
            yield
        model.train(mode=mode)


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
