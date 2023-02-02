# Learn more:
# https://stackoverflow.com/questions/33533148/how-do-i-specify-that-the-return-type-of-a-method-is-the-same-as-the-class-itsel
from __future__ import annotations

import collections
import contextlib
import copy
import dataclasses
import enum
import functools
import html
import io
import itertools
import logging
import math
import numbers
import os
import pathlib
import platform
import pprint
import random
import resource
import sys
import time
import typing
from datetime import timedelta
from functools import partial

import config as cf
import numpy
import torch
import torch._C
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import torch.utils.data
import torch.utils.data._utils.worker
from third_party import LazyLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _MultiProcessingDataLoaderIter
from torchnlp.encoders.text import SequenceBatch

import lib
import run
from lib.distributed import ListedDict, is_master
from lib.environment import load, load_most_recent_file
from lib.utils import dataclass_as_dict, flatten_2d, seconds_to_str
from run._config import (
    Cadence,
    DatasetType,
    Device,
    Label,
    get_dataset_label,
    get_model_label,
    get_timer_label,
    load_spacy_nlp,
)
from run._models.spectrogram_model import Inputs, Mode, Preds, SpectrogramModel
from run._utils import Dataset, get_datasets
from run.data._loader.structures import Language, Session, Speaker

if typing.TYPE_CHECKING:  # pragma: no cover
    import comet_ml
    import matplotlib.figure
    import threadpoolctl
else:
    comet_ml = LazyLoader("comet_ml", globals(), "comet_ml")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")
    threadpoolctl = LazyLoader("threadpoolctl", globals(), "threadpoolctl")


lib.environment.enable_fault_handler()
logger = logging.getLogger(__name__)
pprinter = pprint.PrettyPrinter(indent=2)


class Context(enum.Enum):
    """Constants and labels for contextualizing the use-case."""

    # NOTE: This includes the entirety of the running script.
    SCRIPT: typing.Final = "script"
    TRAIN: typing.Final = "train"
    EVALUATE: typing.Final = "evaluate"
    EVALUATE_INFERENCE: typing.Final = "evaluate_inference"


class CometMLExperiment:
    """Create a `comet_ml.Experiment` or `comet_ml.ExistingExperiment` object with several
    adjustments.

    TODO: `auto_output_logging` and `RecordStandardStreams` will mutate `sys.stdout`. This can also
    cause issues where the logs are cut off. Create a toggle to avoid this mutation during
    debugging.

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

  img {
    vertical-align: middle;
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
            self._experiment = comet_ml.Experiment(display_summary_level=0, **kwargs)
            self._experiment.log_html(self._BASE_HTML_STYLING)
        else:
            self._experiment = comet_ml.ExistingExperiment(
                previous_experiment=experiment_key, display_summary_level=0, **kwargs
            )

        self.log_asset = self._experiment.log_asset
        self.log_html = self._experiment.log_html
        self.get_key = self._experiment.get_key
        self.set_model_graph = self._experiment.set_model_graph

        self._last_step_time: float = math.nan
        self._last_step: float = math.nan
        self._last_epoch_time: float = math.nan
        self._last_epoch_step: float = math.nan
        self._first_epoch_time: float = math.nan
        self._first_epoch_step: float = math.nan

        self._log_environment()

    @property
    def curr_step(self) -> typing.Optional[int]:
        return typing.cast(typing.Optional[int], self._experiment.curr_step)

    @property
    def curr_epoch(self) -> typing.Optional[int]:
        return typing.cast(typing.Optional[int], self._experiment.curr_epoch)

    @property
    def context(self) -> typing.Optional[str]:
        return typing.cast(typing.Optional[str], self._experiment.context)

    def _log_environment(self):
        """
        TODO: Collection additional information via:
        torch.cuda.get_device_capability
        torch.cuda.get_arch_list
        torch.cuda.get_device_name
        torch.cuda.get_gencode_flags

        TODO: Collect additional environment details like CUDA, CUDANN, NVIDIA Driver versions
        with this script:
        https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
        """
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
        """
        NOTE: Ensure that the variable `_last_step` is updated before `self.log_metric` is called.
        This prevents infinite recursion via
        `not math.isnan(seconds_per_step) and seconds_per_step > 0`.
        """
        self._experiment.set_step(step)
        if self.curr_step is not None:
            num_steps = self.curr_step - self._last_step
            num_seconds = time.time() - self._last_step_time
            seconds_per_step = num_seconds / num_steps if num_steps > 0 else math.nan
            self._last_step_time = time.time()
            self._last_step = self.curr_step
            if not math.isnan(seconds_per_step) and seconds_per_step > 0:
                label = get_timer_label("seconds_per_step", cadence=Cadence.STEP)
                self.log_metric(label, seconds_per_step)

    @contextlib.contextmanager
    def context_manager(self, context: Context):
        with self._experiment.context_manager(str(context.value)):
            yield self

    def log_current_epoch(self, epoch: int):
        assert self.curr_step is not None
        self._last_epoch_step = self.curr_step
        self._last_epoch_time = time.time()
        if math.isnan(self._first_epoch_time) and math.isnan(self._first_epoch_step):
            self._first_epoch_step = self.curr_step
            self._first_epoch_time = time.time()
        self._experiment.log_current_epoch(epoch)
        self._experiment.set_epoch(epoch)
        if not self._experiment.alive:
            self._experiment.curr_epoch = typing.cast(numbers.Number, epoch)

    def log_epoch_end(self, epoch: int):
        assert self.curr_step is not None

        # NOTE: Logs an average `steps_per_second` for each epoch.
        label = get_timer_label("steps_per_second", cadence=Cadence.MULTI_STEP)
        num_seconds = time.time() - self._last_epoch_time
        steps_per_second = (self.curr_step - self._last_epoch_step) / num_seconds
        if not math.isnan(steps_per_second):
            self.log_metric(label, steps_per_second)

        # NOTE: Logs an average `steps_per_second` since the training started.
        with self.context_manager(Context.SCRIPT):
            label = get_timer_label("steps_per_second", cadence=Cadence.RUN)
            num_seconds = time.time() - self._first_epoch_time
            steps_per_second = (self.curr_step - self._first_epoch_step) / num_seconds
            if not math.isnan(steps_per_second):
                self.log_metric(label, steps_per_second)

        self._experiment.log_epoch_end(epoch)

    def log_npy(
        self,
        name: str,
        speaker: run.data._loader.Speaker,
        array: typing.Union[numpy.ndarray, torch.Tensor],
    ) -> typing.Optional[str]:
        """Log a `ndarray` or `tensor` as a `.npy` file and return asset url."""
        file_name = f"step={self.curr_step},speaker={speaker.label},"
        file_name += f"name={name},experiment={self.get_key()}.npy"
        array = array.detach().cpu().numpy() if isinstance(array, torch.Tensor) else array
        file_ = io.BytesIO()
        numpy.save(file_, array, allow_pickle=False)
        file_.seek(0)
        asset = self.log_asset(file_, file_name=file_name)
        return asset["web"] if asset is not None else asset

    def _upload_audio(
        self, file_name: str, data: typing.Union[numpy.ndarray, torch.Tensor]
    ) -> typing.Optional[str]:
        """Upload the audio and return the URL."""
        file_ = io.BytesIO()
        lib.audio.write_audio(file_, data, **cf.get())
        asset = self.log_asset(file_, file_name=file_name)
        return asset["web"] if asset is not None else asset

    def log_html_audio(
        self,
        session: run.data._loader.Session,
        audio: typing.Dict[str, typing.Union[numpy.ndarray, torch.Tensor]] = {},
        **kwargs,
    ):
        """Audio with related metadata to Comet in the HTML tab.

        Args:
            audio
            **kwargs: Additional metadata to include.
        """
        items = [f"<p><b>Step:</b> {self.curr_step}</p>"]
        param_label = lambda s: s.title().replace("_", " ") if " " not in s else s
        html_repr = lambda v: v if isinstance(v, str) else html.escape(repr(v))
        kwargs = dict(session=session, **kwargs)
        items.extend([f"<p><b>{param_label(k)}:</b> {html_repr(v)}</p>" for k, v in kwargs.items()])
        for key, data in audio.items():
            name = param_label(key)
            file_name = f"step={self.curr_step},speaker={session[0].label},"
            file_name += f"name={name},experiment={self.get_key()}.wav"
            url = self._upload_audio(file_name, data)
            items.append(f"<p><b>{name}:</b></p>")
            if url is None:
                items.append(f"Failed to upload: {file_name}")
            else:
                items.append(f'<audio controls preload="none" src="{url}"></audio>')
        self.log_html("<section>{}</section>".format("\n".join(items)))

    def _handle_param(self, key: run._config.Label, value: typing.Any, max_len: int = 50) -> str:
        """Format and log complex objects in standard out."""
        if isinstance(value, (list, tuple, dict, set)) and len(repr(value)) > max_len:
            message = f"Comet parameter `{key}` is:\n{pprinter.pformat(value)}"
            lib.utils.call_once(logger.info, message)
            return "<<<Printed in standard out.>>>"
        if hasattr(value, "__qualname__"):
            return f"<function {value.__qualname__}>"  # type: ignore
        return repr(value)

    def log_parameter(self, key: run._config.Label, value: typing.Any):
        self._experiment.log_parameter(key, self._handle_param(key, value))

    def log_parameters(self, dict_: typing.Dict[run._config.Label, typing.Any]):
        """
        NOTE: Comet doesn't support `typing.Any` so we need to convert to a string representation.
        For example, Comet will silently fail and not log parameters with `numpy` or `torch` values.
        """
        self._experiment.log_parameters({k: self._handle_param(k, v) for k, v in dict_.items()})

    def log_other(self, key: run._config.Label, value: typing.Union[str, int, float]):
        self._experiment.log_other(key, value)

    def log_metrics(self, dict_: typing.Dict[run._config.Label, float]):
        [self.log_metric(k, v) for k, v in dict_.items()]

    def log_metric(self, name: run._config.Label, value: typing.Union[int, float]):
        self._experiment.log_metric(name, value)

    def log_figure(
        self, name: run._config.Label, figure: matplotlib.figure.Figure
    ) -> typing.Optional[str]:
        asset = self._experiment.log_figure(str(name), figure)
        return asset["web"] if asset is not None else asset

    def log_figures(
        self, dict_: typing.Dict[run._config.Label, matplotlib.figure.Figure]
    ) -> typing.Dict[run._config.Label, typing.Optional[str]]:
        """Log multiple figures from `dict_` via `experiment.log_figure`."""
        return {k: self.log_figure(k, v) for k, v in dict_.items()}

    def set_name(self, name: str):
        logger.info('Experiment name set to "%s"', name)
        self._experiment.set_name(name)

    def add_tags(self, tags: typing.List[str]):
        logger.info("Added tags to experiment: %s", tags)
        if len(tags) != 0:
            self._experiment.add_tags(tags)


@dataclasses.dataclass(frozen=True)
class Checkpoint:

    comet_experiment_key: str
    step: int


def _get_dataset_stats(
    train: Dataset, dev: Dataset
) -> typing.Dict[run._config.Label, typing.Union[str, int, float]]:
    """Get `train` and `dev` dataset statistics."""
    stats: typing.Dict[run._config.Label, typing.Union[int, str, float]] = {}
    data: Dataset
    for data, type_ in [(train, DatasetType.TRAIN), (dev, DatasetType.DEV)]:
        label_ = functools.partial(get_dataset_label, cadence=Cadence.STATIC, type_=type_)
        passages_ = flatten_2d([[p for p in v] for v in data.values()])
        for speaker, passages in itertools.chain(list(data.items()), [(None, passages_)]):
            label = label_ if speaker is None else functools.partial(label_, speaker=speaker)
            stats[label("num_audio_files")] = len(set(p.audio_file for p in passages))
            stats[label("num_passages")] = len(passages)
            stats[label("num_characters")] = sum(len(p.script) for p in passages)
            num_seconds = sum(p.segmented_audio_length() for p in passages)
            stats[label("num_seconds")] = seconds_to_str(num_seconds)
    return stats


def _run_experiment(
    comet: CometMLExperiment, debug: bool = False
) -> typing.Tuple[Dataset, Dataset]:
    """Helper function for `start_experiment` and  `resume_experiment`."""
    lib.environment.check_module_versions()
    train_dataset, dev_dataset = get_datasets(debug)
    comet.log_parameters(_get_dataset_stats(train_dataset, dev_dataset))
    return train_dataset, dev_dataset


def _maybe_make_experiment_directories_from_checkpoint(
    checkpoint_path: pathlib.Path,
    *args,
    run_prefix: str = "RUN_",
    run_suffix: str = lib.environment.bash_time_label(add_pid=False),
    checkpoints_directory_name: str = "checkpoints",
    **kwargs,
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """For checkpoints saved with the `_maybe_make_experiment_directories` directory structure,
    this creates another "run" under the original experiment.
    """
    message = "Unexpected directory structure."
    assert checkpoint_path.parent.name == checkpoints_directory_name, message
    assert checkpoint_path.parent.parent.name.startswith(run_prefix), message
    return _maybe_make_experiment_directories(
        checkpoint_path.parent.parent.parent,
        *args,
        run_name=run_prefix + run_suffix,
        checkpoints_directory_name=checkpoints_directory_name,
        **kwargs,
    )


def _maybe_make_experiment_directories(
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


def _setup_experiment() -> lib.environment.RecordStandardStreams:
    """
    TODO: For checkpointed runs, should we triple check the same parameters are getting
    configured? Should we throw an error if not? Or should we create a new experiment, and
    ensure that each experiments parameters are immutable?

    TODO: `RecordStandardStreams` should be started after `CometMLExperiment`; otherwise,
    `CometMLExperiment` will not be able to monitor the standard streams. Can this be fixed?
    """
    recorder = lib.environment.RecordStandardStreams()
    logger.info("Command line args: %s", str(sys.argv))  # NOTE: Command line args are recorded.
    run._config.configure()
    return recorder


def start_experiment(
    directory: pathlib.Path,
    project: str,
    name: str = "",
    tags: typing.List[str] = [],
    min_disk_space: float = 0.2,
    **kwargs,
) -> typing.Tuple[pathlib.Path, Dataset, Dataset, CometMLExperiment]:
    """Start a training run in a comet `project` named `name` with `tags`. The training run
    results are saved in `directory`."""
    lib.environment.assert_enough_disk_space(min_disk_space)
    lib.environment.set_basic_logging_config()
    comet = CometMLExperiment(project_name=project)
    comet.set_name(name)
    comet.add_tags(tags)
    recorder = _setup_experiment()
    experiment_root = directory / lib.environment.bash_time_label()
    run_root, checkpoints_path = _maybe_make_experiment_directories(experiment_root, recorder)
    comet.log_other(run._config.get_environment_label("directory"), str(run_root))
    return checkpoints_path, *_run_experiment(comet, **kwargs), comet


def resume_experiment(
    directory: pathlib.Path, checkpoint: typing.Optional[pathlib.Path], **kwargs
) -> typing.Tuple[pathlib.Path, Dataset, Dataset, CometMLExperiment, pathlib.Path]:
    """Resume training from `checkpoint`. If `checkpoint` is not given, the most recent checkpoint
    file is loaded from `directory`."""
    lib.environment.set_basic_logging_config()
    pattern = str(directory / f"**/*{lib.environment.PT_EXTENSION}")
    if checkpoint:
        loaded = load(checkpoint)
    else:
        checkpoint, loaded = load_most_recent_file(pattern, load)
    checkpoint_ = typing.cast(Checkpoint, loaded)
    comet = CometMLExperiment(experiment_key=checkpoint_.comet_experiment_key)
    recorder = _setup_experiment()
    _, checkpoints_path = _maybe_make_experiment_directories_from_checkpoint(checkpoint, recorder)
    return checkpoints_path, *_run_experiment(comet, **kwargs), comet, checkpoint


@contextlib.contextmanager
def set_train_mode(
    model: torch.nn.Module,
    mode: bool,
    ema: typing.Optional[lib.optimizers.ExponentialMovingParameterAverage] = None,
):
    original = model.training
    model.train(mode=mode)
    with contextlib.nullcontext() if ema is None or mode else ema:
        with torch.set_grad_enabled(mode=mode):
            yield
    model.train(mode=original)


@contextlib.contextmanager
def set_context(
    context: Context,
    comet: CometMLExperiment,
    *models: torch.nn.Module,
    ema: typing.Optional[lib.optimizers.ExponentialMovingParameterAverage] = None,
):
    with contextlib.ExitStack() as stack:
        stack.enter_context(comet.context_manager(context))
        logger.info("Setting context to '%s'.", context.value)
        is_training = context == Context.TRAIN
        for model in models:
            stack.enter_context(set_train_mode(model, is_training))
        stack.enter_context(contextlib.nullcontext() if ema is None or is_training else ema)
        yield


@contextlib.contextmanager
def set_epoch(comet: CometMLExperiment, step: int, steps_per_epoch: int):
    epoch = int(step // steps_per_epoch)
    message = "[%s] Running Epoch %d (Step %d)"
    logger.info(message, comet.context, epoch, step)
    comet.set_step(typing.cast(int, step))
    comet.log_current_epoch(epoch)
    yield
    comet.log_epoch_end(epoch)


def set_run_seed(seed: int):
    lib.environment.set_seed(seed)


def save_checkpoint(
    checkpoint: Checkpoint,
    checkpoints_directory: pathlib.Path,
    name: str,
    suffix=lib.environment.PT_EXTENSION,
) -> pathlib.Path:
    path = checkpoints_directory / f"{name}{suffix}"
    if is_master():
        lib.environment.save(path, checkpoint)
    return path


_ApplyToTensorsVar = typing.TypeVar("_ApplyToTensorsVar")


def apply_to_tensors(
    data: _ApplyToTensorsVar, call: typing.Callable[[torch.Tensor], torch.Tensor], is_return=True
) -> typing.Optional[_ApplyToTensorsVar]:
    """
    Args:
        data: An object holding data, either a `dataclass` or `NamedTuple`.
    """
    is_named_tuple = hasattr(data, "_fields") and hasattr(data, "_asdict")
    if not dataclasses.is_dataclass(data) and not is_named_tuple:
        return data

    dict_: dict = data._asdict() if is_named_tuple else dataclass_as_dict(data)  # type: ignore
    apply = lambda v: call(v) if torch.is_tensor(v) else apply_to_tensors(v, call, is_return)
    if is_return:
        return data.__class__(**{k: apply(v) for k, v in dict_.items()})
    else:
        [apply(value) for value in dict_.values()]


BatchType = typing.TypeVar("BatchType", bound="Batch")


@dataclasses.dataclass(frozen=True)
class Batch:
    def apply(self: BatchType, call: typing.Callable[[torch.Tensor], torch.Tensor]) -> BatchType:
        """Apply `call` to `SequenceBatch` in `Batch`."""
        # TODO: Given that this has a specific use case with `SequenceBatch` it shouldn't
        # have a generic name like `apply`.
        apply = lambda o: apply_to_tensors(o, call, True) if isinstance(o, SequenceBatch) else o
        dict_ = lib.utils.dataclass_as_dict(self)
        return dataclasses.replace(self, **{k: apply(v) for k, v in dict_.items()})

    def pin_memory(self: BatchType) -> BatchType:
        """Learn more about this special function:
        https://pytorch.org/docs/stable/data.html#memory-pinning

        NOTE: Filtering `v` by `SequenceBatch` will cause these errors:
        `RuntimeError: received 0 items of ancdata`
        ... followed up with ...
        `RuntimeError: Pin memory thread exited unexpectedly`
        The issue can also be resolved by increasing `ulimit`, like so:
        `ulimit -S -n 4096`. It can also be resolved by not filtering.
        Learn more: https://github.com/pytorch/pytorch/issues/973
        """
        return self.apply(lambda t: t.pin_memory())


def set_num_threads(num_threads: int):
    """Set number of threads for `torch`, `numpy`, `scipy` and `scikit`."""
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    threadpoolctl.threadpool_limits(limits=num_threads)
    assert torch.get_num_threads() == num_threads, "Failed to set `num_threads`."
    assert torch.get_num_interop_threads() == num_threads, "Failed to set `num_threads`."
    info = threadpoolctl.threadpool_info()
    assert any("/numpy" in i["filepath"] for i in info), "Failed to find `numpy`."
    assert any("/scipy" in i["filepath"] for i in info), "Failed to find `scipy`."
    assert all(i["num_threads"] == num_threads for i in info), "Failed to set `num_threads`."


def _worker_init_fn(
    _,
    configuration: cf.Config,
    worker_init_fn: typing.Optional[typing.Callable],
    rank: int,
    num_threads: int = 1,
):
    cf.enable_fast_trace()
    cf.add(configuration)
    info = torch.utils.data.get_worker_info()
    assert isinstance(info, torch.utils.data._utils.worker.WorkerInfo)
    lib.environment.set_basic_logging_config()
    # NOTE: Set `num_threads` to ensure that these workers share resources with the main process.
    set_num_threads(num_threads)
    logger.info("Worker %d/%d started for rank %d.", info.id + 1, info.num_workers, rank)
    if worker_init_fn is not None:
        worker_init_fn()


DataLoaderVar = typing.TypeVar("DataLoaderVar", bound=Batch)


class DataLoader(typing.Iterable[DataLoaderVar], typing.Generic[DataLoaderVar]):
    """Load and batch spans given a dataset `iterator`.

    NOTE: The `DataLoader` by default will create a sequential sampler. It'll use that sampler
    to queue up batches from `DataIterator`, in order.

    NOTE: Each `DataLoader` worker replicates the dataset, and other objects. As of
    02/04/2020, about half of our memory (30 gb) was used by `DataLoader` workers. This
    can be resolved with memory sharing like "fork" and `gc.freeze`.

    NOTE: `DataLoader` isn't compatible with "fork" because NCCL isn't fork safe. There
    are also issues with OMP and CUDA. They have issues with fork, as well. Learn more:
    > Unfortunately Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
    likely experience deadlocks if you don't change this setting.
    https://github.com/pytorch/pytorch/pull/4766
    > After OpenMP features are utilized, a fork is only allowed if the child process does not
    > use OpenMP features, or it does so as a completely new process (such as after exec()).
    https://bisqwit.iki.fi/story/howto/openmp/#OpenmpAndFork
    https://github.com/pytorch/pytorch/issues/42444
    > The CUDA runtime does not support the fork start method
    https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

    TODO: Remove `copy.deepcopy` after this issue is fixed:
    https://github.com/pytorch/pytorch/issues/51849
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        num_steps_per_epoch: int,
        worker_init_fn: typing.Optional[typing.Callable],
        cuda_prefetch: int = 16,
        **kwargs,
    ):
        self._set_r_limit()
        self.device = device
        self.stream = torch.cuda.streams.Stream() if torch.cuda.is_available() else None
        self.loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=typing.cast(int, None),
            worker_init_fn=functools.partial(
                _worker_init_fn,
                configuration=copy.deepcopy(cf.export()),
                worker_init_fn=worker_init_fn,
                rank=lib.distributed.get_rank(),
            ),
            collate_fn=lib.utils.identity,
            **kwargs,
        )
        self.iter: typing.Optional[_BaseDataLoaderIter] = None
        self.num_steps_per_epoch = num_steps_per_epoch
        self.prefetched = []
        self.cuda_prefetch = cuda_prefetch

    @staticmethod
    def _set_r_limit(soft_limit=4096):
        """Increase the number of available file descriptors to `soft_limit`.

        Learn more: https://github.com/pytorch/pytorch/issues/973
        """
        if platform.system() == "Linux":
            rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, rlimit[1]))

    def process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        NOTE: Tensors are moved to CUDA outside of the `DataLoader` workers. Learn more:
        > It is generally not recommended to return CUDA tensors in multi-process loading
        > because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing
        https://pytorch.org/docs/stable/data.html#multi-process-data-loading

        NOTE: `torch.utils.data.dataloader.DataLoader` doesn't pin tensors if CUDA isn't
        available.
        """
        message = f"Expecting `tensor` ({tensor.shape}, {tensor.dtype}) memory to be "
        message += "pinned before moving."
        assert not torch.cuda.is_available() or tensor.is_pinned(), message
        return tensor.to(device=self.device, non_blocking=True)

    def prefetch(self):
        """Prefetch next example, and move it asynchronously to the correct device.

        TODO: Coordinate `num_steps_per_epoch` with `prefetch`.

        Learn more:
        https://github.com/PyTorchLightning/lightning-bolts/pull/127
        https://github.com/NVIDIA/apex/issues/304
        """
        assert self.iter is not None
        for _ in range(self.cuda_prefetch - len(self.prefetched)):
            next_: Batch = next(self.iter)
            self.prefetched.append(next_.apply(self.process_tensor))

    def __iter__(self) -> typing.Iterator[DataLoaderVar]:
        if self.iter is None:
            self.iter = typing.cast(_BaseDataLoaderIter, iter(self.loader))

        if not torch.cuda.is_available():
            yield from (next(self.iter) for _ in range(self.num_steps_per_epoch))

        self.prefetch()
        for _ in range(self.num_steps_per_epoch):
            next_ = self.prefetched.pop(0)
            self.prefetch()
            yield next_


def _init_distributed(
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
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank, timeout=timeout
    )
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


class _RunWorker(typing.Protocol):
    def __call__(
        self,
        device: torch.device,
        comet: CometMLExperiment,
        checkpoint: typing.Optional[Checkpoint],
        *args,
    ) -> typing.NoReturn:
        ...


def _run_workers_helper(
    device_index: int,
    comet_partial: typing.Callable[..., CometMLExperiment],
    configuration: cf.Config,
    checkpoint: typing.Optional[pathlib.Path],
    run_worker: _RunWorker,
    *args,
):
    lib.environment.set_basic_logging_config(device_index)
    device = _init_distributed(device_index)
    comet = comet_partial(disabled=not is_master(), auto_output_logging=False)
    cf.enable_fast_trace()
    cf.add(configuration)
    set_run_seed(**cf.get())
    checkpoint_ = None if checkpoint is None else load(checkpoint, device=device)
    return run_worker(device, comet, checkpoint_, *args)


def run_workers(
    run_worker: _RunWorker,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path],
    *args,
):
    """Spawn workers for each GPU, and setup their environment.

    TODO: Remove `copy.deepcopy` after this issue is fixed:
    https://github.com/pytorch/pytorch/issues/51849
    """
    partial_ = functools.partial(CometMLExperiment, experiment_key=comet.get_key())
    args = (partial_, copy.deepcopy(cf.export()), checkpoint, run_worker, *args)
    logger.info("Spawning workers %s", lib.utils.mazel_tov())
    return lib.distributed.spawn(_run_workers_helper, args=args)  # type: ignore


@dataclasses.dataclass(frozen=True)
class MetricsKey:

    label: str


MetricsKeyTypeVar = typing.TypeVar("MetricsKeyTypeVar", bound=MetricsKey)
MetricsReduceOp = typing.Callable[[typing.List[float]], float]
# NOTE: `MetricsSelect` selects a subset of `ListedDict` values.
MetricsSelect = typing.Callable[
    [ListedDict[MetricsKeyTypeVar, float]], ListedDict[MetricsKeyTypeVar, float]
]


class Metrics(lib.distributed.DictStore, typing.Generic[MetricsKeyTypeVar]):
    """Metrics collated accross different processes."""

    DATA_QUEUE_SIZE, *_ = tuple([str(i) for i in range(100)])

    MIN_DATA_LOADER_QUEUE_SIZE = partial(get_dataset_label, "min_data_loader_queue_size")

    GRADIENT_INFINITY_NORM = partial(get_model_label, "grad_norm/inf")
    GRADIENT_MAX_NORM = partial(get_model_label, "grad_norm/max_norm")
    GRADIENT_NORM = partial(get_model_label, "grad_norm")
    LR = partial(get_model_label, "lr")

    def __init__(self, comet: CometMLExperiment, *args, **kwargs):
        self.data: ListedDict[MetricsKeyTypeVar, float]
        super().__init__(*args, **kwargs, cache_keys=True)
        self.comet = comet

    def update(self, data: typing.Dict[MetricsKeyTypeVar, float]):
        return super().update(data)

    @staticmethod
    def _to_list(tensor: torch.Tensor) -> typing.List[float]:
        assert len(tensor.squeeze().shape) <= 1, "Tensor must be 1-dimensional."
        return tensor.view(-1).tolist()

    def get_data_loader_values(
        self, data_loader: DataLoader
    ) -> typing.Dict[MetricsKeyTypeVar, float]:
        """
        NOTE: `qsize` is not implemented on MacOS, learn more:
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
        """
        is_multiprocessing = isinstance(data_loader.iter, _MultiProcessingDataLoaderIter)
        if is_multiprocessing and platform.system() != "Darwin":
            iterator = typing.cast(_MultiProcessingDataLoaderIter, data_loader.iter)
            make_key = typing.get_args(self.__class__.__orig_bases__[0])[0]  # type: ignore
            return {make_key(self.DATA_QUEUE_SIZE): iterator._data_queue.qsize()}
        return {}

    def _reduce(
        self, key: MetricsKeyTypeVar, select: MetricsSelect, op: MetricsReduceOp = sum
    ) -> float:
        """Reduce `self.data[key]` measurements to a float."""
        flat: typing.List[float] = flatten_2d(select(self.data)[key] if key in self.data else [])
        assert all(not math.isnan(val) for val in flat), f"Encountered NaN value for metric {key}."
        return math.nan if len(flat) == 0 else op(flat)

    def _div(
        self,
        num: typing.Union[MetricsKeyTypeVar, float],
        denom: typing.Union[MetricsKeyTypeVar, float],
        **kwargs,
    ) -> float:
        """Reduce and divide `self.data[num] / self.data[denom]`."""
        reduced_denom = denom if isinstance(denom, float) else self._reduce(denom, **kwargs)
        if reduced_denom == 0:
            return math.nan
        reduced_num = num if isinstance(num, float) else self._reduce(num, **kwargs)
        return reduced_num / reduced_denom

    def log_optim_metrics(
        self,
        parameter_norm: float,
        parameter_norm_inf: float,
        optimizer: typing.Union[torch.optim.Adam, torch.optim.AdamW],
        clipper: lib.optimizers.AdaptiveGradientNormClipper,
        **kwargs,
    ):
        """Log optimizer metrics for `optimizer` and `clipper`. The model parameters have already
        been sync'd; therefore, there is no need to further sync parameters.
        """
        if is_master():
            message = "Expecting only 1 learning rate."
            assert len(set(g["lr"] for g in optimizer.param_groups)) == 1, message
            metrics = {
                self.GRADIENT_NORM: parameter_norm,
                self.GRADIENT_INFINITY_NORM: parameter_norm_inf,
                self.LR: optimizer.param_groups[0]["lr"],
            }
            self.comet.log_metrics({k(**kwargs): v for k, v in metrics.items()})
            if math.isfinite(clipper.max_norm):  # NOTE: Initially, `max_norm` will be `inf`.
                self.comet.log_metric(self.GRADIENT_MAX_NORM(**kwargs), clipper.max_norm)


class _TimerEvent(typing.NamedTuple):
    name: str
    cpu: float
    cuda: typing.Optional[torch._C._CudaEventBase]  # type: ignore


class Timer:
    """Record and time the time elapsed between the below events."""

    LOAD_DATA = "load_data"
    MODEL_FORWARD = "model_forward"
    MODEL_BACKWARD = "model_backward"
    MODEL_STEP = "model_step"
    VISUALIZE_PREDICTIONS = "visualize_predictions"
    MEASURE_METRICS = "measure_metrics"
    GATHER_METRICS = "gather_metrics"
    REDUCE_METRICS = "reduce_metrics"
    LOG_METRICS = "log_metrics"
    _LAST_EVENT = "last_event"

    def __init__(self, prefix="seconds/"):
        self.events: typing.List[_TimerEvent] = []
        self.prefix = prefix

    def record_event(self, name: str):
        event = None
        if torch.cuda.is_available():
            event = torch.cuda.streams.Event(enable_timing=True)
            event.record(torch.cuda.default_stream())
        self.events.append(_TimerEvent(name, time.perf_counter(), event))
        return self

    def get_timers(self, **kwargs) -> typing.Dict[Label, float]:
        self.record_event(self._LAST_EVENT)
        times: typing.Dict[Label, float] = collections.defaultdict(float)
        for prev, next in zip(self.events, self.events[1:]):
            name = f"{self.prefix}{prev.name}"
            times[get_timer_label(name, **kwargs)] += next.cpu - prev.cpu
            if torch.cuda.is_available():
                assert prev.cuda is not None
                assert next.cuda is not None
                prev.cuda.synchronize()
                next.cuda.synchronize()
                label = get_timer_label(name, device=Device.CUDA, **kwargs)
                times[label] += prev.cuda.elapsed_time(next.cuda) / 1000
        return dict(times)


def process_select_cases(
    model: SpectrogramModel,
    avail_sessions: typing.Set[Session],
    cases: typing.List[typing.Tuple[Language, str]],
    speakers: typing.Set[Speaker],
    num_cases: int = 5,
) -> typing.Tuple[Inputs, Preds]:
    """Get the spectrogram model prediction for `num_cases` sampled from `cases` limited to
    `speakers`."""
    cases = [random.choice(cases) for _ in range(num_cases)]
    docs = [load_spacy_nlp(l)(t) for (l, t) in cases]
    # NOTE: `seshs` is sorted so `random.choice` produces consistent results.
    vocab = sorted(avail_sessions)
    seshs = [[s for s in vocab if s[0].language is l and s[0] in speakers] for l, _ in cases]
    seshs = [random.choice(choices) for choices in seshs]
    inputs_ = Inputs(seshs, docs)
    return inputs_, model(inputs_, mode=Mode.INFER)
