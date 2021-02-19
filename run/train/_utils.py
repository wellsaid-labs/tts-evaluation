import contextlib
import copy
import dataclasses
import enum
import functools
import io
import itertools
import logging
import os
import pathlib
import sys
import time
import typing
from datetime import timedelta

import hparams.hparams
import numpy
import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import torch.utils.data
import typer
from hparams import HParam, HParams, configurable, get_config, parse_hparam_args
from third_party import LazyLoader
from torchnlp.utils import tensors_to

import lib
import run
from lib.distributed import is_master
from lib.environment import load, load_most_recent_file
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


def _maybe_make_experiment_directories_from_checkpoint(
    checkpoint: Checkpoint, *args, **kwargs
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """For checkpoints saved in the `maybe_make_experiment_directories` directory structure,
    this creates another "run" under the original experiment.
    """
    return _maybe_make_experiment_directories(
        checkpoint.checkpoints_directory.parent.parent, *args, **kwargs
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


def _get_dataset_stats(
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


def _nested_to_flat_config_helper(
    config: typing.Dict[str, typing.Any], delimitator: str, keys: typing.List[str]
) -> typing.Dict[str, typing.Any]:
    ret_ = {}
    for key in config:
        if isinstance(config[key], dict) and not isinstance(config, HParams):
            ret_.update(_nested_to_flat_config_helper(config[key], delimitator, keys + [key]))
        else:
            ret_[delimitator.join(keys + [key])] = config[key]
    return ret_


def _nested_to_flat_config(
    config: typing.Dict[str, typing.Any], delimitator: str = "."
) -> typing.Dict[str, typing.Any]:
    """Convert nested `hparam` configuration a flat configuration by concatenating keys with a
    `delimitator`.

    Args:
        ...
        delimitator: Delimitator used to join keys.
    """
    return _nested_to_flat_config_helper(config=config, delimitator=delimitator, keys=[])


def get_config_parameters() -> typing.Dict[run._config.Label, typing.Any]:
    """Get `hparams` configuration as a flat dictionary that can be logged."""
    flat = _nested_to_flat_config(get_config())
    return {run._config.get_config_label(k): v for k, v in flat.items()}


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
            self._experiment = comet_ml.Experiment(**kwargs, display_summary_level=0)
            self._experiment.log_html(self._BASE_HTML_STYLING)
        else:
            self._experiment = comet_ml.ExistingExperiment(
                previous_experiment=experiment_key, **kwargs, display_summary_level=0
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

    def log_parameter(self, key: run._config.Label, value: typing.Any):
        self._experiment.log_parameter(key, repr(value))

    def log_parameters(self, dict_: typing.Dict[run._config.Label, typing.Any]):
        """
        NOTE: Comet doesn't support `typing.Any` so we need to convert to a string representation.
        For example, Comet will silently fail and not log parameters with `numpy` or `torch` values.
        """
        self._experiment.log_parameters({k: repr(v) for k, v in dict_.items()})

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


@configurable
def set_run_seed(seed=HParam()):
    lib.environment.set_seed(seed)


def _worker_init_fn(_, config):
    # TODO: Add a method for transfering global configuration between processes without private
    # variables.
    # TODO: After the global configuration is transfered, the functions need to be rechecked
    # like for a configuration, just in case the configuration is on a new process.
    hparams.hparams._configuration = config
    info = torch.utils.data.get_worker_info()
    lib.environment.set_basic_logging_config()
    logger.info("Worker %d/%d iterator started.", info.id + 1, info.num_workers)
    set_run_seed()  # NOTE: Each worker needs the same random seed to be deterministic.


DataLoaderVar = typing.TypeVar("DataLoaderVar")


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
    likely experience deadlocks if you don’t change this setting.
    https://github.com/pytorch/pytorch/pull/4766
    > After OpenMP features are utilized, a fork is only allowed if the child process does not
    > use OpenMP features, or it does so as a completely new process (such as after exec()).
    https://bisqwit.iki.fi/story/howto/openmp/#OpenmpAndFork
    https://github.com/pytorch/pytorch/issues/42444
    > The CUDA runtime does not support the fork start method
    https://pytorch.org/docs/stable/notes/multiprocessing.html#cuda-in-multiprocessing

    TODO: The `DataLoader` runs `make_span_batch` and `iterator` in each worker. For performance,
    we could move `make_span_batch` to `DataIterator` and preprocess larger batches at the
    same time. The `collate_fn` function could be replaced with an `identity` function, and
    everything could be processed in the `DataIterator` efficiently. Learn more:
    https://github.com/pytorch/pytorch/blob/272f4db043ec2c63ecfe6d2759e7893cb842a3c3/torch/utils/data/_utils/fetch.py#L35
    https://pytorch.org/docs/stable/data.html#disable-automatic-batching
    This should also help with code locality. Also, if we'd like to run a more expensive dataset
    filtering, it is more doable in batches.

    TODO: Remove `copy.deepcopy` after this issue is fixed:
    https://github.com/pytorch/pytorch/issues/51849
    """

    def __init__(self, dataset: typing.Mapping, device: torch.device, **kwargs):
        logger.info("Creating `DataLoader`...")
        self.device = device
        loader = torch.utils.data.dataloader.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            pin_memory=True,
            batch_size=None,
            worker_init_fn=functools.partial(_worker_init_fn, config=copy.deepcopy(get_config())),
            collate_fn=lib.utils.identity,
            **kwargs,
        )
        self.loader = iter(loader)
        logger.info("Created `DataLoader`.")

    def process_batch(self, batch: DataLoaderVar) -> DataLoaderVar:
        # NOTE: Tensors are moved to CUDA outside of the `DataLoader` workers. Learn more:
        # > It is generally not recommended to return CUDA tensors in multi-process loading
        # > because of many subtleties in using CUDA and sharing CUDA tensors in multiprocessing
        # https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        return typing.cast(DataLoaderVar, tensors_to(batch, device=self.device, non_blocking=True))

    def __iter__(self) -> typing.Iterator[DataLoaderVar]:
        first = time.time()

        while True:
            yield self.process_batch(next(self.loader))

            # NOTE: The first batch will take the longest to load, so we log the time.
            if first is not None:
                elapsed = lib.utils.seconds_to_string(time.time() - first)
                logger.info("Time to first batch was %s.", elapsed)
                first = None


class _RunApp(typing.Protocol):
    def __call__(
        self,
        checkpoints_path: pathlib.Path,
        checkpoint: typing.Optional[pathlib.Path],
        train_dataset: run._config.Dataset,
        dev_dataset: run._config.Dataset,
        comet: CometMLExperiment,
        cli_config: typing.Dict[str, typing.Any],
        debug: bool = False,
    ):
        """Callback to run an experiment.

        Args:
            checkpoints_path: The directory to store checkpoints.
            checkpoint: The path of the loaded checkpoint.
            ...
            cli_config: Additional configuration passed through the command line.
            debug: Flag indicating if to start a debugging session.
        """
        ...


def make_app(run_app: _RunApp, directory: pathlib.Path, *args, **kwargs):
    """ Make CLI application for running an experiment, and saving it's details at `directory`."""
    app = typer.Typer(*args, **kwargs)

    lib.environment.enable_fault_handler()

    def _run(
        checkpoints_path: pathlib.Path,
        cli_config: typing.Dict[str, typing.Any],
        comet: CometMLExperiment,
        checkpoint: typing.Optional[pathlib.Path] = None,
        debug: bool = False,
    ):
        """Run spectrogram model training. """
        lib.environment.check_module_versions()

        datasets = run._config.DATASETS
        datasets = {k: v for k, v in list(datasets.items())[:1]} if debug else datasets

        # NOTE: Load, preprocess, and cache dataset values.
        dataset = run._utils.get_dataset(datasets)
        train_dataset, dev_dataset = run._utils.split_dataset(dataset)
        comet.log_parameters(_get_dataset_stats(train_dataset, dev_dataset))

        return run_app(
            checkpoints_path, checkpoint, train_dataset, dev_dataset, comet, cli_config, debug
        )

    def _setup_config(
        extra_cli_args: typing.List[str], debug: bool
    ) -> typing.Tuple[typing.Dict[str, typing.Any], lib.environment.RecordStandardStreams]:
        """
        TODO: For checkpointed runs, should we triple check the same parameters are getting
        configured? Should we throw an error if not? Or should we create a new experiment, and
        ensure that each experiments parameters are immutable?

        TODO: `RecordStandardStreams` should be started after `CometMLExperiment`; otherwise,
        `CometMLExperiment` will not be able to monitor the standard streams. Can this be fixed?
        """
        recorder = lib.environment.RecordStandardStreams()
        # NOTE: Ensure command line args are captured in the logs.
        logger.info("Command line args: %s", str(sys.argv))
        parsed = parse_hparam_args(extra_cli_args)
        run._config.configure()
        return parsed, recorder

    @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    def resume(
        context: typer.Context,
        checkpoint: typing.Optional[pathlib.Path] = typer.Argument(
            None, help="Checkpoint file to restart training from."
        ),
        debug: bool = typer.Option(False, help="Turn on debugging mode."),
    ):
        """Resume training from CHECKPOINT. If CHECKPOINT is not given, the most recent checkpoint
        file is loaded."""
        lib.environment.set_basic_logging_config()
        pattern = str(directory / f"**/*{lib.environment.PT_EXTENSION}")
        if checkpoint:
            loaded = load(checkpoint)
        else:
            checkpoint, loaded = load_most_recent_file(pattern, load)
        checkpoint_ = typing.cast(Checkpoint, loaded)
        comet = CometMLExperiment(experiment_key=checkpoint_.comet_experiment_key)
        cli_config, recorder = _setup_config(context.args, debug)
        _, checkpoints_path = _maybe_make_experiment_directories_from_checkpoint(
            checkpoint_, recorder
        )
        _run(checkpoints_path, cli_config, comet, checkpoint, debug=debug)

    @app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
    def start(
        context: typer.Context,
        project: str = typer.Argument(..., help="Experiment project name."),
        name: str = typer.Argument("", help="Experiment name."),
        tags: typing.List[str] = typer.Option([], help="Experiment tags."),
        debug: bool = typer.Option(False, help="Turn on debugging mode."),
        min_disk_space: float = 0.2,
    ):
        """ Start a training run in PROJECT named NAME with TAGS. """
        lib.environment.assert_enough_disk_space(min_disk_space)
        lib.environment.set_basic_logging_config()
        comet = CometMLExperiment(project_name=project)
        comet.set_name(name)
        comet.add_tags(tags)
        cli_config, recorder = _setup_config(context.args, debug)
        experiment_root = directory / lib.environment.bash_time_label()
        run_root, checkpoints_path = _maybe_make_experiment_directories(experiment_root, recorder)
        comet.log_other(run._config.get_environment_label("directory"), str(run_root))
        _run(checkpoints_path, cli_config, comet, debug=debug)

    return app


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


class _RunWorker(typing.Protocol):
    def __call__(self, device: torch.device, comet: CometMLExperiment, *args):
        ...


def _run_workers_helper(
    device_index: int,
    comet_partial: typing.Callable[..., CometMLExperiment],
    config: typing.Dict[str, typing.Any],
    run_worker: _RunWorker,
    *args,
):
    lib.environment.set_basic_logging_config(device_index)
    device = _init_distributed(device_index)
    comet = comet_partial(disabled=not is_master(), auto_output_logging=False)
    hparams.hparams._configuration = config
    set_run_seed()
    return run_worker(device, comet, *args)


def run_workers(run_worker: _RunWorker, comet: CometMLExperiment, *args):
    """Spawn workers for each GPU, and setup their environment."""
    logger.info("Spawning workers %s", lib.utils.mazel_tov())
    partial_ = functools.partial(CometMLExperiment, experiment_key=comet.get_key())
    args = (partial_, get_config(), run_worker, *args)
    return lib.distributed.spawn(_run_workers_helper, args=args)  # type: ignore
