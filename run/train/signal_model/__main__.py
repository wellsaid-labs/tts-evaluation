import logging
import pathlib
import typing
from unittest.mock import MagicMock

import config as cf

from run._config import (
    SIGNAL_MODEL_EXPERIMENTS_PATH,
    config_sig_model_training_from_datasets,
    get_config_label,
)
from run._utils import Dataset
from run.train._utils import CometMLExperiment, resume_experiment, run_workers, start_experiment
from run.train.signal_model import _worker

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import typer

    app = typer.Typer()
else:
    try:
        import typer

        app = typer.Typer()
    except (ModuleNotFoundError, NameError):
        app = MagicMock()
        typer = MagicMock()
        logger.info("Ignoring optional `typer` dependency.")


def _run_app(
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path],
    spectrogram_model_checkpoint: typing.Optional[pathlib.Path],
    cli_config: cf.Config,
    debug: bool,
):
    """Run signal model training."""
    config_sig_model_training_from_datasets(train_dataset, dev_dataset, debug)
    cf.add(cli_config)
    comet.log_parameters({get_config_label(k): v for k, v in cf.log(lambda x: x).items()})
    return run_workers(
        _worker.run_worker,  # type: ignore
        comet,
        checkpoint,
        checkpoints_directory,
        spectrogram_model_checkpoint,
        train_dataset,
        dev_dataset,
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def resume(
    context: typer.Context,
    checkpoint: typing.Optional[pathlib.Path] = typer.Argument(
        None, help="Checkpoint file to restart training from.", exists=True, dir_okay=False
    ),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Resume training from CHECKPOINT. If CHECKPOINT is not given, the most recent checkpoint
    file is loaded."""
    args = resume_experiment(SIGNAL_MODEL_EXPERIMENTS_PATH, checkpoint, debug=debug)
    cli_config = cf.parse_cli_args(context.args)
    _run_app(*args, None, cli_config, debug)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    checkpoint: pathlib.Path = typer.Argument(
        ...,
        help="Spectrogram model checkpoint file to generate training data with.",
        exists=True,
        dir_okay=False,
    ),
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Option([], help="Experiment tags."),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Start a training run in PROJECT named NAME with TAGS."""
    args = start_experiment(SIGNAL_MODEL_EXPERIMENTS_PATH, project, name, tags, debug=debug)
    cli_config = cf.parse_cli_args(context.args)
    _run_app(*args, None, checkpoint, cli_config, debug)


if __name__ == "__main__":  # pragma: no cover
    app()
