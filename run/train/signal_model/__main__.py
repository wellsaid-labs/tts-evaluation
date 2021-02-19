import dataclasses
import logging
import pathlib
import typing

# NOTE: `comet_ml` needs to be imported before torch
import comet_ml  # type: ignore # noqa
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import add_config, configurable

import lib
from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, Dataset
from run.train import _utils
from run.train._utils import CometMLExperiment, get_config_parameters, make_app, run_workers

logger = logging.getLogger(__name__)
torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for signal model training."""
    return {}


@dataclasses.dataclass(frozen=True)
class Checkpoint(_utils.Checkpoint):
    """Checkpoint used to checkpoint spectrogram model training."""

    model: lib.signal_model.SignalModel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    scheduler: torch.optim.lr_scheduler.LambdaLR


def _run_worker(
    device: torch.device,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[Checkpoint],
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
) -> typing.NoReturn:
    """Train and evaluate the signal model in a loop."""
    while True:
        pass


def _run_app(
    checkpoints_path: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    cli_config: typing.Dict[str, typing.Any],
    debug: bool = False,
):
    """Run signal model training."""
    add_config(_make_configuration(train_dataset, dev_dataset, debug))
    add_config(cli_config)
    comet.log_parameters(get_config_parameters())
    return run_workers(_run_worker, comet, checkpoint, checkpoints_path, train_dataset, dev_dataset)


if __name__ == "__main__":  # pragma: no cover
    app = make_app(_run_app, SIGNAL_MODEL_EXPERIMENTS_PATH)
    app()
