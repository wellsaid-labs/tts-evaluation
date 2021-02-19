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
from hparams import HParam, add_config, configurable

from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, Dataset
from run.train._utils import CometMLExperiment, get_config_parameters, make_app, run_workers

logger = logging.getLogger(__name__)
torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for signal model training."""
    return {}


@configurable
def _run_worker(
    device: torch.device,
    comet: CometMLExperiment,
    checkpoints_directory: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
    train_steps_per_epoch: int = HParam(),
    dev_steps_per_epoch: int = HParam(),
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
    partial_ = _run_worker.get_configured_partial()
    return run_workers(partial_, comet, checkpoints_path, checkpoint, train_dataset, dev_dataset)


if __name__ == "__main__":  # pragma: no cover
    app = make_app(_run_app, SIGNAL_MODEL_EXPERIMENTS_PATH)
    app()
