import logging
import pathlib
import typing
from functools import partial

# NOTE: `comet_ml` needs to be imported before torch
import comet_ml  # type: ignore # noqa
import hparams.hparams
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, add_config, configurable, get_config

import lib
from lib.distributed import is_master
from run._config import SIGNAL_MODEL_EXPERIMENTS_PATH, Dataset
from run.train._utils import (
    CometMLExperiment,
    get_config_parameters,
    init_distributed,
    make_app,
    set_run_seed,
)

logger = logging.getLogger(__name__)
torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for spectrogram model training."""
    return {}


@configurable
def _run_worker(
    device_index: int,
    checkpoints_directory: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet_partial: typing.Callable[..., CometMLExperiment],
    config: typing.Dict[str, typing.Any],
    train_steps_per_epoch: int = HParam(),
    dev_steps_per_epoch: int = HParam(),
) -> typing.NoReturn:
    """Train and evaluate the spectrogram model on a loop.

    TODO: Should we checkpoint `metrics` so that metrics like `num_frames_per_speaker`,
    `num_spans_per_text_length`, or `max_num_frames` can be computed accross epochs?
    """
    lib.environment.set_basic_logging_config(device_index)
    device = init_distributed(device_index)
    comet = comet_partial(disabled=not is_master(), auto_output_logging=False)
    hparams.hparams._configuration = config
    set_run_seed()
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
    logger.info("Spawning workers %s", lib.utils.mazel_tov())
    partial_ = partial(CometMLExperiment, experiment_key=comet.get_key())
    args = (checkpoints_path, checkpoint, train_dataset, dev_dataset, partial_, get_config())
    return lib.distributed.spawn(_run_worker.get_configured_partial(), args=args)  # type: ignore


if __name__ == "__main__":  # pragma: no cover
    app = make_app(_run_app, SIGNAL_MODEL_EXPERIMENTS_PATH)
    app()
