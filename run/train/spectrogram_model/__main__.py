import logging
import pathlib
import typing
from functools import partial
from unittest.mock import MagicMock

import torch
import torch.optim
from hparams import HParams, add_config, parse_hparam_args
from third_party import LazyLoader

import lib
from run._config import (
    NUM_FRAME_CHANNELS,
    PHONEME_SEPARATOR,
    RANDOM_SEED,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    Dataset,
)
from run.train._utils import (
    CometMLExperiment,
    get_config_parameters,
    resume_experiment,
    run_workers,
    set_run_seed,
    start_experiment,
)
from run.train.spectrogram_model import _metrics, _worker
from run.train.spectrogram_model._data import InputEncoder

if typing.TYPE_CHECKING:  # pragma: no cover
    import typer
else:
    typer = LazyLoader("typer", globals(), "typer")

logger = logging.getLogger(__name__)

try:
    app = typer.Typer()
except (ModuleNotFoundError, NameError):
    app = MagicMock()
    typer = MagicMock()
    logger.info("Ignoring optional `typer` dependency.")

def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for spectrogram model training."""

    train_size = sum(sum(p.segmented_audio_length() for p in d) for d in train_dataset.values())
    dev_size = sum(sum(p.segmented_audio_length() for p in d) for d in dev_dataset.values())
    ratio = train_size / dev_size
    logger.info("The training dataset is approx %fx bigger than the development dataset.", ratio)
    train_batch_size = 28 if debug else 56
    batch_size_ratio = 4
    dev_batch_size = train_batch_size * batch_size_ratio
    dev_steps_per_epoch = 1 if debug else 64
    train_steps_per_epoch = int(round(dev_steps_per_epoch * batch_size_ratio * ratio))
    train_steps_per_epoch = 1 if debug else train_steps_per_epoch
    assert train_batch_size % lib.distributed.get_device_count() == 0
    assert dev_batch_size % lib.distributed.get_device_count() == 0

    return {
        set_run_seed: HParams(seed=RANDOM_SEED),
        _worker._State._get_optimizers: HParams(
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.AdamW,
            exclude_from_decay=_worker.exclude_from_decay,
        ),
        _worker._run_step: HParams(
            # NOTE: This scalar calibrates the loss so that it's scale is similar to Tacotron-2.
            spectrogram_loss_scalar=1 / 100,
            # NOTE: Learn more about this parameter here: https://arxiv.org/abs/2002.08709
            # NOTE: This value is the minimum loss the test set achieves before the model
            # starts overfitting on the train set.
            # TODO: Try increasing the stop token minimum loss because it still overfit.
            stop_token_min_loss=0.027,
            # NOTE: This value is the average spectrogram length in the training dataset.
            average_spectrogram_length=117.5,
        ),
        _worker._get_data_loaders: HParams(
            # SOURCE: Tacotron 2
            # To train the feature prediction network, we apply the standard maximum-likelihood
            # training procedure (feeding in the correct output instead of the predicted output on
            # the decoder side, also referred to as teacher-forcing) with a batch size of 64 on a
            # single GPU.
            # NOTE: Batch size parameters set after experimentation on a 2 Px100 GPU.
            train_batch_size=train_batch_size,
            dev_batch_size=dev_batch_size,
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=int(dev_steps_per_epoch),
            is_train_balanced=False,
            is_dev_balanced=True,
            num_workers=2,
            prefetch_factor=2 if debug else 10,
        ),
        _metrics.Metrics._get_model_metrics: HParams(num_frame_channels=NUM_FRAME_CHANNELS),
        # NOTE: Based on the alignment visualizations, if the maximum alignment is less than 30%
        # then a misalignment has likely occured.
        _metrics.get_num_small_max: HParams(threshold=0.3),
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−6 learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        # NOTE: No L2 regularization performed better based on Comet experiments in March 2020.
        torch.optim.AdamW.__init__: HParams(
            eps=10 ** -6,
            weight_decay=0.01,
            lr=10 ** -3,
            amsgrad=False,
            betas=(0.9, 0.999),
        ),
        InputEncoder.__init__: HParams(phoneme_separator=PHONEME_SEPARATOR),
    }


def _run_app(
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path],
    cli_config: typing.Dict[str, typing.Any],
    debug: bool,
):
    """Run spectrogram model training.

    TODO: PyTorch-Lightning makes strong recommendations to not use `spawn`. Learn more:
    https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    https://github.com/PyTorchLightning/pytorch-lightning/pull/2029
    https://github.com/PyTorchLightning/pytorch-lightning/issues/5772
    Also, it's less normal to use `spawn` because it wouldn't work with multiple nodes, so
    we should consider using `torch.distributed.launch`.
    TODO: Should we consider setting OMP num threads similarly:
    https://github.com/pytorch/pytorch/issues/22260
    """
    add_config(_make_configuration(train_dataset, dev_dataset, debug))
    add_config(cli_config)
    comet.log_parameters(get_config_parameters())
    return run_workers(
        _worker.run_worker, comet, checkpoint, checkpoints_directory, train_dataset, dev_dataset
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
    args = resume_experiment(SPECTROGRAM_MODEL_EXPERIMENTS_PATH, checkpoint, debug=debug)
    cli_config = parse_hparam_args(context.args)
    _run_app(*args, cli_config, debug)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Option(
        [
            "recording_sessions",
            "promo",
            "monotonic",
            "no_balance_train",
            "no_slash",
            "no_long_pauses",
            "stop_token_0.027_min_loss",
            "encoder_2_lstm",
            "ema",
            "avg_spec_len_117.5",
        ],
        help="Experiment tags.",
    ),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """ Start a training run in PROJECT named NAME with TAGS. """
    args = start_experiment(SPECTROGRAM_MODEL_EXPERIMENTS_PATH, project, name, tags, debug=debug)
    cli_config = parse_hparam_args(context.args)
    _run_app(*args, None, cli_config, debug)


if __name__ == "__main__":  # pragma: no cover
    app()
