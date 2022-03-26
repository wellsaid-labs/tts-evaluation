import logging
import pathlib
import typing
from functools import partial
from unittest.mock import MagicMock

import config as cf
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data

import lib
from run._config import FRAME_HOP, RANDOM_SEED, SIGNAL_MODEL_EXPERIMENTS_PATH, get_config_label
from run._utils import Dataset, get_window
from run.train._utils import (
    CometMLExperiment,
    resume_experiment,
    run_workers,
    set_run_seed,
    start_experiment,
)
from run.train.signal_model import _metrics, _worker

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


def _make_configuration(train_dataset: Dataset, dev_dataset: Dataset, debug: bool) -> cf.Config:
    """Make additional configuration for signal model training."""
    train_size = sum(sum(p.segmented_audio_length() for p in d) for d in train_dataset.values())
    dev_size = sum(sum(p.segmented_audio_length() for p in d) for d in dev_dataset.values())
    ratio = train_size / dev_size
    logger.info("The training dataset is approx %fx bigger than the development dataset.", ratio)
    train_batch_size = int((32 if debug else 128) / lib.distributed.get_device_count())
    train_slice_size = 8192
    dev_slice_size = 32768
    batch_size_ratio = 1 / 2
    dev_batch_size = int(train_batch_size * train_slice_size / dev_slice_size / 2)
    oversample = 2
    dev_steps_per_epoch = 1 if debug else 256
    train_steps_per_epoch = int(round(dev_steps_per_epoch * batch_size_ratio * ratio * oversample))
    train_steps_per_epoch = 1 if debug else train_steps_per_epoch

    # NOTE: The `num_mel_bins` must be proportional to `fft_length`,
    # learn more:
    # https://stackoverflow.com/questions/56929874/what-is-the-warning-empty-filters-detected-in-mel-frequency-basis-about
    signal_to_spectrogram_params = [
        dict(
            fft_length=length,
            frame_hop=length // 4,
            window=get_window("hann", length, length // 4),
            num_mel_bins=length // 8,
        )
        for length in (256, 1024, 4096)
    ]

    real_label = True
    fake_label = False
    threshold = 0.5
    return {
        set_run_seed: cf.Args(seed=RANDOM_SEED),
        _worker._get_data_loaders: cf.Args(
            # SOURCE (Tacotron 2):
            # We train with a batch size of 128 distributed across 32 GPUs with
            # synchronous updates, using the Adam optimizer with β1 = 0.9, β2 =
            # 0.999, eps = 10−8 and a fixed learning rate of 10−4
            # NOTE: Parameters set after experimentation on a 8 V100 GPUs.
            train_batch_size=train_batch_size,
            # SOURCE: Efficient Neural Audio Synthesis
            # The WaveRNN models are trained on sequences of 960 audio samples.
            # SOURCE: Parallel WaveNet: Fast High-Fidelity Speech Synthesis
            # The teacher WaveNet network was trained for 1,000,000 steps with
            # the ADAM optimiser [14] with a minibatch size of 32 audio clips,
            # each containing 7,680 timesteps (roughly 320ms).
            # NOTE: The `spectrogram_slice_size` must be larger than the
            # `fft_length - frame_hop` of the largest `SpectrogramLoss`;
            # otherwise, the loss can't be computed.
            train_slice_size=int(train_slice_size / FRAME_HOP),
            dev_batch_size=dev_batch_size,
            dev_slice_size=int(dev_slice_size / FRAME_HOP),
            train_span_bucket_size=32,
            dev_span_bucket_size=32,
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=dev_steps_per_epoch,
            num_workers=2 if debug else 4,
            prefetch_factor=2 if debug else 16,
        ),
        _worker._State._get_optimizers: cf.Args(
            optimizer=partial(torch.optim.Adam, lr=10 ** -4, amsgrad=False, betas=(0.9, 0.999)),
            # NOTE: We employ a small warmup because the model can be unstable
            # at the start of it's training.
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
        ),
        _worker._State._get_signal_to_spectrogram_modules: cf.Args(
            kwargs=signal_to_spectrogram_params
        ),
        _worker._State._get_discrims: cf.Args(
            args=[(p["fft_length"], p["num_mel_bins"]) for p in signal_to_spectrogram_params]
        ),
        _worker._State._get_discrim_optimizers: cf.Args(
            optimizer=partial(torch.optim.Adam, lr=10 ** -3)
        ),
        _worker._run_discriminator: cf.Args(real_label=real_label, fake_label=fake_label),
        _metrics.Metrics.get_discrim_values: cf.Args(
            real_label=real_label, fake_label=fake_label, threshold=threshold
        ),
    }


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
    cf.add(_make_configuration(train_dataset, dev_dataset, debug))
    cf.add(cli_config)
    comet.log_parameters({get_config_label(k): v for k, v in cf.log().items()})
    return run_workers(
        _worker.run_worker,
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
