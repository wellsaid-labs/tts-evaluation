import logging
import pathlib
import typing
from functools import partial

import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import typer
from hparams import HParams, add_config, configurable, parse_hparam_args

import lib
from lib.distributed import get_world_size
from run._config import FRAME_HOP, RANDOM_SEED, SIGNAL_MODEL_EXPERIMENTS_PATH, Dataset
from run._utils import get_window
from run.train._utils import (
    CometMLExperiment,
    get_config_parameters,
    resume_experiment,
    run_workers,
    set_run_seed,
    start_experiment,
)
from run.train.signal_model import _metrics, _worker

lib.environment.enable_fault_handler()
logger = logging.getLogger(__name__)
app = typer.Typer()

if not hasattr(torch.optim.Adam.__init__, "_configurable"):
    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for signal model training."""

    train_size = sum([sum([p.aligned_audio_length() for p in d]) for d in train_dataset.values()])
    dev_size = sum([sum([p.aligned_audio_length() for p in d]) for d in dev_dataset.values()])
    ratio = train_size / dev_size
    logger.info("The training dataset is approx %fx bigger than the development dataset.", ratio)
    train_batch_size = 32 if debug else 128
    train_slice_size = 8192
    dev_slice_size = 32768
    batch_size_ratio = 1 / 2
    dev_batch_size = train_batch_size * train_slice_size / dev_slice_size / 2
    dev_steps_per_epoch = 1 if debug else 16
    train_steps_per_epoch = 1 if debug else dev_steps_per_epoch * batch_size_ratio * round(ratio)
    assert train_batch_size % get_world_size() == 0
    assert dev_batch_size % get_world_size() == 0

    # NOTE: The `num_mel_bins` must be proportional to `fft_length`,
    # learn more:
    # https://stackoverflow.com/questions/56929874/what-is-the-warning-empty-filters-detected-in-mel-frequency-basis-about
    signal_to_spectrogram_params = [
        dict(
            fft_length=2048,
            frame_hop=256,
            window=get_window("hann", 1024, 256),
            num_mel_bins=128,
        ),
        dict(
            fft_length=1024,
            frame_hop=128,
            window=get_window("hann", 512, 128),
            num_mel_bins=64,
        ),
        dict(
            fft_length=512,
            frame_hop=64,
            window=get_window("hann", 256, 64),
            num_mel_bins=32,
        ),
    ]

    real_label = True
    fake_label = False
    threshold = 0.5
    return {
        set_run_seed: HParams(seed=RANDOM_SEED),
        _worker._get_data_loaders: HParams(
            # SOURCE (Tacotron 2):
            # We train with a batch size of 128 distributed across 32 GPUs with
            # synchronous updates, using the Adam optimizer with β1 = 0.9, β2 =
            # 0.999, eps = 10−8 and a fixed learning rate of 10−4
            # NOTE: Parameters set after experimentation on a 8 V100 GPUs.
            train_batch_size=128,
            # SOURCE: Efficient Neural Audio Synthesis
            # The WaveRNN models are trained on sequences of 960 audio samples.
            # SOURCE: Parallel WaveNet: Fast High-Fidelity Speech Synthesis
            # The teacher WaveNet network was trained for 1,000,000 steps with
            # the ADAM optimiser [14] with a minibatch size of 32 audio clips,
            # each containing 7,680 timesteps (roughly 320ms).
            # NOTE: The `spectrogram_slice_size` must be larger than the
            # `fft_length - frame_hop` of the largest `SpectrogramLoss`;
            # otherwise, the loss can't be computed.
            train_slice_size=int(8192 / FRAME_HOP),
            dev_batch_size=16,
            dev_slice_size=int(32768 / FRAME_HOP),
            train_span_bucket_size=32,
            dev_span_bucket_size=32,
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=dev_steps_per_epoch,
            num_workers=2 if debug else 4,
            prefetch_factor=2 if debug else 4,
        ),
        _worker._State._get_optimizers: HParams(
            optimizer=torch.optim.Adam,
            # NOTE: We employ a small warmup because the model can be unstable
            # at the start of it's training.
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
        ),
        _worker._State._get_signal_to_spectrogram_modules: HParams(
            kwargs=signal_to_spectrogram_params
        ),
        _worker._State._get_discrims: HParams(
            args=[(p["fft_length"], p["num_mel_bins"]) for p in signal_to_spectrogram_params]
        ),
        _worker._State._get_discrim_optimizers: HParams(
            optimizer=partial(torch.optim.Adam, lr=10 ** -3)
        ),
        _worker._run_discriminator: HParams(real_label=real_label, fake_label=fake_label),
        _metrics.Metrics.__init__: HParams(
            fft_lengths=[p["fft_length"] for p in signal_to_spectrogram_params]
        ),
        _metrics.Metrics.get_discrim_values: HParams(
            real_label=real_label, fake_label=fake_label, threshold=threshold
        ),
        torch.optim.Adam.__init__: HParams(
            eps=10 ** -6,
            weight_decay=0,
            lr=10 ** -4,
            amsgrad=True,
            betas=(0.9, 0.999),
        ),
    }


def _run_app(
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path],
    spectrogram_model_checkpoint: typing.Optional[pathlib.Path],
    cli_config: typing.Dict[str, typing.Any],
    debug: bool,
):
    """Run signal model training."""
    add_config(_make_configuration(train_dataset, dev_dataset, debug))
    add_config(cli_config)
    comet.log_parameters(get_config_parameters())
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
        None, help="Checkpoint file to restart training from."
    ),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Resume training from CHECKPOINT. If CHECKPOINT is not given, the most recent checkpoint
    file is loaded."""
    args = resume_experiment(SIGNAL_MODEL_EXPERIMENTS_PATH, checkpoint, debug=debug)
    cli_config = parse_hparam_args(context.args)
    _run_app(*args, None, cli_config, debug)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Option([], help="Experiment tags."),
    checkpoint: pathlib.Path = typer.Argument(
        None, help="Spectrogram model checkpoint file to generate training data with."
    ),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
):
    """Start a training run in PROJECT named NAME with TAGS."""
    args = start_experiment(SIGNAL_MODEL_EXPERIMENTS_PATH, project, name, tags, debug=debug)
    cli_config = parse_hparam_args(context.args)
    _run_app(*args, None, checkpoint, cli_config, debug)


if __name__ == "__main__":  # pragma: no cover
    app()
