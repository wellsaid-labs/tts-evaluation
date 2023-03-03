import logging
from functools import partial

import config as cf
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data

import lib
import run
from run._config.audio import FRAME_HOP, FRAME_SIZE, NUM_FRAME_CHANNELS
from run._config.environment import RANDOM_SEED
from run._utils import Dataset

logger = logging.getLogger(__name__)


def _get_ratio(train_dataset: Dataset, dev_dataset: Dataset) -> float:
    train_size = sum(sum(p.segmented_audio_length() for p in d) for d in train_dataset.values())
    dev_size = sum(sum(p.segmented_audio_length() for p in d) for d in dev_dataset.values())
    ratio = train_size / dev_size
    logger.info("The training dataset is approx %fx bigger than the development dataset.", ratio)
    return ratio


def _get_num_sessions(train_dataset: Dataset) -> int:
    return len(set(psg.session for data in train_dataset.values() for psg in data))


def _get_spec_model_training_configs(train_dataset: Dataset, dev_dataset: Dataset, debug: bool):
    """Get configurations for various values like batch size and steps per epoch using the train
    and dev datasets."""
    ratio = _get_ratio(train_dataset, dev_dataset)
    train_batch_size = 28 if debug else 56
    batch_size_ratio = 4
    dev_batch_size = train_batch_size * batch_size_ratio
    dev_steps_per_epoch = 1 if debug else 64
    oversample = 3
    train_steps_per_epoch = int(round(dev_steps_per_epoch * batch_size_ratio * ratio * oversample))
    train_steps_per_epoch = 1 if debug else train_steps_per_epoch
    assert train_batch_size % lib.distributed.get_device_count() == 0
    assert dev_batch_size % lib.distributed.get_device_count() == 0
    num_sesh = _get_num_sessions(train_dataset)
    return train_batch_size, dev_batch_size, train_steps_per_epoch, dev_steps_per_epoch, num_sesh


def exclude_from_decay(
    param_name: str, param: torch.nn.parameter.Parameter, module: torch.nn.Module
) -> bool:
    """
    NOTE: Learn more about removing regularization from bias terms or `LayerNorm`:
    https://stats.stackexchange.com/questions/153605/no-regularisation-term-for-bias-unit-in-neural-network/153650
    https://github.com/huggingface/transformers/issues/4360
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994

    Args:
        param_name: The parameter name as returned by `torch.nn.Module.named_parameters`.
        param: The parameter name as returned by `torch.nn.Module.parameters`.
        module: The parent module for this parameter.
    """
    deny_list = (torch.nn.modules.normalization.LayerNorm,)
    return ".bias" in param_name or any(isinstance(module, m) for m in deny_list)


def _config_spec_model_training(
    train_batch_size: int,
    dev_batch_size: int,
    train_steps_per_epoch: int,
    dev_steps_per_epoch: int,
    num_sesh: int,
    debug: bool,
    **kwargs,
):
    """Make additional configurations for spectrogram model training."""
    spectrogram_model = run.train.spectrogram_model
    config = {
        run.train._utils.set_run_seed: cf.Args(seed=RANDOM_SEED),
        spectrogram_model._worker._State._get_optimizers: cf.Args(
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=10_000
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.AdamW,
            exclude_from_decay=spectrogram_model._worker.exclude_from_decay,
        ),
        spectrogram_model._worker._run_step: cf.Args(
            # NOTE: This scalar calibrates the loss so that it's scale is similar to Tacotron-2.
            spectrogram_loss_scalar=1 / 100,
            # NOTE: This value is the average spectrogram length in the training dataset.
            # NOTE: This value was computed with a reference frame size of 4096, and it scales
            # linearly with frame size.
            average_spectrogram_length=87.485 * (4096 / FRAME_SIZE),
            # NOTE: This starts to decay the stop token loss as soon as it converges so it doesn't
            # overfit. Also, this ensures that the model doesn't unnecessarily prioritize the stop
            # token loss when it has already converged.
            stop_token_loss_multiplier=partial(
                lib.optimizers.exponential_decay_lr_multiplier_schedule,
                warmup=0,
                start_decay=10_000,
                end_decay=50_000,
                multiplier=0.001,
            ),
        ),
        spectrogram_model._worker._get_data_generator: cf.Args(
            train_get_weight=spectrogram_model._data.train_get_weight,
            dev_get_weight=spectrogram_model._data.dev_get_weight,
        ),
        spectrogram_model._worker._get_data_processors: cf.Args(
            train_batch_size=train_batch_size,
            dev_batch_size=dev_batch_size,
        ),
        spectrogram_model._worker._get_data_loaders: cf.Args(
            # SOURCE: Tacotron 2
            # To train the feature prediction network, we apply the standard maximum-likelihood
            # training procedure (feeding in the correct output instead of the predicted output on
            # the decoder side, also referred to as teacher-forcing) with a batch size of 64 on a
            # single GPU.
            # NOTE: Batch size parameters set after experimentation on a 2 Px100 GPU.
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=int(dev_steps_per_epoch),
            num_workers=2,
            prefetch_factor=2 if debug else 10,
        ),
        spectrogram_model._metrics.Metrics._get_model_metrics: cf.Args(
            num_frame_channels=NUM_FRAME_CHANNELS
        ),
        # NOTE: This parameter was set via the workbook `run/review/tts/batch_griffin_lim.py`. It
        # is closely aligned to the number of silent frames at the end of each sequence; however,
        # it's a bit more robust.
        spectrogram_model._metrics.get_hang_time: cf.Args(threshold=0.03),
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−6 learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        # NOTE: No L2 regularization performed better based on Comet experiments in March 2020.
        # NOTE: In May 2022, we found that halfing the learning rate had a significant performance
        # improvement on the loudness and pausing of the model.
        torch.optim.AdamW: cf.Args(
            eps=10**-6,
            weight_decay=0.04,
            lr=5e-4,
            amsgrad=False,
            betas=(0.9, 0.999),
        ),
        run._models.spectrogram_model.wrapper.SpectrogramModelWrapper: cf.Args(
            max_sessions=num_sesh
        ),
    }
    cf.add(config, **kwargs)


def config_spec_model_training_from_datasets(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
):
    configs = _get_spec_model_training_configs(train_dataset, dev_dataset, debug)
    _config_spec_model_training(*configs, debug)


def config_sig_model_training_from_datasets(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool, **kwargs
):
    """Make additional configuration for signal model training."""
    ratio = _get_ratio(train_dataset, dev_dataset)

    train_batch_size = int((32 if debug else 128) / lib.distributed.get_device_count())
    train_slice_size = FRAME_HOP * 8
    dev_slice_size = train_slice_size * 4
    batch_size_ratio = 1 / 2
    dev_batch_size = int(train_batch_size * train_slice_size / dev_slice_size / 2)
    oversample = 6
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
            window=run._utils.get_window("hann", length, length // 4),
            num_mel_bins=length // 8,
        )
        for length in (256, 1024, 4096)
    ]

    real_label = True
    fake_label = False
    threshold = 0.5

    signal_model = run.train.signal_model
    _State, _worker = signal_model._worker._State, signal_model._worker

    config = {
        run.train._utils.set_run_seed: cf.Args(seed=RANDOM_SEED),
        signal_model._worker._get_data_loaders: cf.Args(
            # SOURCE (Tacotron 2):
            # We train with a batch size of 128 distributed across 32 GPUs with
            # synchronous updates, using the Adam optimizer with β1 = 0.9, β2 =
            # 0.999, eps = 10−8 and a fixed learning rate of 10−4
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
            num_workers=2,
            prefetch_factor=2 if debug else 16,
        ),
        _State._get_optimizers: cf.Args(
            optimizer=partial(torch.optim.Adam, lr=10**-4, amsgrad=False, betas=(0.9, 0.999)),
            # NOTE: We employ a small warmup because the model can be unstable
            # at the start of it's training.
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
        ),
        _State._get_signal_to_spectrogram_modules: cf.Args(kwargs=signal_to_spectrogram_params),
        _State._get_discrims: cf.Args(
            args=[(p["fft_length"], p["num_mel_bins"]) for p in signal_to_spectrogram_params]
        ),
        _State._get_discrim_optimizers: cf.Args(optimizer=partial(torch.optim.Adam, lr=10**-3)),
        _worker._run_discriminator: cf.Args(real_label=real_label, fake_label=fake_label),
        signal_model._metrics.Metrics.get_discrim_values: cf.Args(
            real_label=real_label, fake_label=fake_label, threshold=threshold
        ),
        run._models.signal_model.wrapper.SignalModelWrapper: cf.Args(
            max_sessions=_get_num_sessions(train_dataset),
        ),
    }
    cf.add(config, **kwargs)
