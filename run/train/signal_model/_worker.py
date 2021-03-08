import contextlib
import dataclasses
import logging
import math
import pathlib
import random
import typing
import warnings
from functools import partial

import torch
import torch.nn
import torch.optim
import torch.utils.data
from hparams import HParam, configurable
from third_party import get_parameter_norm
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss
from torch.nn.parallel import DistributedDataParallel
from torchnlp.utils import get_total_parameters

import lib
from lib.audio import SignalTodBMelSpectrogram
from lib.distributed import get_rank, is_master
from lib.signal_model import SignalModel, SpectrogramDiscriminator
from lib.visualize import plot_mel_spectrogram, plot_spectrogram
from run._config import (
    RANDOM_SEED,
    Cadence,
    Dataset,
    DatasetType,
    get_dataset_label,
    get_model_label,
)
from run.train import _utils, spectrogram_model
from run.train._utils import (
    CometMLExperiment,
    Context,
    DataLoader,
    Timer,
    save_checkpoint,
    set_context,
    set_epoch,
)
from run.train.signal_model._data import Batch, DataProcessor
from run.train.signal_model._metrics import Metrics, MetricsValues

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Checkpoint(_utils.Checkpoint):
    """Checkpoint used to checkpoint signal model training."""

    model: SignalModel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    ema: lib.optimizers.ExponentialMovingParameterAverage
    scheduler: torch.optim.lr_scheduler.LambdaLR
    discrims: typing.List[SpectrogramDiscriminator]
    discrim_optimizers: typing.List[torch.optim.Adam]
    spectrogram_model_checkpoint_path: pathlib.Path


@dataclasses.dataclass(frozen=True)
class _State:
    model: DistributedDataParallel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    ema: lib.optimizers.ExponentialMovingParameterAverage
    scheduler: torch.optim.lr_scheduler.LambdaLR
    signal_to_spectrogram_modules: typing.List[SignalTodBMelSpectrogram]
    discrims: typing.List[DistributedDataParallel]
    discrim_optimizers: typing.List[torch.optim.Adam]
    spectrogram_model_checkpoint: spectrogram_model._worker.Checkpoint
    spectrogram_model_checkpoint_path: pathlib.Path
    comet: CometMLExperiment
    device: torch.device
    step: torch.Tensor = torch.tensor(0, dtype=torch.long)

    def __post_init__(self):
        """ Check datastructure invariants. """
        assert len(self.discrims) == len(self.discrim_optimizers)
        assert len(self.discrims) == len(self.signal_to_spectrogram_modules)
        assert self.scheduler._step_count == self.step.item() + 1
        assert self.scheduler.last_epoch == self.step.item()
        assert self.scheduler.optimizer == self.optimizer
        assert self.ema.step == self.step.item() + 1
        ids = set(id(p) for p in self.model.parameters() if p.requires_grad)
        assert set(id(p) for p in self.ema.parameters) == ids
        assert set(id(p) for p in self.optimizer.param_groups[0]["params"]) == ids
        assert set(id(p) for p in self.clipper.parameters) == ids
        for discrim, discrim_optimizer in zip(self.discrims, self.discrim_optimizers):
            ids = set(id(p) for p in discrim.parameters() if p.requires_grad)
            assert set(id(p) for p in discrim_optimizer.param_groups[0]["params"]) == ids

    @staticmethod
    def _get_model(
        device: torch.device,
        comet: CometMLExperiment,
    ) -> SignalModel:
        model = SignalModel().to(device, non_blocking=True)
        comet.set_model_graph(str(model))
        label = get_model_label("num_parameters", Cadence.STATIC)
        comet.log_parameter(label, get_total_parameters(model))
        label = get_model_label("parameter_sum", Cadence.STATIC)
        parameter_sum = torch.stack([param.sum() for param in model.parameters()]).sum().item()
        comet.log_parameter(label, parameter_sum)
        return model

    @staticmethod
    @configurable
    def _get_optimizers(
        model: torch.nn.Module,
        optimizer: typing.Type[torch.optim.Adam] = HParam(),
        lr_multiplier_schedule: typing.Callable[[int], float] = HParam(),
    ) -> typing.Tuple[
        torch.optim.Adam,
        lib.optimizers.AdaptiveGradientNormClipper,
        lib.optimizers.ExponentialMovingParameterAverage,
        torch.optim.lr_scheduler.LambdaLR,
    ]:
        """Initialize model optimizers."""
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer_ = optimizer(params)
        clipper = lib.optimizers.AdaptiveGradientNormClipper(params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_multiplier_schedule)
        ema = lib.optimizers.ExponentialMovingParameterAverage(params)
        return optimizer_, clipper, ema, scheduler

    @staticmethod
    @configurable
    def _get_signal_to_spectrogram_modules(
        device: torch.device,
        kwargs: typing.List[typing.Dict] = HParam(),
    ) -> typing.List[SignalTodBMelSpectrogram]:
        with warnings.catch_warnings():
            message = r".*Overwriting configured argument.*"
            warnings.filterwarnings("ignore", module=r".*hparams", message=message)
            return [SignalTodBMelSpectrogram(**k).to(device, non_blocking=True) for k in kwargs]

    @staticmethod
    @configurable
    def _get_discrims(
        device: torch.device, args: typing.List[typing.Tuple[int, int]] = HParam()
    ) -> typing.List[SpectrogramDiscriminator]:
        return [SpectrogramDiscriminator(*a).to(device, non_blocking=True) for a in args]

    @staticmethod
    @configurable
    def _get_discrim_optimizers(
        discrims: typing.List[DistributedDataParallel],
        optimizer: typing.Callable[..., torch.optim.Adam] = HParam(),
    ):
        is_include = lambda p: p.requires_grad
        return [optimizer(filter(is_include, d.parameters())) for d in discrims]

    def to_checkpoint(self):
        """ Create a checkpoint to save the signal model training state. """
        return Checkpoint(
            comet_experiment_key=self.comet.get_key(),
            model=typing.cast(SignalModel, self.model.module),
            optimizer=self.optimizer,
            clipper=self.clipper,
            ema=self.ema,
            scheduler=self.scheduler,
            discrims=[typing.cast(SpectrogramDiscriminator, d.module) for d in self.discrims],
            discrim_optimizers=self.discrim_optimizers,
            spectrogram_model_checkpoint_path=self.spectrogram_model_checkpoint_path,
            step=int(self.step.item()),
        )

    @property
    def models(self) -> typing.Tuple[torch.nn.Module]:
        return tuple([self.model] + self.discrims)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """Recreate the signal model training state from a `checkpoint`.

        NOTE: `dataclasses.astuple` isn't compatible with PyTorch, learn more:
        https://github.com/pytorch/pytorch/issues/52127
        """
        discrims = [DistributedDataParallel(d, [device], device) for d in checkpoint.discrims]
        return cls(
            DistributedDataParallel(checkpoint.model, [device], device),
            checkpoint.optimizer,
            checkpoint.clipper,
            checkpoint.ema,
            checkpoint.scheduler,
            cls._get_signal_to_spectrogram_modules(device),
            discrims,
            checkpoint.discrim_optimizers,
            lib.environment.load(checkpoint.spectrogram_model_checkpoint_path),
            checkpoint.spectrogram_model_checkpoint_path,
            comet,
            device,
            torch.tensor(checkpoint.step),
        )

    @classmethod
    def make(
        cls,
        spectrogram_model_checkpoint_path: pathlib.Path,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """Initialize signal model training state."""
        distribute = partial(DistributedDataParallel, device_ids=[device], output_device=device)
        model = distribute(cls._get_model(device, comet))
        discrims = [distribute(d) for d in cls._get_discrims(device)]
        return cls(
            model,
            *cls._get_optimizers(model),
            cls._get_signal_to_spectrogram_modules(device),
            discrims,
            cls._get_discrim_optimizers(discrims),
            lib.environment.load(spectrogram_model_checkpoint_path),
            spectrogram_model_checkpoint_path,
            comet,
            device,
        )


def _worker_init_fn():
    # NOTE: Each worker needs a different random seed to generate unique data.
    info = torch.utils.data.get_worker_info()
    seed = RANDOM_SEED + get_rank() * info.num_workers + info.id
    lib.environment.set_seed(seed)
    logger.info("Worker random seed set to %d", seed)


@configurable
def _get_data_loaders(
    state: _State,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    train_batch_size: int = HParam(),
    dev_batch_size: int = HParam(),
    train_slice_size: int = HParam(),
    dev_slice_size: int = HParam(),
    train_span_bucket_size: int = HParam(),
    dev_span_bucket_size: int = HParam(),
    train_steps_per_epoch: int = HParam(),
    dev_steps_per_epoch: int = HParam(),
    num_workers: int = HParam(),
    prefetch_factor: int = HParam(),
) -> typing.Tuple[DataLoader, DataLoader]:
    """ Initialize training and development data loaders.  """
    model = state.model.module if isinstance(state.model, DistributedDataParallel) else state.model
    padding = typing.cast(int, model.padding)
    checkpoint = state.spectrogram_model_checkpoint
    processor = partial(DataProcessor, slice_padding=padding, checkpoint=checkpoint)
    train = processor(train_dataset, train_slice_size, train_batch_size, train_span_bucket_size)
    dev = processor(dev_dataset, dev_slice_size, dev_batch_size, dev_span_bucket_size)
    kwargs = dict(
        num_workers=num_workers,
        device=state.device,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn,
    )
    return (
        DataLoader(train, num_steps_per_epoch=train_steps_per_epoch, **kwargs),
        DataLoader(dev, num_steps_per_epoch=dev_steps_per_epoch, **kwargs),
    )


@configurable
def _run_discriminator(
    state: _State,
    metrics: Metrics,
    batch: Batch,
    i: int,
    fake_db_mel_spectrogram: torch.Tensor,
    fake_db_spectrogram: torch.Tensor,
    fake_spectrogram: torch.Tensor,
    real_db_mel_spectrogram: torch.Tensor,
    real_db_spectrogram: torch.Tensor,
    real_spectrogram: torch.Tensor,
    real_label: bool = HParam(),
    fake_label: bool = HParam(),
) -> typing.Tuple[torch.Tensor, typing.Callable[..., MetricsValues]]:
    """Discriminate between fake and real spectrograms.

    Learn more about this approach:
    https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    NOTE: This approach proved successful in Comet experiment f590fe3c51a04130ad65736f8aa5fd81 run
    in February 2020.

    Args:
        ...
        i: The index of the discriminator and discriminator optimizer to use.
        ...
        fake_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
        fake_db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
        fake_db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
        real_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
        real_db_spectrogram (torch.FloatTensor [batch_size, num_frames, fft_length // 2 + 1])
        real_db_mel_spectrogram (torch.FloatTensor [batch_size, num_frames, num_mel_bins])
        ...

    Returns:
        generator_loss: (torch.FloatTensor [batch_size]): The generator loss on `fake`.
        ...
    """
    discriminator = state.discrims[i]
    discriminator_optimizer = state.discrim_optimizers[i]
    batch_size = fake_spectrogram.shape[0]
    device = fake_spectrogram.device

    real_labels = torch.full((batch_size,), float(real_label), device=device)
    fake_labels = torch.full((batch_size,), float(fake_label), device=device)
    labels = torch.cat([real_labels, fake_labels])

    # NOTE: `detach` to avoid updating the generator.
    db_mel_spectrogram = torch.cat([real_db_mel_spectrogram, fake_db_mel_spectrogram.detach()])
    db_spectrogram = torch.cat([real_db_spectrogram, fake_db_spectrogram.detach()])
    spectrogram = torch.cat([real_spectrogram, fake_spectrogram.detach()])
    predictions = discriminator(spectrogram, db_spectrogram, db_mel_spectrogram)
    predictions = typing.cast(torch.Tensor, predictions)
    discriminator_loss = binary_cross_entropy_with_logits(predictions, labels, reduction="none")
    get_discrim_values = partial(
        metrics.get_discrim_values,
        fft_length=discriminator.module.fft_length,
        batch=batch,
        real_logits=predictions[:batch_size],
        fake_logits=predictions[batch_size:],
        discrim_real_losses=discriminator_loss[:batch_size],
        discrim_fake_losses=discriminator_loss[batch_size:],
    )

    if discriminator.training:
        discriminator.zero_grad(set_to_none=True)
        discriminator_loss.mean().backward()
        discriminator_optimizer.step()

    # NOTE: Use real labels instead of fake to flip the gradient for the generator.
    predictions = discriminator(fake_spectrogram, fake_db_spectrogram, fake_db_mel_spectrogram)
    predictions = typing.cast(torch.Tensor, predictions)
    generator_loss = binary_cross_entropy_with_logits(predictions, real_labels, reduction="none")
    get_discrim_values = partial(
        get_discrim_values,
        generator_logits=predictions,
        generator_losses=generator_loss,
    )
    return generator_loss, get_discrim_values


def __run_step(state: _State, timer: Timer, metrics: Metrics):
    timer.record_event(timer.MODEL_STEP)

    # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
    assert len(state.optimizer.param_groups) == 1, "Expecting only 1 group of parameters."
    params = state.optimizer.param_groups[0]["params"]
    assert all([p.grad is not None for p in params]), "`None` gradients found."

    # NOTE: Measure the "grad_norm" before `state.step_()`.
    norm_inf = get_parameter_norm(params, math.inf) if is_master() else torch.tensor(math.nan)
    assert not is_master() or torch.isfinite(norm_inf), f"Gradient was too large {norm_inf}."

    norm = state.clipper.clip()
    state.optimizer.step()
    state.ema.update()
    state.scheduler.step()
    state.step.add_(1)
    state.comet.set_step(typing.cast(int, state.step.item()))

    timer.record_event(timer.LOG_METRICS)
    norm_inf_ = float(norm_inf.item())
    metrics.log_optim_metrics(norm, norm_inf_, state.optimizer, state.clipper, cadence=Cadence.STEP)


def _run_step(
    state: _State,
    metrics: Metrics,
    batch: Batch,
    data_loader: DataLoader,
    timer: Timer,
):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    TODO: Example padding shouldn't affect the model loss, at all. For example, at the moment,
    it affects the demoninator of the loss reduction operation.

    TODO: For the target signal, the `signal_to_spec` can be computed during data processing.

    TODO: Parallelize loop with multiple independent discriminators.
    """
    timer.record_event(timer.MODEL_FORWARD)
    signal = state.model(
        spectrogram=batch.spectrogram.tensor,
        spectrogram_mask=batch.spectrogram_mask.tensor,
        pad_input=False,
    )
    signal = typing.cast(torch.Tensor, signal)

    loss = torch.tensor(0.0, device=signal.device)
    get_values_partials = []
    for i, signal_to_spectrogram_module in enumerate(state.signal_to_spectrogram_modules):
        signal_to_spec = partial(signal_to_spectrogram_module, intermediate=True)
        Specs = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        predicted_specs = typing.cast(Specs, signal_to_spec(signal))
        target_specs = typing.cast(Specs, signal_to_spec(batch.target_signal.tensor))

        l1_loss_ = l1_loss(predicted_specs[0], target_specs[0], reduction="none").mean(dim=1)
        mse_loss_ = mse_loss(predicted_specs[0], target_specs[0], reduction="none").mean(dim=1)
        generator_loss, get_discrim_values = _run_discriminator(
            state, metrics, batch, i, *predicted_specs, *target_specs
        )
        loss += l1_loss_.mean() + mse_loss_.mean() + generator_loss.mean()

        get_model_values = partial(
            metrics.get_model_values,
            fft_length=signal_to_spectrogram_module.fft_length,
            batch=batch,
            l1_losses=l1_loss_,
            mse_losses=mse_loss_,
        )
        get_values_partials.extend([get_discrim_values, get_model_values])

    if state.model.training:
        state.model.zero_grad(set_to_none=True)
        timer.record_event(timer.MODEL_BACKWARD)
        (loss / len(state.signal_to_spectrogram_modules)).backward()
        __run_step(state, timer, metrics)

    timer.record_event(timer.MEASURE_METRICS)
    values: _utils.MetricsValues = {k: v for p in get_values_partials for k, v in p().items()}
    values.update(metrics.get_dataset_values(batch))
    values.update(metrics.get_data_loader_values(data_loader))
    timer.record_event(timer.GATHER_METRICS)
    metrics.update(values)


def _log_specs(state: _State, target: torch.Tensor, predicted: torch.Tensor, **kwargs):
    """Log the various spectrograms produced by `state.signal_to_spectrogram_modules`."""
    get_dataset_label_ = partial(get_dataset_label, **kwargs)
    get_model_label_ = partial(get_model_label, **kwargs)
    for signal_to_spectrogram_module in state.signal_to_spectrogram_modules:
        signal_to_spec = partial(signal_to_spectrogram_module, intermediate=True)
        Specs = typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        target_specs = typing.cast(Specs, signal_to_spec(target))
        predicted_specs = typing.cast(Specs, signal_to_spec(predicted))
        iterator = zip((get_dataset_label_, get_model_label_), (target_specs, predicted_specs))
        for get_label, specs in iterator:
            suffix = "{fft_length}_spectrogram"
            for name, spec in zip((f"db_mel_{suffix}", f"db_{suffix}", suffix), specs):
                plot = plot_mel_spectrogram if "_mel_" in name else plot_spectrogram
                fft_length = signal_to_spectrogram_module.fft_length
                state.comet.log_figure(get_label(name, fft_length=fft_length), plot(spec))


@lib.utils.log_runtime
def _visualize_inferred(state: _State, data_loader: DataLoader, dataset_type: DatasetType):
    """Run in inference mode and visualize results."""
    if not is_master():
        return

    batch = typing.cast(Batch, next(iter(data_loader)))
    item = random.randint(0, len(batch.batch) - 1)
    length = batch.batch.predicted_spectrogram.lengths[:, item]
    spectrogram = batch.batch.predicted_spectrogram.tensor[:length, item]
    predicted = typing.cast(torch.Tensor, state.model.module(spectrogram))
    target = batch.batch.audio[item]
    state.comet.log_html_audio(
        audio={"gold_audio": target.cpu().numpy(), "predicted_audio": predicted.cpu().numpy()},
        context=state.comet.context,
        text=batch.batch.spans[item].script,
        speaker=batch.batch.spans[item].speaker,
    )
    get_label = partial(get_dataset_label, cadence=Cadence.STEP, type_=dataset_type)
    state.comet.log_figure(get_label("input_spectrogram"), plot_mel_spectrogram(spectrogram))
    _log_specs(state, target.to(state.device), predicted, cadence=Cadence.STEP, type_=dataset_type)


@lib.utils.log_runtime
def _visualize_inferred_end_to_end(
    state: _State, data_loader: DataLoader, dataset_type: DatasetType
):
    """Run spectrogram and signal model in inference mode and visualize results."""
    if not is_master():
        return

    batch = typing.cast(Batch, next(iter(data_loader)))
    item = random.randint(0, len(batch.batch) - 1)
    num_tokens = batch.batch.encoded_phonemes.lengths[:, item]
    spectrogram_model = state.spectrogram_model_checkpoint.model
    spectrogram_model = spectrogram_model.train(False).to(state.device)
    predicted_spectrogram, predicted_stop_token, predicted_alignments, _, _ = spectrogram_model(
        tokens=batch.batch.encoded_phonemes.tensor[:num_tokens, item],
        speaker=batch.batch.encoded_speaker.tensor[:, item],
        mode=lib.spectrogram_model.Mode.INFER,
    )
    predicted = typing.cast(torch.Tensor, state.model.module(predicted_spectrogram))
    target = batch.batch.audio[item]
    model_label_ = partial(get_model_label, cadence=Cadence.STEP)
    dataset_label_ = partial(get_dataset_label, cadence=Cadence.STEP, type_=dataset_type)
    num_frames = batch.batch.spectrogram.lengths[:, item]
    gold_spectrogram = batch.batch.spectrogram.tensor[:num_frames, item]
    figures = {
        dataset_label_("gold_spectrogram"): plot_mel_spectrogram(gold_spectrogram),
        model_label_("predicted_spectrogram"): plot_mel_spectrogram(predicted_spectrogram),
        model_label_("alignment"): lib.visualize.plot_alignments(predicted_alignments),
        model_label_("stop_token"): lib.visualize.plot_logits(predicted_stop_token),
    }
    state.comet.log_figures(figures)
    audio = {
        "predicted_griffin_lim_audio": lib.audio.griffin_lim(predicted_spectrogram.cpu().numpy()),
        "gold_griffin_lim_audio": lib.audio.griffin_lim(gold_spectrogram.cpu().numpy()),
        "predicted_signal_model_audio": predicted.cpu().numpy(),
        "gold_audio": target.cpu().numpy(),
    }
    span = batch.batch.spans[item]
    state.comet.log_html_audio(
        audio=audio, context=state.comet.context, text=span.script, speaker=span.speaker
    )
    _log_specs(state, target.to(state.device), predicted, cadence=Cadence.STEP, type_=dataset_type)


class _HandleBatch(typing.Protocol):
    def __call__(
        self, state: _State, metrics: Metrics, batch: Batch, data_loader: DataLoader, timer: Timer
    ) -> None:
        ...


def _run_steps(
    store: torch.distributed.TCPStore,
    state: _State,
    context: Context,
    dataset_type: DatasetType,
    data_loader: DataLoader,
    handle_batch: _HandleBatch,
):
    """Run the `handle_batch` in a loop over `data_loader` batches."""
    with set_context(context, state.comet, *state.models):
        with contextlib.nullcontext() if context == Context.TRAIN else state.ema:
            speakers = state.spectrogram_model_checkpoint.input_encoder.speaker_encoder.vocab
            metrics = Metrics(store, state.comet, speakers)
            iterator = iter(data_loader)
            while True:
                timer = Timer()
                timer.record_event(timer.LOAD_DATA)
                batch = next(iterator, None)
                if batch is None:
                    break

                handle_batch(state, metrics, batch, data_loader, timer)

                if Context.TRAIN == context:
                    metrics.log(lambda l: l[-1:], timer, type_=dataset_type, cadence=Cadence.STEP)
                    state.comet.log_metrics(timer.get_timers(cadence=Cadence.STEP))

            metrics.log(is_verbose=True, type_=dataset_type, cadence=Cadence.MULTI_STEP)


def run_worker(
    device: torch.device,
    store: torch.distributed.TCPStore,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[Checkpoint],
    checkpoints_directory: pathlib.Path,
    spectrogram_model_checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
) -> typing.NoReturn:
    """Train and evaluate the signal model in a loop.

    TODO: Support training from ground truth spectrograms.
    """
    conditions = [checkpoint is None, spectrogram_model_checkpoint is None]
    message = "Either signal model or spectrogram model checkpoint needs to be defined."
    assert any(conditions) and not all(conditions), message
    state = (
        _State.make(typing.cast(pathlib.Path, spectrogram_model_checkpoint), comet, device)
        if checkpoint is None
        else _State.from_checkpoint(checkpoint, comet, device)
    )
    train_loader, dev_loader = _get_data_loaders(state, train_dataset, dev_dataset)
    contexts: typing.List[typing.Tuple[Context, DatasetType, DataLoader, _HandleBatch]] = [
        (Context.TRAIN, DatasetType.TRAIN, train_loader, _run_step),
        (Context.EVALUATE, DatasetType.DEV, dev_loader, _run_step),
    ]
    while True:
        steps_per_epoch = train_loader.num_steps_per_epoch
        with set_epoch(comet, step=state.step.item(), steps_per_epoch=steps_per_epoch):
            [_run_steps(store, state, *args) for args in contexts]

            with state.ema:
                with set_context(Context.EVALUATE_INFERENCE, state.comet, *state.models):
                    _visualize_inferred(state, dev_loader, DatasetType.DEV)

                with set_context(Context.EVALUATE_END_TO_END, state.comet, *state.models):
                    _visualize_inferred_end_to_end(state, dev_loader, DatasetType.DEV)

            name = f"step_{state.step.item()}"
            save_checkpoint(state.to_checkpoint(), checkpoints_directory, name)
