import contextlib
import copy
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
from lib.audio import SignalTodBMelSpectrogram, Spectrograms
from lib.distributed import get_rank, get_world_size, is_master
from lib.signal_model import SignalModel, SpectrogramDiscriminator, generate_waveform
from lib.visualize import plot_mel_spectrogram, plot_spectrogram
from run._config import (
    RANDOM_SEED,
    Cadence,
    Dataset,
    DatasetType,
    configurable_,
    get_config_label,
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
    set_train_mode,
)
from run.train.signal_model._data import Batch, DataProcessor
from run.train.signal_model._metrics import Metrics, MetricsValues

logger = logging.getLogger(__name__)
torch.optim.Adam.__init__ = configurable_(torch.optim.Adam.__init__)


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

    def check_invariants(self):
        """ Check datastructure invariants. """
        assert len(self.discrims) == len(self.discrim_optimizers)
        assert self.scheduler._step_count == self.step + 1
        assert self.scheduler.last_epoch == self.step
        assert self.scheduler.optimizer == self.optimizer
        assert self.ema.step == self.step + 1
        ptrs = set(p.data_ptr() for p in self.model.parameters() if p.requires_grad)
        assert len(self.optimizer.param_groups) == 1
        assert set(p.data_ptr() for p in self.optimizer.param_groups[0]["params"]) == ptrs
        assert self.scheduler.get_last_lr() == [self.optimizer.param_groups[0]["lr"]]
        assert set(p.data_ptr() for p in self.clipper.parameters) == ptrs
        assert set(p.data_ptr() for p in self.ema.parameters) == ptrs
        for discrim, discrim_optimizer in zip(self.discrims, self.discrim_optimizers):
            ptrs = set(p.data_ptr() for p in discrim.parameters() if p.requires_grad)
            assert len(discrim_optimizer.param_groups) == 1
            assert set(p.data_ptr() for p in discrim_optimizer.param_groups[0]["params"]) == ptrs
            assert discrim.training
            assert all([p.grad is None for p in discrim.parameters()])
        assert self.ema.backup == []  # NOTE: Ensure EMA hasn't been applied.
        assert self.model.training  # NOTE: Ensure `model` is in training mode.
        # NOTE: Ensure there are no gradients.
        assert all([p.grad is None for p in self.model.parameters()])

    def __post_init__(self):
        self.check_invariants()

    def export(self) -> lib.signal_model.SignalModel:
        """Export inference ready `InputEncoder` and `SpectrogramModel` without needing additional
        context managers."""
        self.check_invariants()
        self.model.grad_enabled = None  # NOTE: For backwards compatibility
        with contextlib.ExitStack() as stack:
            stack.enter_context(set_train_mode(self.model, False, self.ema))
            model = copy.deepcopy(self.model)
            model.set_grad_enabled(False)
            model.remove_weight_norm_()
        self.check_invariants()
        return model


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
    spectrogram_model_input_encoder: spectrogram_model._worker.InputEncoder
    # NOTE: An additional underscore was added because Pylance had issues with the former naming.
    spectrogram_model_: lib.spectrogram_model.SpectrogramModel
    spectrogram_model_checkpoint_path: pathlib.Path
    comet: CometMLExperiment
    device: torch.device
    step: torch.Tensor = torch.tensor(0, dtype=torch.long)

    def __post_init__(self):
        """ Check datastructure invariants. """
        ptrs = set(p.data_ptr() for p in self.model.parameters() if p.requires_grad)
        assert set(p.data_ptr() for p in self.model.module.parameters() if p.requires_grad) == ptrs
        assert self.model.training
        for discrim in self.discrims:
            ptrs = set(p.data_ptr() for p in discrim.parameters() if p.requires_grad)
            assert set(p.data_ptr() for p in discrim.module.parameters() if p.requires_grad) == ptrs
            assert discrim.training
        assert len(self.discrims) == len(self.signal_to_spectrogram_modules)
        self.to_checkpoint().check_invariants()

    @staticmethod
    def _get_model(
        input_encoder: spectrogram_model._data.InputEncoder,
        device: torch.device,
        comet: CometMLExperiment,
    ) -> SignalModel:
        num_speakers = input_encoder.speaker_encoder.vocab_size
        num_sessions = input_encoder.session_encoder.vocab_size
        model = SignalModel(num_speakers, num_sessions).to(device, non_blocking=True)
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
        optimizer: typing.Callable[..., torch.optim.Adam] = HParam(),
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
        """
        TODO: How can we prevent this warning?
        - Can we move `configurable` functions outside of `lib`?
        - Can we make `hparams` less strict? Especially, for default values, like this one.
        - Can we, instead of configuring `SignalTodBMelSpectrogram`, configure a different
          function?
        """
        with warnings.catch_warnings():
            message = r".*Overwriting configured argument.*"
            warnings.filterwarnings("ignore", module=r".*hparams", message=message)
            return [SignalTodBMelSpectrogram(**k).to(device, non_blocking=True) for k in kwargs]

    @staticmethod
    @configurable
    def _get_discrims(
        input_encoder: spectrogram_model._data.InputEncoder,
        device: torch.device,
        args: typing.List[typing.Tuple[int, int]] = HParam(),
    ) -> typing.List[SpectrogramDiscriminator]:
        num_speakers = input_encoder.speaker_encoder.vocab_size
        num_sessions = input_encoder.session_encoder.vocab_size
        return [
            SpectrogramDiscriminator(*a, num_speakers, num_sessions).to(device, non_blocking=True)
            for a in args
        ]

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
    def models(self) -> typing.Tuple[torch.nn.Module, ...]:
        return tuple([self.model] + self.discrims)

    @staticmethod
    def load_spectrogram_model(
        comet: CometMLExperiment, spectrogram_model_checkpoint_path: pathlib.Path
    ) -> typing.Tuple[
        spectrogram_model._worker.InputEncoder, lib.spectrogram_model.SpectrogramModel
    ]:
        checkpoint = lib.environment.load(spectrogram_model_checkpoint_path)
        comet.log_other(
            get_config_label("spectrogram_model_experiment_key"),
            checkpoint.comet_experiment_key,
        )
        return checkpoint.export()

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
        checkpoint.model.grad_enabled = None  # NOTE: For backwards compatibility
        discrims = [DistributedDataParallel(d, [device], device) for d in checkpoint.discrims]
        input_encoder, spectrogram_model = cls.load_spectrogram_model(
            comet, checkpoint.spectrogram_model_checkpoint_path
        )
        return cls(
            DistributedDataParallel(checkpoint.model, [device], device),
            checkpoint.optimizer,
            checkpoint.clipper,
            checkpoint.ema,
            checkpoint.scheduler,
            cls._get_signal_to_spectrogram_modules(device),
            discrims,
            checkpoint.discrim_optimizers,
            input_encoder,
            spectrogram_model,
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
        input_encoder, spectrogram_model = cls.load_spectrogram_model(
            comet, spectrogram_model_checkpoint_path
        )
        model = distribute(cls._get_model(input_encoder, device, comet))
        discrims = [distribute(d) for d in cls._get_discrims(input_encoder, device)]
        return cls(
            model,
            *cls._get_optimizers(model),
            cls._get_signal_to_spectrogram_modules(device),
            discrims,
            cls._get_discrim_optimizers(discrims),
            input_encoder,
            spectrogram_model,
            spectrogram_model_checkpoint_path,
            comet,
            device,
        )


def _worker_init_fn(
    step: int,
    rank: int,
    world_size: int,
):
    # NOTE: Each worker needs a different random seed to generate unique data.
    info = torch.utils.data.get_worker_info()
    seed = RANDOM_SEED
    seed += world_size * info.num_workers * step
    seed += rank * info.num_workers
    seed += info.id
    lib.environment.set_seed(seed)


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
) -> typing.Tuple[DataLoader[Batch], DataLoader[Batch]]:
    """ Initialize training and development data loaders.  """
    model = state.model.module if isinstance(state.model, DistributedDataParallel) else state.model
    padding = typing.cast(int, model.padding)
    processor = partial(
        DataProcessor,
        slice_padding=padding,
        spectrogram_model_input_encoder=state.spectrogram_model_input_encoder,
        spectrogram_model=state.spectrogram_model_,
    )
    train = processor(train_dataset, train_slice_size, train_batch_size, train_span_bucket_size)
    dev = processor(dev_dataset, dev_slice_size, dev_batch_size, dev_span_bucket_size)
    kwargs = dict(
        num_workers=num_workers,
        device=state.device,
        prefetch_factor=prefetch_factor,
        worker_init_fn=partial(
            _worker_init_fn,
            step=int(state.step.item()),
            rank=get_rank(),
            world_size=get_world_size(),
        ),
    )
    return (
        DataLoader(train, num_steps_per_epoch=train_steps_per_epoch, **kwargs),
        DataLoader(dev, num_steps_per_epoch=dev_steps_per_epoch, **kwargs),
    )


class _HandleBatchArgs(typing.NamedTuple):
    state: _State
    data_loader: DataLoader
    context: Context
    dataset_type: DatasetType
    metrics: Metrics
    timer: Timer
    batch: Batch
    cadence: Cadence = Cadence.STEP


@configurable
def _run_discriminator(
    args: _HandleBatchArgs,
    i: int,
    fake_specs: Spectrograms,
    real_specs: Spectrograms,
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

    Returns:
        generator_loss: (torch.FloatTensor [batch_size]): The generator loss on `fake`.
        ...
    """
    args.timer.record_event(args.timer.MODEL_FORWARD)
    discriminator = args.state.discrims[i]
    discriminator_optimizer = args.state.discrim_optimizers[i]
    batch_size = fake_specs.amp.shape[0]
    device = fake_specs.amp.device

    real_labels = torch.full((batch_size,), float(real_label), device=device)
    fake_labels = torch.full((batch_size,), float(fake_label), device=device)
    labels = torch.cat([real_labels, fake_labels])

    # NOTE: `detach` to avoid updating the generator.
    db_mel = torch.cat([real_specs.db_mel, fake_specs.db_mel.detach()])
    db = torch.cat([real_specs.db, fake_specs.db.detach()])
    amp = torch.cat([real_specs.amp, fake_specs.amp.detach()])
    speaker = torch.cat([args.batch.speaker, args.batch.speaker])
    session = torch.cat([args.batch.session, args.batch.session])
    predictions = discriminator(amp, db, db_mel, speaker, session)
    predictions = typing.cast(torch.Tensor, predictions)
    discriminator_loss = binary_cross_entropy_with_logits(predictions, labels, reduction="none")
    get_discrim_values = partial(
        args.metrics.get_discrim_values,
        fft_length=discriminator.module.fft_length,
        batch=args.batch,
        real_logits=predictions[:batch_size],
        fake_logits=predictions[batch_size:],
        discrim_real_losses=discriminator_loss[:batch_size],
        discrim_fake_losses=discriminator_loss[batch_size:],
    )

    if discriminator.training:
        args.timer.record_event(args.timer.MODEL_BACKWARD)
        assert all([p.grad is None for p in discriminator_optimizer.param_groups[0]["params"]])
        discriminator_loss.mean().backward()
        discriminator_optimizer.step()
        discriminator.zero_grad(set_to_none=True)

    args.timer.record_event(args.timer.MODEL_FORWARD)
    # NOTE: Use real labels instead of fake to flip the gradient for the generator.
    predictions = discriminator(
        fake_specs.amp,
        fake_specs.db,
        fake_specs.db_mel,
        args.batch.speaker,
        args.batch.session,
    )
    predictions = typing.cast(torch.Tensor, predictions)
    generator_loss = binary_cross_entropy_with_logits(predictions, real_labels, reduction="none")
    get_discrim_values = partial(
        get_discrim_values,
        generator_logits=predictions,
        generator_losses=generator_loss,
    )
    return generator_loss, get_discrim_values


def _random_volume(
    signal: torch.Tensor, target_signal: torch.Tensor
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Randomly select a volume level for `target_signal` and `signal`."""
    peek = torch.max(torch.max(signal.abs()), torch.max(target_signal.abs()))
    volume = torch.zeros(1, device=signal.device).uniform_(0.0, 1.0 / float(peek.item()))
    return (signal * volume, target_signal * volume)


def _run_step(args: _HandleBatchArgs):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    TODO: Example padding shouldn't affect the model loss, at all. For example, at the moment,
    it affects the demoninator of the loss reduction operation.

    TODO: For the target signal, the `signal_to_spec` can be computed during data processing.

    TODO: Parallelize loop with multiple independent discriminators. Or, we could batch together
    the multiple resolutions via padding or proportional frame hop. And, we'd could use grouped
    convolutions, in order, to process the spectrograms seperately.
    """
    args.timer.record_event(args.timer.MODEL_FORWARD)
    signal = args.state.model(
        spectrogram=args.batch.spectrogram.tensor,
        speaker=args.batch.speaker,
        session=args.batch.session,
        spectrogram_mask=args.batch.spectrogram_mask.tensor,
        pad_input=False,
    )
    signal = typing.cast(torch.Tensor, signal)
    signal, target_signal = _random_volume(signal, args.batch.target_signal.tensor)

    loss = torch.tensor(0.0, device=signal.device)
    get_values = []
    for i, signal_to_spectrogram in enumerate(args.state.signal_to_spectrogram_modules):
        pred_specs = signal_to_spectrogram(signal, intermediate=True)
        gold_specs = signal_to_spectrogram(target_signal, intermediate=True)
        generator_loss, get_discrim_values = _run_discriminator(args, i, pred_specs, gold_specs)
        l1_loss_ = l1_loss(pred_specs.db_mel, gold_specs.db_mel, reduction="none").mean(dim=1)
        mse_loss_ = mse_loss(pred_specs.db_mel, gold_specs.db_mel, reduction="none").mean(dim=1)
        loss += l1_loss_.mean() + mse_loss_.mean() + generator_loss.mean()

        fft_length = signal_to_spectrogram.fft_length
        get_model_values = args.metrics.get_model_values
        get_values.append(partial(get_model_values, fft_length, args.batch, l1_loss_, mse_loss_))
        get_values.append(get_discrim_values)

    if args.state.model.training:
        args.timer.record_event(args.timer.MODEL_BACKWARD)
        params = args.state.optimizer.param_groups[0]["params"]
        discrim_optimizers = args.state.discrim_optimizers
        discrim_params = [p for o in discrim_optimizers for p in o.param_groups[0]["params"]]
        assert all([p.grad is None for s in (params, discrim_params) for p in s])
        (loss / len(args.state.signal_to_spectrogram_modules)).backward()

        args.timer.record_event(args.timer.MODEL_STEP)
        # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
        assert all([p.grad is not None for p in params]), "`None` gradients found."
        # NOTE: Measure the "grad_norm" before `state.step_()`.
        norm_inf = get_parameter_norm(params, math.inf) if is_master() else torch.tensor(math.nan)
        assert not is_master() or torch.isfinite(norm_inf), f"Gradient was too large {norm_inf}."

        norm = args.state.clipper.clip()
        args.state.optimizer.step()
        args.state.ema.update()
        args.state.scheduler.step()
        args.state.step.add_(1)
        args.state.comet.set_step(typing.cast(int, args.state.step.item()))
        args.state.model.zero_grad(set_to_none=True)
        [discrim.zero_grad(set_to_none=True) for discrim in args.state.discrims]

        args.timer.record_event(args.timer.LOG_METRICS)
        norm_inf_ = float(norm_inf.item())
        args.metrics.log_optim_metrics(
            norm, norm_inf_, args.state.optimizer, args.state.clipper, cadence=args.cadence
        )

    args.timer.record_event(args.timer.MEASURE_METRICS)
    values: _utils.MetricsValues = {}
    for func in get_values:
        for key, value in func().items():
            values[key] = values[key] + value if key in values else value
    values.update(args.metrics.get_dataset_values(args.batch))
    values.update(args.metrics.get_data_loader_values(args.data_loader))
    args.timer.record_event(args.timer.GATHER_METRICS)
    args.metrics.update(values)


def _log_specs(state: _State, gold: torch.Tensor, pred: torch.Tensor, **kwargs):
    """Log the various spectrograms produced by `state.signal_to_spectrogram_modules`."""
    get_labels = (partial(get_dataset_label, **kwargs), partial(get_model_label, **kwargs))
    signals = (gold, pred)
    for signal_to_spectrogram in state.signal_to_spectrogram_modules:
        for get_label, signal in zip(get_labels, signals):
            for key, spec in signal_to_spectrogram(signal, intermediate=True)._asdict().items():
                fft_length = signal_to_spectrogram.fft_length
                label = get_label(f"{key}_{fft_length}_spectrogram", fft_length=fft_length)
                plot = plot_mel_spectrogram if "_mel" in key else plot_spectrogram
                state.comet.log_figure(label, plot(spec))


@lib.utils.log_runtime
def _visualize_inferred(
    state: _State, data_loader: DataLoader, dataset_type: DatasetType, split_size: int = 32
):
    """Run in inference mode and visualize results."""
    if not is_master():
        return

    batch = typing.cast(Batch, next(iter(data_loader)))
    item = random.randint(0, len(batch.batch) - 1)
    length = batch.batch.predicted_spectrogram.lengths[:, item]
    spectrogram = batch.batch.predicted_spectrogram.tensor[:length, item].to(state.device)
    speaker = batch.batch.encoded_speaker.tensor[:, item].to(state.device)
    session = batch.batch.encoded_session.tensor[:, item].to(state.device)
    splits = spectrogram.split(split_size)
    predicted = generate_waveform(state.model.module, splits, speaker, session)
    predicted = list(predicted)
    predicted = typing.cast(torch.Tensor, torch.cat(predicted, dim=-1))
    target = batch.batch.audio[item]
    state.comet.log_html_audio(
        audio={"gold_audio": target.numpy(), "predicted_audio": predicted.cpu().numpy()},
        context=state.comet.context,
        text=batch.batch.spans[item].script,
        speaker=batch.batch.spans[item].speaker,
    )
    get_label = partial(get_dataset_label, cadence=Cadence.STEP, type_=dataset_type)
    state.comet.log_figure(get_label("input_spectrogram"), plot_mel_spectrogram(spectrogram))
    _log_specs(state, target.to(state.device), predicted, cadence=Cadence.STEP, type_=dataset_type)


@lib.utils.log_runtime
def _visualize_inferred_end_to_end(
    state: _State, data_loader: DataLoader, dataset_type: DatasetType, split_size: int = 32
):
    """Run spectrogram and signal model in inference mode and visualize results."""
    if not is_master():
        return

    batch = typing.cast(Batch, next(iter(data_loader)))
    item = random.randint(0, len(batch.batch) - 1)
    num_tokens = batch.batch.encoded_phonemes.lengths[:, item]
    params = lib.spectrogram_model.Params(
        tokens=batch.batch.encoded_phonemes.tensor[:num_tokens, item],
        speaker=batch.batch.encoded_speaker.tensor[:, item],
        session=batch.batch.encoded_session.tensor[:, item],
    )
    # NOTE: The `spectrogram_model` runs on CPU to conserve GPU memory.
    preds = state.spectrogram_model_(params=params, mode=lib.spectrogram_model.Mode.INFER)
    preds = typing.cast(lib.spectrogram_model.Infer, preds)
    splits = preds.frames.to(state.device).split(split_size)
    speaker = batch.batch.encoded_speaker.tensor[:, item].to(state.device)
    session = batch.batch.encoded_session.tensor[:, item].to(state.device)
    predicted = list(generate_waveform(state.model.module, splits, speaker, session))
    predicted = typing.cast(torch.Tensor, torch.cat(predicted, dim=-1))
    target = batch.batch.audio[item]
    model_label_ = partial(get_model_label, cadence=Cadence.STEP)
    dataset_label_ = partial(get_dataset_label, cadence=Cadence.STEP, type_=dataset_type)
    num_frames = batch.batch.spectrogram.lengths[:, item]
    gold_spectrogram = batch.batch.spectrogram.tensor[:num_frames, item]
    figures = {
        dataset_label_("gold_spectrogram"): plot_mel_spectrogram(gold_spectrogram),
        model_label_("predicted_spectrogram"): plot_mel_spectrogram(preds.frames),
        model_label_("alignment"): lib.visualize.plot_alignments(preds.alignments),
        model_label_("stop_token"): lib.visualize.plot_logits(preds.stop_tokens),
    }
    state.comet.log_figures(figures)
    audio = {
        "predicted_griffin_lim_audio": lib.audio.griffin_lim(preds.frames.numpy()),
        "gold_griffin_lim_audio": lib.audio.griffin_lim(gold_spectrogram.numpy()),
        "predicted_signal_model_audio": predicted.cpu().numpy(),
        "gold_audio": target.numpy(),
    }
    span = batch.batch.spans[item]
    state.comet.log_html_audio(
        audio=audio, context=state.comet.context, text=span.script, speaker=span.speaker
    )
    _log_specs(state, target.to(state.device), predicted, cadence=Cadence.STEP, type_=dataset_type)


_HandleBatch = typing.Callable[[_HandleBatchArgs], None]


@lib.utils.log_runtime
def _run_steps(
    state: _State,
    context: Context,
    dataset_type: DatasetType,
    data_loader: DataLoader,
    handle_batch: _HandleBatch,
    **kwargs,
):
    """Run the `handle_batch` in a loop over `data_loader` batches."""
    make_args = partial(_HandleBatchArgs, state, data_loader, context, dataset_type)
    with contextlib.ExitStack() as stack:
        stack.enter_context(set_context(context, state.comet, *state.models, ema=state.ema))
        stack.enter_context(set_epoch(state.comet, step=state.step.item(), **kwargs))

        metrics = Metrics(state.comet, state.spectrogram_model_input_encoder.speaker_encoder.vocab)
        timer = Timer().record_event(Timer.LOAD_DATA)
        iterator = iter(data_loader)
        while True:
            batch = next(iterator, None)
            if batch is None:
                break

            handle_batch(make_args(metrics, timer, batch))

            if Context.TRAIN == context:
                metrics.log(lambda l: l[-1:], timer, type_=dataset_type, cadence=Cadence.STEP)
                state.comet.log_metrics(timer.get_timers(cadence=Cadence.STEP))

            timer = Timer().record_event(Timer.LOAD_DATA)

        metrics.log(is_verbose=True, type_=dataset_type, cadence=Cadence.MULTI_STEP)


def run_worker(
    device: torch.device,
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
    save_checkpoint(state.to_checkpoint(), checkpoints_directory, f"step_{state.step.item()}")
    while True:
        steps_per_epoch = train_loader.num_steps_per_epoch
        [_run_steps(state, *args, steps_per_epoch=steps_per_epoch) for args in contexts]

        with set_context(Context.EVALUATE_INFERENCE, state.comet, *state.models, ema=state.ema):
            _visualize_inferred(state, dev_loader, DatasetType.DEV)

        with set_context(Context.EVALUATE_END_TO_END, state.comet, *state.models, ema=state.ema):
            _visualize_inferred_end_to_end(state, dev_loader, DatasetType.DEV)

        save_checkpoint(state.to_checkpoint(), checkpoints_directory, f"step_{state.step.item()}")
