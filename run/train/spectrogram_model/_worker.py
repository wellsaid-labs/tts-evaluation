import atexit
import collections
import contextlib
import copy
import dataclasses
import logging
import math
import pathlib
import random
import types
import typing
from functools import partial

import config as cf
import torch
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from third_party import get_parameter_norm
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchnlp.utils import get_total_parameters

import lib
from lib.audio import griffin_lim
from lib.distributed import is_master
from lib.utils import log_runtime
from lib.visualize import plot_alignments, plot_logits, plot_mel_spectrogram
from run._config import Cadence, DatasetType, get_dataset_label, get_model_label
from run._models.spectrogram_model import Mode, Preds, SpectrogramModel
from run._utils import Dataset, SpanGeneratorGetWeight
from run.data._loader.structures import Speaker
from run.train import _utils
from run.train._utils import (
    CometMLExperiment,
    Context,
    DataLoader,
    Timer,
    save_checkpoint,
    set_context,
    set_epoch,
    set_run_seed,
    set_train_mode,
)
from run.train.spectrogram_model._data import Batch, DataProcessor
from run.train.spectrogram_model._metrics import (
    Metrics,
    MetricsKey,
    MetricsValues,
    get_alignment_norm,
    get_average_db_rms_level,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Checkpoint(_utils.Checkpoint):
    """Checkpoint used to checkpoint spectrogram model training."""

    model: SpectrogramModel
    optimizer: torch.optim.AdamW
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    ema: lib.optimizers.ExponentialMovingParameterAverage
    scheduler: torch.optim.lr_scheduler.LambdaLR

    def check_invariants(self):
        """Check datastructure invariants."""
        assert self.scheduler._step_count == self.step + 1  # type: ignore
        assert self.scheduler.last_epoch == self.step  # type: ignore
        assert self.scheduler.optimizer == self.optimizer  # type: ignore
        assert self.ema.step == self.step + 1
        ptrs = set(p.data_ptr() for p in self.model.parameters() if p.requires_grad)
        assert len(self.optimizer.param_groups) == 2
        assert set(p.data_ptr() for g in self.optimizer.param_groups for p in g["params"]) == ptrs
        assert set(self.scheduler.get_last_lr()) == set(
            g["lr"] for g in self.optimizer.param_groups
        )
        assert set(p.data_ptr() for p in self.clipper.parameters) == ptrs
        assert set(p.data_ptr() for p in self.ema.parameters) == ptrs
        assert self.ema.backup == []  # NOTE: Ensure EMA hasn't been applied.
        assert self.model.training  # NOTE: Ensure `model` is in training mode
        # NOTE: Ensure there are no gradients.
        assert all([p.grad is None for p in self.model.parameters()])

    def __post_init__(self):
        self.check_invariants()

    def export(self) -> SpectrogramModel:
        """Export inference ready `SpectrogramModel` without needing additional context managers."""
        logger.info("Exporting spectrogram model...")
        self.check_invariants()
        model = None
        with contextlib.ExitStack() as stack:
            stack.enter_context(set_train_mode(self.model, False, self.ema))
            model = copy.deepcopy(self.model)
            model.set_grad_enabled(False)
            model.allow_unk_on_eval(False)
        self.check_invariants()
        assert model is not None
        return model


ExcludeFromDecay = typing.Callable[[str, torch.nn.parameter.Parameter, torch.nn.Module], bool]


@dataclasses.dataclass(frozen=True)
class _State:
    model: torch.nn.parallel.DistributedDataParallel
    optimizer: torch.optim.AdamW
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    ema: lib.optimizers.ExponentialMovingParameterAverage
    scheduler: torch.optim.lr_scheduler.LambdaLR
    comet: CometMLExperiment
    device: torch.device
    step: torch.Tensor = torch.tensor(0, dtype=torch.long)

    def __post_init__(self):
        """Check datastructure invariants."""
        ptrs = set(p.data_ptr() for p in self.model.parameters() if p.requires_grad)
        assert set(p.data_ptr() for p in self.model.module.parameters() if p.requires_grad) == ptrs
        assert self.model.training
        self.to_checkpoint().check_invariants()

    @staticmethod
    def _get_model(comet: CometMLExperiment, device: torch.device) -> SpectrogramModel:
        """Initialize a model onto `device`."""
        model = cf.partial(SpectrogramModel)().to(device, non_blocking=True)
        comet.set_model_graph(str(model))
        label = get_model_label("num_parameters", Cadence.STATIC)
        comet.log_parameter(label, get_total_parameters(model))
        label = get_model_label("parameter_sum", Cadence.STATIC)
        parameter_sum = torch.stack([param.sum() for param in model.parameters()]).sum().item()
        comet.log_parameter(label, parameter_sum)
        return model

    @staticmethod
    def _make_optimizer_groups(
        model: torch.nn.Module, exclude_from_decay: ExcludeFromDecay
    ) -> typing.List[typing.Dict[str, typing.Any]]:
        """Create optimizer groups with optimizer options per group.

        Args:
            ...
            exclude_from_decay: Given the paramter name, object, and parent module this returns
                if the parameter should have weight decay.
        """
        param_to_module = {p: m for m in model.modules() for p in m.parameters(recurse=False)}
        named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        exclude: typing.Callable[[str, torch.nn.parameter.Parameter], bool]
        exclude = lambda n, p: exclude_from_decay(n, p, param_to_module[p])
        no_decay_names, no_decay_params = tuple(zip(*[p for p in named_params if exclude(*p)]))
        decay_names, decay_params = tuple(zip(*[p for p in named_params if not exclude(*p)]))
        logger.info("Parameters excluded from weight decay: %s", no_decay_names)
        logger.info("Parameters with weight decay: %s", decay_names)
        return [{"params": no_decay_params, "weight_decay": 0.0}, {"params": decay_params}]

    @staticmethod
    def _get_optimizers(
        model: torch.nn.Module,
        optimizer: typing.Type[torch.optim.AdamW],
        lr_multiplier_schedule: typing.Callable[[int], float],
        exclude_from_decay: ExcludeFromDecay,
    ) -> typing.Tuple[
        torch.optim.AdamW,
        lib.optimizers.AdaptiveGradientNormClipper,
        lib.optimizers.ExponentialMovingParameterAverage,
        torch.optim.lr_scheduler.LambdaLR,
    ]:
        """Initialize model optimizers.

        NOTE: These optimizers cannot be moved easily between devices; therefore, the model weights
        should already be on the appropriate device. Learn more:
        https://github.com/pytorch/pytorch/issues/2830
        """
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optim_groups = _State._make_optimizer_groups(model, exclude_from_decay)
        optimizer_ = cf.partial(optimizer)(optim_groups)
        clipper = cf.partial(lib.optimizers.AdaptiveGradientNormClipper)(params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_multiplier_schedule)
        ema = cf.partial(lib.optimizers.ExponentialMovingParameterAverage)(params)
        return optimizer_, clipper, ema, scheduler

    def to_checkpoint(self):
        """Create a checkpoint to save the spectrogram training state."""
        return Checkpoint(
            comet_experiment_key=self.comet.get_key(),
            model=typing.cast(SpectrogramModel, self.model.module),
            optimizer=self.optimizer,
            clipper=self.clipper,
            ema=self.ema,
            scheduler=self.scheduler,
            step=int(self.step.item()),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Checkpoint,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """Recreate the spectrogram training state from a `checkpoint`.

        NOTE: `dataclasses.astuple` isn't compatible with PyTorch, learn more:
        https://github.com/pytorch/pytorch/issues/52127
        NOTE: Learn more about `DistributedDataParallel` here:
        https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
        """
        return cls(
            torch.nn.parallel.DistributedDataParallel(checkpoint.model, [device], device),
            checkpoint.optimizer,
            checkpoint.clipper,
            checkpoint.ema,
            checkpoint.scheduler,
            comet,
            device,
            torch.tensor(checkpoint.step),
        )

    @classmethod
    def from_scratch(cls, comet: CometMLExperiment, device: torch.device):
        """Create new spectrogram training state."""
        model = cls._get_model(comet, device)
        # NOTE: Even if `_get_model` is initialized differently in each process, the parameters
        # will be synchronized. Learn more:
        # https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564/2
        model = torch.nn.parallel.DistributedDataParallel(model, [device], device)
        return cls(model, *cf.partial(cls._get_optimizers)(model), comet, device)


def _worker_init_fn():
    # NOTE: Each worker needs the same random seed to be in-sync.
    set_run_seed(**cf.get())


def _get_data_loaders(
    state: _State,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    train_batch_size: int,
    dev_batch_size: int,
    train_steps_per_epoch: int,
    dev_steps_per_epoch: int,
    train_get_weight: SpanGeneratorGetWeight,
    dev_get_weight: SpanGeneratorGetWeight,
    num_workers: int,
    prefetch_factor: int,
) -> typing.Tuple[DataLoader[Batch], DataLoader[Batch]]:
    """Initialize training and development data loaders."""
    step = int(state.step.item())
    train = DataProcessor(train_dataset, train_batch_size, step, get_weight=train_get_weight)
    dev = DataProcessor(dev_dataset, dev_batch_size, step, get_weight=dev_get_weight)
    kwargs: typing.Dict[str, typing.Any] = dict(
        num_workers=num_workers,
        device=state.device,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init_fn,
    )
    train = typing.cast(torch.utils.data.Dataset, train)
    dev = typing.cast(torch.utils.data.Dataset, dev)
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
    visualize: bool
    cadence: Cadence = Cadence.STEP


def _visualize_source_vs_target(args: _HandleBatchArgs, preds: Preds):
    """Visualize predictions as compared to the original `batch`.

    TODO: Add `pick` so that we can find examples which perform poorly in training, and hence are
    probably bad examples. Also, this should log the audio, for analysis.
    """
    if not is_master():
        return

    item = random.randint(0, len(args.batch) - 1)
    spectrogram_length = int(args.batch.spectrogram.lengths[0, item].item())
    text_length = int(preds.num_tokens[item].item())

    # predicted_spectrogram, gold_spectrogram [num_frames, frame_channels]
    predicted_spectrogram = preds.frames[:spectrogram_length, item]
    gold_spectrogram = args.batch.spectrogram.tensor[:spectrogram_length, item]

    predicted_delta = abs(gold_spectrogram - predicted_spectrogram)
    predicted_alignments = preds.alignments[:spectrogram_length, item, :text_length]
    predicted_stop_token = preds.stop_tokens[:spectrogram_length, item]
    model = partial(get_model_label, cadence=args.cadence)
    dataset = partial(get_dataset_label, cadence=args.cadence, type_=args.dataset_type)
    figures = {
        model("spectrogram_delta"): plot_mel_spectrogram(predicted_delta, **cf.get()),
        model("predicted_spectrogram"): plot_mel_spectrogram(predicted_spectrogram, **cf.get()),
        model("alignment"): plot_alignments(predicted_alignments),
        model("stop_token"): plot_logits(predicted_stop_token),
        dataset("gold_spectrogram"): plot_mel_spectrogram(gold_spectrogram, **cf.get()),
    }
    args.state.comet.log_figures(figures)


def _run_step(
    args: _HandleBatchArgs,
    spectrogram_loss_scalar: float,
    average_spectrogram_length: float,
    stop_token_loss_multiplier: typing.Callable[[int], float],
):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    TODO: Run the model with a dummy batch that's larger than the largest sequence length, learn
    more:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
    TODO: Maybe enable the CUDNN auto tuner:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    TODO: Use multithreading to overlap measure, gather, and reduction metrics operations with
    the `backwards` model pass. The metrics overhead takes 21 milliseconds (3% of runtime), on
    average. Learn more:
    https://github.com/NVIDIA/nccl/issues/338
    https://github.com/horovod/horovod/issues/665
    https://discuss.pytorch.org/t/how-to-overlap-h2d-and-training/93635/5
    https://github.com/pytorch/pytorch/issues/23729#issuecomment-518616242
    TODO: Log spectrogram loss per speaker, in order to monitor overfitting.

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        average_spectrogram_length: The training dataset average spectrogram length. It is used
            to normalize the loss magnitude.
        stop_token_loss_multiplier: A callable to determine the stop token multiplier based on the
            step.
    """
    args.timer.record_event(args.timer.MODEL_FORWARD)
    preds = args.state.model(
        inputs=args.batch.inputs,
        target_frames=args.batch.spectrogram.tensor,
        target_mask=args.batch.spectrogram_mask.tensor,
        mode=Mode.FORWARD,
    )
    preds = typing.cast(Preds, preds)

    # SOURCE: Tacotron 2
    # We minimize the summed mean squared error (MSE) from before and after the post-net to aid
    # convergence.
    spectrogram_loss = mse_loss(preds.frames, args.batch.spectrogram.tensor, reduction="none")
    spectrogram_loss *= args.batch.spectrogram_mask.tensor.unsqueeze(2)

    # SOURCE (Tacotron 2 Author):
    # The author confirmed they used BCE loss in Google Chat.
    target = args.batch.stop_token.tensor
    stop_token_loss = binary_cross_entropy_with_logits(preds.stop_tokens, target, reduction="none")
    stop_token_loss *= args.batch.spectrogram_mask.tensor

    if args.state.model.training:
        # NOTE: We sum over the `num_frames` dimension to ensure that we don't bias based on
        # `num_frames`. For example, a larger `num_frames` means that the denominator is larger;
        # therefore, the loss value for each element is smaller.
        # NOTE: We average accross `batch_size`, `num_frames` and `frame_channels` so that the loss
        # magnitude is invariant to those variables.

        # spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
        spectrogram_loss_ = (spectrogram_loss.sum(dim=0) / average_spectrogram_length).mean()
        spectrogram_loss_ *= spectrogram_loss_scalar

        # stop_token_loss [num_frames, batch_size] → [1]
        multiplier = stop_token_loss_multiplier(int(args.state.step.item()))
        delim = average_spectrogram_length
        stop_token_loss_ = ((stop_token_loss.sum(dim=0) * multiplier) / delim).mean()

        args.timer.record_event(args.timer.MODEL_BACKWARD)
        params = [p for g in args.state.optimizer.param_groups for p in g["params"]]
        assert all([p.grad is None for p in params])
        (spectrogram_loss_ + stop_token_loss_).backward()

        args.timer.record_event(args.timer.MODEL_STEP)
        # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
        assert all([p.grad is not None for p in params]), "`None` gradients found."
        # NOTE: Measure the "grad_norm" before `clipper.clip()`.
        norm_inf = get_parameter_norm(params, math.inf) if is_master() else torch.tensor(math.nan)
        assert not is_master() or torch.isfinite(norm_inf), f"Gradient was too large {norm_inf}."
        norm = args.state.clipper.clip()
        args.state.optimizer.step()
        args.state.ema.update()
        args.state.scheduler.step()
        args.timer.record_event(args.timer.LOG_METRICS)
        args.state.step.add_(1)
        args.state.comet.set_step(typing.cast(int, args.state.step.item()))
        args.state.model.zero_grad(set_to_none=True)

        norm_inf_ = float(norm_inf.item())
        args.metrics.log_optim_metrics(
            norm, norm_inf_, args.state.optimizer, args.state.clipper, cadence=args.cadence
        )

    if args.visualize:
        args.timer.record_event(args.timer.VISUALIZE_PREDICTIONS)
        _visualize_source_vs_target(args, preds)

    args.timer.record_event(args.timer.MEASURE_METRICS)
    model = typing.cast(SpectrogramModel, args.state.model.module)
    values: MetricsValues = {
        **args.metrics.get_dataset_values(args.batch, model, preds),
        **args.metrics.get_alignment_values(args.batch, model, preds),
        **args.metrics.get_loudness_values(args.batch, model, preds),
        **args.metrics.get_data_loader_values(args.data_loader),
        **args.metrics.get_stop_token_values(args.batch, model, preds),
        MetricsKey(args.metrics.SPECTROGRAM_LOSS_SUM): float(spectrogram_loss.sum().item()),
        MetricsKey(args.metrics.STOP_TOKEN_LOSS_SUM): float(stop_token_loss.sum().item()),
    }
    args.timer.record_event(args.timer.GATHER_METRICS)
    args.metrics.update(values)


def _min_alignment_norm(_: _HandleBatchArgs, preds: Preds) -> int:
    """Get the index of the prediction that has the smallest alignment norm."""
    return int(torch.argmin(get_alignment_norm(preds)))


def _max_num_frames_diff(args: _HandleBatchArgs, preds: Preds) -> int:
    """Get the index of the prediction that most deviates from the original spectrogram length."""
    return int(torch.argmax((args.batch.spectrogram.lengths - preds.num_frames).abs()))


def _random_sequence(args: _HandleBatchArgs, preds: Preds) -> int:
    """Get a random batch index."""
    return random.randint(0, len(args.batch) - 1)


class _Pick(typing.Protocol):
    """Get a batch index given the arguments and predictions."""

    def __call__(self, args: _HandleBatchArgs, preds: Preds) -> int:
        ...


def _visualize_inferred(args: _HandleBatchArgs, preds: Preds, pick: _Pick = _random_sequence):
    """Run in inference mode and visualize results.

    TODO: Visualize any related text annotations.
    """
    if not is_master():
        return

    pick_label = typing.cast(types.MethodType, pick).__name__
    item = pick(args, preds)
    num_frames = int(args.batch.spectrogram.lengths[0, item].item())
    num_frames_predicted = int(preds.num_frames[item].item())
    text_length = int(preds.num_tokens[item].item())
    # gold_spectrogram [num_frames, frame_channels]
    gold_spectrogram = args.batch.spectrogram.tensor[:num_frames, item]
    # spectrogram [num_frames, frame_channels]
    predicted_spectrogram = preds.frames[:num_frames_predicted, item]
    predicted_alignments = preds.alignments[:num_frames_predicted, item, :text_length]
    predicted_stop_token = preds.stop_tokens[:num_frames_predicted, item]

    model = lambda n: get_model_label(f"{n}/{pick_label}", cadence=args.cadence)
    dataset = lambda n: get_dataset_label(
        f"{n}/{pick_label}", cadence=args.cadence, type_=args.dataset_type
    )
    _plot_mel_spectrogram = cf.partial(plot_mel_spectrogram)
    figures = (
        (dataset("gold_spectrogram"), _plot_mel_spectrogram, gold_spectrogram),
        (model("predicted_spectrogram"), _plot_mel_spectrogram, predicted_spectrogram),
        (model("alignment"), plot_alignments, predicted_alignments),
        (model("stop_token"), plot_logits, predicted_stop_token),
    )
    assets = args.state.comet.log_figures({l: v(n) for l, v, n in figures})
    audio = {
        "predicted_griffin_lim_audio": griffin_lim(predicted_spectrogram.cpu().numpy(), **cf.get()),
        "gold_griffin_lim_audio": griffin_lim(gold_spectrogram.cpu().numpy(), **cf.get()),
        "gold_audio": args.batch.audio[item].cpu().numpy(),
    }
    log_npy = args.state.comet.log_npy
    npy_urls = {f"{l} Array": log_npy(l, args.batch.spans[item].speaker, a) for l, _, a in figures}
    link = lambda h: "Failed to upload." if h is None else f'<a href="{h}">{h}</a>'
    args.state.comet.log_html_audio(
        audio=audio,
        context=args.state.comet.context,
        text=args.batch.spans[item].script,
        reconstructed_text=args.batch.inputs.reconstruct_text(item)[args.batch.inputs.slices[item]],
        reconstructed_text_context=args.batch.inputs.reconstruct_text(item),
        session=args.batch.spans[item].session,
        predicted_loudness=get_average_db_rms_level(predicted_spectrogram.unsqueeze(1)).item(),
        gold_loudness=get_average_db_rms_level(gold_spectrogram.unsqueeze(1)).item(),
        pick_function=pick_label,
        **{f"{k} Figure": link(v) for k, v in assets.items()},
        **{k: link(v) for k, v in npy_urls.items()},
    )


def _run_inference(args: _HandleBatchArgs):
    """Run the model in inference mode, and measure it's results.

    TODO: Over multiple steps, track the example with the smallest `_min_alignment_norm` or largest
    `_max_num_frames_diff`, and visualize it.
    """
    args.timer.record_event(args.timer.MODEL_FORWARD)
    model = typing.cast(SpectrogramModel, args.state.model.module)
    preds = model(args.batch.inputs, mode=Mode.INFER)
    preds = typing.cast(Preds, preds)

    if args.visualize:
        args.timer.record_event(args.timer.VISUALIZE_PREDICTIONS)
        for pick in [_random_sequence, _max_num_frames_diff, _min_alignment_norm]:
            _visualize_inferred(args, preds, pick)

    args.timer.record_event(args.timer.MEASURE_METRICS)
    values: MetricsValues = {
        **args.metrics.get_dataset_values(args.batch, model, preds),
        **args.metrics.get_alignment_values(args.batch, model, preds),
        **args.metrics.get_loudness_values(args.batch, model, preds),
        **args.metrics.get_data_loader_values(args.data_loader),
    }
    args.timer.record_event(args.timer.GATHER_METRICS)
    args.metrics.update(values)


def _visualize_select_cases(state: _State, dataset_type: DatasetType, cadence: Cadence, **kw):
    """Run spectrogram model in inference mode and visualize a test case."""
    if not is_master():
        return

    model = typing.cast(SpectrogramModel, state.model.module)
    inputs, preds = cf.partial(_utils.process_select_cases)(model, model.session_embed.vocab, **kw)

    for item in range(len(inputs.doc)):
        text_length = int(preds.num_tokens[item].item())
        num_frames_predicted = int(preds.num_frames[item].item())
        # spectrogram [num_frames, frame_channels]
        predicted_spectrogram = preds.frames[:num_frames_predicted, item]
        predicted_alignments = preds.alignments[:num_frames_predicted, item, :text_length]
        predicted_stop_token = preds.stop_tokens[:num_frames_predicted, item]

        model = partial(get_model_label, cadence=cadence)
        _plot_mel_spectrogram = cf.partial(plot_mel_spectrogram)
        figures = (
            (model("predicted_spectrogram"), _plot_mel_spectrogram, predicted_spectrogram),
            (model("alignment"), plot_alignments, predicted_alignments),
            (model("stop_token"), plot_logits, predicted_stop_token),
        )
        assets = state.comet.log_figures({l: v(n) for l, v, n in figures})
        audio = cf.partial(griffin_lim)(predicted_spectrogram.cpu().numpy())
        log_npy = state.comet.log_npy
        npy_urls = {f"{l} Array": log_npy(l, inputs.session[item][0], a) for l, _, a in figures}
        link = lambda h: "Failed to upload." if h is None else f'<a href="{h}">{h}</a>'
        state.comet.log_html_audio(
            randomly_sampled_case=item,
            audio={"predicted_griffin_lim_audio": audio},
            context=state.comet.context,
            text=str(inputs.doc[item]),
            session=inputs.session[item],
            predicted_loudness=get_average_db_rms_level(predicted_spectrogram.unsqueeze(1)).item(),
            dataset_type=dataset_type,
            **{f"{k} Figure": link(v) for k, v in assets.items()},
            **{k: link(v) for k, v in npy_urls.items()},
        )


_HandleBatch = typing.Callable[[_HandleBatchArgs], None]


def _log_vocab(state: _State, dataset_type: DatasetType):
    """Log the model vocabulary to Comet."""
    if not is_master():
        return

    label = partial(get_dataset_label, cadence=Cadence.RUN, type_=dataset_type)
    model = typing.cast(SpectrogramModel, state.model.module)
    session_vocab = model.session_embed.tokens()
    parameters = {
        label("token_vocab_size"): len(model.token_embed.tokens()),
        label("token_vocab"): sorted(model.token_embed.tokens()),
        label("speaker_vocab_size"): len(model.speaker_embed.tokens()),
        label("speaker_vocab"): sorted(s for s in model.speaker_embed.tokens()),
        label("session_vocab_size"): len(session_vocab),
        label("dialect_vocab_size"): len(model.dialect_embed.tokens()),
        label("dialect_vocab"): sorted(d.value[1] for d in model.dialect_embed.tokens()),
        label("style_vocab_size"): len(model.style_embed.tokens()),
        label("style_vocab"): sorted(s.value for s in model.style_embed.tokens()),
        label("language_vocab_size"): len(model.language_embed.tokens()),
        label("language_vocab"): sorted(l.value for l in model.language_embed.tokens()),
    }
    state.comet.log_parameters(parameters)

    sessions: typing.Dict[Speaker, typing.List[str]] = collections.defaultdict(list)
    for session in session_vocab:
        sessions[session[0]].append(session[1])

    for speaker, session_label in sessions.items():
        parameters = {
            label("num_sessions", speaker=speaker): len(session_label),
            label("sessions", speaker=speaker): sorted(session_label),
        }
        state.comet.log_parameters(parameters)


@log_runtime
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
        stack.enter_context(set_context(context, state.comet, state.model, ema=state.ema))
        stack.enter_context(set_epoch(state.comet, step=int(state.step.item()), **kwargs))

        metrics = Metrics(state.comet)
        timer = Timer().record_event(Timer.LOAD_DATA)
        iterator = enumerate(iter(data_loader))
        while True:
            item = next(iterator, None)
            if item is None:
                break

            index, batch = typing.cast(typing.Tuple[int, Batch], item)
            handle_batch(make_args(metrics, timer, batch, index == 0))

            if Context.TRAIN == context:
                metrics.log(lambda l: l[-1:], timer, type_=dataset_type, cadence=Cadence.STEP)
                state.comet.log_metrics(timer.get_timers(cadence=Cadence.STEP))

            timer = Timer().record_event(Timer.LOAD_DATA)

        metrics.log(is_verbose=True, type_=dataset_type, cadence=Cadence.MULTI_STEP)
        if Context.TRAIN == context:
            _log_vocab(state, dataset_type)


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
    return ".bias" in param_name or type(module).__name__ in (torch.nn.LayerNorm,)


def run_worker(
    device: torch.device,
    comet: CometMLExperiment,
    checkpoint: typing.Optional[_utils.Checkpoint],
    checkpoints_directory: pathlib.Path,
    train_dataset: Dataset,
    dev_dataset: Dataset,
) -> typing.NoReturn:
    """Train and evaluate the spectrogram model in a loop.

    TODO: Should we checkpoint `metrics` so that metrics like `num_frames_per_speaker`,
    `num_spans_per_text_length`, or `max_num_frames` can be computed accross epochs?
    """
    if is_master():
        atexit.register(cf.purge)

    assert isinstance(checkpoint, Checkpoint) or checkpoint is None
    state = (
        _State.from_scratch(comet, device)
        if checkpoint is None
        else _State.from_checkpoint(checkpoint, comet, device)
    )
    train_loader, dev_loader = _get_data_loaders(state, train_dataset, dev_dataset, **cf.get())
    contexts: typing.List[typing.Tuple[Context, DatasetType, DataLoader, _HandleBatch]] = [
        (Context.TRAIN, DatasetType.TRAIN, train_loader, cf.partial(_run_step)),
        (Context.EVALUATE, DatasetType.DEV, dev_loader, cf.partial(_run_step)),
        (Context.EVALUATE_INFERENCE, DatasetType.DEV, dev_loader, _run_inference),
    ]
    save_checkpoint(state.to_checkpoint(), checkpoints_directory, f"step_{state.step.item()}")
    while True:
        steps_per_epoch = train_loader.num_steps_per_epoch
        [_run_steps(state, *args, steps_per_epoch=steps_per_epoch) for args in contexts]

        with set_context(Context.EVALUATE_INFERENCE, state.comet, state.model, ema=state.ema):
            _visualize_select_cases(state, DatasetType.TEST, Cadence.MULTI_STEP)

        save_checkpoint(state.to_checkpoint(), checkpoints_directory, f"step_{state.step.item()}")
