import dataclasses
import logging
import math
import pathlib
import random
import typing
from functools import partial
from itertools import chain

import torch
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, configurable
from third_party import get_parameter_norm
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torchnlp.utils import get_total_parameters, lengths_to_mask

import lib
from lib.distributed import is_master
from lib.environment import save
from run._config import (
    DATASET_PHONETIC_CHARACTERS,
    Cadence,
    Dataset,
    DatasetType,
    get_dataset_label,
    get_model_label,
)
from run.train import _utils
from run.train._utils import CometMLExperiment, Context, DataLoader, Timer, set_context, set_epoch
from run.train.spectrogram_model._data import DataProcessor, InputEncoder, SpanBatch
from run.train.spectrogram_model._metrics import Metrics, get_average_db_rms_level

logger = logging.getLogger(__name__)

if not hasattr(torch.optim.Adam.__init__, "_configurable"):
    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


@dataclasses.dataclass(frozen=True)
class Checkpoint(_utils.Checkpoint):
    """Checkpoint used to checkpoint spectrogram model training."""

    input_encoder: InputEncoder
    model: lib.spectrogram_model.SpectrogramModel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    scheduler: torch.optim.lr_scheduler.LambdaLR


@dataclasses.dataclass(frozen=True)
class _State:
    input_encoder: InputEncoder
    model: torch.nn.parallel.DistributedDataParallel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    scheduler: torch.optim.lr_scheduler.LambdaLR
    comet: CometMLExperiment
    device: torch.device
    step: torch.Tensor = torch.tensor(0, dtype=torch.long)

    @staticmethod
    def _get_input_encoder(
        train_dataset: Dataset,
        dev_dataset: Dataset,
        comet: CometMLExperiment,
    ) -> InputEncoder:
        """Initialize an input encoder to encode model input.

        TODO: For some reason, Comet doesn't log: "phoneme_vocab".
        """
        passages = chain(*tuple(chain(train_dataset.values(), dev_dataset.values())))
        input_encoder = InputEncoder(
            [p.script for p in passages],
            DATASET_PHONETIC_CHARACTERS,
            list(train_dataset.keys()) + list(dev_dataset.keys()),
        )

        label = partial(get_dataset_label, cadence=Cadence.STATIC, type_=DatasetType.TRAIN)
        stats = {
            label("grapheme_vocab_size"): input_encoder.grapheme_encoder.vocab_size,
            label("grapheme_vocab"): sorted(input_encoder.grapheme_encoder.vocab),
            label("phoneme_vocab_size"): input_encoder.phoneme_encoder.vocab_size,
            label("phoneme_vocab"): sorted(input_encoder.phoneme_encoder.vocab),
            label("num_speakers"): input_encoder.speaker_encoder.vocab_size,
            label("speakers"): sorted([s.label for s in input_encoder.speaker_encoder.vocab]),
        }
        comet.log_parameters(stats)

        label = partial(label, type_=DatasetType.DEV)
        stats = {
            label("num_speakers"): len(list(dev_dataset.keys())),
            label("speakers"): sorted([s.label for s in dev_dataset.keys()]),
        }
        comet.log_parameters(stats)

        return input_encoder

    @staticmethod
    def _get_model(
        device: torch.device,
        comet: CometMLExperiment,
        input_encoder: InputEncoder,
    ) -> lib.spectrogram_model.SpectrogramModel:
        """Initialize a model onto `device`.

        NOTE: Learn more about `DistributedDataParallel` here:
        https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
        """
        model = lib.spectrogram_model.SpectrogramModel(
            input_encoder.phoneme_encoder.vocab_size,
            input_encoder.speaker_encoder.vocab_size,
        ).to(device, non_blocking=True)
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
        torch.optim.lr_scheduler.LambdaLR,
    ]:
        """Initialize model optimizers.

        NOTE: These optimizers cannot be moved easily between devices; therefore, the model weights
        should already be on the appropriate device. Learn more:
        https://github.com/pytorch/pytorch/issues/2830
        """
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer_ = optimizer(params)
        clipper = lib.optimizers.AdaptiveGradientNormClipper(params)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_, lr_multiplier_schedule)
        return optimizer_, clipper, scheduler

    def to_checkpoint(self, **kwargs):
        """ Create a checkpoint to save the spectrogram training state. """
        return Checkpoint(
            comet_experiment_key=self.comet.get_key(),
            input_encoder=self.input_encoder,
            model=typing.cast(lib.spectrogram_model.SpectrogramModel, self.model.module),
            optimizer=self.optimizer,
            clipper=self.clipper,
            scheduler=self.scheduler,
            step=int(self.step.item()),
            **kwargs,
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
        """
        return cls(
            checkpoint.input_encoder,
            torch.nn.parallel.DistributedDataParallel(checkpoint.model, [device], device),
            checkpoint.optimizer,
            checkpoint.clipper,
            checkpoint.scheduler,
            comet,
            device,
            torch.tensor(checkpoint.step),
        )

    @classmethod
    def from_dataset(
        cls,
        train_dataset: Dataset,
        dev_dataset: Dataset,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """ Create spectrogram training state from the `train_dataset`. """
        input_encoder = cls._get_input_encoder(train_dataset, dev_dataset, comet)
        model = cls._get_model(device, comet, input_encoder)
        # NOTE: Even if `_get_model` is initialized differently in each process, the parameters
        # will be synchronized. Learn more:
        # https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564/2
        model = torch.nn.parallel.DistributedDataParallel(model, [device], device)
        return cls(input_encoder, model, *cls._get_optimizers(model), comet, device)


@configurable
def _get_data_loaders(
    state: _State,
    train_dataset: Dataset,
    dev_dataset: Dataset,
    train_batch_size: int = HParam(),
    dev_batch_size: int = HParam(),
    train_steps_per_epoch: int = HParam(),
    dev_steps_per_epoch: int = HParam(),
    num_workers: int = HParam(),
    prefetch_factor: int = HParam(),
) -> typing.Tuple[DataLoader, DataLoader]:
    """ Initialize training and development data loaders.  """
    input_encoder, step = state.input_encoder, int(state.step.item())
    train = DataProcessor(train_dataset, train_batch_size, input_encoder=input_encoder, step=step)
    dev = DataProcessor(dev_dataset, dev_batch_size, input_encoder=input_encoder, step=step)
    kwargs = dict(num_workers=num_workers, device=state.device, prefetch_factor=prefetch_factor)
    return (
        DataLoader(train, num_steps_per_epoch=train_steps_per_epoch, **kwargs),
        DataLoader(dev, num_steps_per_epoch=dev_steps_per_epoch, **kwargs),
    )


def _visualize_source_vs_target(
    state: _State,
    batch: SpanBatch,
    predicted_spectrogram: torch.Tensor,
    predicted_stop_token: torch.Tensor,
    predicted_alignments: torch.Tensor,
    dataset_type: DatasetType,
    cadence: Cadence,
):
    """Visualize predictions as compared to the original `batch`.

    Args:
        ...
        predicted_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Spectrogram frames.
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Stopping probability for
            each frame.
        predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
            alignments between `frames` and `tokens`.
        ...
    """
    if not is_master():
        return

    item = random.randint(0, batch.length - 1)
    spectrogram_length = int(batch.spectrogram.lengths[0, item].item())
    text_length = int(batch.encoded_phonemes.lengths[0, item].item())

    # predicted_spectrogram, gold_spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:spectrogram_length, item]
    gold_spectrogram = batch.spectrogram.tensor[:spectrogram_length, item]

    predicted_delta = abs(gold_spectrogram - predicted_spectrogram)
    predicted_alignments = predicted_alignments[:spectrogram_length, item, :text_length]
    predicted_stop_token = predicted_stop_token[:spectrogram_length, item]
    model = partial(get_model_label, cadence=cadence)
    dataset = partial(get_dataset_label, cadence=cadence, type_=dataset_type)
    figures = {
        model("spectrogram_delta"): lib.visualize.plot_mel_spectrogram(predicted_delta),
        model("predicted_spectrogram"): lib.visualize.plot_mel_spectrogram(predicted_spectrogram),
        model("alignment"): lib.visualize.plot_alignments(predicted_alignments),
        model("stop_token"): lib.visualize.plot_logits(predicted_stop_token),
        dataset("gold_spectrogram"): lib.visualize.plot_mel_spectrogram(gold_spectrogram),
    }
    state.comet.log_figures(figures)


@configurable
def _run_step(
    state: _State,
    metrics: Metrics,
    batch: SpanBatch,
    data_loader: DataLoader,
    dataset_type: DatasetType,
    timer: Timer,
    visualize: bool = False,
    spectrogram_loss_scalar: float = HParam(),
    stop_token_min_loss: float = HParam(),
    average_spectrogram_length: float = HParam(),
):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    TODO: Run the model with a dummy batch that's larger than the largest sequence length, learn
    more:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
    TODO: Maybe enable the CUDNN auto tuner:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    TODO: Use `asycio` to measure, gather, and reduce metrics without blocking training. This could
    help improve performance by as much as ~9% (70ms of 800ms).

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        stop_token_min_loss: This thresholds the stop token loss to prevent overfitting.
        average_spectrogram_length: The training dataset average spectrogram length. It is used
            to normalize the loss magnitude.
    """
    timer.record_event(timer.MODEL_FORWARD)
    frames, stop_token, alignments = state.model(
        tokens=batch.encoded_phonemes.tensor,
        speaker=batch.encoded_speaker.tensor,
        target_frames=batch.spectrogram.tensor,
        num_tokens=batch.encoded_phonemes.lengths,
        tokens_mask=batch.encoded_phonemes_mask.tensor,
        target_mask=batch.spectrogram_mask.tensor,
        mode=lib.spectrogram_model.Mode.FORWARD,
    )

    # SOURCE: Tacotron 2
    # We minimize the summed mean squared error (MSE) from before and after the post-net to aid
    # convergence.
    spectrogram_loss = mse_loss(frames, batch.spectrogram.tensor, reduction="none")
    spectrogram_loss *= batch.spectrogram_mask.tensor.unsqueeze(2)

    # SOURCE (Tacotron 2 Author):
    # The author confirmed they used BCE loss in Google Chat.
    target = batch.stop_token.tensor
    stop_token_loss = binary_cross_entropy_with_logits(stop_token, target, reduction="none")
    stop_token_loss *= batch.spectrogram_mask.tensor

    if state.model.training:
        state.model.zero_grad(set_to_none=True)

        # NOTE: We sum over the `num_frames` dimension to ensure that we don't bias based on
        # `num_frames`. For example, a larger `num_frames` means that the denominator is larger;
        # therefore, the loss value for each element is smaller.
        # NOTE: We average accross `batch_size`, `num_frames` and `frame_channels` so that the loss
        # magnitude is invariant to those variables.

        # spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
        spectrogram_loss_ = (spectrogram_loss.sum(dim=0) / average_spectrogram_length).mean()
        spectrogram_loss_ *= spectrogram_loss_scalar

        # stop_token_loss [num_frames, batch_size] → [1]
        stop_token_loss_ = (stop_token_loss.sum(dim=0) / average_spectrogram_length).mean()
        stop_token_loss_ = (stop_token_loss_ - stop_token_min_loss).abs() + stop_token_min_loss

        timer.record_event(timer.MODEL_BACKWARD)
        (spectrogram_loss_ + stop_token_loss_).backward()

        timer.record_event(timer.MODEL_STEP)
        # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
        assert len(state.optimizer.param_groups) == 1, "Expecting only 1 group of parameters."
        params = state.optimizer.param_groups[0]["params"]
        assert all([p.grad is not None for p in params]), "`None` gradients found."
        # NOTE: Measure the "grad_norm" before `clipper.clip()`.
        norm_inf = get_parameter_norm(params, math.inf) if is_master() else torch.tensor(math.nan)
        assert not is_master() or torch.isfinite(norm_inf), f"Gradient was too large {norm_inf}."
        norm = state.clipper.clip()
        state.optimizer.step()
        state.scheduler.step()
        timer.record_event(timer.LOG_METRICS)
        state.step.add_(1)
        state.comet.set_step(typing.cast(int, state.step.item()))
        norm_inf_ = float(norm_inf.item())
        metrics.log_optimizer_metrics(
            norm, norm_inf_, state.optimizer, state.clipper, cadence=Cadence.STEP
        )

    if visualize:
        timer.record_event(timer.VISUALIZE_PREDICTIONS)
        _visualize_source_vs_target(
            state, batch, frames, stop_token, alignments, dataset_type, Cadence.STEP
        )

    timer.record_event(timer.MEASURE_METRICS)
    stop_threshold = typing.cast(float, state.model.module.stop_threshold)
    spectrogram_mask = batch.spectrogram_mask.tensor
    spectrogram_lengths = batch.spectrogram.lengths
    values: _utils.MetricsValues = {
        **metrics.get_dataset_values(batch),
        **metrics.get_alignment_values(batch, alignments, spectrogram_lengths, spectrogram_mask),
        **metrics.get_loudness_values(batch, frames, spectrogram_mask),
        **metrics.get_data_loader_values(data_loader),
        **metrics.get_stop_token_values(batch, stop_token, stop_threshold),
        metrics.SPECTROGRAM_LOSS_SUM: float(spectrogram_loss.sum().item()),
        metrics.STOP_TOKEN_LOSS_SUM: float(stop_token_loss.sum().item()),
    }
    timer.record_event(timer.GATHER_METRICS)
    metrics.update(values)


def _visualize_inferred(
    state: _State,
    batch: SpanBatch,
    predicted_spectrogram: torch.Tensor,
    predicted_stop_token: torch.Tensor,
    predicted_alignments: torch.Tensor,
    predicted_lengths: torch.Tensor,
    dataset_type: DatasetType,
    cadence: Cadence,
):
    """Run in inference mode and visualize results.

    TODO: Visualize any related text annotations.

    Args:
        ...
        predicted_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels]):
            Spectrogram frames.
        predicted_stop_token (torch.FloatTensor [num_frames, batch_size]): Stopping probability for
            each frame.
        predicted_alignments (torch.FloatTensor [num_frames, batch_size, num_tokens]): Attention
            alignments between `frames` and `tokens`.
        predicted_lengths (torch.LongTensor [1, batch_size]): The sequence length.
        ...
    """
    if not is_master():
        return

    item = random.randint(0, batch.length - 1)
    num_frames = int(batch.spectrogram.lengths[0, item].item())
    num_frames_predicted = int(predicted_lengths[0, item].item())
    text_length = int(batch.encoded_phonemes.lengths[0, item].item())
    # gold_spectrogram [num_frames, frame_channels]
    gold_spectrogram = batch.spectrogram.tensor[:num_frames, item]
    # spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:num_frames_predicted, item]
    predicted_alignments = predicted_alignments[:num_frames_predicted, item, :text_length]
    predicted_stop_token = predicted_stop_token[:num_frames_predicted, item]

    model = partial(get_model_label, cadence=cadence)
    dataset = partial(get_dataset_label, cadence=cadence, type_=dataset_type)
    figures = {
        dataset("gold_spectrogram"): lib.visualize.plot_mel_spectrogram(gold_spectrogram),
        model("predicted_spectrogram"): lib.visualize.plot_mel_spectrogram(predicted_spectrogram),
        model("alignment"): lib.visualize.plot_alignments(predicted_alignments),
        model("stop_token"): lib.visualize.plot_logits(predicted_stop_token),
    }
    state.comet.log_figures(figures)
    audio = {
        "predicted_griffin_lim_audio": lib.audio.griffin_lim(predicted_spectrogram.cpu().numpy()),
        "gold_griffin_lim_audio": lib.audio.griffin_lim(gold_spectrogram.cpu().numpy()),
        "gold_audio": batch.audio[item].cpu().numpy(),
    }
    state.comet.log_html_audio(
        audio=audio,
        context=state.comet.context,
        text=batch.spans[item].script,
        speaker=batch.spans[item].speaker,
        predicted_loudness=get_average_db_rms_level(predicted_spectrogram.unsqueeze(1)).item(),
        gold_loudness=get_average_db_rms_level(gold_spectrogram.unsqueeze(1)).item(),
    )


def _run_inference(
    state: _State,
    metrics: Metrics,
    batch: SpanBatch,
    data_loader: DataLoader,
    dataset_type: DatasetType,
    timer: Timer,
    visualize: bool = False,
):
    """Run the model in inference mode, and measure it's results.

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
    """
    timer.record_event(timer.MODEL_FORWARD)
    frames, stop_tokens, alignments, lengths, reached_max = state.model.module(
        batch.encoded_phonemes.tensor,
        batch.encoded_speaker.tensor,
        batch.encoded_phonemes.lengths,
        mode=lib.spectrogram_model.Mode.INFER,
    )

    if visualize:
        timer.record_event(timer.VISUALIZE_PREDICTIONS)
        _visualize_inferred(
            state, batch, frames, stop_tokens, alignments, lengths, dataset_type, Cadence.STEP
        )

    timer.record_event(timer.MEASURE_METRICS)
    mask = lengths_to_mask(lengths, device=lengths.device).transpose(0, 1)
    values: _utils.MetricsValues = {
        **metrics.get_dataset_values(batch, reached_max),
        **metrics.get_alignment_values(batch, alignments, lengths, mask, reached_max),
        **metrics.get_loudness_values(batch, frames, mask, reached_max),
        **metrics.get_data_loader_values(data_loader),
    }
    timer.record_event(timer.GATHER_METRICS)
    metrics.update(values)


_HandleBatch = typing.Callable[
    [_State, Metrics, SpanBatch, DataLoader, DatasetType, Timer, bool], None
]


def _run_steps(
    store: torch.distributed.TCPStore,
    context: Context,
    state: _State,
    dataset_type: DatasetType,
    data_loader: DataLoader,
    handle_batch: _HandleBatch,
):
    """Run the `handle_batch` in a loop over `data_loader` batches."""
    with set_context(context, model=state.model, comet=state.comet):
        metrics = Metrics(store, state.comet, state.input_encoder.speaker_encoder.vocab)
        iterator = enumerate(iter(data_loader))
        while True:
            timer = Timer()
            timer.record_event(timer.LOAD_DATA)
            item = next(iterator, None)
            if item is None:
                break

            index, batch = item
            handle_batch(state, metrics, batch, data_loader, dataset_type, timer, index == 0)

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
    train_dataset: Dataset,
    dev_dataset: Dataset,
) -> typing.NoReturn:
    """Train and evaluate the spectrogram model in a loop.

    TODO: Should we checkpoint `metrics` so that metrics like `num_frames_per_speaker`,
    `num_spans_per_text_length`, or `max_num_frames` can be computed accross epochs?
    """
    state = (
        _State.from_dataset(train_dataset, dev_dataset, comet, device)
        if checkpoint is None
        else _State.from_checkpoint(checkpoint, comet, device)
    )
    train_loader, dev_loader = _get_data_loaders(state, train_dataset, dev_dataset)
    contexts: typing.List[typing.Tuple[Context, DatasetType, DataLoader, _HandleBatch]] = [
        (Context.TRAIN, DatasetType.TRAIN, train_loader, _run_step),
        (Context.EVALUATE, DatasetType.DEV, dev_loader, _run_step),
        (Context.EVALUATE_INFERENCE, DatasetType.DEV, dev_loader, _run_inference),
    ]
    while True:
        with set_epoch(
            comet, step=state.step.item(), steps_per_epoch=train_loader.num_steps_per_epoch
        ):
            for context, dataset_type, data_loader, handle_batch in contexts:
                _run_steps(store, context, state, dataset_type, data_loader, handle_batch)

            if is_master():
                name = f"step_{state.step.item()}{lib.environment.PT_EXTENSION}"
                path = checkpoints_directory / name
                save(path, state.to_checkpoint(checkpoints_directory=checkpoints_directory))
