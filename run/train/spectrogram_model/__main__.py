import dataclasses
import logging
import pathlib
import random
import sys
import typing
from functools import partial
from itertools import chain

# NOTE: `comet_ml` needs to be imported before torch
import comet_ml  # type: ignore # noqa
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import tqdm
import typer
from hparams import HParam, HParams, add_config, configurable, get_config, parse_hparam_args
from torchnlp.utils import get_total_parameters, lengths_to_mask

import lib
from lib.distributed import get_world_size, is_master
from lib.environment import load, load_most_recent_file, save
from lib.utils import flatten
from run._config import (
    DATASET_PHONETIC_CHARACTERS,
    DATASETS,
    NUM_FRAME_CHANNELS,
    PHONEME_SEPARATOR,
    RANDOM_SEED,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    Cadence,
    Dataset,
    DatasetType,
    configure,
    get_config_label,
    get_dataset_label,
    get_environment_label,
    get_model_label,
)
from run._utils import get_dataset, nested_to_flat_config, split_dataset
from run.train._utils import (
    Checkpoint,
    CometMLExperiment,
    Context,
    get_dataset_stats,
    init_distributed,
    maybe_make_experiment_directories,
    maybe_make_experiment_directories_from_checkpoint,
    set_context,
)
from run.train.spectrogram_model._data import DataIterator, DataLoader, InputEncoder, SpanBatch
from run.train.spectrogram_model._metrics import DistributedMetrics, get_average_db_rms_level
from run.train.spectrogram_model._utils import set_seed

lib.environment.enable_fault_handler()
logger = logging.getLogger(__name__)
app = typer.Typer()


def _configure(
    more_config: typing.Dict[str, typing.Any], debug: bool
) -> typing.Dict[str, typing.Any]:
    """ Configure modules for spectrogram model training, and return parameters. """
    configure()

    train_batch_size = 28 if debug else 56
    dev_batch_size = train_batch_size * 4
    train_steps_per_epoch = 64 if debug else 1024
    # NOTE: This parameter was set approximately based on the size of each respective
    # dataset. The development dataset is about 16 times smaller than the training dataset
    # based on the number of characters in each dataset.
    dev_steps_per_epoch = (train_steps_per_epoch / (dev_batch_size / train_batch_size)) / 16
    assert dev_steps_per_epoch % 1 == 0, "The number of steps must be an integer."
    assert train_batch_size % get_world_size() == 0
    assert dev_batch_size % get_world_size() == 0

    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)  # type: ignore
    config = {
        set_seed: HParams(seed=RANDOM_SEED),
        _State._get_optimizers: HParams(
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.Adam,
        ),
        _run_worker: HParams(
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=int(dev_steps_per_epoch),
        ),
        _run_step: HParams(
            # NOTE: This scalar calibrates the loss so that it's scale is similar to Tacotron-2.
            spectrogram_loss_scalar=1 / 100,
            # NOTE: Learn more about this parameter here: https://arxiv.org/abs/2002.08709
            # NOTE: This value is the minimum loss the test set achieves before the model
            # starts overfitting on the train set.
            # TODO: Try increasing the stop token minimum loss because it still overfit.
            stop_token_min_loss=0.0105,
        ),
        _get_data_loaders: HParams(
            # SOURCE: Tacotron 2
            # To train the feature prediction network, we apply the standard maximum-likelihood
            # training procedure (feeding in the correct output instead of the predicted output on
            # the decoder side, also referred to as teacher-forcing) with a batch size of 64 on a
            # single GPU.
            # NOTE: Batch size parameters set after experimentation on a 2 Px100 GPU.
            train_batch_size=train_batch_size,
            dev_batch_size=dev_batch_size,
            num_workers=2 if debug else 4,
        ),
        DistributedMetrics.get_model_metrics: HParams(num_frame_channels=NUM_FRAME_CHANNELS),
        # SOURCE (Tacotron 2):
        # We use the Adam optimizer with β1 = 0.9, β2 = 0.999, eps = 10−6 learning rate of 10−3
        # We also apply L2 regularization with weight 10−6
        # NOTE: No L2 regularization performed better based on Comet experiments in March 2020.
        torch.optim.Adam.__init__: HParams(
            eps=10 ** -6,
            weight_decay=0,
            lr=10 ** -3,
            amsgrad=True,
            betas=(0.9, 0.999),
        ),
        InputEncoder.__init__: HParams(phoneme_separator=PHONEME_SEPARATOR),
    }
    add_config(config)
    add_config(more_config)
    set_seed()
    return nested_to_flat_config(get_config())


@dataclasses.dataclass(frozen=True)
class Checkpoint(Checkpoint):
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
    num_examples: torch.Tensor = torch.tensor(0, dtype=torch.long)

    def update_num_examples(self, count: int):
        self.num_examples.add_(int(lib.distributed.reduce(count, self.device)))

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
            flatten([p.script for p in passages]),
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
            num_examples=int(self.num_examples.item()),
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
            torch.tensor(checkpoint.num_examples),
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
    num_workers: int = HParam(),
) -> typing.Tuple[DataLoader, DataLoader]:
    """ Initialize training and development data loaders.  """
    kwargs = dict(num_workers=num_workers, device=state.device, input_encoder=state.input_encoder)
    DataLoaderPartial = partial(DataLoader, **kwargs)
    return (
        DataLoaderPartial(DataIterator(train_dataset, train_batch_size)),
        DataLoaderPartial(DataIterator(dev_dataset, dev_batch_size)),
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
            alignment between `frames` and `tokens`.
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
    metrics: DistributedMetrics,
    batch: SpanBatch,
    data_loader: DataLoader,
    dataset_type: DatasetType,
    visualize: bool = False,
    spectrogram_loss_scalar: float = HParam(),
    stop_token_min_loss: float = HParam(),
):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    TODO: Run the model with a dummy batch that's larger than the largest sequence length, learn
    more:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#pre-allocate-memory-in-case-of-variable-input-length
    TODO: Maybe enable the CUDNN auto tuner:
    https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        stop_token_min_loss: This thresholds the stop token loss to prevent overfitting.
    """
    frames, stop_token, alignment, spectrogram_loss, stop_token_loss = state.model(
        tokens=batch.encoded_phonemes.tensor,
        speaker=batch.encoded_speaker.tensor,
        target_frames=batch.spectrogram.tensor,
        target_stop_token=batch.stop_token.tensor,
        num_tokens=batch.encoded_phonemes.lengths,
        tokens_mask=batch.encoded_phonemes_mask.tensor,
        target_mask=batch.spectrogram_mask.tensor,
        mode=lib.spectrogram_model.Mode.FORWARD,
    )

    if state.model.training:
        state.model.zero_grad(set_to_none=True)

        # NOTE: We sum over the `num_frames` dimension to ensure that we don't bias based on
        # `num_frames`. For example, a larger `num_frames` means that the denominator is larger;
        # therefore, the loss value for each element is smaller.
        # NOTE: We average accross `batch_size` and `frame_channels` so that the loss magnitude is
        # invariant to those variables.

        average_spectrogram_length = data_loader.average_spectrogram_length

        # spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
        spectrogram_loss_ = (spectrogram_loss.sum(dim=0) / average_spectrogram_length).mean()
        spectrogram_loss_ *= spectrogram_loss_scalar

        # stop_token_loss [num_frames, batch_size] → [1]
        stop_token_loss_ = (stop_token_loss.sum(dim=0) / average_spectrogram_length).mean()
        stop_token_loss_ = (stop_token_loss_ - stop_token_min_loss).abs() + stop_token_min_loss

        (spectrogram_loss_ + stop_token_loss_).backward()

        # NOTE: Measure the "grad_norm" before `clipper.clip()`.
        metrics.log_optimizer_metrics(state.optimizer, state.clipper)

        # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
        params = state.optimizer.param_groups[0]["params"]
        assert all([p.grad is not None for p in params]), "No gradients found."
        state.clipper.clip()
        state.optimizer.step()
        state.step.add_(1)
        state.update_num_examples(batch.length)
        state.scheduler.step()
        state.comet.set_step(typing.cast(int, state.step.item()))

    if visualize:
        _visualize_source_vs_target(
            state, batch, frames, stop_token, alignment, dataset_type, Cadence.STEP
        )

    # Update metrics, and log those updates.
    metrics.update_dataset_metrics(batch, state.input_encoder)
    metrics.append(metrics.num_frames_predicted, batch.spectrogram.lengths)
    metrics.update_alignment_metrics(
        alignment,
        batch.spectrogram_mask.tensor,
        batch.encoded_phonemes_mask.tensor,
        batch.encoded_phonemes.lengths,
        batch.encoded_speaker.tensor,
        state.input_encoder,
    )
    metrics.update_stop_token_accuracy(
        batch.stop_token.tensor,
        stop_token,
        typing.cast(float, state.model.module.stop_threshold),
        batch.spectrogram_mask.tensor,
    )
    metrics.update_data_queue_size(data_loader)
    metrics.append(metrics.spectrogram_loss, spectrogram_loss)
    metrics.append(metrics.stop_token_loss, stop_token_loss)


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
            alignment between `frames` and `tokens`.
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
    metrics: DistributedMetrics,
    batch: SpanBatch,
    data_loader: DataLoader,
    dataset_type: DatasetType,
    visualize: bool = False,
):
    """Run the model in inference mode, and measure it's results.

    TODO: Consider calling `update_dataset_metrics`, and filtering the spans which overflowed.

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
    """
    frames, stop_tokens, alignments, lengths, reached_max = state.model.module(
        batch.encoded_phonemes.tensor,
        batch.encoded_speaker.tensor,
        batch.encoded_phonemes.lengths,
        mode=lib.spectrogram_model.Mode.INFER,
    )

    if visualize:
        _visualize_inferred(
            state, batch, frames, stop_tokens, alignments, lengths, dataset_type, Cadence.STEP
        )

    # NOTE: Remove predictions that diverged (reached max) as to not skew other metrics. We
    # count these sequences separatly with `reached_max_frames`.
    bool_ = ~reached_max.view(-1)
    if bool_.sum() > 0:
        max_frames = lengths[:, bool_].max()
        max_tokens = batch.encoded_phonemes.lengths[:, bool_].max()
        # NOTE: `lengths_to_mask` moves data from gpu to cpu, so it causes a sync
        predicted_mask = lengths_to_mask(lengths[:, bool_], device=lengths.device).transpose(0, 1)
    else:
        max_frames, max_tokens = 0, 0
        predicted_mask = torch.ones(0, 0, dtype=torch.bool, device=lengths.device)
    metrics.append(metrics.batch_size, batch.length - reached_max.sum().item())
    metrics.append(metrics.num_frames, batch.spectrogram.lengths[:, bool_])
    metrics.append(metrics.num_frames_predicted, lengths[:, bool_])
    metrics.update_rms_metrics(
        batch.spectrogram.tensor[:max_frames, bool_],
        frames[:max_frames, bool_],
        batch.spectrogram_mask.tensor[:max_frames, bool_],
        predicted_mask,
    )
    metrics.update_alignment_metrics(
        alignments[:max_frames, bool_, :max_tokens],
        predicted_mask,
        batch.encoded_phonemes_mask.tensor[:max_tokens, bool_],
        batch.encoded_phonemes.lengths[:max_tokens, bool_],
        batch.encoded_speaker.tensor[:, bool_],
        state.input_encoder,
    )
    metrics.update_data_queue_size(data_loader)
    metrics.update_reached_max_metrics(batch, state.input_encoder, reached_max)


_BatchHandler = typing.Callable[
    [_State, DistributedMetrics, SpanBatch, DataLoader, DatasetType, bool], None
]


@configurable
def _run_worker(
    device_index: int,
    checkpoints_directory: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet_partial: typing.Callable[..., CometMLExperiment],
    config: typing.Dict[str, typing.Any],
    debug: bool,
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
    _configure(config, debug)
    if checkpoint is None:
        state = _State.from_dataset(train_dataset, dev_dataset, comet, device)
    else:
        checkpoint_ = typing.cast(Checkpoint, load(checkpoint, device=device))
        state = _State.from_checkpoint(checkpoint_, comet, device)
    train_loader, dev_loader = _get_data_loaders(state, train_dataset, dev_dataset)
    _set_context = partial(set_context, model=state.model, comet=comet)
    dev_args = (DatasetType.DEV, dev_loader, dev_steps_per_epoch)
    contexts: typing.List[typing.Tuple[Context, DatasetType, DataLoader, int, _BatchHandler]] = [
        (Context.TRAIN, DatasetType.TRAIN, train_loader, train_steps_per_epoch, _run_step),
        (Context.EVALUATE, *dev_args, _run_step),
        (Context.EVALUATE_INFERENCE, *dev_args, _run_inference),
    ]

    while True:
        epoch = int(state.step.item() // train_steps_per_epoch)
        message = "Running Epoch %d (Step %d, Example %d)"
        logger.info(message, epoch, state.step.item(), state.num_examples.item())
        comet.set_step(typing.cast(int, state.step.item()))
        comet.log_current_epoch(epoch)

        for context, dataset_type, data_loader, num_steps, handle_batch in contexts:
            with _set_context(context):
                metrics = DistributedMetrics(comet, state.device)
                loader = zip(range(num_steps), data_loader)
                for i, batch in tqdm.tqdm(loader, total=num_steps) if is_master() else loader:
                    batch = typing.cast(SpanBatch, batch)
                    handle_batch(state, metrics, batch, data_loader, dataset_type, i == 0)
                    if Context.TRAIN == context:
                        metrics.log(lambda l: l[-1], dataset_type, Cadence.STEP)
                metrics.log(sum, dataset_type, Cadence.MULTI_STEP)

        if is_master():
            path = checkpoints_directory / f"step_{state.step.item()}{lib.environment.PT_EXTENSION}"
            save(path, state.to_checkpoint(checkpoints_directory=checkpoints_directory))
        comet.log_epoch_end(epoch)


def _run(
    checkpoints_path: pathlib.Path,
    config: typing.Dict[str, typing.Any],
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path] = None,
    debug: bool = False,
):
    """Run spectrogram model training. """
    lib.environment.check_module_versions()

    datasets = DATASETS
    datasets = {k: v for k, v in list(datasets.items())[:1]} if debug else datasets

    # NOTE: Load, preprocess, and cache dataset values.
    dataset = get_dataset(datasets)
    train_dataset, dev_dataset = split_dataset(dataset)
    comet.log_parameters(get_dataset_stats(train_dataset, dev_dataset))

    logger.info("Spawning workers %s", lib.utils.mazel_tov())
    # TODO: PyTorch-Lightning makes strong recommendations to not use `spawn`. Learn more:
    # https://pytorch-lightning.readthedocs.io/en/stable/multi_gpu.html#distributed-data-parallel
    # https://github.com/PyTorchLightning/pytorch-lightning/pull/2029
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5772
    # Also, it's less normal to use `spawn` because it wouldn't work with multiple nodes, so
    # we should consider using `torch.distributed.launch`.
    # TODO: Should we consider setting OMP num threads similarly:
    # https://github.com/pytorch/pytorch/issues/22260
    return lib.distributed.spawn(
        _run_worker.get_configured_partial(),  # type: ignore
        args=(
            checkpoints_path,
            checkpoint,
            train_dataset,
            dev_dataset,
            partial(CometMLExperiment, experiment_key=comet.get_key()),
            config,
            debug,
        ),
    )


def _setup_config(
    comet: CometMLExperiment, config: typing.List[str], debug: bool
) -> typing.Tuple[typing.Dict[str, typing.Any], lib.environment.RecordStandardStreams]:
    """
    TODO: For checkpointed runs, should we triple check the same parameters are getting
    configured? Should we throw an error if not? Or should we create a new experiment, and ensure
    that each experiments parameters are immutable?

    TODO: `RecordStandardStreams` should be started after `CometMLExperiment`; otherwise,
    `CometMLExperiment` will not be able to monitor the standard streams. Can this be fixed?
    """
    recorder = lib.environment.RecordStandardStreams()
    # NOTE: Ensure command line args are captured in the logs.
    logger.info("Command line args: %s", str(sys.argv))
    parsed = parse_hparam_args(config)
    parameters = _configure(parsed, debug)
    params = {get_config_label(k): v for k, v in parameters.items()}
    comet.log_parameters(params)
    return parsed, recorder


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
    lib.environment.set_basic_logging_config()
    pattern = str(SPECTROGRAM_MODEL_EXPERIMENTS_PATH / f"**/*{lib.environment.PT_EXTENSION}")
    if checkpoint:
        loaded = load(checkpoint)
    else:
        checkpoint, loaded = load_most_recent_file(pattern, load)
    checkpoint_ = typing.cast(Checkpoint, loaded)
    comet = CometMLExperiment(experiment_key=checkpoint_.comet_experiment_key)
    config, recorder = _setup_config(comet, context.args, debug)
    _, checkpoints_path = maybe_make_experiment_directories_from_checkpoint(checkpoint_, recorder)
    _run(checkpoints_path, config, comet, checkpoint, debug=debug)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Option([], help="Experiment tags."),
    debug: bool = typer.Option(False, help="Turn on debugging mode."),
    min_disk_space: float = 0.2,
):
    """ Start a training run in PROJECT named NAME with TAGS. """
    lib.environment.assert_enough_disk_space(min_disk_space)
    lib.environment.set_basic_logging_config()
    comet = CometMLExperiment(project_name=project)
    comet.set_name(name)
    comet.add_tags(tags)
    config, recorder = _setup_config(comet, context.args, debug)
    experiment_root = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / lib.environment.bash_time_label()
    run_root, checkpoints_path = maybe_make_experiment_directories(experiment_root, recorder)
    comet.log_other(get_environment_label("directory"), str(run_root))
    _run(checkpoints_path, config, comet, debug=debug)


if __name__ == "__main__":  # pragma: no cover
    app()
