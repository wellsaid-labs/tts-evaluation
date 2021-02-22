import dataclasses
import logging
import os
import pathlib
import random
import typing
from functools import partial
from itertools import chain

# NOTE: `comet_ml` needs to be imported before torch
import comet_ml  # type: ignore # noqa
import torch
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, HParams, add_config, configurable
from torchnlp.utils import get_total_parameters, lengths_to_mask

import lib
from lib.distributed import get_world_size, is_master
from lib.environment import save
from run._config import (
    DATASET_PHONETIC_CHARACTERS,
    NUM_FRAME_CHANNELS,
    PHONEME_SEPARATOR,
    RANDOM_SEED,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    Cadence,
    Dataset,
    DatasetType,
    get_dataset_label,
    get_model_label,
)
from run.train import _utils
from run.train._utils import (
    CometMLExperiment,
    Context,
    DataLoader,
    get_config_parameters,
    make_app,
    run_workers,
    set_context,
    set_epoch,
    set_run_seed,
)
from run.train.spectrogram_model._data import DataProcessor, InputEncoder, SpanBatch
from run.train.spectrogram_model._metrics import Metrics, get_average_db_rms_level

logger = logging.getLogger(__name__)
torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)


def _make_configuration(
    train_dataset: Dataset, dev_dataset: Dataset, debug: bool
) -> typing.Dict[typing.Callable, typing.Any]:
    """Make additional configuration for spectrogram model training."""

    train_size = sum([sum([p.aligned_audio_length() for p in d]) for d in train_dataset.values()])
    dev_size = sum([sum([p.aligned_audio_length() for p in d]) for d in dev_dataset.values()])
    ratio = round(train_size / dev_size)
    train_batch_size = 28 if debug else 56
    dev_batch_size = train_batch_size * 4
    train_steps_per_epoch = 64 if debug else 1024
    dev_steps_per_epoch = (train_steps_per_epoch / (dev_batch_size / train_batch_size)) / ratio
    assert dev_steps_per_epoch % 1 == 0, "The number of steps must be an integer."
    assert train_batch_size % get_world_size() == 0
    assert dev_batch_size % get_world_size() == 0

    return {
        set_run_seed: HParams(seed=RANDOM_SEED),
        _State._get_optimizers: HParams(
            lr_multiplier_schedule=partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.Adam,
        ),
        _run_step: HParams(
            # NOTE: This scalar calibrates the loss so that it's scale is similar to Tacotron-2.
            spectrogram_loss_scalar=1 / 100,
            # NOTE: Learn more about this parameter here: https://arxiv.org/abs/2002.08709
            # NOTE: This value is the minimum loss the test set achieves before the model
            # starts overfitting on the train set.
            # TODO: Try increasing the stop token minimum loss because it still overfit.
            stop_token_min_loss=0.0105,
            # NOTE: This value is the average spectrogram length in the training dataset.
            average_spectrogram_length=315.0,
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
            train_steps_per_epoch=train_steps_per_epoch,
            dev_steps_per_epoch=int(dev_steps_per_epoch),
            num_workers=2 if debug else 4,
            prefetch_factor=2 if debug else 4,
        ),
        Metrics._get_model_metrics: HParams(num_frame_channels=NUM_FRAME_CHANNELS),
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
    max_parallel = int(os.cpu_count() // get_world_size())
    partial_ = partial(DataProcessor, input_encoder=state.input_encoder, max_parallel=max_parallel)
    kwargs = dict(num_workers=num_workers, device=state.device, prefetch_factor=prefetch_factor)
    make_loader = lambda d, b, s: DataLoader(partial_(d, b), num_steps_per_epoch=s, **kwargs)
    return (
        make_loader(train_dataset, train_batch_size, train_steps_per_epoch),
        make_loader(dev_dataset, dev_batch_size, dev_steps_per_epoch),
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

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        stop_token_min_loss: This thresholds the stop token loss to prevent overfitting.
        average_spectrogram_length: The training dataset average spectrogram length. It is used
            to normalize the loss magnitude.
    """
    frames, stop_token, alignments, spectrogram_loss, stop_token_loss = state.model(
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
        # NOTE: We average accross `batch_size`, `num_frames` and `frame_channels` so that the loss
        # magnitude is invariant to those variables.

        # spectrogram_loss [num_frames, batch_size, frame_channels] → [1]
        spectrogram_loss_ = (spectrogram_loss.sum(dim=0) / average_spectrogram_length).mean()
        spectrogram_loss_ *= spectrogram_loss_scalar

        # stop_token_loss [num_frames, batch_size] → [1]
        stop_token_loss_ = (stop_token_loss.sum(dim=0) / average_spectrogram_length).mean()
        stop_token_loss_ = (stop_token_loss_ - stop_token_min_loss).abs() + stop_token_min_loss

        (spectrogram_loss_ + stop_token_loss_).backward()

        # NOTE: Measure the "grad_norm" before `clipper.clip()`.
        metrics.log_optimizer_metrics(state.optimizer, state.clipper, cadence=Cadence.STEP)

        # NOTE: `optimizer` will not error if there are no gradients so we check beforehand.
        params = state.optimizer.param_groups[0]["params"]
        assert all([p.grad is not None for p in params]), "No gradients found."
        state.clipper.clip()
        state.optimizer.step()
        state.step.add_(1)
        state.scheduler.step()
        state.comet.set_step(typing.cast(int, state.step.item()))

    if visualize:
        _visualize_source_vs_target(
            state, batch, frames, stop_token, alignments, dataset_type, Cadence.STEP
        )

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

    mask = lengths_to_mask(lengths).transpose(0, 1)
    values: _utils.MetricsValues = {
        **metrics.get_dataset_values(batch, reached_max),
        **metrics.get_alignment_values(batch, alignments, lengths, mask, reached_max),
        **metrics.get_loudness_values(batch, frames, mask, reached_max),
        **metrics.get_data_loader_values(data_loader),
    }
    metrics.update(values)


_BatchHandler = typing.Callable[[_State, Metrics, SpanBatch, DataLoader, DatasetType, bool], None]


def _run_worker(
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
    contexts: typing.List[typing.Tuple[Context, DatasetType, DataLoader, _BatchHandler]] = [
        (Context.TRAIN, DatasetType.TRAIN, train_loader, _run_step),
        (Context.EVALUATE, DatasetType.DEV, dev_loader, _run_step),
        (Context.EVALUATE_INFERENCE, DatasetType.DEV, dev_loader, _run_inference),
    ]
    while True:
        with set_epoch(
            comet, step=state.step.item(), steps_per_epoch=train_loader.num_steps_per_epoch
        ):
            for context, dataset_type, data_loader, handle_batch in contexts:
                with set_context(context, model=state.model, comet=comet):
                    metrics = Metrics(store, comet, state.input_encoder.speaker_encoder.vocab)
                    for i, batch in enumerate(data_loader):
                        handle_batch(state, metrics, batch, data_loader, dataset_type, i == 0)
                        if Context.TRAIN == context:
                            metrics.log(lambda l: l[-1:], type_=dataset_type, cadence=Cadence.STEP)
                    metrics.log(is_verbose=True, type_=dataset_type, cadence=Cadence.MULTI_STEP)

            if is_master():
                name = f"step_{state.step.item()}{lib.environment.PT_EXTENSION}"
                path = checkpoints_directory / name
                save(path, state.to_checkpoint(checkpoints_directory=checkpoints_directory))


def _run_app(
    checkpoints_path: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: Dataset,
    dev_dataset: Dataset,
    comet: CometMLExperiment,
    cli_config: typing.Dict[str, typing.Any],
    debug: bool = False,
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
    return run_workers(_run_worker, comet, checkpoint, checkpoints_path, train_dataset, dev_dataset)


if __name__ == "__main__":  # pragma: no cover
    app = make_app(_run_app, SPECTROGRAM_MODEL_EXPERIMENTS_PATH)
    app()
