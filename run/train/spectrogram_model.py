import collections
import dataclasses
import functools
import logging
import math
import pathlib
import platform
import random
import sqlite3
import sys
import typing
from itertools import chain

# NOTE: `comet_ml` needs to be imported before torch
import comet_ml  # type: ignore # noqa
import hparams
import hparams.hparams
import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
import typer
from hparams import HParam, HParams, add_config, configurable, get_config, parse_hparam_args
from third_party import get_parameter_norm
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter  # type: ignore
from torchnlp.utils import get_total_parameters, lengths_to_mask, tensors_to

import lib
import run
from lib.environment import load, load_most_recent_file, save
from lib.utils import flatten
from run._config import (
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    Cadence,
    DatasetType,
    Label,
    get_config_label,
    get_dataset_label,
    get_model_label,
)
from run._utils import (
    CometMLExperiment,
    Context,
    SpectrogramModelCheckpoint,
    SpectrogramModelExampleBatch,
    get_rms_level,
    get_spectrogram_example,
    maybe_make_experiment_directories,
    maybe_make_experiment_directories_from_checkpoint,
    set_context,
)

logger = logging.getLogger(__name__)
app = typer.Typer()


def _configure(more_config: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """ Configure modules for spectrogram model training, and return parameters. """
    run._config.configure()
    train_batch_size = 56
    torch.optim.Adam.__init__ = configurable(torch.optim.Adam.__init__)  # type: ignore
    config = {
        _State._get_optimizers: HParams(
            lr_multiplier_schedule=functools.partial(
                lib.optimizers.warmup_lr_multiplier_schedule, warmup=500
            ),
            # SOURCE (Tacotron 2):
            # We use the Adam optimizer [29] with β1 = 0.9, β2 = 0.999
            optimizer=torch.optim.Adam,
        ),
        _run_worker: HParams(num_steps_per_epoch=1000),
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
            dev_batch_size=train_batch_size * 4,
            bucket_size_multiplier=10,
            num_workers=8,
        ),
        _DistributedMetrics.log: HParams(num_frame_channels=run._config.NUM_FRAME_CHANNELS),
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
        lib.spectrogram_model.InputEncoder.__init__: HParams(
            phoneme_separator=run._config.PHONEME_SEPARATOR
        ),
    }
    add_config(config)
    add_config(more_config)
    lib.environment.set_seed()
    return lib.utils.nested_to_flat_dict(get_config())


@dataclasses.dataclass(frozen=True)
class _State:
    input_encoder: lib.spectrogram_model.InputEncoder
    model: torch.nn.parallel.DistributedDataParallel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    scheduler: torch.optim.lr_scheduler.LambdaLR
    comet: CometMLExperiment
    device: torch.device
    step: torch.Tensor = torch.tensor(0, dtype=torch.long)

    @staticmethod
    def _get_input_encoder(
        train_dataset: run._config.Dataset,
        dev_dataset: run._config.Dataset,
        connection: sqlite3.Connection,
        comet: CometMLExperiment,
    ) -> lib.spectrogram_model.InputEncoder:
        """ Initialize an input encoder to encode model input. """
        examples = chain(*tuple(chain(train_dataset.values(), dev_dataset.values())))
        input_encoder = lib.spectrogram_model.InputEncoder(
            flatten([e.text for e in examples]),
            run._config.DATASET_PHONETIC_CHARACTERS,
            list(train_dataset.keys()) + list(dev_dataset.keys()),
        )
        label = functools.partial(
            get_dataset_label, cadence=Cadence.STATIC, type_=DatasetType.TRAIN
        )
        stats = {
            label("grapheme_vocab_size"): input_encoder.grapheme_encoder.vocab_size,
            label("grapheme_vocab"): sorted(input_encoder.grapheme_encoder.vocab),
            label("phoneme_vocab_size"): input_encoder.phoneme_encoder.vocab_size,
            label("phoneme_vocab"): sorted(input_encoder.phoneme_encoder.vocab),
            label("num_speakers"): input_encoder.speaker_encoder.vocab_size,
            label("speakers"): sorted(input_encoder.speaker_encoder.vocab),
        }
        comet.log_parameters(stats)
        return input_encoder

    @staticmethod
    def _get_model(
        device: torch.device,
        comet: CometMLExperiment,
        input_encoder: lib.spectrogram_model.InputEncoder,
    ) -> lib.spectrogram_model.SpectrogramModel:
        """Initialize a model onto `device`.

        NOTE: Learn more about `DistributedDataParallel` here:
        https://discuss.pytorch.org/t/proper-distributeddataparallel-usage/74564
        """
        model = lib.spectrogram_model.SpectrogramModel(
            input_encoder.grapheme_encoder.vocab_size,
            input_encoder.speaker_encoder.vocab_size,
        ).to(device)
        comet.set_model_graph(str(model))
        label = get_model_label("num_parameters", Cadence.STATIC)
        comet.log_parameter(label, get_total_parameters(model))
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
        return SpectrogramModelCheckpoint(
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
        checkpoint: SpectrogramModelCheckpoint,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """ Recreate the spectrogram training state from a `checkpoint`. """
        _, _, step, encoder, model, optimizer, clipper, scheduler = dataclasses.astuple(checkpoint)
        model_ = torch.nn.parallel.DistributedDataParallel(model, [device], device)
        step = torch.tensor(step)
        return cls(encoder, model_, optimizer, clipper, scheduler, comet, device, step)

    @classmethod
    def from_dataset(
        cls,
        train_dataset: run._config.Dataset,
        dev_dataset: run._config.Dataset,
        connection: sqlite3.Connection,
        comet: CometMLExperiment,
        device: torch.device,
    ):
        """ Create spectrogram training state from the `train_dataset`. """
        input_encoder = cls._get_input_encoder(train_dataset, dev_dataset, connection, comet)
        model = cls._get_model(device, comet, input_encoder)
        distributed_model = torch.nn.parallel.DistributedDataParallel(model, [device], device)
        return cls(input_encoder, distributed_model, *cls._get_optimizers(model), comet, device)


class _DataIterator(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset: run._config.Dataset,
        database: pathlib.Path,
        bucket_size: int,
        input_encoder: lib.spectrogram_model.InputEncoder,
        comet: CometMLExperiment,
        text_length_bucket_size: int = 10,
    ):
        """Generate examples from `run._config.Dataset`.

        TODO: Filter examples based on additional data from `get_spectrogram_example`.

        Args:
            ...
            bucket_size: A batch of examples is sampled and sorted to minimize padding.
            ...
            text_length_bucket_size: The buckets size for tracking the text length distribution.
        """
        super().__init__()
        self.dataset = dataset
        self.bucket_size = bucket_size
        self.database = database
        self.input_encoder = input_encoder
        self.config = get_config()

    def __iter__(self) -> typing.Iterator[run._utils.SpectrogramModelExample]:
        # TODO: Add a method for transfering global configuration between processes without private
        # variables.
        # TODO: After the global configuration is transfered, the functions need to be rechecked
        # like for a configuration, just in case the configuration is on a new process.
        hparams.hparams._configuration = self.config
        # NOTE: `sqlite3.Connection` cannot be pickled, so it needs to be created on each process.
        connection = run._utils.connect(self.database)
        generator = run._config.get_dataset_generator(self.dataset)
        while True:
            examples = [
                get_spectrogram_example(next(generator), connection, self.input_encoder)
                for _ in range(self.bucket_size)
            ]
            yield from sorted(examples, key=lambda e: e.spectrogram.shape[0])


class _DataLoader(collections.Iterable):
    """Load and batch examples given a dataset `iterator`."""

    def __init__(self, iterator: _DataIterator, batch_size: int, device: torch.device, **kwargs):
        self.device = device
        loader = torch.utils.data.dataloader.DataLoader(
            iterator,
            pin_memory=True,
            batch_size=batch_size,
            worker_init_fn=functools.partial(
                run._utils.worker_init_fn, seed=run._config.RANDOM_SEED, device_index=device.index
            ),
            collate_fn=run._utils.batch_spectrogram_examples,
            **kwargs,
        )
        self.loader = iter(loader)
        self.num_frames = 0
        self.num_examples = 0

    @property
    def average_spectrogram_length(self) -> float:
        return self.num_frames / self.num_examples

    def __iter__(self) -> typing.Iterator[SpectrogramModelExampleBatch]:
        while True:
            batch = next(self.loader)
            self.num_frames += batch.spectrogram.lengths.float().sum().item()
            self.num_examples += batch.length
            yield typing.cast(
                SpectrogramModelExampleBatch,
                tensors_to(batch, device=self.device, non_blocking=True),
            )


@configurable
def _get_data_loaders(
    state: _State,
    train_dataset: run._config.Dataset,
    dev_dataset: run._config.Dataset,
    train_batch_size: int = HParam(),
    dev_batch_size: int = HParam(),
    bucket_size_multiplier: int = HParam(),
    num_workers: int = HParam(),
    database: pathlib.Path = run._config.DATABASE_PATH,
) -> typing.Tuple[_DataLoader, _DataLoader]:
    """ Initialize training and development data loaders.  """
    bucket_size = bucket_size_multiplier * train_batch_size
    _DataIteratorPartial = functools.partial(
        _DataIterator,
        input_encoder=state.input_encoder,
        comet=state.comet,
        bucket_size=bucket_size,
        database=database,
    )
    DataLoaderPartial = functools.partial(_DataLoader, num_workers=num_workers, device=state.device)
    return (
        DataLoaderPartial(_DataIteratorPartial(train_dataset), train_batch_size),
        DataLoaderPartial(_DataIteratorPartial(dev_dataset), dev_batch_size),
    )


@dataclasses.dataclass(frozen=True)
class _DistributedMetrics:
    """Track metrics with measurements taken on every process for every step.

    Args:
        batch_size: The batch size at each step.
        data_queue_size: This measures the data loader queue each step. This metric should be a
            positive integer indicating that the `data_loader` is loading faster than the data is
            getting ingested; otherwise, the `data_loader` is bottlenecking training by loading too
            slowly.
        predicted_frame_alignment_norm: This measures the p-norm of an alignment from a frame to the
            tokens. As the alignment per frame consolidates on a couple tokens in the input, this
            metric goes from zero to one.
        predicted_frame_alignment_std: This measures the discrete standard deviation of an alignment
            from a frame to the tokens. As the alignment per frame is localized to a couple
            sequential tokens in the input, this metric goes to zero.
        num_skips_per_speaker: In the predicted alignment, this tracks the number of tokens
            that were skipped per speaker. This could indicate that the model has issues, or that
            the dataset is flawed.
        num_tokens_per_speaker: The number of tokens per speaker for each step.
        frame_rms_level: This measures the sum of the RMS level for each frame in each step.
        text_length_bucket_size: This is a constant value bucket size for reporting the text
            length distribution.
        num_examples_per_text_length: For each text length bucket, this counts the number of
            examples.
        num_frames_per_speaker: For each speaker, this counts the number of spectrogram frames
            each step.
        num_frames_predicted: This measures the number of frames predicte each step.
        num_frames: This measures the number of frames in each step.
        num_reached_max_frames: This measures the number of predicted spectrograms that reach max
            frames each step.
        predicted_frame_rms_level: This measures the sum of the RMS level for each predicted frame
            in each step.
        spectrogram_loss: This measures the difference between the original and predicted
            spectrogram each step.
        stop_token_loss: This measures the difference between the original and predicted stop token
            distribution each step.
        stop_token_num_correct: This measures the number of correct stop token predictions each
            step.
    """

    batch_size: typing.List[float] = dataclasses.field(default_factory=list)
    data_queue_size: typing.List[float] = dataclasses.field(default_factory=list)
    predicted_frame_alignment_norm: typing.List[float] = dataclasses.field(default_factory=list)
    predicted_frame_alignment_std: typing.List[float] = dataclasses.field(default_factory=list)
    num_skips_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_tokens_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    frame_rms_level: typing.List[float] = dataclasses.field(default_factory=list)
    text_length_bucket_size: int = 10
    num_examples_per_text_length: typing.Dict[int, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_frames_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_frames_predicted: typing.List[float] = dataclasses.field(default_factory=list)
    num_frames: typing.List[float] = dataclasses.field(default_factory=list)
    num_reached_max_frames: typing.List[float] = dataclasses.field(default_factory=list)
    predicted_frame_rms_level: typing.List[float] = dataclasses.field(default_factory=list)
    spectrogram_loss: typing.List[float] = dataclasses.field(default_factory=list)
    stop_token_loss: typing.List[float] = dataclasses.field(default_factory=list)
    stop_token_num_correct: typing.List[float] = dataclasses.field(default_factory=list)

    @staticmethod
    def append(metric: typing.List[float], value: typing.Union[int, float, torch.Tensor]):
        """Append measurement to a `metric`.

        NOTE: The measurements will accrue on the master process only.
        """
        value = float(value.sum().item() if isinstance(value, torch.Tensor) else value)
        metric.append(lib.distributed.reduce_(value))

    def update_dataset_metrics(
        self,
        batch: SpectrogramModelExampleBatch,
        input_encoder: lib.spectrogram_model.InputEncoder,
    ):
        self.append(self.batch_size, batch.length)
        self.append(self.num_frames, batch.spectrogram.lengths)

        for text in flatten(lib.distributed.gather_list([len(t) for t in batch.text])):
            self.num_examples_per_text_length[text // self.text_length_bucket_size] += 1

        lambda_ = lambda t: flatten(lib.distributed.gather_list(t.squeeze().tolist()))
        iterator = zip(lambda_(batch.encoded_speaker.tensor), lambda_(batch.spectrogram.lengths))
        for speaker_index, num_frames in iterator:
            speaker = input_encoder.speaker_encoder.index_to_token[speaker_index]
            self.num_frames_per_speaker[speaker] += num_frames

    def update_alignment_metrics(
        self,
        alignments: torch.Tensor,
        spectrogram_mask: torch.Tensor,
        token_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        speakers: torch.Tensor,
        input_encoder: lib.spectrogram_model.InputEncoder,
        norm: float = math.inf,
    ):
        """
        TODO: Reduce the boiler plate required to track a metric per speaker.

        Args:
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            spectrogram_mask (torch.BoolTensor [num_frames, batch_size])
            token_mask (torch.BoolTensor [batch_size, num_tokens])
            num_tokens (torch.LongTensor [batch_size])
            speakers (torch.LongTensor [batch_size])
            ...
        """
        if spectrogram_mask.sum() == 0 or token_mask.sum() == 0:
            return

        mask = lambda t: t.masked_select(spectrogram_mask)
        weighted_std = lib.utils.get_weighted_std(alignments, dim=2)
        self.append(self.predicted_frame_alignment_std, mask(weighted_std))
        self.append(self.predicted_frame_alignment_norm, mask(alignments.norm(norm, dim=2)))

        num_skipped = run._utils.get_num_skipped(alignments, token_mask, spectrogram_mask)

        iterate = lambda t: flatten(lib.distributed.gather_list(t.squeeze().tolist()))
        iterator = zip(iterate(speakers), iterate(num_skipped), iterate(num_tokens))
        for speaker_index, _num_skipped, _num_tokens in iterator:
            speaker = input_encoder.speaker_encoder.index_to_token[speaker_index]
            self.num_skips_per_speaker[speaker] += _num_skipped
            self.num_tokens_per_speaker[speaker] += _num_tokens

    def update_rms_level_metrics(
        self,
        target_spectrogram: torch.Tensor,
        predicted_spectrogram: torch.Tensor,
        target_mask: typing.Optional[torch.Tensor] = None,
        predicted_mask: typing.Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            target (torch.FloatTensor [num_frames, batch_size, frame_channels])
            predicted (torch.FloatTensor [num_frames, batch_size, frame_channels])
            target_mask (torch.FloatTensor [num_frames, batch_size])
            predicted_mask (torch.FloatTensor [num_frames, batch_size])
            **kwargs: Additional key word arguments passed to `get_rms_level`.
        """
        lambda_ = lambda t, m: get_rms_level(t, m, **kwargs)
        self.append(self.frame_rms_level, lambda_(target_spectrogram, target_mask))
        self.append(self.predicted_frame_rms_level, lambda_(predicted_spectrogram, predicted_mask))

    def update_stop_token_accuracy(
        self,
        target: torch.Tensor,
        predicted_logits: torch.Tensor,
        stop_threshold: float,
        mask: torch.Tensor,
    ):
        """
        Args:
            target (torch.FloatTensor [num_frames, batch_size])
            predicted_logits (torch.FloatTensor [num_frames, batch_size])
            stop_threshold
            mask (torch.BoolTensor [num_frames, batch_size])
        """
        bool_ = lambda t: (t > stop_threshold).masked_select(mask)
        is_correct = bool_(target) == bool_(torch.sigmoid(predicted_logits))
        self.append(self.stop_token_num_correct, is_correct)

    def update_data_queue_size(self, data_loader: _DataLoader):
        # NOTE: `qsize` is not implemented on MacOS, learn more:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
        if (
            isinstance(data_loader.loader, _MultiProcessingDataLoaderIter)
            and platform.system() != "Darwin"
        ):
            self.append(
                self.data_queue_size,
                typing.cast(_MultiProcessingDataLoaderIter, data_loader.loader)._data_queue.qsize(),
            )

    @configurable
    def log(
        self,
        comet: CometMLExperiment,
        reduce_: typing.Callable[[typing.List[float]], float],
        dataset_type: DatasetType,
        cadence: Cadence,
        num_frame_channels=HParam(),
    ):
        """Log `self` to `comet`.

        Args:
            comet
            reduce_: If there is a list of measurements, this callable is used to reduce it.
            ...
        """
        div = lambda n, d, r=reduce_: r(n) / r(d) if len(n) > 0 and len(d) > 0 else None
        power_to_db = lambda t: lib.audio.power_to_db(torch.tensor(reduce_(t)))
        predicted_rms = div(self.predicted_frame_rms_level, self.num_frames_predicted, power_to_db)
        rms = div(self.frame_rms_level, self.num_frames, power_to_db)

        Stats = typing.Dict[str, typing.Optional[float]]
        model_stats: Stats = {
            "alignment_norm": div(self.predicted_frame_alignment_norm, self.num_frames_predicted),
            "alignment_std": div(self.predicted_frame_alignment_std, self.num_frames_predicted),
            "average_relative_speed": div(self.num_frames_predicted, self.num_frames),
            "stop_token_accuracy": div(self.stop_token_num_correct, self.num_frames),
            "stop_token_loss": div(self.stop_token_loss, self.num_frames),
            "reached_max_frames": div(self.num_reached_max_frames, self.batch_size),
            "spectrogram_loss": div(self.spectrogram_loss, self.num_frames * num_frame_channels),
            "average_predicted_rms_level": predicted_rms,
            "average_rms_level_delta": (
                predicted_rms - rms if predicted_rms is not None and rms is not None else None
            ),
        }
        dataset_stats: Stats = {
            "data_loader_queue_size": div(self.data_queue_size, [1] * len(self.data_queue_size)),
            "average_rms_level": rms,
        }
        partial = functools.partial(get_dataset_label, type_=dataset_type)
        iterator: typing.List[typing.Tuple[Stats, typing.Callable[..., Label]]]
        iterator = [(model_stats, get_model_label), (dataset_stats, partial)]
        for stats, get_label in iterator:
            for name, value in stats.items():
                if value is not None:
                    comet.log_metric(get_label(name, cadence=cadence), value)

        for speaker, count in self.num_frames_per_speaker.items():
            label = partial("frequency", cadence=cadence, speaker=speaker)
            comet.log_metric(label, count / sum(self.num_frames_per_speaker.values()))

        for bucket, count in self.num_examples_per_text_length.items():
            lower = bucket * self.text_length_bucket_size
            upper = (bucket + 1) * self.text_length_bucket_size
            label = partial(f"{lower}_{upper}", cadence=cadence)
            comet.log_metric(label, count / sum(self.num_examples_per_text_length.values()))

        zip_ = zip(self.num_tokens_per_speaker.items(), self.num_skips_per_speaker.values())
        for (speaker, num_tokens), num_skips in zip_:
            comet.log_metric(get_model_label("skips", cadence, speaker), num_skips / num_tokens)


def _visualize_source_vs_target(
    state: _State,
    batch: SpectrogramModelExampleBatch,
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
    item = random.randint(0, batch.length - 1)
    spectrogram_length = int(batch.spectrogram.lengths[0, item].item())
    text_length = int(batch.encoded_text.lengths[0, item].item())

    # predicted_spectrogram, gold_spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:spectrogram_length, item]
    gold_spectrogram = batch.spectrogram.tensor[:spectrogram_length, item]

    predicted_delta = abs(gold_spectrogram - predicted_spectrogram)
    predicted_alignments = predicted_alignments[:spectrogram_length, item, :text_length]
    predicted_stop_token = predicted_stop_token[:spectrogram_length, item]
    model = functools.partial(get_model_label, cadence=cadence)
    dataset = functools.partial(get_dataset_label, cadence=cadence, type_=dataset_type)
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
    metrics: _DistributedMetrics,
    batch: SpectrogramModelExampleBatch,
    data_loader: _DataLoader,
    dataset_type: DatasetType,
    visualize: bool = False,
    spectrogram_loss_scalar: float = HParam(),
    stop_token_min_loss: float = HParam(),
):
    """Run the `model` on the next batch from `data_loader`, and maybe update it.

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
        spectrogram_loss_scalar: This scales the spectrogram loss by some value.
        stop_token_min_loss: This thresholds the stop token loss to prevent overfitting.
    """
    frames, stop_token, alignment, spectrogram_loss, stop_token_loss = state.model(
        tokens=batch.encoded_text.tensor,
        speaker=batch.encoded_speaker.tensor,
        target_frames=batch.spectrogram.tensor,
        target_stop_token=batch.stop_token.tensor,
        num_tokens=batch.encoded_text.lengths,
        target_lengths=batch.spectrogram.lengths,
        mode=lib.spectrogram_model.Mode.FORWARD,
    )

    if state.model.training:
        state.optimizer.zero_grad()

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

        log_metric = lambda n, v: state.comet.log_metric(
            get_model_label(n, cadence=Cadence.STEP), v
        )
        log_metric("grad_norm/two", get_parameter_norm(state.model.parameters(), 2))
        log_metric("grad_norm/inf", get_parameter_norm(state.model.parameters(), math.inf))
        log_metric("grad_norm/max_norm", state.clipper.max_norm)
        [log_metric("parameter_%d/lr", g["lr"]) for i, g in enumerate(state.optimizer.param_groups)]

        state.clipper.clip()
        state.optimizer.step()
        state.step.add_(1)
        state.scheduler.step()
        state.comet.set_step(typing.cast(int, state.step.item()))

    if visualize:
        _visualize_source_vs_target(
            state, batch, frames, stop_token, alignment, dataset_type, Cadence.STEP
        )

    # Update metrics, and log those updates.
    metrics.update_dataset_metrics(batch, state.input_encoder)
    metrics.update_alignment_metrics(
        alignment,
        batch.spectrogram_mask.tensor,
        batch.encoded_text_mask.tensor,
        batch.encoded_text.lengths,
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
    metrics.log(state.comet, lambda l: l[-1], dataset_type, Cadence.STEP)


def _visualize_inferred(
    state: _State,
    batch: SpectrogramModelExampleBatch,
    predicted_spectrogram: torch.Tensor,
    predicted_stop_token: torch.Tensor,
    predicted_alignments: torch.Tensor,
    dataset_type: DatasetType,
    cadence: Cadence,
):
    """Run in inference mode and visualize results.

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
    item = random.randint(0, batch.length - 1)
    num_frames = int(batch.spectrogram.lengths[0, item].item())
    text_length = int(batch.encoded_text.lengths[0, item].item())
    # spectrogram [num_frames, frame_channels]
    predicted_spectrogram = predicted_spectrogram[:num_frames, item]
    # gold_spectrogram [num_frames, frame_channels]
    gold_spectrogram = batch.spectrogram.tensor[:num_frames, item]
    predicted_alignments = predicted_alignments[:num_frames, item, :text_length]

    model = functools.partial(get_model_label, cadence=cadence)
    dataset = functools.partial(get_dataset_label, cadence=cadence, type_=dataset_type)
    figures = {
        dataset("gold_spectrogram"): lib.visualize.plot_mel_spectrogram(gold_spectrogram),
        model("predicted_spectrogram"): lib.visualize.plot_mel_spectrogram(predicted_spectrogram),
        model("alignment"): lib.visualize.plot_alignments(predicted_alignments),
        model("stop_token"): lib.visualize.plot_logits(predicted_stop_token[:num_frames, item]),
    }
    state.comet.log_figures(figures)
    audio = {
        "predicted_griffin_lim_audio": lib.audio.griffin_lim(predicted_spectrogram.cpu().numpy()),
        "gold_griffin_lim_audio": lib.audio.griffin_lim(gold_spectrogram.numpy()),
        "gold_audio": batch.audio[item].numpy(),
    }
    state.comet.log_html_audio(
        audio=audio,
        context=state.comet.context,
        text=batch.text[item],
        speaker=batch.speaker[item],
        predicted_loudness=get_rms_level(predicted_spectrogram.unsqueeze(1)),
        gold_loudness=get_rms_level(gold_spectrogram.unsqueeze(1)),
    )


def _run_inference(
    state: _State,
    metrics: _DistributedMetrics,
    batch: SpectrogramModelExampleBatch,
    data_loader: _DataLoader,
    dataset_type: DatasetType,
    visualize: bool = False,
):
    """Run the model in inference mode, and measure it's results.

    TODO: Consider calling `update_dataset_metrics`, and filtering the examples which overflowed.

    Args:
        ...
        visualize: If `True` visualize the results with `comet`.
    """
    frames, stop_tokens, alignments, lengths, reached_max = state.model(
        batch.encoded_text.tensor,
        batch.encoded_speaker.tensor,
        batch.encoded_text.lengths,
        mode=lib.spectrogram_model.Mode.INFER,
    )

    if visualize:
        _visualize_inferred(
            state, batch, frames, stop_tokens, alignments, dataset_type, Cadence.STEP
        )

    if lengths.numel() > 0:
        # NOTE: Remove predictions that diverged (reached max) as to not skew other metrics. We
        # count these sequences seperatly with `reached_max_frames`.
        bool_ = ~reached_max.view(-1)
        max_frames = int(lengths[:, bool_].max())
        max_tokens = int(batch.encoded_text.lengths[:, bool_].max())
        metrics.append(metrics.batch_size, batch.length - reached_max.sum().item())
        metrics.append(metrics.num_frames, batch.spectrogram.lengths[:, bool_])
        metrics.append(metrics.num_frames_predicted, lengths[:, bool_])
        metrics.update_rms_level_metrics(
            batch.spectrogram.tensor[:max_frames, bool_],
            frames[:max_frames, bool_],
            batch.spectrogram_mask.tensor[:max_frames, bool_],
            lengths_to_mask(lengths[:, bool_], device=lengths.device).transpose(0, 1),
        )
        metrics.update_alignment_metrics(
            alignments[:max_frames, bool_, :max_tokens],
            batch.spectrogram_mask.tensor[:max_frames, bool_],
            batch.encoded_text_mask.tensor[:max_tokens, bool_],
            batch.encoded_text.lengths[:max_tokens, bool_],
            batch.encoded_speaker.tensor[:, bool_],
            state.input_encoder,
        )
    metrics.update_data_queue_size(data_loader)
    metrics.append(metrics.num_reached_max_frames, reached_max)


_BatchHandler = typing.Callable[
    [_State, _DistributedMetrics, SpectrogramModelExampleBatch, _DataLoader, DatasetType, bool],
    None,
]


@configurable
def _run_worker(
    device_index: int,
    run_root: pathlib.Path,
    checkpoints_directory: pathlib.Path,
    checkpoint: typing.Optional[pathlib.Path],
    train_dataset: run._config.Dataset,
    dev_dataset: run._config.Dataset,
    comet_partial: typing.Callable[..., CometMLExperiment],
    config: typing.Dict[str, typing.Any],
    num_steps_per_epoch: int = HParam(),
) -> typing.NoReturn:
    """ Train and evaluate the spectrogram model on a loop. """
    lib.environment.set_basic_logging_config(device_index)
    device = run._utils.init_distributed(device_index)
    comet = comet_partial(disabled=not lib.distributed.is_master(), auto_output_logging=False)
    _configure(config)
    connection = run._utils.connect(run._config.DATABASE_PATH)
    if checkpoint is None:
        state = _State.from_dataset(train_dataset, dev_dataset, connection, comet, device)
    else:
        checkpoint_ = typing.cast(SpectrogramModelCheckpoint, load(checkpoint, device=device))
        state = _State.from_checkpoint(checkpoint_, comet, device)
    train_loader, dev_loader = _get_data_loaders(state, train_dataset, dev_dataset)
    _set_context = functools.partial(set_context, model=state.model, comet=comet)
    while True:
        epoch = int(state.step.item() // num_steps_per_epoch)
        logger.info("Running Epoch %d, Step %d", epoch, state.step.item())
        comet.log_current_epoch(epoch)

        iterator: typing.List[typing.Tuple[Context, DatasetType, _DataLoader, _BatchHandler]] = [
            (Context.TRAIN, DatasetType.TRAIN, train_loader, _run_step),
            (Context.EVALUATE, DatasetType.DEV, dev_loader, _run_step),
            (Context.EVALUATE_INFERENCE, DatasetType.DEV, dev_loader, _run_inference),
        ]
        for context, dataset_type, data_loader, handle_batch in iterator:
            with _set_context(context):
                # TODO: `metrics` are not propagated and we might want to incorperate that for
                # dataset metrics like `num_frames_per_speaker` or `num_examples_per_text_length`.
                # In order to do so, we'd also need to checkpoint those metrics.
                metrics = _DistributedMetrics()
                for i, batch in zip(range(num_steps_per_epoch), data_loader):
                    handle_batch(state, metrics, batch, data_loader, dataset_type, i == 0)
                    metrics.log(comet, lambda l: l[-1], dataset_type, Cadence.STEP)
                metrics.log(comet, sum, dataset_type, Cadence.MULTI_STEP)

        save(
            checkpoints_directory / f"step_{state.step.item()}{lib.environment.PT_EXTENSION}",
            state.to_checkpoint(checkpoints_directory=checkpoints_directory),
        )
        comet.log_epoch_end(epoch)


def _run(
    run_root: pathlib.Path,
    checkpoints_path: pathlib.Path,
    config: typing.Dict[str, typing.Any],
    comet: CometMLExperiment,
    checkpoint: typing.Optional[pathlib.Path] = None,
    minimum_disk_space: float = 0.2,
):
    """ Run spectrogram model training. """
    lib.environment.check_module_versions()
    lib.environment.assert_enough_disk_space(minimum_disk_space)

    # NOTE: Load, preprocess, and cache dataset values.
    connection = run._utils.connect(run._config.DATABASE_PATH)
    dataset = run._config.get_dataset()
    all_examples = lambda: list(chain(*tuple(dataset.values())))
    run._utils.normalize_audio(dataset)
    run._utils.update_audio_file_metadata(connection, [e.audio_path for e in all_examples()])
    run._utils.handle_null_alignments(connection, dataset)
    train_dataset, dev_dataset = run._config.split_dataset(dataset)
    comet.log_parameters(run._utils.get_dataset_stats(train_dataset, dev_dataset))

    return lib.distributed.spawn(
        _run_worker.get_configured_partial(),  # type: ignore
        args=(
            run_root,
            checkpoints_path,
            checkpoint,
            train_dataset,
            dev_dataset,
            functools.partial(run._utils.CometMLExperiment, experiment_key=comet.get_key()),
            config,
        ),
    )


def _setup(
    comet: CometMLExperiment, config: typing.List[str]
) -> typing.Tuple[typing.Dict[str, typing.Any], lib.environment.RecordStandardStreams]:
    """ Setup the environment logging and modules. """
    lib.environment.set_basic_logging_config()
    recorder = lib.environment.RecordStandardStreams()
    logger.info("Command line args: %s", str(sys.argv))  # NOTE: Ensure command line args are logged
    parsed = parse_hparam_args(config)
    # TODO: For checkpointed runs, should we triple check the same parameters are getting
    # configured? Should we throw an error if not? Or should we create a new experiment, and ensure
    # that each experiments parameters are immutable?
    parameters = _configure(parsed)
    params = {get_config_label(k): v for k, v in parameters.items()}
    comet.log_parameters(params)
    return parsed, recorder


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def resume(
    context: typer.Context,
    checkpoint: typing.Optional[pathlib.Path] = typer.Argument(
        None, help="Checkpoint file to restart training from."
    ),
):
    """Resume training from CHECKPOINT. If CHECKPOINT is not given, the most recent checkpoint
    file is loaded."""
    pattern = str(SPECTROGRAM_MODEL_EXPERIMENTS_PATH / f"**/*{lib.environment.PT_EXTENSION}")
    if checkpoint:
        loaded = load(checkpoint)
    else:
        checkpoint, loaded = load_most_recent_file(pattern, load)
    checkpoint_ = typing.cast(SpectrogramModelCheckpoint, loaded)
    comet = run._utils.CometMLExperiment(experiment_key=checkpoint_.comet_experiment_key)
    config, recorder = _setup(comet, context.args)
    paths = maybe_make_experiment_directories_from_checkpoint(checkpoint_, recorder)
    _run(*paths, config, comet, checkpoint)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def start(
    context: typer.Context,
    project: str = typer.Argument(..., help="Experiment project name."),
    name: str = typer.Argument("", help="Experiment name."),
    tags: typing.List[str] = typer.Argument([], help="Experiment tags."),
):
    """ Start a training run in PROJECT named NAME with TAGS. """
    comet = run._utils.CometMLExperiment(project_name=project)
    comet.set_name(name)
    comet.add_tags(tags)
    config, recorder = _setup(comet, context.args)
    experiment_root = SPECTROGRAM_MODEL_EXPERIMENTS_PATH / lib.environment.bash_time_label()
    run_root, checkpoints_path = maybe_make_experiment_directories(experiment_root, recorder)
    comet.log_other(run._config.get_environment_label("directory"), str(run_root))
    _run(run_root, checkpoints_path, config, comet)


if __name__ == "__main__":  # pragma: no cover
    app()
