import collections
import itertools
import math
import typing
from functools import partial

import torch
import torch.distributed
from hparams import HParam, configurable

import lib
import run
from lib.distributed import is_master
from run._config import GetLabel, get_dataset_label, get_signal_model_label
from run.train import _utils
from run.train._utils import CometMLExperiment, MetricsValues, Timer
from run.train.signal_model._data import Batch

_GetMetrics = typing.Dict[GetLabel, float]


class Metrics(_utils.Metrics):
    """
    Vars:
        ...
        AVERAGE_NUM_SAMPLES: The average number of samples trained on.
        FREQUENCY_NUM_SAMPLES: The frequency of each speaker based on the number of samples.
        FREQUENCY_NUM_SECONDS: The frequency of each speaker based on the number of seconds.
        MIN_DATA_LOADER_QUEUE_SIZE: The minimum data loader queue size.
        DISCRIM_ACCURACY: The number of correct discriminations over the total
            discriminations.
        GENERATOR_LOSS: Given the predicted frames, this loss measures how confident the
            discriminator is. The more confident the higher the loss.
        GRADIENT_INFINITY_NORM: The total infinity norm of all parameter gradients.
        GRADIENT_TWO_NORM: The total 2-norm of all parameter gradients.
        GRADIENT_MAX_NORM: The maximum gradient norm used for clipping gradients.
        L1_LOSS: The l1 loss between a target and predicted spectrogram.
        LR: The learning rate.
        MSE_LOSS: The l2 loss between a target and predicted spectrogram.
    """

    (
        DATA_QUEUE_SIZE,
        DISCRIM_NUM_FAKE_CORRECT,
        DISCRIM_FAKE_LOSS_SUM,
        DISCRIM_NUM_REAL_CORRECT,
        DISCRIM_REAL_LOSS_SUM,
        GENERATOR_LOSS_SUM,
        GENERATOR_NUM_CORRECT,
        L1_LOSS_SUM,
        MSE_LOSS_SUM,
        NUM_SAMPLES_MIN_,
        NUM_FRAMES,
        NUM_SAMPLES,
        NUM_SLICES,
        *_,
    ) = tuple([str(i) for i in range(100)])

    MIN_NUM_SAMPLES = partial(get_dataset_label, "min_num_samples")
    AVERAGE_NUM_SAMPLES = partial(get_dataset_label, "average_num_samples")
    FREQUENCY_NUM_SAMPLES = partial(get_dataset_label, "frequency/num_samples")
    FREQUENCY_NUM_SECONDS = partial(get_dataset_label, "frequency/num_seconds")

    DISCRIM_FAKE_ACCURACY = partial(get_signal_model_label, "discriminator_fake_accuracy")
    DISCRIM_REAL_ACCURACY = partial(get_signal_model_label, "discriminator_real_accuracy")
    DISCRIM_FAKE_LOSS = partial(get_signal_model_label, "discriminator_fake_loss")
    DISCRIM_REAL_LOSS = partial(get_signal_model_label, "discriminator_real_loss")
    GENERATOR_LOSS = partial(get_signal_model_label, "generator_loss")
    GENERATOR_ACCURACY = partial(get_signal_model_label, "generator_accuracy")
    L1_LOSS = partial(get_signal_model_label, "l1_loss")
    MSE_LOSS = partial(get_signal_model_label, "mse_loss")

    @configurable
    def __init__(
        self,
        comet: CometMLExperiment,
        speakers: typing.List[run.data._loader.Speaker],
        fft_lengths: typing.List[int] = HParam(),
        **kwargs,
    ):
        super().__init__(comet, **kwargs)
        self.speakers = speakers
        self.fft_lengths = fft_lengths

    def get_dataset_values(self, batch: Batch) -> MetricsValues:
        values: typing.Dict[str, float] = collections.defaultdict(float)

        for index, num_samples, num_frames in zip(
            batch.indicies,
            self._to_list(batch.target_signal.lengths),
            self._to_list(batch.spectrogram.lengths),
        ):
            span = batch.batch.spans[index]
            assert span.speaker in self.speakers
            for suffix in ["", f"/{span.speaker.label}"]:
                format_ = lambda s: f"{s}{suffix}"
                values[format_(self.NUM_FRAMES)] += num_frames
                values[format_(self.NUM_SAMPLES)] += num_samples
                values[format_(self.NUM_SLICES)] += 1
                label = format_(self.NUM_SAMPLES_MIN_)
                values[label] = min(values[label], num_samples) if label in values else num_samples

        return dict(values)

    @configurable
    def get_discrim_values(
        self,
        fft_length: int,
        batch: Batch,
        real_logits: torch.Tensor,
        fake_logits: torch.Tensor,
        generator_logits: torch.Tensor,
        discrim_real_losses: torch.Tensor,
        discrim_fake_losses: torch.Tensor,
        generator_losses: torch.Tensor,
        real_label: bool = HParam(),
        fake_label: bool = HParam(),
        threshold: float = HParam(),
    ) -> MetricsValues:
        """
        Args:
            fft_length: The resolution of the spectrogram discriminated.
            ...
            real_logits (torch.FloatTensor [batch_size]): The logits predicted given the target
                spectrogram.
            fake_logits (torch.FloatTensor [batch_size]): The logits predicted given the generated
                spectrogram.
            generator_logits (torch.FloatTensor [batch_size]): The logits predicted given the
                generated spectrogram after the discriminator weights have been updated.
            discrim_real_losses (torch.FloatTensor [batch_size]): The discriminator loss given
                a target spectrogram.
            discrim_fake_losses (torch.FloatTensor [batch_size]): The discriminator loss given
                a generated spectrogram.
            generator_losses (torch.FloatTensor [batch_size]): The negated discriminator loss given
                a generated spectrogram after the discriminator weights have been updated.
            real_label: The boolean value assigned to real inputs.
            fake_label: The boolean value assigned to fake inputs.
            threshold: Given a probability, this threshold decides if the input is real or fake.
        """
        values: typing.Dict[str, float] = collections.defaultdict(float)
        assert fft_length in self.fft_lengths
        prefixes = ["", f"{fft_length}/"]
        for index, fake_pred, real_pred, gen_pred, real_loss, fake_loss, gen_loss in zip(
            batch.indicies,
            self._to_list(torch.sigmoid(fake_logits) > threshold),
            self._to_list(torch.sigmoid(real_logits) > threshold),
            self._to_list(torch.sigmoid(generator_logits) > threshold),
            self._to_list(discrim_real_losses),
            self._to_list(discrim_fake_losses),
            self._to_list(generator_losses),
        ):
            span = batch.batch.spans[index]
            assert span.speaker in self.speakers
            for prefix, suffix in itertools.product(prefixes, ["", f"/{span.speaker.label}"]):
                format_ = lambda s: f"{prefix}{s}{suffix}"
                values[format_(self.DISCRIM_NUM_FAKE_CORRECT)] += fake_pred == fake_label
                values[format_(self.DISCRIM_FAKE_LOSS_SUM)] += fake_loss
                values[format_(self.DISCRIM_NUM_REAL_CORRECT)] += real_pred == real_label
                values[format_(self.DISCRIM_REAL_LOSS_SUM)] += real_loss
                values[format_(self.GENERATOR_NUM_CORRECT)] += gen_pred == fake_label
                values[format_(self.GENERATOR_LOSS_SUM)] += gen_loss

        return dict(values)

    def get_model_values(
        self,
        fft_length: int,
        batch: Batch,
        l1_losses: torch.Tensor,
        mse_losses: torch.Tensor,
    ) -> MetricsValues:
        """
        Args:
            fft_length: The resolution of the spectrogram discriminated.
            ...
            l1_losses (torch.FloatTensor [batch_size]): The absolute difference between the
                predicted and target spectrogram.
            mse_losses (torch.FloatTensor [batch_size]): The squared difference between the
                predicted and target spectrogram.
        """
        values: typing.Dict[str, float] = collections.defaultdict(float)
        assert fft_length in self.fft_lengths
        prefixes = ["", f"{fft_length}/"]
        for index, l1_loss, mse_loss in zip(
            batch.indicies,
            self._to_list(l1_losses),
            self._to_list(mse_losses),
        ):
            span = batch.batch.spans[index]
            assert span.speaker in self.speakers
            for prefix, suffix in itertools.product(prefixes, ["", f"/{span.speaker.label}"]):
                format_ = lambda s: f"{prefix}{s}{suffix}"
                values[format_(self.L1_LOSS_SUM)] += l1_loss
                values[format_(self.MSE_LOSS_SUM)] += mse_loss

        return dict(values)

    def _iter_permutations(self, select: _utils.MetricsSelect, is_verbose: bool = True):
        """Iterate over permutations of metric names and return convenience operations.

        Args:
            is_verbose: If `True`, iterate over more permutations.
        """
        speakers = itertools.chain([None], self.speakers if is_verbose else [])
        fft_lengths = itertools.chain([None], self.fft_lengths)
        for fft_length, speaker in itertools.product(fft_lengths, speakers):
            suffix = "" if speaker is None else f"/{speaker.label}"
            prefix = "" if fft_length is None else f"{fft_length}/"
            format_ = lambda s: f"{prefix}{s}{suffix}" if isinstance(s, str) else s
            reduce_: typing.Callable[[str], float]
            reduce_ = lambda k: self._reduce(format_(k), select=select)
            div: typing.Callable[[typing.Union[float, str], typing.Union[float, str]], float]
            div = lambda n, d: self._div(format_(n), format_(d), select=select)
            yield fft_length, speaker, suffix, reduce_, div

    def _get_discrim_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        metrics = {}
        for fft_length, speaker, suffix, _, div in self._iter_permutations(select, is_verbose):
            num_slices = self._reduce(f"{self.NUM_SLICES}{suffix}", select=select)
            num_slices = len(self.fft_lengths) * num_slices if fft_length is None else num_slices
            update = {
                self.DISCRIM_FAKE_ACCURACY: div(self.DISCRIM_NUM_FAKE_CORRECT, num_slices),
                self.DISCRIM_REAL_ACCURACY: div(self.DISCRIM_NUM_REAL_CORRECT, num_slices),
                self.DISCRIM_FAKE_LOSS: div(self.DISCRIM_FAKE_LOSS_SUM, num_slices),
                self.DISCRIM_REAL_LOSS: div(self.DISCRIM_REAL_LOSS_SUM, num_slices),
                self.GENERATOR_LOSS: div(self.GENERATOR_LOSS_SUM, num_slices),
                self.GENERATOR_ACCURACY: div(self.GENERATOR_NUM_CORRECT, num_slices),
            }
            kwargs = dict(speaker=speaker, fft_length=fft_length)
            metrics.update({partial(k, **kwargs): v for k, v in update.items()})
        return metrics

    def _get_model_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        """
        TODO: The `L1_LOSS_SUM` and `MSE_LOSS_SUM` need to be normalized by the number of frames
        instead of the number of examples, in order to stay consistent with the training loss.
        TODO: Answer... Do spectrograms with different resolutions have the same total energy? And
        how would that impact our normalization approach?
        """
        metrics = {}
        for fft_length, speaker, suffix, _, div in self._iter_permutations(select, is_verbose):
            num_slices = self._reduce(f"{self.NUM_SLICES}{suffix}", select=select)
            num_slices = len(self.fft_lengths) * num_slices if fft_length is None else num_slices
            update = {
                self.L1_LOSS: div(self.L1_LOSS_SUM, num_slices),
                self.MSE_LOSS: div(self.MSE_LOSS_SUM, num_slices),
            }
            kwargs = dict(speaker=speaker, fft_length=fft_length)
            metrics.update({partial(k, **kwargs): v for k, v in update.items()})
        return metrics

    def _get_dataset_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        reduce = partial(self._reduce, select=select)
        metrics = {
            self.MIN_DATA_LOADER_QUEUE_SIZE: reduce(self.DATA_QUEUE_SIZE, op=min),
            self.MIN_NUM_SAMPLES: reduce(self.NUM_SAMPLES_MIN_, op=min),
            self.AVERAGE_NUM_SAMPLES: self._div(self.NUM_SAMPLES, self.NUM_SLICES, select=select),
        }

        total_frames = reduce(self.NUM_FRAMES)
        total_seconds = reduce(self.NUM_SAMPLES)
        for _, speaker, _, _reduce, _ in self._iter_permutations(select, is_verbose):
            update = {
                self.FREQUENCY_NUM_SAMPLES: _reduce(self.NUM_FRAMES) / total_frames,
                self.FREQUENCY_NUM_SECONDS: _reduce(self.NUM_SAMPLES) / total_seconds,
            }
            metrics.update({partial(k, speaker=speaker): v for k, v in update.items()})
        return metrics

    def log(
        self,
        select: _utils.MetricsSelect = lib.utils.identity,
        timer: typing.Optional[Timer] = None,
        is_verbose: bool = False,
        **kwargs,
    ):
        """Log metrics to `self.comet`.

        Args:
            ...
            is_verbose: Comet will throttle experiments or lag if too many metrics are logged.
                This flag controls the number of metrics logged.
            **kwargs: Key-word arguments passed to `get_model_label` and `get_dataset_label`.
        """
        if is_master():
            record_event = lambda e: None if timer is None else timer.record_event(e)
            record_event(Timer.REDUCE_METRICS)
            metrics = {
                **self._get_model_metrics(select=select, is_verbose=is_verbose),
                **self._get_dataset_metrics(select=select, is_verbose=is_verbose),
                **self._get_discrim_metrics(select=select, is_verbose=is_verbose),
            }
            record_event(Timer.LOG_METRICS)
            self.comet.log_metrics(
                {k(**kwargs): v for k, v in metrics.items() if not math.isnan(v)}
            )
