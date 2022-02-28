import collections
import dataclasses
import itertools
import math
import typing
from functools import partial
from operator import add

import torch
import torch.distributed
from hparams import HParam, configurable

import lib
from lib.distributed import is_master
from run._config import GetLabel, get_dataset_label, get_signal_model_label
from run.data._loader import Speaker
from run.train import _utils
from run.train._utils import Timer
from run.train.signal_model._data import Batch

_GetMetrics = typing.Dict[GetLabel, float]


@dataclasses.dataclass(frozen=True)
class MetricsKey(_utils.MetricsKey):
    speaker: typing.Optional[Speaker] = None
    fft_length: typing.Optional[int] = None


MetricsValues = typing.Dict[MetricsKey, float]


class Metrics(_utils.Metrics[MetricsKey]):
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

    @staticmethod
    def _make_values():
        values: MetricsValues = collections.defaultdict(float)

        def _reduce(
            label: str,
            speaker: typing.Optional[Speaker] = None,
            fft_length: typing.Optional[int] = None,
            v: float = 0,
            op: typing.Callable[[float, float], float] = add,
        ):
            key = MetricsKey(label, speaker, fft_length)
            value = v if key not in values and op in (min, max) else values[key]
            values[key] = op(value, v)

        return values, _reduce

    def get_dataset_values(self, batch: Batch) -> MetricsValues:
        values, _reduce = self._make_values()

        for index, num_samples, num_frames in zip(
            batch.indicies,
            self._to_list(batch.target_signal.lengths),
            self._to_list(batch.spectrogram.lengths),
        ):
            span = batch.batch.spans[index]
            for speaker in [None, span.speaker]:
                _reduce(self.NUM_FRAMES, speaker, v=num_frames)
                _reduce(self.NUM_SAMPLES, speaker, v=num_samples)
                _reduce(self.NUM_SLICES, speaker, v=1)
                _reduce(self.NUM_SAMPLES_MIN_, speaker, v=num_samples, op=min)

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
        values, _reduce = self._make_values()

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
            for args in itertools.product([None, span.speaker], [None, fft_length]):
                _reduce(self.DISCRIM_NUM_FAKE_CORRECT, *args, v=fake_pred == fake_label)
                _reduce(self.DISCRIM_FAKE_LOSS_SUM, *args, v=fake_loss)
                _reduce(self.DISCRIM_NUM_REAL_CORRECT, *args, v=real_pred == real_label)
                _reduce(self.DISCRIM_REAL_LOSS_SUM, *args, v=real_loss)
                _reduce(self.GENERATOR_NUM_CORRECT, *args, v=gen_pred == fake_label)
                _reduce(self.GENERATOR_LOSS_SUM, *args, v=gen_loss)

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
        values, _reduce = self._make_values()
        for index, l1_loss, mse_loss in zip(
            batch.indicies,
            self._to_list(l1_losses),
            self._to_list(mse_losses),
        ):
            span = batch.batch.spans[index]
            for args in itertools.product([None, span.speaker], [None, fft_length]):
                _reduce(self.L1_LOSS_SUM, *args, v=l1_loss)
                _reduce(self.MSE_LOSS_SUM, *args, v=mse_loss)

        return dict(values)

    def _iter_permutations(self, select: _utils.MetricsSelect, is_verbose: bool = True):
        """Iterate over permutations of metric names and return convenience operations.

        Args:
            is_verbose: If `True`, iterate over more permutations.

        Returns:
            kwargs: Key-word arguments for formatting a metrics label.
            num_slices: The total number of slices trained on.
            reduce_: Wrapper around `self._reduce` with an update signature.
            div: Wrapper around `self._div` with an update signature.
        """
        speakers = set(key.speaker for key in self.data.keys()) if is_verbose else [None]
        fft_lengths = set(key.fft_length for key in self.data.keys())
        for args in itertools.product(speakers, fft_lengths):
            reduce_: typing.Callable[[str], float]
            reduce_ = lambda a, **k: self._reduce(MetricsKey(a, *args), select=select, **k)
            process: typing.Callable[[typing.Union[float, str]], typing.Union[float, MetricsKey]]
            process = lambda a: a if isinstance(a, float) else MetricsKey(a, *args)
            div: typing.Callable[[typing.Union[float, str], typing.Union[float, str]], float]
            div = lambda n, d, **k: self._div(process(n), process(d), select=select, **k)
            num_slices = self._reduce(MetricsKey(self.NUM_SLICES, args[0]), select=select)
            num_slices = len(fft_lengths) * num_slices if args[1] is None else num_slices
            yield dict(speaker=args[0], fft_length=args[1]), num_slices, reduce_, div

    def _get_discrim_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        metrics = {}
        for kwargs, num_slices, _, div in self._iter_permutations(select, is_verbose):
            update = {
                self.DISCRIM_FAKE_ACCURACY: div(self.DISCRIM_NUM_FAKE_CORRECT, num_slices),
                self.DISCRIM_REAL_ACCURACY: div(self.DISCRIM_NUM_REAL_CORRECT, num_slices),
                self.DISCRIM_FAKE_LOSS: div(self.DISCRIM_FAKE_LOSS_SUM, num_slices),
                self.DISCRIM_REAL_LOSS: div(self.DISCRIM_REAL_LOSS_SUM, num_slices),
                self.GENERATOR_LOSS: div(self.GENERATOR_LOSS_SUM, num_slices),
                self.GENERATOR_ACCURACY: div(self.GENERATOR_NUM_CORRECT, num_slices),
            }
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
        for kwargs, num_slices, _, div in self._iter_permutations(select, is_verbose):
            update = {
                self.L1_LOSS: div(self.L1_LOSS_SUM, num_slices),
                self.MSE_LOSS: div(self.MSE_LOSS_SUM, num_slices),
            }
            metrics.update({partial(k, **kwargs): v for k, v in update.items()})
        return metrics

    def _get_dataset_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        reduce = lambda *a, **k: self._reduce(MetricsKey(*a), select=select, **k)

        metrics = {
            self.MIN_DATA_LOADER_QUEUE_SIZE: reduce(self.DATA_QUEUE_SIZE, op=min),
            self.MIN_NUM_SAMPLES: reduce(self.NUM_SAMPLES_MIN_, op=min),
            self.AVERAGE_NUM_SAMPLES: self._div(
                MetricsKey(self.NUM_SAMPLES), MetricsKey(self.NUM_SLICES), select=select
            ),
        }

        total_frames = reduce(self.NUM_FRAMES)
        total_seconds = reduce(self.NUM_SAMPLES)
        for kwargs, _, _reduce, _ in self._iter_permutations(select, is_verbose):
            update = {
                self.FREQUENCY_NUM_SAMPLES: _reduce(self.NUM_FRAMES) / total_frames,
                self.FREQUENCY_NUM_SECONDS: _reduce(self.NUM_SAMPLES) / total_seconds,
            }
            metrics.update({partial(k, **kwargs): v for k, v in update.items()})

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
