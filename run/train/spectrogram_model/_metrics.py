import collections
import collections.abc
import dataclasses
import math
import typing
from functools import partial
from operator import add

import config as cf
import torch
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data

import lib
from lib.audio import amp_to_power, db_to_power, power_spectrogram_to_framed_rms, power_to_db
from lib.distributed import is_master
from run._config import GetLabel, get_dataset_label, get_model_label
from run._models.spectrogram_model import Preds, SpectrogramModel
from run.data._loader import Speaker
from run.train import _utils
from run.train._utils import Context, Timer
from run.train.spectrogram_model._data import Batch


def get_hang_time(
    stop_token: torch.Tensor, mask: torch.Tensor, num_frames: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Get the number of frames after stop token reached `threshold`, this is metric is closely
    aligned to the number of non-speech frames there at the end.

    Args:
        stop_token (torch.FloatTensor [num_frames, batch_size])
        mask (torch.FloatTensor [batch_size, num_frames])
        num_frames (torch.FloatTensor [batch_size])

    Returns:
        torch.FloatTensor [batch_size]
    """
    if stop_token.numel() == 0:
        return torch.empty(0, device=stop_token.device)

    # [num_frames, batch_size]
    is_potential_stop = torch.sigmoid(stop_token) >= threshold
    is_potential_stop = is_potential_stop.masked_fill(~mask.transpose(0, 1), 0)
    # [num_frames, batch_size] → [batch_size]
    had_potential_stop = is_potential_stop.sum(dim=0) != 0
    first_potential_stop = is_potential_stop.long().argmax(dim=0)
    return (num_frames - first_potential_stop - 1) * had_potential_stop


"""
TODO: In order to support `get_rms_level`, the signal used to compute the spectrogram should
be padded appropriately. At the moment, the spectrogram is padded such that it's length
is a multiple of the signal length.

The current spectrogram does not work with `get_rms_level` because the boundary samples are
underrepresented in the resulting spectrogram. Except for the boundaries, every sample
appears 4 times in the resulting spectrogram assuming there is a 75% overlap between frames. The
boundary samples appear less than 4 times. For example, the first and last sample appear
only once in the spectrogram.

In order to correct this, we'd need to add 3 additional frames to either end of the spectrogram.
Unfortunately, adding a constant number of frames to either end is incompatible with the orignal
impelementation where the resulting spectrogram length is a multiple of the signal length.

Fortunately, this requirement can be relaxed. The signal model can be adjusted to accept 3
additional frames on either side of the spectrogram. This would also ensure that the signal model
has adequate information about the boundary samples.

NOTE: Our approximate implementation of RMS-level is consistent with EBU R128 / ITU-R BS.1770
loudness implementations. For example, see these implementations:
https://github.com/BrechtDeMan/loudness.py
https://github.com/csteinmetz1/pyloudnorm
They don't even pad the signal to ensure every sample is represented.
NOTE: For most signals, the underrepresentation of the first 85 milliseconds (assuming a frame
length of 2048, frame hop of 512 and a sample rate of 24000), doesn't practically matter.
"""


def get_loudness_in_amps(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor], **kwargs
):
    """Get the RMS AMP loudness for each frame in the `db_spectrogram`.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])
        **kwargs: Additional key word arguments passed to `power_spectrogram_to_framed_rms`.

    Returns:
        (torch.FloatTensor [batch_size, num_frames])
    """
    # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
    power_frames = db_to_power(db_spectrogram.transpose(0, 1))
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    frame_rms_level = cf.call(power_spectrogram_to_framed_rms, power_frames, **kwargs)
    return frame_rms_level if mask is None else frame_rms_level * mask


def get_num_sil_frames(
    db_spectrogram: torch.Tensor,
    sil_threshold: float,
    mask: typing.Optional[torch.Tensor] = None,
    **kwargs,
):
    """Get the number of silent frames with a loudness lower than `sil_threshold`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    # [batch_size, num_frames]
    db_loudness = lib.audio.amp_to_db(get_loudness_in_amps(db_spectrogram, mask, **kwargs))
    return ((db_loudness < sil_threshold) * mask).sum(dim=1)


def get_power_rms_level_sum(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, **kwargs
) -> torch.Tensor:
    """Get the sum of the power RMS level for each frame in the spectrogram.

    Returns:
        torch.FloatTensor [batch_size]
    """
    return amp_to_power(get_loudness_in_amps(db_spectrogram, mask, **kwargs)).sum(dim=1)


def get_average_db_rms_level(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get the average, over spectrogram frames, RMS level (dB) for each spectrogram.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])
        **kwargs: Additional key word arguments passed to `get_power_rms_level_sum`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    num_elements = db_spectrogram.shape[0] if mask is None else mask.sum(dim=1)
    cum_power_rms_level = get_power_rms_level_sum(db_spectrogram, mask)
    return power_to_db(cum_power_rms_level / num_elements)


def get_num_pause_frames(
    db_spectrogram: torch.Tensor,
    mask: typing.Optional[torch.Tensor],
    max_loudness: float,
    min_length: float,
    frame_hop: int,
    sample_rate: int,
    **kwargs,
) -> typing.List[int]:
    """Count the number of frames inside a pause.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [batch_size, num_frames])
        max_loudness: The maximum loudness a pause can be.
        min_length: The minimum length for a pause to be considered a pause.
        ...
    """
    # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
    power_spec = lib.audio.db_to_power(db_spectrogram).transpose(0, 1)
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    framed_rms_level = cf.partial(power_spectrogram_to_framed_rms)(power_spec, **kwargs)
    is_silent = framed_rms_level < lib.audio.db_to_amp(max_loudness)  # [batch_size, num_frames]
    is_silent = is_silent if mask is None else is_silent * mask
    frames_threshold = min_length * sample_rate / frame_hop
    batch_size = is_silent.shape[0]
    num_frames = [0] * batch_size
    for i in range(batch_size):
        _is_silent, count = torch.unique_consecutive(is_silent[i], return_counts=True)
        count = _is_silent.float() * count
        count = (count >= frames_threshold).float() * count
        num_frames[i] = int(count.sum().item())
    return num_frames


def get_alignment_norm(preds: Preds) -> torch.Tensor:
    """The inf-norm of an alignment. The more focused an alignment is the higher this metric. The
    metric is bounded at [0, 1].

    Returns:
        torch.FloatTensor [batch_size]
    """
    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    alignments = alignments.norm(dim=2, p=math.inf)  # type: ignore
    alignments = alignments.masked_fill(~preds.frames_mask.transpose(0, 1), 0)
    return alignments.sum(dim=0)


def get_alignment_std(preds: Preds) -> torch.Tensor:
    """This metric measures the standard deviation of an alignment. As the alignment is more
    focused, this metrics goes to zero.

    Returns:
        torch.FloatTensor [batch_size]
    """
    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    alignments = lib.utils.get_weighted_std(alignments, dim=2)
    alignments = alignments.masked_fill(~preds.frames_mask.transpose(0, 1), 0)
    assert not torch.any(torch.isnan(alignments)), "NaN detected."
    return alignments.sum(dim=0)


_GetMetrics = typing.Dict[GetLabel, float]


@dataclasses.dataclass(frozen=True)
class MetricsKey(_utils.MetricsKey):
    speaker: typing.Optional[Speaker] = None
    text_length_bucket: typing.Optional[int] = None


MetricsValues = typing.Dict[MetricsKey, float]


class Metrics(_utils.Metrics[MetricsKey]):
    """
    Vars:
        ...
        AVERAGE_NUM_FRAMES: The average number of frames per spectrogram.
        PAUSE_FRAMES: The percentage of frames inside of a pause.
        AVERAGE_RMS_LEVEL: The average loudness per frame.
        MAX_NUM_FRAMES: The maximum number of frames in a spectrogram.
        MIN_DATA_LOADER_QUEUE_SIZE: The minimum data loader queue size.
        FREQUENCY_NUM_FRAMES: The frequency of each speaker based on the number of frames.
        FREQUENCY_NUM_SECONDS: The frequency of each speaker based on the number of seconds.
        FREQUENCY_TEXT_LENGTH: The frequency of each text length bucket.
        ALIGNMENT_NORM: The p-norm of an alignment. The more focused an alignment is the higher this
            metric. The metric is bounded at [0, 1].
        ALIGNMENT_STD: This metric measures the standard deviation of an alignment. As the alignment
            is more focused, this metrics goes to zero.
        STOP_TOKEN_HANG_TIME_SUM: The metric measures the number of frames after the model started
            to predict a stop. Essentially, this measures how many silence or breathing frames
            there are before the model stops generating.
        PRED_PAUSE_FRAMES: The percentage of predicted frames inside of a pause.
        PRED_RATIO_SIL_FRAMES: The percentage of silent frames.
        AVERAGE_PRED_RMS_LEVEL: The average loudness per predicted frame.
        AVERAGE_RELATIVE_SPEED: The number of predicted frames divided by the number of frames.
        AVERAGE_RELATIVE_SPEED_TRIMMED: This similar to `AVERAGE_RELATIVE_SPEED`, except we
            remove any hanging frames from predicted frames. A hanging frame is defined by
            `STOP_TOKEN_HANG_TIME_SUM`. This metric will always be slightly under 100% due
            to the trimming; however, it may be more stable than `AVERAGE_RELATIVE_SPEED`.
        AVERAGE_RELATIVE_SPEED_NON_SIL: This similar to `AVERAGE_RELATIVE_SPEED`, except we
            remove any silent frames from predicted frames. This metric will always be slightly
            under 100%; however, it may be more stable than `AVERAGE_RELATIVE_SPEED`.
        AVERAGE_RMS_LEVEL_DELTA: The delta between `AVERAGE_PRED_RMS_LEVEL` and
            `AVERAGE_RMS_LEVEL`.
        GRADIENT_INFINITY_NORM: The total infinity norm of all parameter gradients.
        GRADIENT_TWO_NORM: The total 2-norm of all parameter gradients.
        GRADIENT_MAX_NORM: The maximum gradient norm used for clipping gradients.
        LR: The learning rate.
        REACHED_MAX_FRAMES: The percentage of perdictions that reached the maximum frames allowed.
        ...
    """

    (
        ALIGNMENT_NORM_SUM,
        ALIGNMENT_STD_SUM,
        STOP_TOKEN_HANG_TIME_SUM,
        DATA_QUEUE_SIZE,
        MAX_FRAMES_PER_TOKEN,
        NUM_CORRECT_STOP_TOKEN,
        NUM_FRAMES_MAX,
        NUM_FRAMES_PRED,
        NUM_FRAMES,
        NUM_REACHED_MAX,
        NUM_SECONDS,
        NUM_SPANS_PER_TEXT_LENGTH,
        NUM_SPANS,
        NUM_TOKENS,
        RMS_SUM_PRED,
        RMS_SUM,
        NUM_PAUSE_FRAMES_PRED,
        NUM_SIL_FRAMES_PRED,
        SPECTROGRAM_LOSS_SUM,
        STOP_TOKEN_LOSS_SUM,
        *_,
    ) = tuple([str(i) for i in range(100)])

    NUM_SPANS_ = partial(get_dataset_label, "num_spans")
    AVERAGE_NUM_FRAMES = partial(get_dataset_label, "average_num_frames")
    AVERAGE_RMS_LEVEL = partial(get_dataset_label, "average_rms_level")
    MAX_NUM_FRAMES = partial(get_dataset_label, "max_num_frames")
    AVERAGE_FRAMES_PER_TOKEN = partial(get_dataset_label, "average_frames_per_token")
    MAX_FRAMES_PER_TOKEN_ = partial(get_dataset_label, "max_frames_per_token")
    MIN_DATA_LOADER_QUEUE_SIZE = partial(get_dataset_label, "min_data_loader_queue_size")
    FREQUENCY_NUM_FRAMES = partial(get_dataset_label, "frequency/num_frames")
    FREQUENCY_NUM_SECONDS = partial(get_dataset_label, "frequency/num_seconds")
    FREQUENCY_TEXT_LENGTH = partial(get_dataset_label, "text_length_bucket_{lower}_{upper}")

    ALIGNMENT_NORM = partial(get_model_label, "alignment_norm")
    ALIGNMENT_STD = partial(get_model_label, "alignment_std")
    STOP_TOKEN_HANG_TIME = partial(get_model_label, "stop_token_hang_time")
    PRED_PAUSE_FRAMES = partial(get_model_label, "predicted_pause_frames")
    PRED_RATIO_SIL_FRAMES = partial(get_model_label, "pred_ratio_sil_frames")
    AVERAGE_PRED_RMS_LEVEL = partial(get_model_label, "average_predicted_rms_level")
    AVERAGE_RELATIVE_SPEED = partial(get_model_label, "average_relative_speed")
    AVERAGE_RELATIVE_SPEED_TRIMMED = partial(get_model_label, "average_relative_speed_trimmed")
    AVERAGE_RELATIVE_SPEED_NON_SIL = partial(get_model_label, "average_relative_speed_non_sil")
    AVERAGE_RMS_LEVEL_DELTA = partial(get_model_label, "average_rms_level_delta")
    REACHED_MAX_FRAMES = partial(get_model_label, "reached_max_frames")
    SPECTROGRAM_LOSS = partial(get_model_label, "spectrogram_loss")
    STOP_TOKEN_ACCURACY = partial(get_model_label, "stop_token_accuracy")
    STOP_TOKEN_LOSS = partial(get_model_label, "stop_token_loss")

    TEXT_LENGTH_BUCKET_SIZE = 25

    @staticmethod
    def _make_values():
        values: MetricsValues = collections.defaultdict(float)

        def _reduce(
            label: str,
            speaker: typing.Optional[Speaker] = None,
            text_length_bucket: typing.Optional[int] = None,
            v: float = 0,
            op: typing.Callable[[float, float], float] = add,
        ):
            key = MetricsKey(label, speaker, text_length_bucket)
            value = v if key not in values and op in (min, max) else values[key]
            values[key] = op(value, v)

        return values, _reduce

    def is_eval_infer(self):
        return self.comet.context == str(Context.EVALUATE_INFERENCE.value)

    def get_dataset_values(self, batch: Batch, preds: Preds) -> MetricsValues:
        """
        TODO: Get dataset metrics on OOV words (spaCy and pronunciation dictionary) in our dataset.
        TODO: Create a `streamlit` for measuring coverage in our dataset, and other datasets.
        """
        values, _reduce = self._make_values()

        for span, num_frames, slice, has_reached_max in zip(
            batch.spans,
            self._to_list(batch.spectrogram.lengths),
            batch.processed.slices,
            self._to_list(preds.reached_max),
        ):
            num_tokens = slice.stop - slice.start
            # NOTE: `has_reached_max` is only applicable during inference; otherwise, the predicted
            # frames lines up with the real frames, so they shouldn't be filtered out unless the
            # same standard is applied to the real data, also.
            has_reached_max = self.is_eval_infer() and has_reached_max

            # NOTE: Create a key for `self.NUM_SPANS` so a value exists, even if zero.
            _reduce(self.NUM_SPANS, v=float(not has_reached_max))
            _reduce(self.NUM_SPANS, span.speaker, v=float(not has_reached_max))

            # NOTE: Remove predictions that diverged (reached max) as to not skew other metrics.
            # TODO: Should we develop a metric that considers the original num of frames in
            # a determination of max frames? We could do a data analysis of how many frames
            # more than the original frames does the model generate before it's too many? That
            # should in theory be more accurate.
            if not has_reached_max:
                index = int(len(span.script) // self.TEXT_LENGTH_BUCKET_SIZE)
                _reduce(self.NUM_SPANS_PER_TEXT_LENGTH, text_length_bucket=index, v=1)
                _reduce(self.NUM_FRAMES_MAX, v=num_frames, op=max)

                speaker: typing.Optional[Speaker]
                for speaker in [None, span.speaker]:
                    _reduce(self.NUM_FRAMES, speaker, v=num_frames)
                    _reduce(self.NUM_SECONDS, speaker, v=span.audio_length)
                    _reduce(self.NUM_TOKENS, speaker, v=num_tokens)
                    _reduce(self.MAX_FRAMES_PER_TOKEN, speaker, v=num_frames / num_tokens, op=max)

        return dict(values)

    def get_alignment_values(self, batch: Batch, preds: Preds) -> MetricsValues:
        values, _reduce = self._make_values()

        for span, std, norm, hang_time, length, has_reached_max in zip(
            batch.spans,
            self._to_list(get_alignment_std(preds)),
            self._to_list(get_alignment_norm(preds)),
            self._to_list(
                cf.partial(get_hang_time)(preds.stop_tokens, preds.frames_mask, preds.num_frames)
            ),
            self._to_list(preds.num_frames),
            self._to_list(preds.reached_max),
        ):
            _reduce(self.NUM_REACHED_MAX, v=has_reached_max)
            _reduce(self.NUM_REACHED_MAX, span.speaker, v=has_reached_max)

            if not (self.is_eval_infer() and has_reached_max):
                speaker: typing.Optional[Speaker]
                for speaker in [None, span.speaker]:
                    _reduce(self.ALIGNMENT_NORM_SUM, speaker, v=norm)
                    _reduce(self.ALIGNMENT_STD_SUM, speaker, v=std)
                    _reduce(self.STOP_TOKEN_HANG_TIME_SUM, speaker, v=hang_time)
                    _reduce(self.NUM_FRAMES_PRED, speaker, v=length)

        return dict(values)

    def get_loudness_values(self, batch: Batch, preds: Preds) -> MetricsValues:
        values, _reduce = self._make_values()

        spectrogram_mask = batch.spectrogram_mask.tensor.transpose(0, 1)
        loudness = get_power_rms_level_sum(batch.spectrogram.tensor, spectrogram_mask)
        for span, loudness, pred_loudness, num_pause_pred, num_sil_pred, has_reached_max in zip(
            batch.spans,
            self._to_list(loudness),
            self._to_list(get_power_rms_level_sum(preds.frames, preds.frames_mask)),
            cf.partial(get_num_pause_frames)(preds.frames, preds.frames_mask),
            self._to_list(cf.partial(get_num_sil_frames)(preds.frames, mask=preds.frames_mask)),
            self._to_list(preds.reached_max),
        ):
            if not (self.is_eval_infer() and has_reached_max):
                for speaker in [None, span.speaker]:
                    _reduce(self.RMS_SUM_PRED, speaker, v=pred_loudness)
                    _reduce(self.RMS_SUM, speaker, v=loudness)
                    _reduce(self.NUM_PAUSE_FRAMES_PRED, speaker, v=num_pause_pred)
                    _reduce(self.NUM_SIL_FRAMES_PRED, speaker, v=num_sil_pred)

        return dict(values)

    def get_stop_token_values(
        self, batch: Batch, model: SpectrogramModel, preds: Preds
    ) -> MetricsValues:
        th = model.stop_threshold
        is_correct = (batch.stop_token.tensor > th) == (torch.sigmoid(preds.stop_tokens) > th)
        is_correct = is_correct.masked_select(batch.spectrogram_mask.tensor)
        return {MetricsKey(self.NUM_CORRECT_STOP_TOKEN): float(is_correct.sum().item())}

    def _iter_permutations(self, select: _utils.MetricsSelect, is_verbose: bool = True):
        """Iterate over permutations of metric names and return convenience operations.

        Args:
            is_verbose: If `True`, iterate over more permutations.
        """
        speakers = set(key.speaker for key in self.data.keys())
        for speaker in speakers if is_verbose else [None]:
            reduce_: typing.Callable[[str], float]
            reduce_ = lambda a, **k: self._reduce(MetricsKey(a, speaker), select=select, **k)
            process: typing.Callable[[typing.Union[float, str]], typing.Union[float, MetricsKey]]
            process = lambda a: a if isinstance(a, float) else MetricsKey(a, speaker)
            div: typing.Callable[[typing.Union[float, str], typing.Union[float, str]], float]
            div = lambda n, d, **k: self._div(process(n), process(d), select=select, **k)
            yield speaker, reduce_, div

    def _get_model_metrics(
        self, select: _utils.MetricsSelect, is_verbose: bool, num_frame_channels: int
    ) -> _GetMetrics:
        metrics = {}
        for speaker, reduce, div in self._iter_permutations(select, is_verbose):
            spectrogram_loss = div(self.SPECTROGRAM_LOSS_SUM, self.NUM_FRAMES) / num_frame_channels
            total_spans = reduce(self.NUM_SPANS) + reduce(self.NUM_REACHED_MAX)
            num_frames_pred = reduce(self.NUM_FRAMES_PRED)
            num_frames_pred_trimmed = num_frames_pred - reduce(self.STOP_TOKEN_HANG_TIME_SUM)
            num_frames_non_sil = num_frames_pred - reduce(self.NUM_SIL_FRAMES_PRED)
            update = {
                self.ALIGNMENT_NORM: div(self.ALIGNMENT_NORM_SUM, self.NUM_FRAMES_PRED),
                self.ALIGNMENT_STD: div(self.ALIGNMENT_STD_SUM, self.NUM_FRAMES_PRED),
                self.STOP_TOKEN_HANG_TIME: div(self.STOP_TOKEN_HANG_TIME_SUM, self.NUM_SPANS),
                self.AVERAGE_RELATIVE_SPEED: div(self.NUM_FRAMES_PRED, self.NUM_FRAMES),
                self.AVERAGE_RELATIVE_SPEED_TRIMMED: div(num_frames_pred_trimmed, self.NUM_FRAMES),
                self.AVERAGE_RELATIVE_SPEED_NON_SIL: div(num_frames_non_sil, self.NUM_FRAMES),
                self.STOP_TOKEN_ACCURACY: div(self.NUM_CORRECT_STOP_TOKEN, self.NUM_FRAMES),
                self.STOP_TOKEN_LOSS: div(self.STOP_TOKEN_LOSS_SUM, self.NUM_FRAMES),
                self.REACHED_MAX_FRAMES: reduce(self.NUM_REACHED_MAX) / total_spans,
                self.SPECTROGRAM_LOSS: spectrogram_loss,
            }
            # TODO: Flesh out various metric invariants.
            accuracy = update[self.STOP_TOKEN_ACCURACY]
            assert math.isnan(accuracy) or accuracy <= 1.0
            metrics.update({partial(k, speaker=speaker): v for k, v in update.items()})
        return metrics

    def _get_dataset_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        """
        NOTE: There is a discrepency between `num_frames_per_speaker` and `num_seconds_per_speaker`
        because `num_seconds_per_speaker` is computed with `Span` and `num_frames_per_speaker`
        is computed with `Batch`. For example, in Feburary 2021, `make_batch` used
        `_pad_and_trim_signal`. "elizabeth_klett", "mary_ann" and "judy_bieber" tend to need
        significantly more trimming than other speakers.
        """
        reduce = lambda *a, **k: self._reduce(MetricsKey(*a), select=select, **k)
        metrics: _GetMetrics = {
            self.MAX_NUM_FRAMES: reduce(self.NUM_FRAMES_MAX, op=max),
            self.MIN_DATA_LOADER_QUEUE_SIZE: reduce(self.DATA_QUEUE_SIZE, op=min),
        }

        total_frames = reduce(self.NUM_FRAMES)
        total_seconds = reduce(self.NUM_SECONDS)
        for speaker, _reduce, _div in self._iter_permutations(select, is_verbose):
            update = {
                self.AVERAGE_NUM_FRAMES: _div(self.NUM_FRAMES, self.NUM_SPANS),
                self.AVERAGE_FRAMES_PER_TOKEN: _div(self.NUM_FRAMES, self.NUM_TOKENS),
                self.MAX_FRAMES_PER_TOKEN_: _reduce(self.MAX_FRAMES_PER_TOKEN, op=max),
                self.FREQUENCY_NUM_FRAMES: _reduce(self.NUM_FRAMES) / total_frames,
                self.FREQUENCY_NUM_SECONDS: _reduce(self.NUM_SECONDS) / total_seconds,
                self.NUM_SPANS_: _reduce(self.NUM_SPANS),
            }
            metrics.update({partial(k, speaker=speaker): v for k, v in update.items()})

        if is_verbose:
            total_spans = reduce(self.NUM_SPANS)
            for key in self.data.keys():
                if self.NUM_SPANS_PER_TEXT_LENGTH is not key.label:
                    continue

                assert isinstance(key.text_length_bucket, int), "Invariant error"
                lower = key.text_length_bucket * self.TEXT_LENGTH_BUCKET_SIZE
                upper = (key.text_length_bucket + 1) * self.TEXT_LENGTH_BUCKET_SIZE
                get_label = partial(self.FREQUENCY_TEXT_LENGTH, lower=lower, upper=upper)
                metrics[get_label] = reduce(key) / total_spans

        return metrics

    def _get_loudness_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        metrics = {}
        for speaker, _, div in self._iter_permutations(select, is_verbose):
            predicted_rms = power_to_db(div(self.RMS_SUM_PRED, self.NUM_FRAMES_PRED))
            rms = power_to_db(div(self.RMS_SUM, self.NUM_FRAMES))
            update = {
                self.AVERAGE_PRED_RMS_LEVEL: predicted_rms,
                self.AVERAGE_RMS_LEVEL: rms,
                self.AVERAGE_RMS_LEVEL_DELTA: predicted_rms - rms,
                self.PRED_PAUSE_FRAMES: div(self.NUM_PAUSE_FRAMES_PRED, self.NUM_FRAMES_PRED),
                self.PRED_RATIO_SIL_FRAMES: div(self.NUM_SIL_FRAMES_PRED, self.NUM_FRAMES_PRED),
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
                **cf.partial(self._get_model_metrics)(select=select, is_verbose=is_verbose),
                **self._get_dataset_metrics(select=select, is_verbose=is_verbose),
                **self._get_loudness_metrics(select=select, is_verbose=is_verbose),
            }
            record_event(Timer.LOG_METRICS)
            self.comet.log_metrics(
                {k(**kwargs): v for k, v in metrics.items() if not math.isnan(v)}
            )
