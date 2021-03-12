import collections
import collections.abc
import itertools
import math
import typing
from functools import partial

import torch
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, configurable

import lib
from lib.distributed import is_master
from run._config import GetLabel, get_dataset_label, get_model_label
from run.train import _utils
from run.train._utils import CometMLExperiment, MetricsValues, Timer
from run.train.spectrogram_model._data import Batch


def get_num_skipped(
    alignments: torch.Tensor, token_mask: torch.Tensor, spectrogram_mask: torch.Tensor
) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this computes the number of tokens that were
    skipped.

    NOTE: This function assumes a token is attended to if it has the most focus of all the other
    tokens for some frame.

    Args:
        alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
        token_mask (torch.BoolTensor [num_tokens, batch_size])
        spectrogram_mask (torch.BoolTensor [num_frames, batch_size])

    Returns:
        torch.FloatTensor [batch_size]
    """
    if alignments.numel() == 0:
        return torch.empty(alignments.shape[1], dtype=torch.float, device=alignments.device)

    indices = alignments.max(dim=2, keepdim=True).indices
    device = alignments.device
    one = torch.ones(*alignments.shape, device=device, dtype=torch.long)
    # [num_frames, batch_size, num_tokens]
    num_skipped = torch.zeros(*alignments.shape, device=device, dtype=torch.long)
    num_skipped = num_skipped.scatter(dim=2, index=indices, src=one)
    # [num_frames, batch_size, num_tokens] → [batch_size, num_tokens]
    num_skipped = num_skipped.masked_fill(~spectrogram_mask.unsqueeze(-1), 0).sum(dim=0)
    token_mask = token_mask.transpose(0, 1)
    return (num_skipped.masked_fill(~token_mask, -1) == 0).float().sum(dim=1)


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


def get_power_rms_level_sum(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, **kwargs
) -> torch.Tensor:
    """Get the sum of the power RMS level for each frame in the spectrogram.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [num_frames, batch_size])
        **kwargs: Additional key word arguments passed to `power_spectrogram_to_framed_rms`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    spectrogram = typing.cast(torch.Tensor, lib.audio.db_to_power(db_spectrogram.transpose(0, 1)))
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    rms = lib.audio.power_spectrogram_to_framed_rms(spectrogram, **kwargs)
    return (rms if mask is None else rms * mask.transpose(0, 1)).pow(2).sum(dim=1)


def get_average_db_rms_level(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, **kwargs
) -> torch.Tensor:
    """Get the average, over spectrogram frames, RMS level (dB) for each spectrogram.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [num_frames, batch_size])
        **kwargs: Additional key word arguments passed to `get_power_rms_level_sum`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    num_elements = db_spectrogram.shape[0] if mask is None else mask.sum(dim=0)
    cumulative_power_rms_level = get_power_rms_level_sum(db_spectrogram, mask, **kwargs)
    return lib.audio.power_to_db(cumulative_power_rms_level / num_elements)


_GetMetrics = typing.Dict[GetLabel, float]


class Metrics(_utils.Metrics):
    """
    Vars:
        ...
        AVERAGE_NUM_FRAMES: The average number of frames per spectrogram.
        AVERAGE_RMS_LEVEL: The average loudness per frame.
        MAX_NUM_FRAMES: The maximum number of frames in a spectrogram.
        MIN_DATA_LOADER_QUEUE_SIZE: The minimum data loader queue size.
        FREQUENCY_NUM_FRAMES: The frequency of each speaker based on the number of frames.
        FREQUENCY_NUM_SECONDS: The frequency of each speaker based on the number of seconds.
        FREQUENCY_TEXT_LENGTH: The frequency of each text length bucket.
        ALIGNMENT_NORM: The p-norm of an alignment. The more focused an alignment is the higher this
            metric. The metric is bounded at [0, 1].
        ALIGNMENT_SKIPS: This metric assumes that each alignment focuses on one token. This measures
            the percentage of tokens skipped by the alignments.
        ALIGNMENT_STD: This metric measures the standard deviation of an alignment. As the alignment
            is more focused, this metrics goes to zero.
        AVERAGE_PREDICTED_RMS_LEVEL: The average loudness per predicted frame.
        AVERAGE_RELATIVE_SPEED: The number of predicted frames divided by the number of frames.
        AVERAGE_RMS_LEVEL_DELTA: The delta between `AVERAGE_PREDICTED_RMS_LEVEL` and
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
        ALIGNMENT_NUM_SKIPS,
        ALIGNMENT_STD_SUM,
        DATA_QUEUE_SIZE,
        NUM_CORRECT_STOP_TOKEN,
        NUM_FRAMES_MAX,
        NUM_FRAMES_PREDICTED,
        NUM_FRAMES,
        NUM_REACHED_MAX,
        NUM_SECONDS,
        NUM_SPANS_PER_TEXT_LENGTH,
        NUM_SPANS,
        NUM_TOKENS,
        RMS_SUM_PREDICTED,
        RMS_SUM,
        SPECTROGRAM_LOSS_SUM,
        STOP_TOKEN_LOSS_SUM,
        *_,
    ) = tuple([str(i) for i in range(100)])

    NUM_SPANS_ = partial(get_dataset_label, "num_spans")
    AVERAGE_NUM_FRAMES = partial(get_dataset_label, "average_num_frames")
    AVERAGE_RMS_LEVEL = partial(get_dataset_label, "average_rms_level")
    MAX_NUM_FRAMES = partial(get_dataset_label, "max_num_frames")
    MIN_DATA_LOADER_QUEUE_SIZE = partial(get_dataset_label, "min_data_loader_queue_size")
    FREQUENCY_NUM_FRAMES = partial(get_dataset_label, "frequency/num_frames")
    FREQUENCY_NUM_SECONDS = partial(get_dataset_label, "frequency/num_seconds")
    FREQUENCY_TEXT_LENGTH = partial(get_dataset_label, "text_length_bucket_{lower}_{upper}")

    ALIGNMENT_NORM = partial(get_model_label, "alignment_norm")
    ALIGNMENT_SKIPS = partial(get_model_label, "alignment_skips")
    ALIGNMENT_STD = partial(get_model_label, "alignment_std")
    AVERAGE_PREDICTED_RMS_LEVEL = partial(get_model_label, "average_predicted_rms_level")
    AVERAGE_RELATIVE_SPEED = partial(get_model_label, "average_relative_speed")
    AVERAGE_RMS_LEVEL_DELTA = partial(get_model_label, "average_rms_level_delta")
    REACHED_MAX_FRAMES = partial(get_model_label, "reached_max_frames")
    SPECTROGRAM_LOSS = partial(get_model_label, "spectrogram_loss")
    STOP_TOKEN_ACCURACY = partial(get_model_label, "stop_token_accuracy")
    STOP_TOKEN_LOSS = partial(get_model_label, "stop_token_loss")

    TEXT_LENGTH_BUCKET_SIZE = 25
    ALIGNMENT_NORM_TYPE = math.inf

    def __init__(
        self,
        comet: CometMLExperiment,
        speakers: typing.List[lib.datasets.Speaker],
        **kwargs,
    ):
        super().__init__(comet, **kwargs)
        self.comet = comet
        self.speakers = speakers

    def get_dataset_values(
        self, batch: Batch, reached_max: typing.Optional[torch.Tensor] = None
    ) -> MetricsValues:
        """
        TODO: Get dataset metrics on OOV words (spaCy and AmEPD) in our dataset.
        TODO: Create a `streamlit` for measuring coverage in our dataset, and other datasets.
        TODO: Measure the difference between punctuation in the phonetic vs grapheme phrases.
        Apart from unique cases, they should have the same punctuation.
        """
        values: typing.Dict[str, float] = collections.defaultdict(float)

        for span, num_frames, num_tokens, has_reached_max in zip(
            batch.spans,
            self._to_list(batch.spectrogram.lengths),
            self._to_list(batch.encoded_phonemes.lengths),
            itertools.repeat(False) if reached_max is None else self._to_list(reached_max),
        ):
            # NOTE: Create a key for `self.NUM_SPANS` so a value exists, even if zero.
            values[self.NUM_SPANS] += float(not has_reached_max)
            values[f"{self.NUM_SPANS}/{span.speaker.label}"] += float(not has_reached_max)

            if not has_reached_max:
                index = int(len(span.script) // self.TEXT_LENGTH_BUCKET_SIZE)
                values[f"{self.NUM_SPANS_PER_TEXT_LENGTH}/{index}"] += 1
                values[self.NUM_FRAMES_MAX] = max(num_frames, values[self.NUM_FRAMES_MAX])

                assert span.speaker in self.speakers
                for suffix in ["", f"/{span.speaker.label}"]:
                    format_ = lambda s: f"{s}{suffix}"
                    values[format_(self.NUM_FRAMES)] += num_frames
                    values[format_(self.NUM_SECONDS)] += span.audio_length
                    values[format_(self.NUM_TOKENS)] += num_tokens

        return dict(values)

    def get_alignment_values(
        self,
        batch: Batch,
        alignments: torch.Tensor,
        spectrogram_lengths: torch.Tensor,
        spectrogram_mask: torch.Tensor,
        reached_max: typing.Optional[torch.Tensor] = None,
    ) -> MetricsValues:
        """
        Args:
            ...
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            spectrogram_lengths (torch.LongTensor [1, batch_size])
            spectrogram_mask (torch.BoolTensor [num_frames, batch_size])
            reached_max (torch.BoolTensor [batch_size]): Remove predictions that diverged
                (reached max) as to not skew other metrics.
        """
        values: typing.Dict[str, float] = collections.defaultdict(float)
        tokens_mask = batch.encoded_phonemes_mask.tensor
        alignments = alignments.masked_fill(~tokens_mask.transpose(0, 1).unsqueeze(0), 0)
        _to_list = lambda a: self._to_list(a.masked_fill(~spectrogram_mask, 0).sum(dim=0))

        for span, num_skipped, alignment_std, alignment_norm, length, has_reached_max in zip(
            batch.spans,
            self._to_list(get_num_skipped(alignments, tokens_mask, spectrogram_mask)),
            _to_list(lib.utils.get_weighted_std(alignments, dim=2)),
            _to_list(alignments.norm(p=self.ALIGNMENT_NORM_TYPE, dim=2)),
            self._to_list(spectrogram_lengths),
            itertools.repeat(False) if reached_max is None else self._to_list(reached_max),
        ):
            assert span.speaker in self.speakers
            values[self.NUM_REACHED_MAX] += has_reached_max
            values[f"{self.NUM_REACHED_MAX}/{span.speaker.label}"] += has_reached_max

            if not has_reached_max:
                for suffix in ["", f"/{span.speaker.label}"]:
                    format_ = lambda s: f"{s}{suffix}"
                    values[format_(self.ALIGNMENT_NORM_SUM)] += alignment_norm
                    values[format_(self.ALIGNMENT_NUM_SKIPS)] += num_skipped
                    values[format_(self.ALIGNMENT_STD_SUM)] += alignment_std
                    values[format_(self.NUM_FRAMES_PREDICTED)] += length

        return dict(values)

    def get_loudness_values(
        self,
        batch: Batch,
        predicted_spectrogram: torch.Tensor,
        spectrogram_mask: torch.Tensor,
        reached_max: typing.Optional[torch.Tensor] = None,
    ) -> MetricsValues:
        """
        Args:
            ...
            predicted (torch.FloatTensor [num_frames, batch_size, frame_channels])
            reached_max (torch.BoolTensor [batch_size])
            spectrogram_mask (torch.FloatTensor [num_frames, batch_size])
            **kwargs: Additional key word arguments passed to `get_power_rms_level_sum`.
        """
        values: typing.Dict[str, float] = collections.defaultdict(float)
        loudness = get_power_rms_level_sum(batch.spectrogram.tensor, batch.spectrogram_mask.tensor)
        for span, loudness, predicted_loudness, has_reached_max in zip(
            batch.spans,
            self._to_list(loudness),
            self._to_list(get_power_rms_level_sum(predicted_spectrogram, spectrogram_mask)),
            itertools.repeat(False) if reached_max is None else self._to_list(reached_max),
        ):
            if not has_reached_max:
                assert span.speaker in self.speakers
                for suffix in ["", f"/{span.speaker.label}"]:
                    format_ = lambda s: f"{s}{suffix}"
                    values[format_(self.RMS_SUM_PREDICTED)] += predicted_loudness
                    values[format_(self.RMS_SUM)] += loudness

        return dict(values)

    def get_stop_token_values(
        self, batch: Batch, predicted_logits: torch.Tensor, stop_threshold: float
    ) -> MetricsValues:
        """
        Args:
            ...
            predicted_logits (torch.FloatTensor [num_frames, batch_size])
            ...
        """
        bool_ = lambda t: (t > stop_threshold).masked_select(batch.spectrogram_mask.tensor)
        is_correct = bool_(batch.stop_token.tensor) == bool_(torch.sigmoid(predicted_logits))
        return {self.NUM_CORRECT_STOP_TOKEN: float(is_correct.sum().item())}

    def _iter_permutations(self, select: _utils.MetricsSelect, is_verbose: bool = True):
        """Iterate over permutations of metric names and return convenience operations.

        Args:
            is_verbose: If `True`, iterate over more permutations.
        """
        for speaker in itertools.chain([None], self.speakers if is_verbose else []):
            suffix = "" if speaker is None else f"/{speaker.label}"
            format_ = lambda s: f"{s}{suffix}"
            reduce = lambda k: self._reduce(format_(k), select=select)
            div = lambda n, d: self._div(format_(n), format_(d), select=select)
            yield speaker, reduce, div

    @configurable
    def _get_model_metrics(
        self, select: _utils.MetricsSelect, is_verbose: bool, num_frame_channels=HParam()
    ) -> _GetMetrics:
        metrics = {}
        for speaker, reduce, div in self._iter_permutations(select, is_verbose):
            spectrogram_loss = div(self.SPECTROGRAM_LOSS_SUM, self.NUM_FRAMES) / num_frame_channels
            total_spans = reduce(self.NUM_SPANS) + reduce(self.NUM_REACHED_MAX)
            update = {
                self.ALIGNMENT_NORM: div(self.ALIGNMENT_NORM_SUM, self.NUM_FRAMES_PREDICTED),
                self.ALIGNMENT_STD: div(self.ALIGNMENT_STD_SUM, self.NUM_FRAMES_PREDICTED),
                self.ALIGNMENT_SKIPS: div(self.ALIGNMENT_NUM_SKIPS, self.NUM_TOKENS),
                self.AVERAGE_RELATIVE_SPEED: div(self.NUM_FRAMES_PREDICTED, self.NUM_FRAMES),
                self.STOP_TOKEN_ACCURACY: div(self.NUM_CORRECT_STOP_TOKEN, self.NUM_FRAMES),
                self.STOP_TOKEN_LOSS: div(self.STOP_TOKEN_LOSS_SUM, self.NUM_FRAMES),
                self.REACHED_MAX_FRAMES: reduce(self.NUM_REACHED_MAX) / total_spans,
                self.SPECTROGRAM_LOSS: spectrogram_loss,
            }
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
        reduce = partial(self._reduce, select=select)
        metrics = {
            self.MAX_NUM_FRAMES: reduce(self.NUM_FRAMES_MAX, op=max),
            self.MIN_DATA_LOADER_QUEUE_SIZE: reduce(self.DATA_QUEUE_SIZE, op=min),
        }

        if is_verbose:
            total_frames = reduce(self.NUM_FRAMES)
            total_seconds = reduce(self.NUM_SECONDS)
            for speaker, _reduce, _div in self._iter_permutations(select):
                update = {
                    self.AVERAGE_NUM_FRAMES: _div(self.NUM_FRAMES, self.NUM_SPANS),
                    self.FREQUENCY_NUM_FRAMES: _reduce(self.NUM_FRAMES) / total_frames,
                    self.FREQUENCY_NUM_SECONDS: _reduce(self.NUM_SECONDS) / total_seconds,
                    self.NUM_SPANS_: _reduce(self.NUM_SPANS),
                }
                metrics.update({partial(k, speaker=speaker): v for k, v in update.items()})

            total_spans = reduce(self.NUM_SPANS)
            for key in self.data.keys():
                if f"{self.NUM_SPANS_PER_TEXT_LENGTH}/" not in key:
                    continue

                bucket = int(key.split("/")[-1])
                lower = bucket * self.TEXT_LENGTH_BUCKET_SIZE
                upper = (bucket + 1) * self.TEXT_LENGTH_BUCKET_SIZE
                get_label = partial(self.FREQUENCY_TEXT_LENGTH, lower=lower, upper=upper)
                metrics[get_label] = reduce(key) / total_spans

        return metrics

    def _get_loudness_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        power_to_db = lambda r: float(lib.audio.power_to_db(torch.tensor(r)).item())
        metrics = {}
        for speaker, _, div in self._iter_permutations(select, is_verbose):
            predicted_rms = power_to_db(div(self.RMS_SUM_PREDICTED, self.NUM_FRAMES_PREDICTED))
            rms = power_to_db(div(self.RMS_SUM, self.NUM_FRAMES))
            update = {
                self.AVERAGE_PREDICTED_RMS_LEVEL: predicted_rms,
                self.AVERAGE_RMS_LEVEL: rms,
                self.AVERAGE_RMS_LEVEL_DELTA: predicted_rms - rms,
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
                **self._get_loudness_metrics(select=select, is_verbose=is_verbose),
            }
            record_event(Timer.LOG_METRICS)
            self.comet.log_metrics(
                {k(**kwargs): v for k, v in metrics.items() if not math.isnan(v)}
            )
