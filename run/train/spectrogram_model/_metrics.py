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
from lib.audio import power_to_db
from lib.distributed import is_master
from lib.spectrogram_model import Preds
from run._config import GetLabel, get_dataset_label, get_model_label
from run.data._loader import Speaker
from run.train import _utils
from run.train._utils import MetricsKey, MetricsValues, Timer
from run.train.spectrogram_model._data import Batch
from run.train.spectrogram_model._model import SpectrogramModel


def get_num_skipped(preds: Preds) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this computes the number of tokens that were
    skipped.

    NOTE: This function assumes a token is attended to if it has the most focus of all the other
    tokens for some frame.

    Returns:
        torch.FloatTensor [batch_size]
    """
    if preds.alignments.numel() == 0:
        return torch.zeros(preds.alignments.shape[1], device=preds.alignments.device)

    indices = preds.alignments.max(dim=2, keepdim=True).indices
    device = preds.alignments.device
    one = torch.ones(*preds.alignments.shape, device=device, dtype=torch.long)
    # [num_frames, batch_size, num_tokens]
    num_skipped = torch.zeros(*preds.alignments.shape, device=device, dtype=torch.long)
    num_skipped = num_skipped.scatter(dim=2, index=indices, src=one)
    # [num_frames, batch_size, num_tokens] → [batch_size, num_tokens]
    num_skipped = num_skipped.masked_fill(~preds.frames_mask.transpose(0, 1).unsqueeze(-1), 0)
    num_skipped = num_skipped.sum(dim=0)
    num_skipped = num_skipped.masked_fill(~preds.tokens_mask, -1)
    return (num_skipped == 0).float().sum(dim=1)


def get_num_jumps(preds: Preds) -> torch.Tensor:
    """Given `alignments` from frames to tokens, the computes the number of "jumps" between frame
    to frame, that would skip at least one token.

    Returns:
        torch.FloatTensor [batch_size]
    """
    if preds.alignments.numel() == 0:
        return torch.zeros(preds.alignments.shape[1], device=preds.alignments.device)

    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    # [num_frames, batch_size, num_tokens] → [num_frames, batch_size]
    indices = alignments.max(dim=2).indices
    start = torch.cat([torch.zeros(1, alignments.shape[1], device=alignments.device), indices[:-1]])
    skip_size = indices - start
    skip_size = skip_size.masked_fill(~preds.frames_mask.transpose(0, 1), 0)
    num_jumps = (skip_size.abs() > 1).float()
    return num_jumps.sum(dim=0)


@configurable
def get_num_small_max(preds: Preds, threshold: float = HParam()) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this computes the number of alignments where no
    token gets no more than `threshold` focus.

    Args:
        preds
        threshold: The percentage focus a token gets.

    Returns:
        torch.FloatTensor [batch_size]
    """
    if preds.alignments.numel() == 0:
        return torch.zeros(preds.alignments.shape[1], device=preds.alignments.device)

    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    # [num_frames, batch_size, num_tokens] → [num_frames, batch_size]
    values = alignments.max(dim=2).values
    values = (values < threshold).float()
    values = values.masked_fill(~preds.frames_mask.transpose(0, 1), 0)
    return values.sum(dim=0)


@configurable
def get_num_repeated(preds: Preds, threshold: float = HParam()) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this gets the number of tokens that get more
    focus than `threshold`.

    Args:
        preds
        threshold: The percentage focus a token gets.

    Returns:
        torch.FloatTensor [batch_size]
    """
    if preds.alignments.numel() == 0:
        return torch.zeros(preds.alignments.shape[1], device=preds.alignments.device)

    alignments = preds.alignments.masked_fill(~preds.tokens_mask.unsqueeze(0), 0)
    alignments = alignments.masked_fill(~preds.frames_mask.transpose(0, 1).unsqueeze(2), 0)
    # [num_frames, batch_size, num_tokens] → [batch_size, num_tokens]
    values = alignments.sum(dim=0)
    values = (values > threshold).float()
    return values.sum(dim=1)


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
        mask (torch.FloatTensor [batch_size, num_frames])
        **kwargs: Additional key word arguments passed to `power_spectrogram_to_framed_rms`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    spectrogram = typing.cast(torch.Tensor, lib.audio.db_to_power(db_spectrogram.transpose(0, 1)))
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    rms = lib.audio.power_spectrogram_to_framed_rms(spectrogram, **kwargs)
    return (rms if mask is None else rms * mask).pow(2).sum(dim=1)


def get_average_db_rms_level(
    db_spectrogram: torch.Tensor, mask: typing.Optional[torch.Tensor] = None, **kwargs
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
    cumulative_power_rms_level = get_power_rms_level_sum(db_spectrogram, mask, **kwargs)
    return power_to_db(cumulative_power_rms_level / num_elements)


@configurable
def get_num_pause_frames(
    db_spectrogram: torch.Tensor,
    mask: typing.Optional[torch.Tensor],
    max_loudness: float = HParam(),
    min_length: float = HParam(),
    frame_hop: int = HParam(),
    sample_rate: int = HParam(),
    **kwargs,
) -> typing.List[int]:
    """Count the number of frames inside a pause.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [num_frames, batch_size])
        max_loudness: The maximum loudness a pause can be.
        min_length: The minimum length for a pause to be considered a pause.
        ...
    """
    # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
    power_spec = lib.audio.db_to_power(db_spectrogram).transpose(0, 1)
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    framed_rms_level = lib.audio.power_spectrogram_to_framed_rms(power_spec, **kwargs)
    is_silent = framed_rms_level < lib.audio.db_to_amp(max_loudness)  # [batch_size, num_frames]
    is_silent = is_silent if mask is None else is_silent * mask.transpose(0, 1)
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
    alignments = alignments.norm(dim=2, p=math.inf)
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
    return alignments.sum(dim=0)


_GetMetrics = typing.Dict[GetLabel, float]


class Metrics(_utils.Metrics):
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
        ALIGNMENT_SMALL_MAX: The percentage of frames which have a small maximum alignment.
        ALIGNMENT_REPEATED: The percentage of tokens which have had a lot of focus.
        ALIGNMENT_SKIPS: This metric assumes that each alignment focuses on one token. This measures
            the percentage of tokens skipped by the alignments.
        ALIGNMENT_JUMPS: This metric assumes that each alignment focuses on one token. This measures
            the percentage of token transitions that jump over a token.
        ALIGNMENT_STD: This metric measures the standard deviation of an alignment. As the alignment
            is more focused, this metrics goes to zero.
        PREDICTED_PAUSE_FRAMES: The percentage of predicted frames inside of a pause.
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
        ALIGNMENT_NUM_SMALL_MAX,
        ALIGNMENT_NUM_REPEATED,
        ALIGNMENT_NUM_SKIPS,
        ALIGNMENT_NUM_JUMPS,
        ALIGNMENT_STD_SUM,
        DATA_QUEUE_SIZE,
        MAX_FRAMES_PER_TOKEN,
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
        NUM_PAUSE_FRAMES_PREDICTED,
        NUM_PAUSE_FRAMES,
        SPECTROGRAM_LOSS_SUM,
        STOP_TOKEN_LOSS_SUM,
        *_,
    ) = tuple([str(i) for i in range(100)])

    NUM_SPANS_ = partial(get_dataset_label, "num_spans")
    AVERAGE_NUM_FRAMES = partial(get_dataset_label, "average_num_frames")
    PAUSE_FRAMES = partial(get_dataset_label, "pause_frames")
    AVERAGE_RMS_LEVEL = partial(get_dataset_label, "average_rms_level")
    MAX_NUM_FRAMES = partial(get_dataset_label, "max_num_frames")
    AVERAGE_FRAMES_PER_TOKEN = partial(get_dataset_label, "average_frames_per_token")
    MAX_FRAMES_PER_TOKEN_ = partial(get_dataset_label, "max_frames_per_token")
    MIN_DATA_LOADER_QUEUE_SIZE = partial(get_dataset_label, "min_data_loader_queue_size")
    FREQUENCY_NUM_FRAMES = partial(get_dataset_label, "frequency/num_frames")
    FREQUENCY_NUM_SECONDS = partial(get_dataset_label, "frequency/num_seconds")
    FREQUENCY_TEXT_LENGTH = partial(get_dataset_label, "text_length_bucket_{lower}_{upper}")

    ALIGNMENT_NORM = partial(get_model_label, "alignment_norm")
    ALIGNMENT_SMALL_MAX = partial(get_model_label, "alignment_num_small_max")
    ALIGNMENT_REPEATED = partial(get_model_label, "alignment_repeated")
    ALIGNMENT_SKIPS = partial(get_model_label, "alignment_skips")
    ALIGNMENT_JUMPS = partial(get_model_label, "alignment_jumps_v2")
    ALIGNMENT_STD = partial(get_model_label, "alignment_std")
    PREDICTED_PAUSE_FRAMES = partial(get_model_label, "predicted_pause_frames")
    AVERAGE_PREDICTED_RMS_LEVEL = partial(get_model_label, "average_predicted_rms_level")
    AVERAGE_RELATIVE_SPEED = partial(get_model_label, "average_relative_speed")
    AVERAGE_RMS_LEVEL_DELTA = partial(get_model_label, "average_rms_level_delta")
    REACHED_MAX_FRAMES = partial(get_model_label, "reached_max_frames")
    SPECTROGRAM_LOSS = partial(get_model_label, "spectrogram_loss")
    STOP_TOKEN_ACCURACY = partial(get_model_label, "stop_token_accuracy")
    STOP_TOKEN_LOSS = partial(get_model_label, "stop_token_loss")

    TEXT_LENGTH_BUCKET_SIZE = 25

    def get_dataset_values(
        self, batch: Batch, model: SpectrogramModel, preds: Preds
    ) -> MetricsValues:
        """
        TODO: Get dataset metrics on OOV words (spaCy and AmEPD) in our dataset.
        TODO: Create a `streamlit` for measuring coverage in our dataset, and other datasets.
        TODO: Measure the difference between punctuation in the phonetic vs grapheme phrases.
        Apart from unique cases, they should have the same punctuation.
        """
        values: MetricsValues = collections.defaultdict(float)
        for span, num_frames, tokens, has_reached_max in zip(
            batch.spans,
            self._to_list(batch.spectrogram.lengths),
            batch.tokens,
            self._to_list(preds.reached_max),
        ):
            # NOTE: Create a key for `self.NUM_SPANS` so a value exists, even if zero.
            values[(self.NUM_SPANS, None)] += float(not has_reached_max)
            values[(self.NUM_SPANS, span.speaker)] += float(not has_reached_max)

            # NOTE: Remove predictions that diverged (reached max) as to not skew other metrics.
            if model.training or not has_reached_max:
                index = int(len(span.script) // self.TEXT_LENGTH_BUCKET_SIZE)
                values[(self.NUM_SPANS_PER_TEXT_LENGTH, index)] += 1
                max_frames = values[(self.NUM_FRAMES_MAX, None)]
                values[(self.NUM_FRAMES_MAX, None)] = max(num_frames, max_frames)
                speaker: typing.Optional[Speaker]
                for speaker in [None, span.speaker]:
                    values[(self.NUM_FRAMES, speaker)] += num_frames
                    values[(self.NUM_SECONDS, speaker)] += span.audio_length
                    values[(self.NUM_TOKENS, speaker)] += len(tokens)
                    label = (self.MAX_FRAMES_PER_TOKEN, speaker)
                    values[label] = max(num_frames / len(tokens), values[label])

        return dict(values)

    def get_alignment_values(
        self, batch: Batch, model: SpectrogramModel, preds: Preds
    ) -> MetricsValues:
        values: MetricsValues = collections.defaultdict(float)
        for span, skipped, jumps, std, norm, small_max, repeated, length, has_reached_max in zip(
            batch.spans,
            self._to_list(get_num_skipped(preds)),
            self._to_list(get_num_jumps(preds)),
            self._to_list(get_alignment_std(preds)),
            self._to_list(get_alignment_norm(preds)),
            self._to_list(get_num_small_max(preds)),
            self._to_list(get_num_repeated(preds)),
            self._to_list(preds.num_frames),
            self._to_list(preds.reached_max),
        ):
            values[(self.NUM_REACHED_MAX, None)] += has_reached_max
            values[(self.NUM_REACHED_MAX, span.speaker)] += has_reached_max

            if model.training or not has_reached_max:
                for speaker in [None, span.speaker]:
                    values[(self.ALIGNMENT_NORM_SUM, speaker)] += norm
                    values[(self.ALIGNMENT_NUM_SMALL_MAX, speaker)] += small_max
                    values[(self.ALIGNMENT_NUM_REPEATED, speaker)] += repeated
                    values[(self.ALIGNMENT_NUM_SKIPS, speaker)] += skipped
                    values[(self.ALIGNMENT_NUM_JUMPS, speaker)] += jumps
                    values[(self.ALIGNMENT_STD_SUM, speaker)] += std
                    values[(self.NUM_FRAMES_PREDICTED, speaker)] += length

        return dict(values)

    def get_loudness_values(
        self, batch: Batch, model: SpectrogramModel, preds: Preds
    ) -> MetricsValues:
        values: MetricsValues = collections.defaultdict(float)
        loudness = get_power_rms_level_sum(batch.spectrogram.tensor, batch.spectrogram_mask.tensor)
        for span, loudness, pred_loudness, num_pause, num_pause_pred, has_reached_max in zip(
            batch.spans,
            self._to_list(loudness),
            self._to_list(get_power_rms_level_sum(preds.frames, preds.frames_mask)),
            get_num_pause_frames(batch.spectrogram.tensor, batch.spectrogram_mask.tensor),
            get_num_pause_frames(preds.frames, preds.frames_mask),
            self._to_list(preds.reached_max),
        ):
            if model.training or not has_reached_max:
                for speaker in [None, span.speaker]:
                    values[(self.RMS_SUM_PREDICTED, speaker)] += pred_loudness
                    values[(self.RMS_SUM, speaker)] += loudness
                    values[(self.NUM_PAUSE_FRAMES, speaker)] += num_pause
                    values[(self.NUM_PAUSE_FRAMES_PREDICTED, speaker)] += num_pause_pred

        return dict(values)

    def get_stop_token_values(
        self, batch: Batch, model: SpectrogramModel, preds: Preds
    ) -> MetricsValues:
        bool_ = lambda t: (t > model.stop_threshold).masked_select(batch.spectrogram_mask.tensor)
        is_correct = bool_(batch.stop_token.tensor) == bool_(torch.sigmoid(preds.stop_tokens))
        return {(self.NUM_CORRECT_STOP_TOKEN, None): float(is_correct.sum().item())}

    def _iter_permutations(self, select: _utils.MetricsSelect, is_verbose: bool = True):
        """Iterate over permutations of metric names and return convenience operations.

        Args:
            is_verbose: If `True`, iterate over more permutations.
        """
        speakers = [spk for _, spk in self.data.keys() if isinstance(spk, Speaker)]
        for speaker in itertools.chain([None], speakers if is_verbose else []):
            reduce_: typing.Callable[[str], float]
            reduce_ = lambda a, **k: self._reduce((a, speaker), select=select, **k)
            process: typing.Callable[[typing.Union[float, str]], typing.Union[float, MetricsKey]]
            process = lambda a: a if isinstance(a, float) else (a, speaker)
            div: typing.Callable[[typing.Union[float, str], typing.Union[float, str]], float]
            div = lambda n, d, **k: self._div(process(n), process(d), select=select, **k)
            yield speaker, reduce_, div

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
                self.ALIGNMENT_SMALL_MAX: div(
                    self.ALIGNMENT_NUM_SMALL_MAX, self.NUM_FRAMES_PREDICTED
                ),
                self.ALIGNMENT_REPEATED: div(self.ALIGNMENT_NUM_REPEATED, self.NUM_TOKENS),
                self.ALIGNMENT_STD: div(self.ALIGNMENT_STD_SUM, self.NUM_FRAMES_PREDICTED),
                self.ALIGNMENT_SKIPS: div(self.ALIGNMENT_NUM_SKIPS, self.NUM_TOKENS),
                self.ALIGNMENT_JUMPS: div(self.ALIGNMENT_NUM_JUMPS, self.NUM_FRAMES_PREDICTED),
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
            self.MAX_NUM_FRAMES: reduce((self.NUM_FRAMES_MAX, None), op=max),
            self.MIN_DATA_LOADER_QUEUE_SIZE: reduce((self.DATA_QUEUE_SIZE, None), op=min),
        }

        total_frames = reduce((self.NUM_FRAMES, None))
        total_seconds = reduce((self.NUM_SECONDS, None))
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
            total_spans = reduce((self.NUM_SPANS, None))
            for label, bucket in self.data.keys():
                if self.NUM_SPANS_PER_TEXT_LENGTH is not label:
                    continue

                assert isinstance(bucket, int), "Invariant error"
                lower = bucket * self.TEXT_LENGTH_BUCKET_SIZE
                upper = (bucket + 1) * self.TEXT_LENGTH_BUCKET_SIZE
                get_label = partial(self.FREQUENCY_TEXT_LENGTH, lower=lower, upper=upper)
                metrics[get_label] = reduce((label, bucket)) / total_spans

        return metrics

    def _get_loudness_metrics(self, select: _utils.MetricsSelect, is_verbose: bool) -> _GetMetrics:
        metrics = {}
        for speaker, _, div in self._iter_permutations(select, is_verbose):
            predicted_rms = power_to_db(div(self.RMS_SUM_PREDICTED, self.NUM_FRAMES_PREDICTED))
            rms = power_to_db(div(self.RMS_SUM, self.NUM_FRAMES))
            update = {
                self.AVERAGE_PREDICTED_RMS_LEVEL: predicted_rms,
                self.AVERAGE_RMS_LEVEL: rms,
                self.AVERAGE_RMS_LEVEL_DELTA: predicted_rms - rms,
                self.PAUSE_FRAMES: div(self.NUM_PAUSE_FRAMES, self.NUM_FRAMES),
                self.PREDICTED_PAUSE_FRAMES: div(
                    self.NUM_PAUSE_FRAMES_PREDICTED, self.NUM_FRAMES_PREDICTED
                ),
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
