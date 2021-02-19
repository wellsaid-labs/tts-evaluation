import collections
import collections.abc
import dataclasses
import math
import platform
import typing
from functools import partial

import torch
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, configurable
from third_party import get_parameter_norm
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

import lib
from lib.distributed import gather_list, is_master
from lib.utils import flatten
from run._config import Cadence, DatasetType, Label, get_dataset_label, get_model_label
from run.train._utils import CometMLExperiment
from run.train.spectrogram_model._data import DataLoader, InputEncoder, SpanBatch


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


def get_cumulative_power_rms_level(
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
        **kwargs: Additional key word arguments passed to `get_cumulative_power_rms_level`.

    Returns:
        torch.FloatTensor [batch_size]
    """
    num_elements = db_spectrogram.shape[0] if mask is None else mask.sum(dim=0)
    cumulative_power_rms_level = get_cumulative_power_rms_level(db_spectrogram, mask, **kwargs)
    return lib.audio.power_to_db(cumulative_power_rms_level / num_elements)


_Measurements = typing.List[float]
# NOTE: `_Reduce` reduces a list of measurements into a metric.
_Reduce = typing.Callable[[_Measurements], float]
_Metrics = typing.Dict[Label, typing.Optional[float]]


@dataclasses.dataclass
class Metrics:
    """Track metrics with measurements taken on every process for every step.

    TODO: Instead of using CUDA tensors, for synchronizing metadata and metrics, it's more natural
    to use a `TCPStore` on CPU. Furthermore, `TCPStore` with `json` could store variable length
    items like lists.
    Furthermore, we could create a generic metric manager. The workers will communicate with the
    manager by sending dictionaries. The master would start a thread that listens for, and
    accumulates metrics from the workers.
    This could also help reduce a lot of complexity this metrics implementation. There is a lot of
    code that's focused on appending.

    Args:
        ...
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
        num_predictions_per_speaker: The number of spectrograms predicted per speaker for each step.
        num_skips_per_speaker: In the predicted alignment, this tracks the number of tokens
            that were skipped per speaker. This could indicate that the model has issues, or that
            the dataset is flawed.
        num_tokens_per_speaker: The number of tokens per speaker for each step.
        frame_rms_level: This measures the sum of the RMS level for each frame in each step.
        text_length_bucket_size: This is a constant value bucket size for reporting the text
            length distribution.
        num_spans_per_text_length: For each text length bucket, this counts the number of spans.
        num_frames_per_speaker: For each speaker, this counts the number of spectrogram frames
            each step.
        num_seconds_per_speaker: For each speaker, this counts the number of seconds each step.
        num_frames_predicted: This measures the number of frames predicte each step.
        num_frames: This measures the number of frames in each step.
        max_num_frames: The maximum number of frames, in a spectrogram, seen.
        num_reached_max: This measures the number of predicted spectrograms that reach max frames
            each step.
        num_reached_max_per_speaker: This measures the number of predicted spectrograms that reach
            max frames each step per speaker.
        predicted_frame_rms_level: This measures the sum of the RMS level for each predicted frame
            in each step.
        spectrogram_loss: This measures the difference between the original and predicted
            spectrogram each step.
        stop_token_loss: This measures the difference between the original and predicted stop token
            distribution each step.
        stop_token_num_correct: This measures the number of correct stop token predictions each
            step.
    """

    comet: CometMLExperiment
    device: torch.device
    batch_size: typing.List[float] = dataclasses.field(default_factory=list)
    data_queue_size: typing.List[float] = dataclasses.field(default_factory=list)
    predicted_frame_alignment_norm: typing.List[float] = dataclasses.field(default_factory=list)
    predicted_frame_alignment_std: typing.List[float] = dataclasses.field(default_factory=list)
    num_predictions_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_skips_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_tokens_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    frame_rms_level: typing.List[float] = dataclasses.field(default_factory=list)
    text_length_bucket_size: int = 25
    num_spans_per_text_length: typing.Dict[int, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_frames_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_seconds_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    num_frames_predicted: typing.List[float] = dataclasses.field(default_factory=list)
    num_frames: typing.List[float] = dataclasses.field(default_factory=list)
    max_num_frames: int = dataclasses.field(default_factory=int)
    num_reached_max: typing.List[float] = dataclasses.field(default_factory=list)
    num_reached_max_per_speaker: typing.Dict[lib.datasets.Speaker, float] = dataclasses.field(
        default_factory=lambda: collections.defaultdict(float)
    )
    predicted_frame_rms_level: typing.List[float] = dataclasses.field(default_factory=list)
    spectrogram_loss: typing.List[float] = dataclasses.field(default_factory=list)
    stop_token_loss: typing.List[float] = dataclasses.field(default_factory=list)
    stop_token_num_correct: typing.List[float] = dataclasses.field(default_factory=list)

    def append(self, metric: typing.List[float], value: typing.Union[int, float, torch.Tensor]):
        """Append measurement to a `metric`.

        NOTE: The measurements will accrue on the master process only.
        """
        value = float(value.sum().item() if isinstance(value, torch.Tensor) else value)
        metric.append(lib.distributed.reduce(value, self.device))

    def gather(
        self, values: typing.Union[typing.List[float], torch.Tensor], **kwargs
    ) -> typing.List[float]:
        values = values.view(-1).tolist() if isinstance(values, torch.Tensor) else values
        return flatten(gather_list(values, device=self.device, **kwargs))

    def update_dataset_metrics(self, batch: SpanBatch, input_encoder: InputEncoder):
        """
        TODO: Get dataset metrics on OOV words (spaCy and AmEPD) in our dataset.

        TODO: Create a `streamlit` for measuring coverage in our dataset, and other datasets.

        TODO: Measure the difference between punctuation in the phonetic vs grapheme phrases.
        Apart from unique cases, they should have the same punctuation.
        """
        self.append(self.batch_size, batch.length)
        self.append(self.num_frames, batch.spectrogram.lengths)

        for text in self.gather([len(s.script) for s in batch.spans]):
            self.num_spans_per_text_length[int(text // self.text_length_bucket_size)] += 1

        iterator = zip(
            self.gather(batch.encoded_speaker.tensor),
            self.gather(batch.spectrogram.lengths),
            self.gather([s.audio_length for s in batch.spans]),
        )
        for speaker_index, num_frames, num_seconds in iterator:
            speaker = input_encoder.speaker_encoder.index_to_token[int(speaker_index)]
            self.num_frames_per_speaker[speaker] += num_frames
            self.num_seconds_per_speaker[speaker] += num_seconds
            self.max_num_frames = max(self.max_num_frames, int(num_frames))

    def update_alignment_metrics(
        self,
        alignments: torch.Tensor,
        spectrogram_mask: torch.Tensor,
        token_mask: torch.Tensor,
        num_tokens: torch.Tensor,
        speakers: torch.Tensor,
        input_encoder: InputEncoder,
        norm: float = math.inf,
    ):
        """
        Args:
            alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
            spectrogram_mask (torch.BoolTensor [num_frames, batch_size])
            token_mask (torch.BoolTensor [num_tokens, batch_size])
            num_tokens (torch.LongTensor [1, batch_size])
            speakers (torch.LongTensor [1, batch_size])
            ...
        """
        mask = lambda t: t.masked_select(spectrogram_mask)
        weighted_std = lib.utils.get_weighted_std(alignments, dim=2)
        self.append(self.predicted_frame_alignment_std, mask(weighted_std))
        self.append(self.predicted_frame_alignment_norm, mask(alignments.norm(norm, dim=2)))

        num_skipped = get_num_skipped(alignments, token_mask, spectrogram_mask)

        assert speakers.numel() == num_skipped.numel()
        assert speakers.numel() == num_tokens.numel()
        iterator = zip(self.gather(speakers), self.gather(num_skipped), self.gather(num_tokens))
        for speaker_index, _num_skipped, _num_tokens in iterator:
            speaker = input_encoder.speaker_encoder.index_to_token[int(speaker_index)]
            self.num_skips_per_speaker[speaker] += _num_skipped
            self.num_tokens_per_speaker[speaker] += _num_tokens

    def update_reached_max_metrics(
        self, batch: SpanBatch, input_encoder: InputEncoder, reached_max: torch.Tensor
    ):
        """
        Args:
            ...
            reached_max (torch.BoolTensor [1, batch_size])
        """
        self.append(self.num_reached_max, reached_max)
        iterator = zip(self.gather(batch.encoded_speaker.tensor), self.gather(reached_max))
        for speaker_index, has_reached_max in iterator:
            speaker = input_encoder.speaker_encoder.index_to_token[int(speaker_index)]
            self.num_reached_max_per_speaker[speaker] += has_reached_max
            self.num_predictions_per_speaker[speaker] += 1

    def update_rms_metrics(
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
        rms_ = lambda s, m: get_cumulative_power_rms_level(s, m, **kwargs)
        self.append(self.frame_rms_level, rms_(target_spectrogram, target_mask))
        self.append(self.predicted_frame_rms_level, rms_(predicted_spectrogram, predicted_mask))

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

    def update_data_queue_size(self, data_loader: DataLoader):
        # NOTE: `qsize` is not implemented on MacOS, learn more:
        # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.Queue.qsize
        is_multiprocessing = isinstance(data_loader.loader, _MultiProcessingDataLoaderIter)
        if is_multiprocessing and platform.system() != "Darwin":
            iterator = typing.cast(_MultiProcessingDataLoaderIter, data_loader.loader)
            self.append(self.data_queue_size, iterator._data_queue.qsize())

    def log_optimizer_metrics(
        self, optimizer: torch.optim.Adam, clipper: lib.optimizers.AdaptiveGradientNormClipper
    ):
        """Log optimizer metrics for `optimizer` and `clipper`. The model parameters have already
        been sync'd; therefore, there is no need to further sync parameters.

        TODO: Incorperate `optimizer_metrics` into the standard `Metrics` usage.
        """
        label_ = partial(get_model_label, cadence=Cadence.STEP)
        log = lambda n, v: self.comet.log_metric(label_(n), v)

        assert len(optimizer.param_groups) == 1, "Expecting only 1 group of parameters."
        param_group = optimizer.param_groups[0]
        parameter_norm = get_parameter_norm(param_group["params"], 2)
        parameter_norm_inf = get_parameter_norm(param_group["params"], math.inf)
        assert torch.isfinite(parameter_norm), f"Gradient was too large {parameter_norm}."
        assert torch.isfinite(parameter_norm_inf), f"Gradient was too large {parameter_norm_inf}."
        log("grad_norm/two", parameter_norm.item())
        log("grad_norm/inf", parameter_norm_inf.item())
        log("lr", param_group["lr"])

        if math.isfinite(clipper.max_norm):  # NOTE: Initially, `max_norm` will be `inf`.
            log("grad_norm/max_norm", clipper.max_norm)

    @staticmethod
    def _div(num: _Measurements, denom: _Measurements, reduce: _Reduce) -> typing.Optional[float]:
        if len(num) == 0 or len(denom) == 0 or reduce(denom) == 0:
            return None
        return reduce(num) / reduce(denom)

    @configurable
    def get_model_metrics(self, reduce: _Reduce, num_frame_channels=HParam(), **kwargs) -> _Metrics:
        """ Get model metrics. """
        div = partial(self._div, reduce=reduce)
        spectrogram_loss = div(self.spectrogram_loss, self.num_frames)
        if spectrogram_loss is not None:
            spectrogram_loss /= num_frame_channels
        metrics = {
            "alignment_norm": div(self.predicted_frame_alignment_norm, self.num_frames_predicted),
            "alignment_std": div(self.predicted_frame_alignment_std, self.num_frames_predicted),
            "average_relative_speed": div(self.num_frames_predicted, self.num_frames),
            "stop_token_accuracy": div(self.stop_token_num_correct, self.num_frames),
            "stop_token_loss": div(self.stop_token_loss, self.num_frames),
            "spectrogram_loss": spectrogram_loss,
        }
        return {get_model_label(k, **kwargs): v for k, v in metrics.items()}

    def get_dataset_metrics(self, reduce: _Reduce, **kwargs) -> _Metrics:
        """ Get generic dataset metrics. """
        div = partial(self._div, reduce=reduce)
        metrics = {
            "data_loader_queue_size": div(self.data_queue_size, [1] * len(self.data_queue_size)),
            "average_num_frames": div(self.num_frames, self.batch_size),
            "max_num_frames": self.max_num_frames,
        }
        return {get_dataset_label(k, **kwargs): v for k, v in metrics.items()}

    @staticmethod
    def _rms(num: _Measurements, denom: _Measurements, reduce: _Reduce) -> typing.Optional[float]:
        power_rms_level = Metrics._div(num, denom, reduce)
        if power_rms_level is not None:
            return float(lib.audio.power_to_db(torch.tensor(power_rms_level)).item())
        return None

    def get_rms_metrics(self, reduce: _Reduce, cadence: Cadence, type_: DatasetType) -> _Metrics:
        """Get loudness metrics in RMS dB."""
        predicted_rms = self._rms(self.predicted_frame_rms_level, self.num_frames_predicted, reduce)
        rms = self._rms(self.frame_rms_level, self.num_frames, reduce)
        delta = None if predicted_rms is None or rms is None else predicted_rms - rms
        return {
            get_model_label("average_predicted_rms_level", cadence=cadence): predicted_rms,
            get_dataset_label("average_rms_level", cadence=cadence, type_=type_): rms,
            get_model_label("average_rms_level_delta", cadence=cadence): delta,
        }

    def get_text_length_metrics(self, **kwargs) -> _Metrics:
        """ Get metrics summarizing text length bucket frequency. """
        metrics = {}
        for bucket, count in self.num_spans_per_text_length.items():
            lower = bucket * self.text_length_bucket_size
            upper = (bucket + 1) * self.text_length_bucket_size
            label = get_dataset_label(f"text_length_bucket_{lower}_{upper}", **kwargs)
            metrics[label] = count / sum(self.num_spans_per_text_length.values())
        return metrics

    def get_speaker_frequency_metrics(self, **kwargs) -> _Metrics:
        """Get metrics summarizing speaker frequency.

        NOTE: There is a discrepency between `num_frames_per_speaker` and `num_seconds_per_speaker`
        because `num_seconds_per_speaker` is computed with `Span` and `num_frames_per_speaker`
        is computed with `SpanBatch`. For example, in Feburary 2021, `make_span_batch` used
        `_pad_and_trim_signal`. "elizabeth_klett", "mary_ann" and "judy_bieber" tend to need
        significantly more trimming than other speakers."""
        metrics = {}
        label = lambda l, **k: get_dataset_label(f"frequency/{l}", **k, **kwargs)

        total = sum(self.num_frames_per_speaker.values())
        for speaker, count in self.num_frames_per_speaker.items():
            metrics[label("num_frames", speaker=speaker)] = count / total

        total = sum(self.num_seconds_per_speaker.values())
        for speaker, count in self.num_seconds_per_speaker.items():
            metrics[label("num_seconds", speaker=speaker)] = count / total

        return metrics

    def get_attention_skip_metrics(self, **kwargs) -> _Metrics:
        """ Get metrics on token skipping per speaker via attention. """
        metrics = {}
        zip_ = zip(self.num_tokens_per_speaker.items(), self.num_skips_per_speaker.values())
        for (speaker, num_tokens), num_skips in zip_:
            metrics[get_model_label("skips", speaker=speaker, **kwargs)] = num_skips / num_tokens
        return metrics

    def get_reached_max_frames(self, reduce: _Reduce):
        """NOTE: The predicted `self.batch_size` does not include predictions that overflowed. The
        total number of predicted spectrograms is equal to `batch_size` plus `num_reached_max`.
        """
        if len(self.num_reached_max) == 0 or len(self.batch_size) == 0:
            return None
        denom = reduce(self.batch_size) + reduce(self.num_reached_max)
        if denom == 0:
            return None
        return reduce(self.num_reached_max) / denom

    def get_reached_max_metrics(self, reduce: _Reduce, **kwargs) -> _Metrics:
        """ Get metrics on model overflow per speaker via attention. """
        name = "reached_max_frames"
        metrics = {get_model_label(name, **kwargs): self.get_reached_max_frames(reduce)}
        for (speaker, num_predictions), num_reached_max in zip(
            self.num_predictions_per_speaker.items(), self.num_reached_max_per_speaker.values()
        ):
            label = get_model_label(name, speaker=speaker, **kwargs)
            metrics[label] = num_reached_max / num_predictions
        return metrics

    def log(self, reduce: _Reduce, dataset_type: DatasetType, cadence: Cadence):
        """Log metrics to `self.comet`.

        NOTE: Comet is limited in the number of metrics it can handle on a step-by-step basis.
        It will throttle experiments reporting too many metrics, or it's UI will lag behind.
        """
        if not is_master():
            return

        metrics = {
            **self.get_model_metrics(reduce=reduce, cadence=cadence),
            **self.get_dataset_metrics(reduce=reduce, cadence=cadence, type_=dataset_type),
        }

        if cadence is not Cadence.STEP:
            more_metrics = {
                **self.get_rms_metrics(reduce=reduce, cadence=cadence, type_=dataset_type),
                **self.get_text_length_metrics(cadence=cadence, type_=dataset_type),
                **self.get_speaker_frequency_metrics(cadence=cadence, type_=dataset_type),
                **self.get_attention_skip_metrics(cadence=cadence),
                **self.get_reached_max_metrics(reduce=reduce, cadence=cadence),
            }
            metrics.update(more_metrics)

        self.comet.log_metrics({k: v for k, v in metrics.items() if v is not None})
