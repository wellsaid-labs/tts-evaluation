import dataclasses
import functools
import logging
import random
import sys
import typing
from concurrent import futures

import config as cf
import numpy
import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from third_party import LazyLoader
from torchnlp.encoders.text import SequenceBatch, stack_and_pad_tensors
from torchnlp.samplers import DeterministicSampler, DistributedBatchSampler

import lib
import run
from lib.audio import sec_to_sample
from lib.distributed import get_rank, get_world_size, is_initialized
from lib.samplers import BucketBatchSampler
from lib.text import XMLType, load_cmudict_syl, respell
from lib.utils import flatten_2d, lengths_to_mask, random_nonoverlapping_intervals
from run._models.spectrogram_model import (
    Inputs,
    PreprocessedInputs,
    SpanAnnotations,
    TokenAnnotations,
    preprocess,
)
from run.data._loader.structures import Alignment, Span, Speaker
from run.train import _utils

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    from scipy import ndimage
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    ndimage = LazyLoader("ndimage", globals(), "scipy.ndimage")


logger = logging.getLogger(__name__)


def _random_nonoverlapping_alignments(
    alignments: typing.Iterable[Alignment],
    avg_alignments: int,
    min_no_intervals_prob: float,
    include_annotation: typing.Callable[[Alignment], bool] = lambda a: True,
) -> typing.Tuple[Alignment, ...]:
    """Generate a random set of non-overlapping alignments.

    NOTE: This will undershoot `avg_alignments` for several reasons because the alignments might
          overlap and be filtered out.

    Args:
        alignments
        avg_alignments: The average number of alignments to return when alignments are returned.
        min_no_intervals_prob: The minimum probability for sampling no intervals. In practice,
            no intervals will be sampled at a slightly higher rate due to the implementation
            quirks.

    Returns: A tuple of non-overlapping alignments that start and end on a boundary. This may
        return no intervals in some cases.
    """
    if random.random() < min_no_intervals_prob:
        return tuple()

    get_ = lambda a, i: tuple([getattr(a, f)[i] for f in Alignment._fields])
    # NOTE: Each of these is a synchronization point along which we can match up the script
    # character, transcript character, and audio sample. We can use any of these points for
    # cutting.
    bounds = flatten_2d([[get_(a, 0), get_(a, -1)] for a in alignments])
    # NOTE: Depending on the parameters, this has some probability of generating no intervals.
    indicies = random_nonoverlapping_intervals(len(bounds), avg_alignments)
    intervals = [(bounds[a], bounds[b]) for (a, b) in indicies]
    # NOTE: Alignments may have overlapping audio segments, we remove those. In practice, this
    # doesn't happen that often in our data, as reviewed in:
    # `run/review/dataset_processing/span_annotation_generation.py`
    ret_ = [Alignment((a[0], b[0]), (a[1], b[1]), (a[2], b[2])) for a, b in intervals]
    ret_ = [a for a in ret_ if a.audio[0] < a.audio[1] and include_annotation(a)]
    return tuple(ret_)


def _get_loudness_annotation(
    audio: numpy.ndarray,
    sample_rate: int,
    alignment: Alignment,
    block_size: float,
    precision: int,
    **kwargs,
) -> typing.Optional[float]:
    """Get the loudness in LUFS for an `alignment` in `audio`.

    NOTE: `integrated_loudness` filters our quiet sections from the loudness computations.
    NOTE: The minimum audio length for calculating loudness is the `block_size` which is typically
    around 400ms.

    Args:
        ...
        precision: The number of decimal places to round LUFS.

    Returns: The loundess in LUFS with a range of 0 to -70 LUFS in alignment with ITU-R BS.1770-4.
        This returns `None` if the loundess cannot be computed.
    """
    meter = lib.audio.get_pyloudnorm_meter(sample_rate, block_size=block_size, **kwargs)
    sec_to_sample_ = functools.partial(sec_to_sample, sample_rate=sample_rate)
    slice_ = audio[sec_to_sample_(alignment.audio[0]) : sec_to_sample_(alignment.audio[1])]
    if slice_.shape[0] >= sec_to_sample_(block_size):
        loudness = round(meter.integrated_loudness(slice_), precision)
        # NOTE: This algorithm returns negative infinity if the loudness is less than -70 LUFS. We
        # return -70 LUFS instead to keep the output finite.
        # NOTE: This number is not parameterized because this specific number is specified in
        # the LUFS algorithm specification, ITU-R BS.1770-4.
        # NOTE: The loudness algorithm can sometimes overflow and return stange values that are
        # significantly outside of the range like in:
        # https://github.com/csteinmetz1/pyloudnorm/issues/42
        loudness = -70 if numpy.isinf(loudness) and loudness < 0 else loudness
        return None if loudness > 0 or loudness < -70 else loudness
    return None


def _random_loudness_annotations(span: Span, signal: numpy.ndarray, **kwargs) -> SpanAnnotations:
    """Create random annotations that represent the loudness in `span.script`."""
    annotations: SpanAnnotations = []
    alignments = cf.partial(_random_nonoverlapping_alignments)(span.speech_segments)
    for alignment in alignments:
        slice_ = slice(alignment.script[0], alignment.script[1])
        sample_rate = span.audio_file.sample_rate
        loudness_ = cf.call(_get_loudness_annotation, signal, sample_rate, alignment, **kwargs)
        if loudness_ is not None:
            annotations.append((slice_, loudness_))
    return annotations


def _random_tempo_annotations(
    span: Span, get_tempo_annotation: typing.Callable[[Span, Alignment], float], **kwargs
) -> SpanAnnotations:
    """Create random annotations that represent the speaking tempo in `span.script`.

    TODO: We should investigate a more accurate speech tempo, there are a couple options here:
    https://en.wikipedia.org/wiki/Speech_tempo
    TODO: We should investiage the correlation of speech tempo to audio length depending on the
    audio length. Are there thresholds for which it is more likely to be incorrect?

    Args:
        span
    """
    annotations: SpanAnnotations = []
    alignments = cf.partial(_random_nonoverlapping_alignments)(span.speech_segments)
    for alignment in alignments:
        slice_ = slice(alignment.script[0], alignment.script[1])
        annotation = get_tempo_annotation(span, alignment, **kwargs)
        annotations.append((slice_, annotation))
    return annotations


def _random_respelling_annotations(span: Span, prob: float, delim: str) -> TokenAnnotations:
    """Create random annotations for different respellings in `span.script`.

    NOTE: We annotate only basic scenarios. A more complex scenario, for example,
          is apostrophes. spaCy, by default, splits some (not all) words on apostrophes while our
          pronunciation dictionary does not; therefore, those words will not be found in it.
    """
    annotations: TokenAnnotations = {}
    tokens = list(span.spacy)
    for prev, token, next_ in zip([None] + tokens[:-1], tokens, tokens[1:] + [None]):
        if random.random() > prob:
            continue
        if prev is not None and len(prev.whitespace_) == 0 and not prev.is_punct:
            continue
        if next_ is not None and len(token.whitespace_) == 0 and not next_.is_punct:
            continue
        respelling = respell(token.text, load_cmudict_syl(), delim)
        if respelling is not None:
            annotations[token] = respelling
    return annotations


def _pad_and_trim_signal(signal: numpy.ndarray) -> torch.Tensor:
    """Pad signal length and trim any extra silence.

    TODO: The RMS function that `librosa.effects.trim` uses mentions that it's likely better to use
    a spectrogram if it's available:
    https://librosa.github.io/librosa/generated/librosa.feature.rms.html?highlight=rms#librosa.feature.rms
    TODO: The RMS function is a naive computation of loudness; therefore, it'd likely
    be more accurate to use our spectrogram for trimming with augmentations like A-weighting.
    TODO: `pad_remainder` could possibly add distortion if it's appended to non-zero samples;
    therefore, it'd likely be beneficial to have a small fade-in and fade-out before
    appending the zero samples.
    TODO: We should consider padding more than just the remainder. We could additionally
    pad a `frame_length` of padding so that further down the pipeline, any additional
    padding does not affect the spectrogram due to overlap between the padding and the
    real audio.
    TODO: Instead of padding with zeros, we should consider padding with real-data.
    TODO: Add some randomness to the audio file trimming so that the stop token cannot overfit.
    """
    signal = lib.audio.pad_remainder(signal, **cf.get())
    _, trim = librosa.effects.trim(signal, **cf.get())
    return torch.tensor(signal[trim[0] : trim[1]], requires_grad=False)


@functools.lru_cache(maxsize=None)
def _get_signal_to_db_mel_spectrogram_helper(**kwargs):
    return lib.audio.SignalTodBMelSpectrogram(**kwargs)


def _get_signal_to_db_mel_spectrogram(**kwargs):
    """Get cached `SignalTodBMelSpectrogram` module."""
    kwargs = {**cf.get(func=lib.audio.SignalTodBMelSpectrogram), **kwargs}
    return _get_signal_to_db_mel_spectrogram_helper(**kwargs)


def _signals_to_spectrograms(
    signals: typing.List[torch.Tensor], **kwargs
) -> typing.Tuple[SequenceBatch, SequenceBatch]:
    """Create a spectrogram batch from a batch of signals.

    Returns:
        spectrogram (SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
            torch.LongTensor [1, batch_size]))
        spectrogram_mask (SequenceBatch[torch.BoolTensor [num_frames, batch_size],
            torch.LongTensor [1, batch_size]))
    """
    signal_to_spectrogram = _get_signal_to_db_mel_spectrogram(**kwargs)
    signals_ = stack_and_pad_tensors(signals)
    db_mel_spectrogram = signal_to_spectrogram(signals_.tensor, aligned=True)
    lengths = torch.div(signals_.lengths, signal_to_spectrogram.frame_hop, rounding_mode="floor")
    mask = lengths_to_mask(lengths)
    db_mel_spectrogram = (db_mel_spectrogram * mask.unsqueeze(-1)).transpose(0, 1)
    return (
        SequenceBatch(db_mel_spectrogram, lengths.unsqueeze(0)),
        SequenceBatch(mask.transpose(0, 1), lengths.unsqueeze(0)),
    )


def _get_normalized_half_gaussian(length: int, standard_deviation: float) -> torch.Tensor:
    """Get a normalized half guassian distribution.

    Learn more:
    https://en.wikipedia.org/wiki/Half-normal_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    Args:
        length: The size of the gaussian filter.
        standard_deviation: The standard deviation of the guassian.

    Returns:
        (torch.FloatTensor [length,])
    """
    kernel = numpy.float_([0] * (length - 1) + [1])  # type: ignore
    kernel = ndimage.gaussian_filter1d(kernel, sigma=standard_deviation)  # type: ignore
    kernel = kernel / kernel.max()
    return torch.tensor(kernel).float()


def _make_stop_token(spectrogram: SequenceBatch, length: int, standard_deviation: float):
    """Create a batch of stop tokens from a spectrogram batch.

    NOTE: The exact stop token distribution is uncertain because there are multiple valid
    stopping points after someone has finished speaking. For example, the audio can be cutoff
    0.1 second or 0.5 seconds after someone has finished speaking. In order to address this
    uncertainty, we naively apply a normal distribution as the stop token ground truth.
    NOTE: This strategy was found to be effective via Comet in January 2020.

    TODO: In the future, it'd likely be more accurate to base the probability for stopping
    based on the loudness of each frame. The maximum loudness is based on a full-scale sine wave
    and the minimum loudness would be -96 Db or so. The probability for stopping is the loudness
    relative to the minimum and maximum loudness. This is assuming that at the end of an audio
    clip it gets progressively quieter.

    TODO: Since some speakers are quieter than others, it might more beneficial to create a
    `stop_token` based on the idea of "voice activity detection". For example, the `stop_token`
    is a 1 if the speaker is speaking and 0 if the speaker isn't speaking. The model will learn
    to stop when the speaker stop speaking. Or we could bake in "voice activity detection" when
    we create the spans, originally.

    Args:
        spectrogram (SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
            torch.LongTensor [1, batch_size]))
        length: The range of uncertainty there is in the exact `stop_token` location.
        standard_deviation: The standard deviation of uncertainty there is in the exact
            `stop_token` location.

    Returns:
        stop_token (SequenceBatch[torch.FloatTensor [num_frames, batch_size],
            torch.LongTensor [1, batch_size]))
    """
    # [num_frames, batch_size, frame_channels] → [num_frames, batch_size]
    stop_token = spectrogram.tensor.new_zeros(spectrogram.tensor.shape[0:2])
    gaussian_kernel = _get_normalized_half_gaussian(length, standard_deviation)
    for i in range(spectrogram.tensor.shape[1]):
        stop_token_length = int(spectrogram.lengths[0, i].item())
        min_ = min(stop_token_length, gaussian_kernel.shape[0])
        slice_ = slice(stop_token_length - min_, stop_token_length)
        stop_token[:, i][slice_] = gaussian_kernel[-min_:]
    return SequenceBatch(stop_token, spectrogram.lengths)


@dataclasses.dataclass(frozen=True)
class Batch(_utils.Batch):
    """Batch of preprocessed `Span` used to training or evaluating the spectrogram model."""

    spans: typing.List[Span]

    audio: typing.List[torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #               torch.LongTensor [1, batch_size])
    spectrogram: SequenceBatch

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    spectrogram_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    stop_token: SequenceBatch

    xmls: typing.List[XMLType]

    processed: PreprocessedInputs

    def apply(self, call: typing.Callable[[torch.Tensor], torch.Tensor]) -> "Batch":
        batch: Batch = super().apply(call)
        token_embed = batch.processed.token_embeddings_padded
        set_ = object.__setattr__
        set_(batch.processed, "token_embeddings_padded", call(token_embed))
        set_(batch.processed, "num_tokens", call(batch.processed.num_tokens))
        set_(batch.processed, "tokens_mask", call(batch.processed.tokens_mask))
        set_(batch.processed, "anno_mask", call(batch.processed.anno_mask))
        set_(batch.processed, "max_audio_len_tensor", call(batch.processed.max_audio_len_tensor))
        return batch

    def __len__(self):
        return len(self.spans)


def make_batch(spans: typing.List[Span], max_workers: int = 6) -> Batch:
    """
    NOTE: In Janurary 2020, this function profiled like so:
    - 27% for `_signals_to_spectrograms`
    - 25% on `nlp.pipe`
    - 13% on `_random_loudness_annotations`
    - 13% on `grapheme_to_phoneme` (Removed in March 2022)
    - 6% for `_pad_and_trim_signal`
    - 5% for `input_encoder.encode` (Removed in Feburary 2022)
    - 4% on `stack_and_pad_tensors` (Removed in Feburary 2022)

    TODO: For `spectrogram_model` training, this function is critical for performance and
    reducing the number of CPUs needed for training. Here are some opportunities for
    performance:
    - Using `jit` or `numpy` or `cuda` for a faster spectrogram calculation.
    - Precomputing `nlp.pipe` and caching the results.
    - Using `multiprocessing` for `grapheme_to_phoneme`.
    - Using the precomputed spectrogram for `_pad_and_trim_signal`.
    """
    assert len(spans) > 0, "Batch must have at least one item."

    with futures.ThreadPoolExecutor(max_workers=min(max_workers, len(spans))) as pool:
        # TODO: Should `audio` be cached so that we do not need to pass it around? It is a slightly
        # redunant to pass it around.
        signals_ = list(pool.map(lambda s: s.audio(), spans))
        signals_ = typing.cast(typing.List[numpy.ndarray], signals_)
    signals = [_pad_and_trim_signal(s) for s in signals_]
    spectrogram, spectrogram_mask = _signals_to_spectrograms(signals)
    inputs = Inputs(
        session=[s.session for s in spans],
        span=[s.spacy for s in spans],
        context=[cf.partial(s.spacy_context)() for s in spans],
        loudness=[_random_loudness_annotations(s, a) for s, a in zip(spans, signals_)],
        tempo=[cf.partial(_random_tempo_annotations)(s) for s in spans],
        respellings=[cf.partial(_random_respelling_annotations)(s) for s in spans],
    )
    # NOTE: `inputs` has a spaCy `Span` which is difficult to `pickle`, so instead, we seralize
    # `inputs` into XML.
    xmls = [inputs.to_xml(i, include_context=True) for i in range(len(inputs))]
    processed = cf.partial(preprocess)(inputs)
    # NOTE: These tensors are not needed, and are taking up memory.
    object.__setattr__(processed, "token_embeddings", None)

    return Batch(
        # NOTE: Prune unused attributes from `Passage` by creating a new `Passage`, in order to
        # reduce batch size, which in turn makes it easier to send to other processes, for example.
        spans=[dataclasses.replace(s, passage=dataclasses.replace(s.passage)) for s in spans],
        audio=signals,
        spectrogram=spectrogram,
        spectrogram_mask=spectrogram_mask,
        stop_token=cf.partial(_make_stop_token)(spectrogram),
        xmls=xmls,
        processed=processed,
    )


class DataProcessor(typing.Mapping[int, Batch]):
    def __init__(self, generator: run._utils.SpanGenerator, batch_size: int, step: int = 0):
        """Given an index, generate the appropriate batch indefinitely.

        NOTE: Our training procedure is similar to BERT, the examples are randomly sampled
        from a large corpus of data with `SpanGenerator`.

        TODO: The `DataProcessor`s state is difficult to checkpoint; therefore, the data pipeline
        isn't deterministic. Here are a couple of ideas to make it deterministic:
        - Implement a conditional `DataProcessor`, so that it's output is deterministic based
          on the step. This is doable as long as we can coordinate `BucketBatchSampler` and
          `DeterministicIterator`. Today, the batch generated depends on these random variables:
          random generator state, the bucket, the bucket batch index. Given a step, we'd need
          to generate the correct bucket and batch.
        - Create `DataProcessor` from scratch each time and fast forward the random state
          to the correct step. We'd likely need to implement a faster fast forward method because
          today it'd take 1 hour to generate 100,000 steps.
        - Checkpoint the random state of every worker, and use it to restart those workers. We'd
          likely need to setup a communication channel with workers, in order to implement this.
        """
        iter_ = BucketBatchSampler(generator, batch_size, False, self._data_iterator_sort_key)
        iter_ = DeterministicSampler(iter_, run._config.RANDOM_SEED + step, cuda=False)
        if is_initialized():
            iter_ = DistributedBatchSampler(iter_, num_replicas=get_world_size(), rank=get_rank())
        iter_ = typing.cast(typing.Iterator[typing.List[Span]], iter_)
        self.index_to_spans = lib.utils.MappedIterator(iter_)

    @staticmethod
    def _data_iterator_sort_key(span: Span):
        return span.audio_length

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(self) -> int:
        return sys.maxsize

    def __getitem__(self, index) -> Batch:
        return make_batch(self.index_to_spans[index])


def train_get_weight(speaker: Speaker, dataset_size: float):
    # TODO: The dictionary datasets are small, making up, roughly 1/17th of the training dataset;
    # however, they have many new words. In attempt to get the model to better learn pronunciation,
    # give 5x more weight to that dataset, so, it'll come up 5x more times during training.
    return dataset_size


def dev_get_weight(speaker: Speaker, dataset_size: float):
    # NOTE: For the dev set, we evaluate each speaker in production, equally.
    return 1.0
