import dataclasses
import functools
import logging
import random
import sys
import typing
from concurrent import futures

import numpy
import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import torch.utils
import torch.utils.data
from hparams import HParam, configurable
from third_party import LazyLoader
from torchnlp.encoders import Encoder, LabelEncoder
from torchnlp.encoders.text import (
    CharacterEncoder,
    DelimiterEncoder,
    SequenceBatch,
    stack_and_pad_tensors,
)
from torchnlp.samplers import DeterministicSampler, DistributedBatchSampler
from torchnlp.utils import lengths_to_mask

import lib
import run
from lib.audio import sec_to_sample
from lib.distributed import get_rank, get_world_size, is_initialized
from lib.samplers import BucketBatchSampler
from lib.utils import Tuple, flatten_2d
from run.data._loader import Alignment, Language, Session, Span, Speaker
from run.train import _utils

if typing.TYPE_CHECKING:  # pragma: no cover
    import librosa
    import spacy.tokens
    from scipy import ndimage
else:
    librosa = LazyLoader("librosa", globals(), "librosa")
    ndimage = LazyLoader("ndimage", globals(), "scipy.ndimage")
    spacy = LazyLoader("spacy", globals(), "spacy")


logger = logging.getLogger(__name__)


def _random_nonoverlapping_alignments(
    alignments: Tuple[Alignment], max_alignments: int
) -> typing.Tuple[Alignment, ...]:
    """Generate a random set of non-overlapping alignments, such that every point in the
    time-series has an equal probability of getting sampled inside an alignment.

    NOTE: The length of the sampled alignments is non-uniform.

    Args:
        alignments
        max_alignments: The maximum number of alignments to generate.
    """
    get_ = lambda a, i: tuple([getattr(a, f)[i] for f in Alignment._fields])
    # NOTE: Each of these is a synchronization point along which we can match up the script
    # character, transcript character, and audio sample. We can use any of these points for
    # cutting.
    bounds = flatten_2d([[get_(a, 0), get_(a, -1)] for a in alignments])
    num_cuts = random.randint(0, int(lib.utils.clamp(max_alignments, min_=0, max_=len(bounds) - 1)))

    if num_cuts == 0:
        return tuple()

    if num_cuts == 1:
        tuple_ = lambda i: (bounds[0][i], bounds[-1][i])
        alignment = Alignment(tuple_(0), tuple_(1), tuple_(2))
        return tuple([alignment]) if random.choice((True, False)) else tuple()

    # NOTE: Functionally, this is similar to a 50% dropout on intervals.
    # NOTE: Each alignment is expected to be included half of the time.
    intervals = bounds[:1] + random.sample(bounds[1:-1], num_cuts - 1) + bounds[-1:]
    return_ = [
        Alignment((a[0], b[0]), (a[1], b[1]), (a[2], b[2]))
        for a, b in zip(intervals, intervals[1:])
        if random.choice((True, False))
    ]
    return tuple(return_)


@configurable
def _get_loudness(
    audio: numpy.ndarray,
    sample_rate: int,
    alignment: Alignment,
    block_size: float = HParam(),
    precision: int = HParam(),
    **kwargs,
) -> typing.Optional[float]:
    """Get the loudness in LUFS for an `alignment` in `audio`.

    TODO: `integrated_loudness` filters our quiet sections from the loudness computations.
    Should this be disabled?

    Args:
        ...
        precision: The number of decimal places to round LUFS.
        ...
    """
    meter = lib.audio.get_pyloudnorm_meter(sample_rate, block_size=block_size, **kwargs)
    sec_to_sample_ = functools.partial(sec_to_sample, sample_rate=sample_rate)
    slice_ = audio[sec_to_sample_(alignment.audio[0]) : sec_to_sample_(alignment.audio[1])]
    if slice_.shape[0] >= sec_to_sample_(block_size):
        return round(meter.integrated_loudness(slice_), precision)
    return None


@configurable
def _random_loudness_annotations(
    span: Span, signal: numpy.ndarray, max_annotations: int = HParam()
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        ...
        max_annotations: The maximum expected loudness intervals within a text segment.
    """
    loudness = torch.zeros(len(span.script))
    loudness_mask = torch.zeros(len(span.script), dtype=torch.bool)
    for alignment in _random_nonoverlapping_alignments(span.alignments, max_annotations):
        slice_ = slice(alignment.script[0], alignment.script[1])
        loudness_ = _get_loudness(signal, span.audio_file.sample_rate, alignment)
        if loudness_ is not None:
            loudness[slice_] = loudness_
            loudness_mask[slice_] = True
    return loudness, loudness_mask


@configurable
def _random_speed_annotations(
    span: Span, max_annotations: int = HParam(), precision: int = HParam()
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        span
        max_annotations: The maximum expected speed intervals within a text segment.
        precision: The number of decimal places to round phonemes per second.
    """
    speed = torch.zeros(len(span.script))
    speed_mask = torch.zeros(len(span.script), dtype=torch.bool)
    for alignment in _random_nonoverlapping_alignments(span.alignments, max_annotations):
        slice_ = slice(alignment.script[0], alignment.script[1])
        # TODO: Instead of using characters per second, we could estimate the number of phonemes
        # with `grapheme_to_phoneme`. This might be slow, so we'd need to do so in a batch.
        # `grapheme_to_phoneme` can only estimate the number of phonemes because we can't
        # incorporate sufficient context to get the actual phonemes pronounced by the speaker.
        second_per_char = (alignment.audio[1] - alignment.audio[0]) / (slice_.stop - slice_.start)
        speed[slice_] = round(second_per_char, precision)
        speed_mask[slice_] = True
    return speed, speed_mask


def _get_char_to_word(doc: spacy.tokens.Doc) -> typing.List[int]:
    """Get a mapping from characters to words in `doc`."""
    char_to_word = [-1] * len(doc.text)
    for token in doc:
        char_to_word[token.idx : token.idx + len(token.text)] = [token.i] * len(token.text)
    return char_to_word


def _get_word_vectors(char_to_word: typing.List[int], doc: spacy.tokens.Doc) -> torch.Tensor:
    """Get word vectors mapped onto a character length vector."""
    zeros = torch.zeros(doc.vector.size)
    word_vectors_ = numpy.stack([zeros if w < 0 else doc[w].vector for w in char_to_word])
    return torch.from_numpy(word_vectors_)


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
    signal = lib.audio.pad_remainder(signal)
    _, trim = librosa.effects.trim(signal)
    return torch.tensor(signal[trim[0] : trim[1]], requires_grad=False)


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
    signal_to_spectrogram = lib.audio.get_signal_to_db_mel_spectrogram(**kwargs)
    signals_ = stack_and_pad_tensors(signals)
    db_mel_spectrogram = signal_to_spectrogram(signals_.tensor, aligned=True)
    lengths = signals_.lengths // signal_to_spectrogram.frame_hop
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


@configurable
def _make_stop_token(
    spectrogram: SequenceBatch, length: int = HParam(), standard_deviation: float = HParam()
):
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

    # TODO: How come this is a `SequenceBatch`? It's not a sequence.
    # SequenceBatch[torch.LongTensor [1, batch_size], torch.LongTensor [1, batch_size])
    encoded_speaker: SequenceBatch

    # SequenceBatch[torch.LongTensor [1, batch_size], torch.LongTensor [1, batch_size])
    encoded_session: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_tokens, batch_size], torch.LongTensor [1, batch_size])
    encoded_tokens: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_tokens, batch_size], torch.LongTensor [1, batch_size])
    encoded_tokens_mask: SequenceBatch

    def __len__(self):
        return len(self.spans)


def _en_span_to_tokens(spans: typing.List[Span]) -> typing.List[str]:
    """Process English `Span`s of text into a list of tokens."""
    nlp = lib.text.load_en_core_web_md(disable=("parser", "ner"))
    en_docs: typing.List[spacy.tokens.Doc] = list(nlp.pipe([s.passage.script for s in spans]))
    for i in range(len(spans)):
        script_slice = spans[i].script_slice
        span = en_docs[i].char_span(script_slice.start, script_slice.stop)  # type: ignore
        assert span is not None, "Invalid `spacy.tokens.Span` selected."
        en_docs[i] = span.as_doc()
    return typing.cast(typing.List[str], lib.text.grapheme_to_phoneme(en_docs))


def make_batch(spans: typing.List[Span], max_workers: int = 6) -> Batch:
    """
    NOTE: spaCy splits some (not all) words on apostrophes while AmEPD does not; therefore,
    those words will not be found in AmEPD. The options are:
    1. Return two different sequences with two different character to word mappings.
    2. Merge the words with apostrophes, and merge the related word vectors.
    (Choosen) 3. Keep the apostrophes separate, and miss some pronunciations.

    NOTE: Contextual word-vectors would likely be more informative than word-vectors; however,
    they are likely not as robust in the presence of OOV words due to intentional misspellings.
    Our users intentionally misspell words to adjust the pronunciation. For that reason, we use
    word-vectors.

    NOTE: In Janurary 2020, this function profiled like so:
    - 27% for `_signals_to_spectrograms`
    - 25% on `nlp.pipe`
    - 13% on `_random_loudness_annotations`
    - 13% on `grapheme_to_phoneme`
    - 6% for `_pad_and_trim_signal`
    - 5% for `input_encoder.encode`
    - 4% on `stack_and_pad_tensors`

    TODO: For `spectrogram_model` training, this function is critical for performance and
    reducing the number of CPUs needed for training. Here are some opportunities for
    performance:
    - Using `jit` or `numpy` or `cuda` for a faster spectrogram calculation.
    - Precomputing `nlp.pipe` and caching the results.
    - Using `multiprocessing` for `grapheme_to_phoneme`.
    - Using the precomputed spectrogram for `_pad_and_trim_signal`.
    """
    _stack = functools.partial(stack_and_pad_tensors, dim=1)
    _make_mask = functools.partial(torch.ones, dtype=torch.bool)

    assert len(spans) > 0, "Batch must have at least one item."

    en_spans = [(i, s) for i, s in enumerate(spans) if s.speaker.language == Language.ENGLISH]
    en_tokens = _en_span_to_tokens([s for _, s in en_spans])

    decoded = [DecodedInput(s.script, s.script, s.speaker, (s.speaker, s.session)) for s in spans]
    for (i, span), token in zip(en_spans, en_tokens):
        decoded[i] = DecodedInput(span.script, token, span.speaker, (span.speaker, span.session))

    encoded = [input_encoder.encode(d) for d in decoded]
    with futures.ThreadPoolExecutor(max_workers=min(max_workers, len(spans))) as pool:
        signals_ = list(pool.map(lambda s: s.audio(), spans))
        signals_ = typing.cast(typing.List[numpy.ndarray], signals_)
    signals = [_pad_and_trim_signal(s) for s in signals_]
    spectrogram, spectrogram_mask = _signals_to_spectrograms(signals)

    return Batch(
        # NOTE: Prune unused attributes from `Passage`, in order to reduce batch size, which in
        # turn makes it easier to send to other processes, for example.
        spans=[dataclasses.replace(s, passage=dataclasses.replace(s.passage)) for s in spans],
        audio=signals,
        spectrogram=spectrogram,
        spectrogram_mask=spectrogram_mask,
        stop_token=_make_stop_token(spectrogram),
        phonemes=[p.split(run._config.PHONEME_SEPARATOR) for p in phonemes],
        encoded_speaker=_stack([s.speaker for s in encoded]),
        encoded_session=_stack([s.session for s in encoded]),
        encoded_tokens=_stack([s.tokens for s in encoded]),
        encoded_tokens_mask=_stack([_make_mask(e.tokens.shape[0]) for e in encoded]),
    )


class DataProcessor(typing.Mapping[int, Batch]):
    def __init__(
        self,
        dataset: run._config.Dataset,
        batch_size: int,
        input_encoder: InputEncoder,
        step: int = 0,
        **kwargs,
    ):
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
        iter_ = run._utils.SpanGenerator(dataset, **kwargs)
        iter_ = BucketBatchSampler(iter_, batch_size, False, self._data_iterator_sort_key)
        iter_ = DeterministicSampler(iter_, run._config.RANDOM_SEED + step, cuda=False)
        if is_initialized():
            iter_ = DistributedBatchSampler(iter_, num_replicas=get_world_size(), rank=get_rank())
        iter_ = typing.cast(typing.Iterator[typing.List[Span]], iter_)
        self.index_to_spans = lib.utils.MappedIterator(iter_)
        self.input_encoder = input_encoder

    @staticmethod
    def _data_iterator_sort_key(span: Span):
        return span.audio_length

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __len__(_) -> int:
        return sys.maxsize

    def __getitem__(self, index) -> Batch:
        return make_batch(self.index_to_spans[index], self.input_encoder)
