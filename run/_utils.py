import contextlib
import dataclasses
import enum
import functools
import hashlib
import io
import logging
import math
import multiprocessing.pool
import os
import pathlib
import random
import time
import typing

import numpy
import torch
import torch.cuda
import torch.distributed
import torch.nn
import torch.optim
import tqdm
from google.cloud import storage
from hparams import HParam, configurable
from third_party import LazyLoader
from torchnlp.encoders.text import SequenceBatch, stack_and_pad_tensors

import lib
import run
from lib.utils import flatten, seconds_to_string
from run._config import Cadence, Dataset, DatasetType, get_dataset_label, get_model_label

if typing.TYPE_CHECKING:  # pragma: no cover
    import comet_ml
    import librosa
    import matplotlib.figure
    from scipy import ndimage
else:
    comet_ml = LazyLoader("comet_ml", globals(), "comet_ml")
    librosa = LazyLoader("librosa", globals(), "librosa")
    matplotlib = LazyLoader("matplotlib", globals(), "matplotlib")
    ndimage = LazyLoader("ndimage", globals(), "scipy.ndimage")


logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Checkpoint:

    checkpoints_directory: pathlib.Path
    comet_experiment_key: str
    step: int


@dataclasses.dataclass(frozen=True)
class SpectrogramModelCheckpoint(Checkpoint):
    """Checkpoint used to checkpoint spectrogram model training."""

    input_encoder: lib.spectrogram_model.InputEncoder
    model: lib.spectrogram_model.SpectrogramModel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipper
    scheduler: torch.optim.lr_scheduler.LambdaLR


@dataclasses.dataclass(frozen=True)
class SignalModelCheckpoint(Checkpoint):
    """Checkpoint used to checkpoint signal model training.

    TODO: Add relevant fields.
    """

    ...


def maybe_make_experiment_directories_from_checkpoint(
    checkpoint: Checkpoint, *args, **kwargs
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """For checkpoints saved in the `maybe_make_experiment_directories` directory structure,
    this creates another "run" under the original experiment.
    """
    return maybe_make_experiment_directories(
        checkpoint.checkpoints_directory.parent.parent, *args, **kwargs
    )


def maybe_make_experiment_directories(
    experiment_root: pathlib.Path,
    recorder: lib.environment.RecordStandardStreams,
    run_name: str = "RUN_" + lib.environment.bash_time_label(add_pid=False),
    checkpoints_directory_name: str = "checkpoints",
    run_log_filename: str = "run.log",
) -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """Create a directory structure to store an experiment run, like so:

      {experiment_root}/
      └── {run_name}/
          ├── run.log
          └── {checkpoints_directory_name}/

    TODO: Could this structure be encoded in some data structure? For example, we could return an
    object called an `ExperimentDirectory` that has `children` called `RunsDirectory`.

    Args:
        experiment_root: Top-level directory to store an experiment, unless a
          checkpoint is provided.
        recorder: This records the standard streams, and saves it.
        run_name: The name of this run.
        checkpoints_directory_name: The name of the directory that stores checkpoints.
        run_log_filename: The run log filename.

    Return:
        run_root: The root directory to store run files.
        checkpoints_directory: The directory to store checkpoints.
    """
    logger.info("Updating directory structure...")
    experiment_root.mkdir(exist_ok=True)
    run_root = experiment_root / run_name
    run_root.mkdir()
    checkpoints_directory = run_root / checkpoints_directory_name
    checkpoints_directory.mkdir()
    recorder.update(run_root, log_filename=run_log_filename)
    return run_root, checkpoints_directory


def _normalize_audio(
    args: typing.Tuple[pathlib.Path, pathlib.Path], callable_: typing.Callable[..., None]
):
    """ Helper function for `normalize_audio`. """
    source, destination = args
    destination.parent.mkdir(exist_ok=True, parents=True)
    callable_(source, destination)


def _normalize_path(path: pathlib.Path) -> pathlib.Path:
    """ Helper function for `normalize_audio`. """
    return path.parent / run._config.TTS_DISK_CACHE_NAME / f"ffmpeg({path.stem}).wav"


def normalize_audio(
    dataset: Dataset, num_processes: int = typing.cast(int, os.cpu_count()), **kwargs
):
    """Normalize audio with ffmpeg in `dataset`.

    TODO: Consider using the ffmpeg SoX resampler, instead.
    """
    logger.info("Normalizing dataset audio...")
    audio_paths_ = [[p.audio_file.path for p in v] for v in dataset.values()]
    audio_paths: typing.Set[pathlib.Path] = set(flatten(audio_paths_))
    partial = lib.audio.normalize_audio.get_configured_partial()  # type: ignore
    partial = functools.partial(partial, **kwargs)
    partial = functools.partial(_normalize_audio, callable_=partial)
    args = [(p, _normalize_path(p)) for p in audio_paths if not _normalize_path(p).exists()]
    with multiprocessing.pool.ThreadPool(num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(partial, args), total=len(args)))

    metadatas = lib.audio.get_audio_metadata([_normalize_path(p) for p in audio_paths])
    lookup = {p: m for p, m in zip(audio_paths, metadatas)}
    for passages in dataset.values():
        for passage in passages:
            passage.audio_file = lookup[passage.audio_file.path]


def init_distributed(
    rank: int,
    backend: str = "nccl",
    init_method: str = "tcp://127.0.0.1:29500",
    world_size: int = torch.cuda.device_count(),
) -> torch.device:
    """Initiate distributed for training.

    Learn more about distributed environments here:
    https://pytorch.org/tutorials/intermediate/dist_tuto.htm
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size
    )
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    logger.info("Worker %d started.", torch.distributed.get_rank())
    logger.info("%d GPUs found.", world_size)
    return device


def split_passages(
    passages: typing.List[lib.datasets.Passage], dev_size: float
) -> typing.Tuple[typing.List[lib.datasets.Passage], typing.List[lib.datasets.Passage]]:
    """Split a dataset into a development and train set.

    Args:
        passages
        dev_size: Number of seconds of audio data in the development set.

    Return:
        train: The rest of the data.
        dev: Dataset with `dev_size` of data.
    """
    passages = passages.copy()
    random.shuffle(passages)
    # NOTE: `len_` assumes that a negligible amount of data is unusable in each passage.
    len_ = lambda p: p[:].audio_length
    dev, train = tuple(lib.utils.split(passages, [dev_size, math.inf], len_))
    dev_size = sum([len_(p) for p in dev])
    train_size = sum([len_(p) for p in train])
    assert train_size >= dev_size, "The `dev` dataset is larger than the `train` dataset."
    assert len(dev) > 0, "The dev dataset has no passages."
    assert len(train) > 0, "The train dataset has no passages."
    return train, dev


class SpectrogramModelSpan(typing.NamedTuple):
    """Preprocessed `Span` used to training or evaluating the spectrogram model."""

    audio_file: lib.audio.AudioFileMetadata
    audio: torch.Tensor  # torch.FloatTensor [num_samples]
    spectrogram: torch.Tensor  # torch.FloatTensor [num_frames, frame_channels]
    spectrogram_mask: torch.Tensor  # torch.FloatTensor [num_frames]
    stop_token: torch.Tensor  # torch.FloatTensor [num_frames]
    speaker: lib.datasets.Speaker
    encoded_speaker: torch.Tensor  # torch.LongTensor [1]
    text: str
    encoded_text: torch.Tensor  # torch.LongTensor [num_characters]
    encoded_text_mask: torch.Tensor  # torch.FloatTensor [num_characters]
    encoded_letter_case: torch.Tensor  # torch.LongTensor [num_characters]
    word_vectors: torch.Tensor  # torch.FloatTensor [num_characters]
    encoded_phonemes: torch.Tensor  # List [num_words] torch.LongTensor [num_phonemes]
    loudness: torch.Tensor  # torch.FloatTensor [num_characters]
    loudness_mask: torch.Tensor  # torch.BoolTensor [num_characters]
    speed: torch.Tensor  # torch.FloatTensor [num_characters]
    speed_mask: torch.Tensor  # torch.BoolTensor [num_characters]
    alignments: typing.Tuple[lib.datasets.Alignment, ...]
    other_metadata: typing.Dict[typing.Union[str, int], typing.Any]


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
    gaussian_kernel = ndimage.gaussian_filter1d(
        numpy.float_([0] * (length - 1) + [1]), sigma=standard_deviation  # type: ignore
    )
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
    return torch.tensor(gaussian_kernel).float()


def _random_nonoverlapping_alignments(
    alignments: typing.Tuple[lib.datasets.Alignment, ...], max_alignments: int
) -> typing.Tuple[lib.datasets.Alignment, ...]:
    """Generate a random set of non-overlapping alignments, such that every point in the
    time-series has an equal probability of getting sampled inside an alignment.

    NOTE: The length of the sampled alignments is non-uniform.

    Args:
        alignments
        max_alignments: The maximum number of alignments to generate.
    """
    get_ = lambda a, i: tuple([getattr(a, f)[i] for f in lib.datasets.Alignment._fields])
    # NOTE: Each of these is a synchronization point along which we can match up the script
    # character, transcript character, and audio sample. We can use any of these points for
    # cutting.
    bounds = flatten([[get_(a, 0), get_(a, -1)] for a in alignments])
    num_cuts = random.randint(0, int(lib.utils.clamp(max_alignments, min_=0, max_=len(bounds) - 1)))

    if num_cuts == 0:
        return tuple()

    if num_cuts == 1:
        tuple_ = lambda i: (bounds[0][i], bounds[-1][i])
        alignment = lib.datasets.Alignment(tuple_(0), tuple_(1), tuple_(2))
        return tuple([alignment]) if random.choice((True, False)) else tuple()

    # NOTE: Functionally, this is similar to a 50% dropout on intervals.
    # NOTE: Each alignment is expected to be included half of the time.
    intervals = bounds[:1] + random.sample(bounds[1:-1], num_cuts - 1) + bounds[-1:]
    return_ = [
        lib.datasets.Alignment((a[0], b[0]), (a[1], b[1]), (a[2], b[2]))
        for a, b in zip(intervals, intervals[1:])
        if random.choice((True, False))
    ]
    return tuple(return_)


seconds_to_samples = lambda seconds, sample_rate: int(round(seconds * sample_rate))


def _get_loudness(
    audio: numpy.ndarray,
    sample_rate: int,
    alignment: lib.datasets.Alignment,
    loudness_implementation: str,
    loudness_precision: int,
) -> float:
    """Get the loudness in LUFS for an `alignment` in `audio`."""
    _seconds_to_samples = functools.partial(seconds_to_samples, sample_rate=sample_rate)
    meter = lib.audio.get_pyloudnorm_meter(sample_rate, loudness_implementation)
    slice_ = slice(_seconds_to_samples(alignment.audio[0]), _seconds_to_samples(alignment.audio[1]))
    return round(meter.integrated_loudness(audio[slice_]), loudness_precision)


def _get_words(
    text: str, start: int, stop: int, **kwargs
) -> typing.Tuple[
    typing.List[int],
    torch.Tensor,
    typing.Tuple[typing.Optional[typing.Tuple[lib.text.AMEPD_ARPABET, ...]], ...],
    str,
]:
    """Get word features for `text[start:stop]`, and a character-to-word mapping.

    NOTE: spaCy splits some (not all) words on apostrophes while AmEPD does not. The options are:
    1. Return two different sequences with two different character to word mappings.
    2. Merge the words with apostrophes, and merge the related word vectors.
    (Choosen) 3. Keep the apostrophes seperate, and miss some pronunciations.
    NOTE: Contextual word-vectors would likely be more informative than word-vectors; however, they
    are likely not as robust in the presence of OOV words due to intentional misspellings. Our
    users intentionally misspell words to adjust the pronunciation.

    TODO: Gather statistics on pronunciations available in the various datasets with additional
    notebooks or scripts.
    TODO: Filter out scripts with ambigious initialisms due to the lack of casing (i.e. everything
    is uppercase)
    TODO: How many OOV words does our dataset have?
    """
    doc = lib.text.load_en_core_web_md(disable=("parser", "ner"))(text)
    stop = len(text) + stop if stop < 0 else stop

    # NOTE: Check that words are not sliced by the boundary.
    assert start >= 0 and start <= len(text) and stop >= 0 and stop <= len(text) and stop >= start
    is_inside = lambda i: i >= start and i < stop
    assert all([is_inside(t.idx) == is_inside(t.idx + len(t.text) - 1) for t in doc])
    slice_ = slice(
        next((i for i, t in enumerate(doc) if is_inside(t.idx)), None),
        next((i + 1 for i, t in reversed(list(enumerate(doc))) if is_inside(t.idx)), None),
    )
    doc = doc[slice_]
    assert len(doc) != 0, "No words were selected."

    character_to_word = [-1] * (stop - start)
    for i, token in enumerate(doc):
        token_start = token.idx - start
        character_to_word[token_start : token_start + len(token.text)] = [i] * len(token.text)

    zeros = torch.zeros(doc[0].vector.size)
    word_vectors_ = numpy.stack([zeros if w < 0 else doc[w].vector for w in character_to_word])
    word_vectors = torch.from_numpy(word_vectors_)
    for token in doc:
        if not token.has_vector:
            lib.utils.call_once(
                logger.warning,  # type: ignore
                "No word vector found for '%s'.",
                token.text,
            )

    word_pronunciations = lib.text.get_pronunciations(doc)

    # NOTE: Since eSpeak, unfortunately, doesn't take extra context to determining the
    # pronunciation. This means it'll sometimes be wrong in ambigious cases.
    phonemes = typing.cast(str, lib.text.grapheme_to_phoneme(doc, **kwargs))

    return character_to_word, word_vectors, word_pronunciations, phonemes


@configurable
def get_spectrogram_model_span(
    span: lib.datasets.Span,
    input_encoder: lib.spectrogram_model.InputEncoder,
    loudness_implementation: str = HParam(),
    max_loudness_annotations: int = HParam(),
    loudness_precision: int = HParam(),
    max_speed_annotations: int = HParam(),
    speed_precision: int = HParam(),
    stop_token_range: int = HParam(),
    stop_token_standard_deviation: float = HParam(),
    sample_rate: int = HParam(),
) -> SpectrogramModelSpan:
    """
    Args:
        span
        input_encoder
        loudness_implementation: See `pyloudnorm.Meter` for various loudness implementations.
        max_loudness_annotations: The maximum expected loudness intervals within a text segment.
        loudness_precision: The number of decimal places to round LUFS.
        max_speed_annotations: The maximum expected speed intervals within a text segment.
        speed_precision: The number of decimal places to round phonemes per second.
        stop_token_range: The range of uncertainty there is in the exact `stop_token` location.
        stop_token_standard_deviation: The standard deviation of uncertainty there is in the exact
            `stop_token` location.
        sample_rate
    """
    lib.audio.assert_audio_normalized(span.audio_file, sample_rate=sample_rate)

    alignments = span.passage.alignments[span.span]
    _, word_vectors, _, phonemes = _get_words(
        span.passage.script, alignments[0].script[0], alignments[-1].script[-1]
    )

    arg = (span.script, phonemes, span.speaker)
    encoded_text, encoded_letter_case, encoded_phonemes, encoded_speaker = input_encoder.encode(arg)

    loudness = torch.zeros(len(span.script))
    loudness_mask = torch.zeros(len(span.script), dtype=torch.bool)
    for alignment in _random_nonoverlapping_alignments(span.alignments, max_loudness_annotations):
        slice_ = slice(alignment.script[0], alignment.script[1])
        loudness[slice_] = _get_loudness(
            span.audio, sample_rate, alignment, loudness_implementation, loudness_precision
        )
        loudness_mask[slice_] = True

    speed = torch.zeros(len(span.script))
    speed_mask = torch.zeros(len(span.script), dtype=torch.bool)
    for alignment in _random_nonoverlapping_alignments(span.alignments, max_speed_annotations):
        slice_ = slice(alignment.script[0], alignment.script[1])
        # TODO: Instead of using characters per second, we could estimate the number of phonemes
        # with `grapheme_to_phoneme`. This might be slow, so we'd need to do so in a batch.
        # `grapheme_to_phoneme` can only estimate the number of phonemes because we can't
        # incorperate sufficient context to get the actual phonemes pronounced by the speaker.
        char_per_second = (slice_.stop - slice_.start) / (alignment.audio[1] - alignment.audio[0])
        speed[slice_] = round(char_per_second, speed_precision)
        speed_mask[slice_] = True

    # TODO: The RMS function that trim uses mentions that it's likely better to use a
    # spectrogram if it's available:
    # https://librosa.github.io/librosa/generated/librosa.feature.rms.html?highlight=rms#librosa.feature.rms
    # TODO: The RMS function is a naive computation of loudness; therefore, it'd likely
    # be more accurate to use our spectrogram for trimming with augmentations like A-weighting.
    # TODO: `pad_remainder` could possibly add distortion if it's appended to non-zero samples;
    # therefore, it'd likely be beneficial to have a small fade-in and fade-out before
    # appending the zero samples.
    # TODO: We should consider padding more than just the remainder. We could additionally
    # pad a `frame_length` of padding so that further down the pipeline, any additional
    # padding does not affect the spectrogram due to overlap between the padding and the
    # real audio.
    # TODO: Instead of padding with zeros, we should consider padding with real-data.
    audio = lib.audio.pad_remainder(span.audio)
    _, trim = librosa.effects.trim(audio)
    audio = torch.tensor(audio[trim[0] : trim[1]], requires_grad=False)

    with torch.no_grad():
        db_mel_spectrogram = lib.audio.get_signal_to_db_mel_spectrogram()(audio, aligned=True)

    # NOTE: The exact stop token distribution is uncertain because there are multiple valid
    # stopping points after someone has finished speaking. For example, the audio can be cutoff
    # 1 second or 2 seconds after someone has finished speaking. In order to address this
    # uncertainty, we naively apply a normal distribution as the stop token ground truth.
    # NOTE: This strategy was found to be effective via Comet in January 2020.
    # TODO: In the future, it'd likely be more accurate to base the probability for stopping
    # based on the loudness of each frame. The maximum loudness is based on a full-scale sine wave
    # and the minimum loudness would be -96 Db or so. The probability for stopping is the loudness
    # relative to the minimum and maximum loudness. This is assuming that at the end of an audio
    # clip it gets progressively quieter.
    stop_token = db_mel_spectrogram.new_zeros((db_mel_spectrogram.shape[0],))
    gaussian_kernel = _get_normalized_half_gaussian(stop_token_range, stop_token_standard_deviation)
    max_len = min(len(stop_token), len(gaussian_kernel))
    stop_token[-max_len:] = gaussian_kernel[-max_len:]

    return SpectrogramModelSpan(
        audio_file=span.audio_file,
        audio=audio,
        spectrogram=db_mel_spectrogram,
        spectrogram_mask=torch.ones(db_mel_spectrogram.shape[0], dtype=torch.bool),
        stop_token=stop_token,
        speaker=span.speaker,
        encoded_speaker=encoded_speaker,
        text=span.script,
        encoded_text=encoded_text,
        encoded_text_mask=torch.ones(encoded_text.shape[0], dtype=torch.bool),
        encoded_letter_case=encoded_letter_case,
        word_vectors=word_vectors,
        encoded_phonemes=encoded_phonemes,
        loudness=loudness,
        loudness_mask=loudness_mask,
        speed=speed,
        speed_mask=speed_mask,
        alignments=span.alignments,
        other_metadata=span.other_metadata,
    )


class SpectrogramModelSpanBatch(typing.NamedTuple):
    """Batch of preprocessed `Span` used to training or evaluating the spectrogram model."""

    length: int

    audio_file: typing.List[lib.audio.AudioFileMetadata]

    audio: typing.List[torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #               torch.LongTensor [1, batch_size])
    spectrogram: SequenceBatch

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    spectrogram_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    stop_token: SequenceBatch

    speaker: typing.List[lib.datasets.Speaker]

    # SequenceBatch[torch.LongTensor [1, batch_size], torch.LongTensor [1, batch_size])
    encoded_speaker: SequenceBatch

    text: typing.List[str]

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    encoded_text_mask: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    encoded_text: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    encoded_letter_case: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    word_vectors: SequenceBatch

    # SequenceBatch[torch.LongTensor [num_phonemes, batch_size], torch.LongTensor [1, batch_size])
    encoded_phonemes: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    loudness: SequenceBatch

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    loudness_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    speed: SequenceBatch

    # NOTE: Mask padding with `False`.
    # SequenceBatch[torch.BoolTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    speed_mask: SequenceBatch

    alignments: typing.List[typing.Tuple[lib.datasets.Alignment, ...]]

    other_metadata: typing.List[typing.Dict[typing.Union[str, int], typing.Any]]


def batch_spectrogram_model_spans(
    spans: typing.List[SpectrogramModelSpan],
) -> SpectrogramModelSpanBatch:
    """
    TODO: For performance reasons, we could consider moving some computations from
    `get_spectrogram_model_span` to this function for batch processing. This technique would be
    efficient to use with `DataLoader` because `collate_fn` runs in the same worker process
    as the basic loader. There is no fancy threading to load multiple spans at the same
    time.
    """
    return SpectrogramModelSpanBatch(
        length=len(spans),
        audio_file=[s.audio_file for s in spans],
        audio=[s.audio for s in spans],
        spectrogram=stack_and_pad_tensors([p.spectrogram for p in spans], dim=1),
        spectrogram_mask=stack_and_pad_tensors([s.spectrogram_mask for s in spans], dim=1),
        stop_token=stack_and_pad_tensors([s.stop_token for s in spans], dim=1),
        speaker=[s.speaker for s in spans],
        encoded_speaker=stack_and_pad_tensors([s.encoded_speaker for s in spans], dim=1),
        text=[s.text for s in spans],
        encoded_text=stack_and_pad_tensors([s.encoded_text for s in spans], dim=1),
        encoded_text_mask=stack_and_pad_tensors([s.encoded_text_mask for s in spans], dim=1),
        encoded_letter_case=stack_and_pad_tensors([s.encoded_letter_case for s in spans], dim=1),
        word_vectors=stack_and_pad_tensors([s.word_vectors for s in spans], dim=1),
        encoded_phonemes=stack_and_pad_tensors([s.encoded_phonemes for s in spans], dim=1),
        loudness=stack_and_pad_tensors([s.loudness for s in spans], dim=1),
        loudness_mask=stack_and_pad_tensors([s.loudness_mask for s in spans], dim=1),
        speed=stack_and_pad_tensors([s.speed for s in spans], dim=1),
        speed_mask=stack_and_pad_tensors([s.speed_mask for s in spans], dim=1),
        alignments=[s.alignments for s in spans],
        other_metadata=[s.other_metadata for s in spans],
    )


def worker_init_fn(worker_id: int, seed: int, device_index: int, digits: int = 8):
    """`worker_init_fn` for `torch.utils.data.DataLoader` that ensures each worker has a
    unique and deterministic random seed."""
    # NOTE: To ensure each worker generates different dataset spans, set a unique seed for
    # each worker.
    # Learn more: https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    seed_ = hashlib.sha256(str([seed, device_index, worker_id]).encode("utf-8")).hexdigest()
    lib.environment.set_seed(int(seed_, 16) % 10 ** digits)


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


@configurable
def get_rms_level(
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
    spectrogram = lib.audio.db_to_power(db_spectrogram.transpose(0, 1))
    # [batch_size, num_frames, frame_channels] → [batch_size, num_frames]
    rms = lib.audio.power_spectrogram_to_framed_rms(spectrogram, **kwargs)
    return (rms if mask is None else rms * mask.transpose(0, 1)).pow(2).sum(dim=1)


def get_dataset_stats(
    train: Dataset, dev: Dataset
) -> typing.Dict[run._config.Label, typing.Union[str, int, float]]:
    """Get `train` and `dev` dataset statistics."""
    # NOTE: `len_` assumes the entire passage is usable.
    len_ = lambda d: sum(p.alignments[-1].audio[-1] - p.alignments[0].audio[0] for p in d)
    stats: typing.Dict[run._config.Label, typing.Union[int, str, float]] = {}
    data: Dataset
    for data, type_ in [(train, DatasetType.TRAIN), (dev, DatasetType.DEV)]:
        label = functools.partial(get_dataset_label, cadence=Cadence.STATIC, type_=type_)
        stats[label("num_passages")] = sum(len(p) for p in data.values())
        stats[label("num_characters")] = sum(sum(len(p.script) for p in v) for v in data.values())
        stats[label("num_seconds")] = seconds_to_string(sum(len_(p) for p in data.values()))
        for speaker, passages in data.items():
            label = functools.partial(label, speaker=speaker)
            stats[label("num_passages")] = len(passages)
            stats[label("num_characters")] = sum(len(p.script) for p in passages)
            stats[label("num_seconds")] = seconds_to_string(len_(passages))
    return stats


def get_num_skipped(
    alignments: torch.Tensor, token_mask: torch.Tensor, spectrogram_mask: torch.Tensor
) -> torch.Tensor:
    """Given `alignments` from frames to tokens, this computes the number of tokens that were
    skipped.

    NOTE: This function assumes a token is attended to if it has the most focus of all the other
    tokens for some frame.

    Args:
        alignments (torch.FloatTensor [num_frames, batch_size, num_tokens])
        token_mask (torch.BoolTensor [batch_size, num_tokens])
        spectrogram_mask (torch.BoolTensor [num_frames, batch_size])

    Returns:
        torch.FloatTensor [batch_size]
    """
    indices = alignments.max(dim=2, keepdim=True).indices
    device = alignments.device
    one = torch.ones(*alignments.shape, device=device, dtype=torch.long)
    # [num_frames, batch_size, num_tokens]
    num_skipped = torch.zeros(*alignments.shape, device=device, dtype=torch.long)
    num_skipped = num_skipped.scatter(dim=2, index=indices, src=one)
    # [num_frames, batch_size, num_tokens] → [batch_size, num_tokens]
    num_skipped = num_skipped.masked_fill(~spectrogram_mask.unsqueeze(-1), 0).sum(dim=0)
    return (num_skipped.masked_fill(~token_mask, -1) == 0).float().sum(dim=1)


class Context(enum.Enum):
    """ Constants and labels for contextualizing the use-case. """

    TRAIN: typing.Final = "train"
    EVALUATE: typing.Final = "evaluate"
    EVALUATE_INFERENCE: typing.Final = "evaluate_inference"


class CometMLExperiment:
    """Create a `comet_ml.Experiment` or `comet_ml.ExistingExperiment` object with several
    adjustments.

    Args:
        project_name
        experiment_key: Existing experiment identifier.
        workspace
        **kwargs: Other kwargs to pass to comet `Experiment` and / or `ExistingExperiment`
    """

    _BASE_HTML_STYLING = """
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/meyer-reset/2.0/reset.css"
      type="text/css">
<style>
  body {
    background-color: #f4f4f5;
  }

  p {
    font-family: 'Roboto', system-ui, sans-serif;
    margin-bottom: .5em;
  }

  b {
    font-weight: bold
  }

  section {
    padding: 1.5em;
    border-bottom: 2px solid #E8E8E8;
    background: white;
  }
</style>
    """

    def __init__(
        self,
        project_name: typing.Optional[str] = None,
        experiment_key: typing.Optional[str] = None,
        workspace: typing.Optional[str] = None,
        **kwargs,
    ):
        if lib.environment.has_untracked_files():
            raise ValueError(
                "Experiment is not reproducible, Comet does not track untracked files. "
                f"Please track these files via `git`:\n{lib.environment.get_untracked_files()}"
            )

        kwargs.update({"project_name": project_name, "workspace": workspace})
        if experiment_key is None:
            self._experiment = comet_ml.Experiment(**kwargs)
            self._experiment.log_html(self._BASE_HTML_STYLING)
        else:
            self._experiment = comet_ml.ExistingExperiment(
                previous_experiment=experiment_key, **kwargs
            )

        self.log_asset = self._experiment.log_asset
        self.log_html = self._experiment.log_html
        self.get_key = self._experiment.get_key
        self.set_model_graph = self._experiment.set_model_graph

        self._last_step_time: typing.Optional[float] = None
        self._last_step: typing.Optional[int] = None
        self._last_epoch_time: typing.Optional[float] = None
        self._last_epoch_step: typing.Optional[int] = None
        self._first_epoch_time: typing.Optional[float] = None
        self._first_epoch_step: typing.Optional[int] = None

        self._log_environment()

    @property
    def curr_step(self) -> typing.Optional[int]:
        return typing.cast(typing.Optional[int], self._experiment.curr_step)

    @property
    def context(self) -> typing.Optional[str]:
        return typing.cast(typing.Optional[str], self._experiment.context)

    def _log_environment(self):
        # TODO: Collect additional environment details like CUDA, CUDANN, NVIDIA Driver versions
        # with this script:
        # https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
        log_other = lambda k, v: self.log_other(run._config.get_environment_label(k), v)

        log_other("last_git_commit_date", lib.environment.get_last_git_commit_date())
        log_other("git_branch", lib.environment.get_git_branch_name())
        log_other("has_git_patch", str(lib.environment.has_tracked_changes()))
        log_other("gpus", lib.environment.get_cuda_gpus())
        log_other("num_gpus", lib.environment.get_num_cuda_gpus())
        log_other("disks", lib.environment.get_disks())
        log_other("unique_cpus", lib.environment.get_unique_cpus())
        log_other("num_cpus", os.cpu_count())
        log_other("total_physical_memory", lib.environment.get_total_physical_memory())

    def set_step(self, step: typing.Optional[int]):
        self._experiment.set_step(step)
        if self.curr_step is not None:
            seconds_per_step = (
                (time.time() - self._last_step_time) / (self.curr_step - self._last_step)
                if self._last_step is not None
                and self._last_step_time is not None
                and self.curr_step > self._last_step
                else None
            )
            self._last_step_time = time.time()
            # NOTE: Ensure that the variable `last_step` is updated before `log_metric` is called.
            # This prevents infinite recursion via `curr_step > last_step`.
            self._last_step = self.curr_step
            if seconds_per_step is not None:
                label = get_model_label("seconds_per_step", Cadence.STEP)
                self.log_metric(label, seconds_per_step)

    @contextlib.contextmanager
    def context_manager(self, context: Context):
        with self._experiment.context_manager(str(context)):
            yield self

    def log_current_epoch(self, epoch: int):
        self._last_epoch_step = self.curr_step
        self._last_epoch_time = time.time()
        if self._first_epoch_time is None and self._first_epoch_step is None:
            self._first_epoch_step = self.curr_step
            self._first_epoch_time = time.time()
        self._experiment.log_current_epoch(epoch)

    def log_epoch_end(self, epoch: int):
        # NOTE: Logs an average `steps_per_second` for each epoch.
        if (
            self._last_epoch_step is not None
            and self._last_epoch_time is not None
            and self.curr_step is not None
        ):
            label = get_model_label("steps_per_second", Cadence.MULTI_STEP)
            metric = (self.curr_step - self._last_epoch_step) / (
                time.time() - self._last_epoch_time
            )
            self.log_metric(label, metric)

        # NOTE: Logs an average `steps_per_second` since the training started.
        if (
            self._first_epoch_time is not None
            and self._first_epoch_step is not None
            and self.curr_step is not None
        ):
            with self.context_manager(None):
                label = get_model_label("steps_per_second", Cadence.RUN)
                metric = (self.curr_step - self._first_epoch_step) / (
                    time.time() - self._first_epoch_time
                )
                self.log_metric(label, metric)

        self._experiment.log_epoch_end(epoch)

    def _upload_audio(
        self, file_name: str, data: typing.Union[numpy.ndarray, torch.Tensor]
    ) -> typing.Optional[str]:
        """Upload the audio and return the URL."""
        file_ = io.BytesIO()
        lib.audio.write_audio(file_, data)
        asset = self.log_asset(file_, file_name=file_name)
        return asset["web"] if asset is not None else asset

    def log_html_audio(
        self,
        audio: typing.Dict[str, typing.Union[numpy.ndarray, torch.Tensor]] = {},
        **kwargs,
    ):
        """Audio with related metadata to Comet in the HTML tab.

        Args:
            audio
            **kwargs: Additional metadata to include.
        """
        items = [f"<p><b>Step:</b> {self.curr_step}</p>"]
        param_to_label = lambda s: s.title().replace("_", " ")
        items.extend([f"<p><b>{param_to_label(k)}:</b> {v}</p>" for k, v in kwargs.items()])
        for key, data in audio.items():
            name = param_to_label(key)
            file_name = f"step={self.curr_step},name={name},experiment={self.get_key()}"
            url = self._upload_audio(file_name, data)
            items.append(f"<p><b>{name}:</b></p>")
            items.append(f'<audio controls preload="metadata" src="{url}"></audio>')
        self.log_html("<section>{}</section>".format("\n".join(items)))

    def log_parameter(self, key: run._config.Label, value: typing.Union[str, int, float]):
        self._experiment.log_parameter(key, value)

    def log_parameters(self, dict_: typing.Dict[run._config.Label, typing.Union[str, int, float]]):
        self._experiment.log_parameters(dict_)

    def log_other(self, key: run._config.Label, value: typing.Union[str, int, float]):
        self._experiment.log_other(key, value)

    def log_metric(self, name: run._config.Label, value: typing.Union[int, float]):
        self._experiment.log_metric(name, value)

    def log_figure(self, name: run._config.Label, figure: matplotlib.figure.Figure):
        self._experiment.log_figure(str(name), figure)

    def log_figures(self, dict_: typing.Dict[run._config.Label, matplotlib.figure.Figure]):
        """ Log multiple figures from `dict_` via `experiment.log_figure`. """
        return [self.log_figure(k, v) for k, v in dict_.items()]

    def set_name(self, name: str):
        logger.info('Experiment name set to "%s"', name)
        self._experiment.set_name(name)

    def add_tags(self, tags: typing.List[str]):
        logger.info("Added tags to experiment: %s", tags)
        self._experiment.add_tags(tags)


@contextlib.contextmanager
def set_context(context: Context, model: torch.nn.Module, comet: CometMLExperiment):
    with comet.context_manager(context.value):
        mode = model.training
        model.train(mode=(context == Context.TRAIN))
        with torch.set_grad_enabled(mode=(context == Context.TRAIN)):
            yield
        model.train(mode=mode)


@functools.lru_cache(maxsize=1)
def get_storage_client() -> storage.Client:
    return storage.Client()


def gcs_uri_to_blob(gcs_uri: str) -> storage.Blob:
    """Parse GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") and return a `Blob`.

    NOTE: This function requires GCS authorization.
    """
    assert len(gcs_uri) > 5, "The URI must be longer than 5 characters to be a valid GCS link."
    assert gcs_uri[:5] == "gs://", "The URI provided is not a valid GCS link."
    path_segments = gcs_uri[5:].split("/")
    bucket = get_storage_client().bucket(path_segments[0])
    name = "/".join(path_segments[1:])
    return bucket.blob(name)


def blob_to_gcs_uri(blob: storage.Blob) -> str:
    """ Get GCS URI (e.g. "gs://cloud-samples-tests/speech/brooklyn.flac") from `blob`. """
    return "gs://" + blob.bucket.name + "/" + blob.name
