import contextlib
import dataclasses
import functools
import hashlib
import io
import json
import logging
import math
import os
import pathlib
import random
import sqlite3
import subprocess
import typing

import comet_ml
import librosa
import numpy
import torch
import tqdm
from hparams import HParam, configurable
from scipy import ndimage
from torchnlp.encoders.text import SequenceBatch, stack_and_pad_tensors

import lib
import run
from lib.utils import flatten

logger = logging.getLogger(__name__)
Dataset = typing.Dict[lib.datasets.Speaker, typing.List[lib.datasets.Example]]


@dataclasses.dataclass(frozen=True)
class Checkpoint:

    checkpoints_directory: pathlib.Path
    comet_ml_experiment_key: str
    comet_ml_project_name: str
    step: int


@dataclasses.dataclass(frozen=True)
class SpectrogramModelCheckpoint(Checkpoint):
    """Checkpoint used to checkpoint spectrogram model training."""

    input_encoder: lib.spectrogram_model.InputEncoder
    model: lib.spectrogram_model.SpectrogramModel
    optimizer: torch.optim.Adam
    clipper: lib.optimizers.AdaptiveGradientNormClipping
    scheduler: torch.optim.lr_scheduler.LambdaLR


@dataclasses.dataclass(frozen=True)
class SignalModelCheckpoint(Checkpoint):
    """Checkpoint used to checkpoint signal model training.

    TODO: Add relevant fields.
    """

    ...


def maybe_make_experiment_directories_from_checkpoint(
    checkpoint: Checkpoint,
    *args,
    **kwargs,
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


def update_audio_file_metadata(
    connection: sqlite3.Connection, audio_paths: typing.List[pathlib.Path]
):
    """ Update table `audio_file_metadata` with metadata for `audio_paths`. """
    cursor = connection.cursor()
    logger.info("Updating audio file metadata...")
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS audio_file_metadata (
      path text PRIMARY KEY,
      sample_rate integer,
      channels integer,
      encoding text,
      length float
    )"""
    )
    cursor.execute("""SELECT path FROM audio_file_metadata""")
    absolute = [a.absolute() for a in audio_paths]
    update = list(set(absolute) - set([pathlib.Path(r[0]) for r in cursor.fetchall()]))
    metadatas = lib.audio.get_audio_metadata(update)
    cursor.executemany(
        """INSERT INTO audio_file_metadata (path, sample_rate, channels, encoding, length)
    VALUES (?,?,?,?,?)""",
        [(str(p.absolute()), s, c, e, l) for (p, s, c, e, l) in metadatas],
    )
    connection.commit()


def fetch_audio_file_metadata(
    connection: sqlite3.Connection, audio_path: pathlib.Path
) -> lib.audio.AudioFileMetadata:
    """ Get `AudioFileMetadata` for `audio_path` from table `audio_file_metadata`. """
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM audio_file_metadata WHERE path=?", (str(audio_path.absolute()),))
    row = cursor.fetchone()
    assert row is not None, f"Metadata for audio path {audio_path} not found."
    return lib.audio.AudioFileMetadata(pathlib.Path(row[0]), *row[1:])


def handle_null_alignments(connection: sqlite3.Connection, dataset: Dataset):
    """Update any `None` alignments with an alignment spaning the entire audio and
    text, in-place."""
    logger.info("Updating null alignments...")
    for speaker, examples in dataset.items():
        updated = []
        logger.info("Updating alignments for %s dataset...", speaker.name)
        for example in tqdm.tqdm(examples):
            if example.alignments is None:
                metadata = fetch_audio_file_metadata(connection, example.audio_path)
                alignment = lib.datasets.Alignment((0, len(example.text)), (0.0, metadata.length))
                example = example._replace(alignments=(alignment,))
            updated.append(example)
        dataset[speaker] = updated


def _normalize_audio(
    args: typing.Tuple[pathlib.Path, str, pathlib.Path],
    encoding: str,
    sample_rate: int,
    channels: int,
):
    """ Helper function for `normalize_audio`. """
    source, audio_filters, destination = args
    destination.parent.mkdir(exist_ok=True, parents=True)
    audio_filters = f"-af {audio_filters}" if audio_filters else ""
    command = (
        f"ffmpeg -i {source.absolute()} -acodec {encoding} -ar {sample_rate} -ac {channels} "
        f"{audio_filters} {destination.absolute()}"
    )
    subprocess.run(command.split(), check=True)


@configurable
def normalize_audio(
    dataset: Dataset,
    encoding: str = HParam(),
    sample_rate: int = HParam(),
    channels: int = HParam(),
    get_audio_filters: typing.Callable[[lib.datasets.Speaker], str] = HParam(),
    num_processes: int = (
        1 if lib.environment.IS_TESTING_ENVIRONMENT else typing.cast(int, os.cpu_count())
    ),
):
    """Normalize audio with ffmpeg in `dataset`.

    TODO: Consider using the ffmpeg SoX resampler, instead.

    Args:
        dataset
        encoding: Input to `ffmpeg` `-acodec` flag.
        sample_rate: Input to `ffmpeg` `-ar` flag.
        channels: Input to `ffmpeg` `-ac` flag.
        get_audio_filters: Callable to generate input to `ffmpeg` `-af` flag.
        num_processes
    """
    logger.info("Normalizing dataset audio...")
    normalize = lambda a: a.parent / run._environment.TTS_DISK_CACHE_NAME / f"ffmpeg({a.stem}).wav"
    partial = functools.partial(
        _normalize_audio, encoding=encoding, sample_rate=sample_rate, channels=channels
    )
    args_ = set(flatten([[(k, e.audio_path) for e in v] for k, v in dataset.items()]))
    args = [(a, get_audio_filters(s), normalize(a)) for s, a in args_ if not normalize(a).exists()]
    with lib.utils.Pool(num_processes) as pool:
        list(tqdm.tqdm(pool.imap_unordered(partial, args), total=len(args)))
    for speaker, examples in dataset.items():
        dataset[speaker] = [e._replace(audio_path=normalize(e.audio_path)) for e in examples]


format_ffmpeg_audio_filter = lambda n, **kw: f"{n}=" + ":".join(["%s=%s" % i for i in kw.items()])


@configurable
def assert_audio_normalized(
    connection: sqlite3.Connection,
    audio_path: pathlib.Path,
    encoding: str = HParam(),
    sample_rate: int = HParam(),
    channels: int = HParam(),
):
    """Assert `audio_path` metadata was normalized. """
    metadata = fetch_audio_file_metadata(connection, audio_path)
    assert metadata.sample_rate == sample_rate
    assert metadata.channels == channels
    assert metadata.encoding == encoding


def _adapt_numpy_array(array: numpy.ndarray) -> sqlite3.Binary:
    """`sqlite` adapter for a `numpy.ndarray`.

    Learn more: http://stackoverflow.com/a/31312102/190597
    """
    out = io.BytesIO()
    numpy.save(out, array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def _convert_numpy_array(binary: bytes) -> numpy.ndarray:
    """`sqlite` converter for a `numpy.ndarray`.

    Learn more: http://stackoverflow.com/a/31312102/190597
    """
    out = io.BytesIO(binary)
    out.seek(0)
    return numpy.load(out)


def _adapt_json(list_) -> str:
    """`sqlite` adapter for a json object.

    TODO: Add typing for `json` once it is supported: https://github.com/python/typing/issues/182
    """
    return json.dumps(list_)


def _convert_json(text: bytes):
    """`sqlite` converter for a json object."""
    return json.loads(text)


def connect(
    *args: typing.Any, detect_types=sqlite3.PARSE_DECLTYPES, **kwargs: typing.Any
) -> sqlite3.Connection:
    """Opens a connection to the SQLite database file."""
    sqlite3.register_adapter(numpy.ndarray, _adapt_numpy_array)
    sqlite3.register_converter("numpy_ndarray", _convert_numpy_array)
    sqlite3.register_adapter(dict, _adapt_json)
    sqlite3.register_adapter(list, _adapt_json)
    sqlite3.register_converter("json", _convert_json)
    # TODO: Update typing once this issue is resolved https://github.com/python/mypy/issues/2582
    connection = sqlite3.connect(*args, detect_types=detect_types, **kwargs)  # type: ignore
    return connection


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


def split_examples(
    examples: typing.List[lib.datasets.Example], dev_size: float
) -> typing.Tuple[typing.List[lib.datasets.Example], typing.List[lib.datasets.Example]]:
    """Split a dataset into a development and train set.

    Args:
        examples
        dev_size: Number of seconds of audio data in the development set.

    Return:
        train: The rest of the data.
        dev: Dataset with `dev_size` of data.
    """
    assert all(e.alignments is not None for e in examples)
    examples = examples.copy()
    random.shuffle(examples)
    # NOTE: `len_` assumes that a negligible amount of data is unusable in each example.
    len_ = lambda e: e.alignments[-1].audio[-1] - e.alignments[0].audio[0]
    dev, train = tuple(lib.utils.accumulate_and_split(examples, [dev_size, math.inf], len_))
    dev_size = sum([len_(e) for e in dev])
    train_size = sum([len_(e) for e in dev])
    assert train_size >= dev_size, "The `dev` dataset is larger than the `train` dataset."
    assert len(dev) > 0, "The dev dataset has no examples."
    assert len(train) > 0, "The train dataset has no examples."
    return train, dev


class SpectrogramModelExample(typing.NamedTuple):
    """Preprocessed `Example` used to training or evaluating the spectrogram model."""

    audio_path: pathlib.Path
    audio: torch.Tensor  # torch.FloatTensor [num_samples]
    spectrogram: torch.Tensor  # torch.FloatTensor [num_frames, frame_channels]
    spectrogram_mask: torch.Tensor  # torch.FloatTensor [num_frames]
    spectrogram_extended_mask: torch.Tensor  # torch.FloatTensor [num_frames, frame_channels]
    stop_token: torch.Tensor  # torch.FloatTensor [num_frames]
    speaker: lib.datasets.Speaker
    encoded_speaker: torch.Tensor  # torch.LongTensor [1]
    text: str
    encoded_text: torch.Tensor  # torch.LongTensor [num_characters]
    encoded_letter_case: torch.Tensor  # torch.LongTensor [num_characters]
    word_vectors: torch.Tensor  # torch.FloatTensor [num_characters]
    encoded_phonemes: torch.Tensor  # List [num_words] torch.LongTensor [num_phonemes]
    loudness: torch.Tensor  # torch.FloatTensor [num_characters]
    loudness_mask: torch.Tensor  # torch.BoolTensor [num_characters]
    speed: torch.Tensor  # torch.FloatTensor [num_characters]
    speed_mask: torch.Tensor  # torch.BoolTensor [num_characters]
    alignments: typing.Tuple[lib.datasets.Alignment, ...]
    metadata: typing.Dict[str, typing.Any]


def _get_normalized_half_gaussian(length: int, standard_deviation: float) -> torch.Tensor:
    """Get a normalized half guassian distribution.

    Learn more:
    https://en.wikipedia.org/wiki/Half-normal_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html

    Args:
        length (int): The size of the gaussian filter.
        standard_deviation (float): The standard deviation of the guassian.

    Returns:
        (torch.FloatTensor [length,])
    """
    gaussian_kernel = ndimage.gaussian_filter1d(
        numpy.float_([0] * (length - 1) + [1]), sigma=standard_deviation
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
    bounds = flatten([[(a.text[0], a.audio[0]), (a.text[-1], a.audio[-1])] for a in alignments])
    num_cuts = random.randint(0, int(lib.utils.clamp(max_alignments, min_=0, max_=len(bounds) - 1)))

    if num_cuts == 0:
        return tuple()

    if num_cuts == 1:
        alignment = lib.datasets.Alignment(
            (bounds[0][0], bounds[-1][0]), (bounds[0][1], bounds[-1][1])
        )
        return tuple([alignment]) if random.choice((True, False)) else tuple()

    # NOTE: Functionally, this is similar to a 50% dropout on intervals.
    # NOTE: Each alignment is expected to be included half of the time.
    intervals = bounds[:1] + random.sample(bounds[1:-1], num_cuts - 1) + bounds[-1:]
    return tuple(
        [
            lib.datasets.Alignment((a[0], b[0]), (a[1], b[1]))
            for a, b in zip(intervals, intervals[1:])
            if random.choice((True, False))
        ]
    )


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
    text: str, start: int, stop: int
) -> typing.Tuple[
    typing.List[int],
    torch.Tensor,
    typing.Tuple[typing.Optional[typing.Tuple[lib.text.AMEPD_ARPABET, ...]], ...],
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
    is_inside = lambda i: i >= start and i < stop
    assert all([is_inside(t.idx) == is_inside(t.idx + len(t.text) - 1) for t in doc])
    doc = [t for t in doc if is_inside(t.idx)]

    character_to_word = [-1] * (stop - start)
    for i, token in enumerate(doc):
        token_start = token.idx - start
        character_to_word[token_start : token_start + len(token.text)] = [i] * len(token.text)

    assert len(doc) != 0, "Text has no words."
    zeros = torch.zeros(doc[0].vector.size)
    word_vectors = torch.from_numpy(
        numpy.stack([zeros if w < 0 else doc[w].vector for w in character_to_word])
    )
    for token in doc:
        if not token.has_vector:
            lib.utils.call_once(logger.warning, "No word vector found for '%s'.", token.text)

    word_pronunciations = lib.text.get_pronunciations(doc)

    return character_to_word, word_vectors, word_pronunciations


@configurable
def get_spectrogram_example(
    example: lib.datasets.Example,
    connection: sqlite3.Connection,
    input_encoder: lib.spectrogram_model.InputEncoder,
    loudness_implementation: str = HParam(),
    max_loudness_annotations: int = HParam(),
    loudness_precision: int = HParam(),
    max_speed_annotations: int = HParam(),
    speed_precision: int = HParam(),
    stop_token_range: int = HParam(),
    stop_token_standard_deviation: float = HParam(),
    sample_rate: int = HParam(),
) -> SpectrogramModelExample:
    """
    Args:
        example
        connection
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
    alignments = example.alignments
    assert alignments is not None
    assert_audio_normalized(connection, example.audio_path, sample_rate=sample_rate)

    num_characters = alignments[-1].text[-1] - alignments[0].text[0]
    num_seconds = alignments[-1].audio[-1] - alignments[0].audio[0]

    text = example.text[alignments[0].text[0] : alignments[-1].text[-1]]
    audio = lib.audio.read_audio_slice(example.audio_path, alignments[0].audio[0], num_seconds)

    character_to_word, word_vectors, _ = _get_words(
        example.text, alignments[0].text[0], alignments[-1].text[-1]
    )

    # NOTE: Since eSpeak is a black-box, we cannot align the phonemes to the graphemes; therefore,
    # unfortunately, determining the correct pronunciation requires context which we can't provide.
    phonemes = lib.text.grapheme_to_phoneme([text])[0]

    arg = (text, phonemes, example.speaker)
    encoded_text, encoded_letter_case, encoded_phonemes, encoded_speaker = input_encoder.encode(arg)

    loudness = torch.zeros(num_characters)
    loudness_mask = torch.ones(num_characters)
    for alignment in _random_nonoverlapping_alignments(alignments, max_loudness_annotations):
        slice_ = slice(alignment.text[0], alignment.text[1])
        loudness[slice_] = _get_loudness(
            audio, sample_rate, alignment, loudness_implementation, loudness_precision
        )
        loudness_mask[slice_] = 0.0

    speed = torch.zeros(num_characters)
    speed_mask = torch.ones(num_characters)
    for alignment in _random_nonoverlapping_alignments(alignments, max_speed_annotations):
        slice_ = slice(alignment.text[0], alignment.text[1])
        # TODO: Instead of using characters per second, we could estimate the number of phonemes
        # with `grapheme_to_phoneme`. This might be slow, so we'd need to do so in a batch.
        # `grapheme_to_phoneme` can only estimate the number of phonemes because we can't
        # incorperate sufficient context to get the actual phonemes pronounced by the speaker.
        speed[slice_] = round(
            (alignment.text[1] - alignment.text[0]) / (alignment.audio[1] - alignment.audio[0]),
            speed_precision,
        )
        speed_mask[slice_] = 0.0

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
    audio = audio.pad_remainder(audio)
    _, trim = librosa.effects.trim(audio)
    audio = audio[trim[0] : trim[1]]

    audio = torch.tensor(audio, requires_grad=False)
    with torch.no_grad():
        db_mel_spectrogram = audio.get_signal_to_db_mel_spectrogram()(audio, aligned=True)

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

    return SpectrogramModelExample(
        audio_path=example.audio_path,
        audio=audio,
        spectrogram=db_mel_spectrogram,
        spectrogram_mask=torch.ones(db_mel_spectrogram.shape[0]),
        spectrogram_extended_mask=torch.ones(*db_mel_spectrogram.shape),
        stop_token=stop_token,
        speaker=example.speaker,
        encoded_speaker=encoded_speaker,
        text=text,
        encoded_text=encoded_text,
        encoded_letter_case=encoded_letter_case,
        word_vectors=word_vectors,
        encoded_phonemes=encoded_phonemes,
        loudness=loudness,
        loudness_mask=loudness_mask,
        speed=speed,
        speed_mask=speed_mask,
        alignments=alignments,
        metadata=example.metadata,
    )


class SpectrogramModelExampleBatch(typing.NamedTuple):
    """Batch of preprocessed `Example` used to training or evaluating the spectrogram model."""

    audio_path: typing.List[pathlib.Path]

    audio: typing.List[torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #                  torch.LongTensor [1, batch_size])
    spectrogram: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    spectrogram_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #                  torch.LongTensor [1, batch_size])
    spectrogram_extended_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    stop_token: SequenceBatch

    speaker: typing.List[lib.datasets.Speaker]

    # SequenceBatch[torch.LongTensor [1, batch_size], torch.LongTensor [1, batch_size])
    encoded_speaker: SequenceBatch

    text: typing.List[str]

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

    # SequenceBatch[torch.BoolTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    loudness_mask: SequenceBatch

    # SequenceBatch[torch.FloatTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    speed: SequenceBatch

    # SequenceBatch[torch.BoolTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    speed_mask: SequenceBatch

    alignments: typing.List[typing.Tuple[lib.datasets.Alignment, ...]]

    metadata: typing.List[typing.Dict[str, typing.Any]]


def batch_spectrogram_examples(
    examples: typing.List[SpectrogramModelExample],
) -> SpectrogramModelExampleBatch:
    """
    TODO: For performance reasons, we could consider moving some computations from
    `get_spectrogram_example` to this function for batch processing. This technique would be
    efficient to use with `DataLoader` because `collate_fn` runs in the same worker process
    as the basic loader. There is no fancy threading to load multiple examples at the same
    time.
    """
    return SpectrogramModelExampleBatch(
        audio_path=[e.audio_path for e in examples],
        audio=[e.audio for e in examples],
        spectrogram=stack_and_pad_tensors([e.spectrogram for e in examples], dim=1),
        spectrogram_mask=stack_and_pad_tensors([e.spectrogram_mask for e in examples], dim=1),
        spectrogram_extended_mask=stack_and_pad_tensors(
            [e.spectrogram_extended_mask for e in examples], dim=1
        ),
        stop_token=stack_and_pad_tensors([e.stop_token for e in examples], dim=1),
        speaker=[e.speaker for e in examples],
        encoded_speaker=stack_and_pad_tensors([e.encoded_speaker for e in examples], dim=1),
        text=[e.text for e in examples],
        encoded_text=stack_and_pad_tensors([e.encoded_text for e in examples], dim=1),
        encoded_letter_case=stack_and_pad_tensors([e.encoded_letter_case for e in examples], dim=1),
        word_vectors=stack_and_pad_tensors([e.word_vectors for e in examples], dim=1),
        encoded_phonemes=stack_and_pad_tensors([e.encoded_phonemes for e in examples], dim=1),
        loudness=stack_and_pad_tensors([e.loudness for e in examples], dim=1),
        loudness_mask=stack_and_pad_tensors([e.loudness_mask for e in examples], dim=1),
        speed=stack_and_pad_tensors([e.speed for e in examples], dim=1),
        speed_mask=stack_and_pad_tensors([e.speed_mask for e in examples], dim=1),
        alignments=[e.alignments for e in examples],
        metadata=[e.metadata for e in examples],
    )


def worker_init_fn(worker_id, seed, device_index, digits=8):
    """`worker_init_fn` for `torch.utils.data.DataLoader` that ensures each worker has a
    unique and deterministic random seed."""
    # NOTE: To ensure each worker generates different dataset examples, set a unique seed for
    # each worker.
    # Learn more: https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    seed = hashlib.sha256(str([seed, device_index, worker_id]).encode("utf-8")).hexdigest()
    lib.environment.set_seed(int(seed, 16) % 10 ** digits)


@contextlib.contextmanager
def model_context(
    model: torch.nn.Module,
    comet_ml: typing.Union[comet_ml.Experiment, comet_ml.ExistingExperiment],
    name: str,
    is_train: bool,
):
    with comet_ml.context_manager(name):
        mode = model.training
        model.train(mode=is_train)
        with torch.set_grad_enabled(mode=is_train):
            yield
        model.train(mode=mode)


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
"""


@configurable
def get_rms_level(
    db_spectrogram: torch.Tensor,
    mask: typing.Optional[torch.Tensor] = None,
    signal_length: typing.Optional[int] = None,
    frame_hop: int = HParam(),
    **kwargs,
) -> float:
    """Get the dB RMS level given a dB spectrogram.

    NOTE: Without the `signal_length` and `frame_hop` parameters, this function estimates the
    RMS level from the `db_spectrogram`. The estimate assumes that each frame is equal without
    any deference to the boundaries. As the `db_spectrogram` grows in length, the estimate
    gets more accurate.

    Args:
        db_spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [num_frames, batch_size])
        **kwargs: Additional key word arguments passed to `power_spectrogram_to_framed_rms`.

    Returns:
        The RMS level in decibels of the dB spectrogram.
    """
    spectrogram = lib.audio.db_to_power(db_spectrogram.transpose(0, 1))
    rms = lib.audio.power_spectrogram_to_framed_rms(spectrogram, **kwargs).numpy()
    mask_ = numpy.ones(rms.shape) if mask is None else mask.transpose(0, 1).numpy()
    divisor = mask_.sum() if signal_length is None else signal_length / frame_hop
    return lib.audio.power_to_db(torch.tensor(((rms * mask_) ** 2).sum() / divisor)).item()
