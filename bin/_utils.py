from contextlib import contextmanager
from functools import lru_cache
from functools import partial

import hashlib
import io
import json
import logging
import pathlib
import random
import subprocess
import typing

from hparams import configurable
from hparams import HParam
from scipy import ndimage
from torchnlp.encoders.text import SequenceBatch
from torchnlp.encoders.text import stack_and_pad_tensors

import comet_ml
import librosa
import numpy
import pyloudnorm
import sqlite3
import torch

from src import environment
from src import spectrogram_model
from src.utils import cumulative_split
from src.utils import flatten

import src

logger = logging.getLogger(__name__)

Dataset = typing.Dict[src.datasets.Speaker, typing.List[src.datasets.Example]]


class SpectrogramModelCheckpoint(typing.NamedTuple):
    """ Checkpoint used to restart spectrogram model training and evaluation.
    """
    checkpoints_directory: pathlib.Path
    comet_ml_experiment_key: str
    comet_ml_project_name: str
    input_encoder: src.spectrogram_model.InputEncoder
    model: torch.nn.Module
    optimizer: src.optimizers.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    step: int


def maybe_make_experiment_directories(
        experiment_root: pathlib.Path,
        recorder: src.environment.RecordStandardStreams,
        checkpoint: typing.Optional[SpectrogramModelCheckpoint] = None,
        run_name: str = 'RUN_' + environment.bash_time_label(add_pid=False),
        checkpoints_directory_name: str = 'checkpoints',
        run_log_filename: str = 'run.log') -> typing.Tuple[pathlib.Path, pathlib.Path]:
    """ Create a directory structure to store an experiment run, like so:

      {experiment_root}/
      └── {run_name}/
          ├── run.log
          └── {checkpoints_directory_name}/

    Args:
        experiment_root: Top-level directory to store an experiment, unless a
          checkpoint is provided.
        recorder: This records the standard streams, and saves it.
        run_name: The name of this run.
        checkpoints_directory_name: The name of the directory that houses checkpoints.
        run_log_filename: The run log filename.
        checkpoint: Prior checkpoint.

    Return:
        run_root: The root directory to store run files.
        checkpoints_directory: The directory to store checkpoints.
    """
    logger.info('Updating directory structure...')
    if checkpoint is not None:
        experiment_root = checkpoint.checkpoints_directory.parent.parent
    run_root = experiment_root / run_name
    run_root.mkdir(parents=checkpoint is None)
    checkpoints_directory = run_root / checkpoints_directory_name
    checkpoints_directory.mkdir()
    recorder.update(run_root, log_filename=run_log_filename)
    return run_root, checkpoints_directory


def update_audio_file_metadata(connection: sqlite3.Connection,
                               audio_paths: typing.List[pathlib.Path]):
    """ Update table `audio_file_metadata` with metadata for `audio_paths`. """
    cursor = connection.cursor()
    logger.info('Updating audio file metadata...')
    cursor.execute("""CREATE TABLE IF NOT EXISTS audio_file_metadata (
      path text PRIMARY KEY,
      sample_rate integer,
      channels integer,
      encoding text,
      length float
    )""")
    cursor.execute("""SELECT path FROM audio_file_metadata""")
    absolute_paths: typing.List[str] = [str(a.absolute()) for a in audio_paths]
    metadatas = src.audio.get_audio_metadata(
        set(absolute_paths) - set([r[0] for r in cursor.fetchall()]))
    cursor.executemany(
        """INSERT INTO audio_file_metadata (path, sample_rate, channels, encoding, length)
    VALUES (?,?,?,?,?)""", [m._replace(path=m.path.absolute()) for m in metadatas])
    connection.commit()


def fetch_audio_length(connection: sqlite3.Connection, audio_path: pathlib.Path) -> float:
    """ Get length for `audio_path` from table `audio_file_metadata`. """
    cursor = connection.cursor()
    cursor.execute('SELECT length FROM audio_file_metadata WHERE path=?',
                   (str(audio_path.absolute()),))
    return cursor.fetchone()[0]


def handle_null_alignments(connection: sqlite3.Connection, dataset: Dataset) -> Dataset:
    """ Update any `None` alignments with an alignment spaning the entire audio and text. """
    logger.info('Updating null alignments...')
    dataset = dataset.copy()
    for speaker, examples in dataset.items():
        updated = []
        for example in examples:
            # TODO: Fetch a batch of audio lengths, instead of fetching one at a time.
            if example.alignments is None:
                length = fetch_audio_length(connection, example.audio_path)
                example._replace(
                    alignments=([
                        src.datasets.Alignment((0, len(example.text)), (0.0, length)),
                    ]))
            updated.append(example)
        dataset[speaker] = updated
    return dataset


# See https://spacy.io/api/annotation#pos-tagging for all available tags.
_SPACY_PUNCT_TAG = 'PUNCT'


def update_word_representations(connection: sqlite3.Connection, texts: typing.List[str],
                                **kwargs: typing.Any):
    """ Update `parsed_text` with various word representations.

    Args:
        cursor
        texts
        **kwargs: Keyword arguments passed to `grapheme_to_phoneme`.
    """
    logger.info('Parsing text with spaCy and eSpeak...')
    cursor = connection.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS parsed_text (
      text text PRIMARY KEY,
      phonemes text,
      character_to_word json,
      word_vectors numpy_ndarray)""")
    cursor.execute('SELECT text FROM parsed_text')
    fetched_texts = set([r[0] for r in cursor.fetchall()])
    texts = list(set(texts) - fetched_texts)
    nlp = src.text.load_en_core_web_md(disable=['parser', 'ner', 'tagger'])
    docs = list(nlp.pipe(texts))
    insert = []
    for text, phonemes, doc in zip(texts, src.text.grapheme_to_phoneme(texts), docs):
        character_to_word = [-1] * len(text)
        for token in doc:
            character_to_word[token.idx:token.idx + len(token.text)] = token.i
        word_vectors = numpy.stack([t.vector for t in doc])
        insert.append((text, phonemes, character_to_word, word_vectors))
    cursor.executemany(
        """INSERT INTO parsed_text (text, phonemes, character_to_word, word_vectors)
      VALUES (?,?,?,?)""", insert)
    connection.commit()


def fetch_texts(connection: sqlite3.Connection) -> typing.List[str]:
    cursor = connection.cursor()
    cursor.execute('SELECT text FROM parsed_text')
    return [r[0] for r in cursor.fetchall()]


def fetch_phonemes(connection: sqlite3.Connection) -> typing.List[str]:
    cursor = connection.cursor()
    cursor.execute('SELECT phoneme FROM grapheme_to_phoneme')
    return [r[0] for r in cursor.fetchall()]


@configurable
def normalize_audio(
    dataset: Dataset,
    encoding: str = HParam(),
    sample_rate: int = HParam(),
    channels: int = HParam(),
    get_audio_filters: typing.Callable[[src.datasets.Speaker], str] = HParam()
) -> Dataset:
    """ Normalize audio with ffmpeg in `dataset`.

    Args:
        dataset
        encoding: Input to `ffmpeg` `-acodec` flag.
        sample_rate: Input to `ffmpeg` `-ar` flag.
        channels: Input to `ffmpeg` `-ac` flag.
        get_audio_filters: Callable to generate input to `ffmpeg` `-af` flag.
    """
    # TODO: Normalize a batch of audio samples at the same tim via threads or something else.
    logger.info('Normalizing dataset audio...')
    dataset = dataset.copy()
    command = 'ffmpeg -i %s -acodec %s -ar %s -ac %s -af %s %s.wav'
    normalized_path = lambda a: a.parent / environment.TTS_DISK_CACHE_NAME / 'ffmpeg({}){}'.format(
        a.stem, a.suffix)
    for speaker, examples in dataset.items():
        for audio_path in set([e.audio_path for e in examples]):
            normalized = normalized_path(audio_path)
            if not normalized.exists():
                args = (audio_path.absolute(), encoding, sample_rate, channels,
                        get_audio_filters(speaker), normalized.absolute())
                subprocess.run((command % args).split(), check=True)
        updated = [e._replace(audio_path=normalized_path(e.audio_path)) for e in examples]
        dataset[speaker] = updated
    return dataset


format_audio_filter = lambda n, **kw: 'n=' + ':'.join(['%s=%s' % i for i in kw.items()])


def adapt_numpy_array(array: numpy.ndarray) -> sqlite3.Binary:
    """ `sqlite` adapter for a `numpy.ndarray`.

    Learn more: http://stackoverflow.com/a/31312102/190597
    """
    out = io.BytesIO()
    numpy.save(out, array)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_numpy_array(binary: bytes) -> numpy.ndarray:
    """ `sqlite` converter for a `numpy.ndarray`.

    Learn more: http://stackoverflow.com/a/31312102/190597
    """
    out = io.BytesIO(binary)
    out.seek(0)
    return numpy.load(out)


def adapt_json(list_) -> str:
    """ `sqlite` adapter for a json object.

    TODO: Add typing for `json` once it is supported: https://github.com/python/typing/issues/182
    """
    return json.dumps(list_)


def convert_json(text: bytes):
    """ `sqlite` converter for a json object.
    """
    return json.loads(text)


def connect(*args: typing.Any,
            detect_types=sqlite3.PARSE_DECLTYPES,
            **kwargs: typing.Any) -> sqlite3.Connection:
    """ Opens a connection to the SQLite database file.
    """
    sqlite3.register_adapter(numpy.ndarray, adapt_numpy_array)
    sqlite3.register_converter('numpy_ndarray', convert_numpy_array)
    sqlite3.register_adapter(dict, adapt_json)
    sqlite3.register_adapter(list, adapt_json)
    sqlite3.register_converter('json', convert_json)
    # TODO: Update typing once this issue is resolved https://github.com/python/mypy/issues/2582
    connection = sqlite3.connect(*args, detect_types=detect_types, **kwargs)  # type: ignore
    return connection


def init_distributed(
    rank: int,
    backend: str = 'nccl',
    init_method: str = 'tcp://127.0.0.1:29500',
    world_size: int = torch.cuda.device_count()
) -> torch.device:
    """ Initiate distributed for training.

    Learn more about distributed environments here:
    https://pytorch.org/tutorials/intermediate/dist_tuto.htm
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    torch.distributed.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    logger.info('Worker %d started.', torch.distributed.get_rank())
    logger.info('%d GPUs found.', world_size)
    return device


def split_examples(
    examples: typing.List[src.datasets.Example], dev_size: float
) -> typing.Tuple[typing.List[src.datasets.Example], typing.List[src.datasets.Example]]:
    """ Split a dataset into a development and train set.

    Args:
        dataset
        dev_size: Number of seconds of audio data in the development set.

    Return:
        train: The rest of the data.
        dev: Dataset with `dev_size` of data.
    """
    examples = examples.copy()
    random.shuffle(examples)
    # NOTE: This assumes that a negligible amount of data is unusable in each example.
    dev, train = cumulative_split(examples, [dev_size],
                                  lambda e: e.alignments[-1][1][1] - e.alignments[0][1][0])
    assert len(dev) > 0, 'The dev dataset has no examples.'
    assert len(train) > 0, 'The train dataset has no examples.'
    return train, dev


class SpectrogramModelExample(typing.NamedTuple):
    """ Preprocessed `Example` used to training or evaluating the spectrogram model.
    """
    audio_path: pathlib.Path
    audio: torch.Tensor  # torch.FloatTensor [num_samples]
    spectrogram: torch.Tensor  # torch.FloatTensor [num_frames, frame_channels]
    spectrogram_mask: torch.Tensor  # torch.FloatTensor [num_frames]
    spectrogram_extended_mask: torch.Tensor  # torch.FloatTensor [num_frames, frame_channels]
    stop_token: torch.Tensor  # torch.FloatTensor [num_frames]
    speaker: src.datasets.Speaker
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
    alignments: typing.List[src.datasets.Alignment]
    metadata: typing.Dict[str, typing.Any]


def _get_normalized_half_gaussian(length: int, standard_deviation: float) -> torch.Tensor:
    """ Get a normalized half guassian distribution.

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
        numpy.float_([0] * (length - 1) + [1]), sigma=standard_deviation)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()
    return torch.tensor(gaussian_kernel).float()


def _random_nonoverlapping_alignments(alignments: typing.List[src.datasets.Alignment],
                                      max_alignments: int) -> typing.List[src.datasets.Alignment]:
    """ Generate a random set of non-overlapping alignments, such that every point in the
    time-series has an equal probability of getting sampled inside an alignment.

    NOTE: The length of the sampled alignments is non-uniform.

    Args:
        alignments
        max_alignments: The maximum number of alignments to generate.
    """
    samples = flatten([[(a.text[0], a.audio[0]), (a.text[-1], a.audio[-1])] for a in alignments])
    num_cuts = random.randint(0, min(max_alignments, len(samples) - 1))
    if num_cuts == 0:
        return []
    if num_cuts == 1:
        return [src.datasets.Alignment(samples[0], samples[-1])] if random.choice(
            (True, False)) else []
    intervals = samples[:1] + random.sample(samples[1:-1], num_cuts - 1) + samples[-1:]
    return [
        src.datasets.Alignment(a, b)
        for a, b in zip(intervals, intervals[1:])
        if random.choice((True, False))
    ]


@lru_cache(maxsize=None)
def _get_pyloudnorm_meter(*args, **kwargs):
    return pyloudnorm.Meter(*args, **kwargs)


seconds_to_samples = lambda seconds, sample_rate: int(round(seconds * sample_rate))


def _get_loudness(audio: numpy.ndarray, sample_rate: int, alignment: src.datasets.Alignment,
                  loudness_implementation: str, loudness_precision: int) -> float:
    """ Get the loudness in LUFS for an `alignment` in `audio`.
    """
    _seconds_to_samples = partial(seconds_to_samples, sample_rate=sample_rate)
    meter = _get_pyloudnorm_meter(sample_rate, loudness_implementation)
    slice_ = slice(_seconds_to_samples(alignment.audio[0]), _seconds_to_samples(alignment.audio[1]))
    return round(meter.integrated_loudness(audio[slice_]), loudness_precision)


@configurable
def get_spectrogram_example(
        example: src.datasets.Example,
        connection: sqlite3.Connection,
        input_encoder: spectrogram_model.InputEncoder,
        format_: str = HParam(),
        encoding: str = HParam(),
        sample_rate: int = HParam(),
        channels: int = HParam(),
        loudness_implementation: str = HParam(),
        max_loudness_annotations: int = HParam(),
        loudness_precision: int = HParam(),
        max_speed_annotations: int = HParam(),
        speed_precision: int = HParam(),
        stop_token_range: int = HParam(),
        stop_token_standard_deviation: float = HParam(),
) -> SpectrogramModelExample:
    """
    Args:
        example
        connection
        input_encoder
        format_: Input to `ffmpeg` `-f` flag.
        encoding: Input to `ffmpeg` `-acodec` flag.
        sample_rate: Input to `ffmpeg` `-ar` flag.
        channels: Input to `ffmpeg` `-ac` flag.
        loudness_implementation: See `pyloudnorm.Meter` for various loudness implementations.
        max_loudness_annotations: The maximum expected loudness intervals within a text segment.
        loudness_precision: The number of decimal places to round LUFS.
        max_speed_annotations: The maximum expected speed intervals within a text segment.
        speed_precision: The number of decimal places to round phonemes per second.
        stop_token_range: The range of uncertainty there is in the exact `stop_token` location.
        stop_token_standard_deviation: The standard deviation of uncertainty there is in the exact
            `stop_token` location.
    """
    assert example.alignments is not None
    alignments = example.alignments
    start_second = alignments[0].audio[0]
    num_seconds = alignments[-1].audio[-1] - alignments[0].audio[0]
    num_characters = alignments[-1].text[-1] - alignments[0].text[0]
    text = example.text[alignments[0].text[0]:alignments[-1].text[-1]]

    command = 'ffmpeg -ss %f -i %s -ss %f -f %s -acodec %s -ar %s -ac %s pipe:' % (
        start_second, example.audio_path, num_seconds, format_, encoding, sample_rate, channels)
    audio = numpy.frombuffer(subprocess.check_output(command.split()), numpy.float32)

    cursor = connection.cursor()
    cursor.execute('SELECT * FROM parsed_text WHERE text = ?', (example.text,))
    _, phonemes, character_to_word, word_vectors = cursor.fetchone()[0]

    encoded_text, encoded_letter_case, encoded_phonemes, encoded_speaker = input_encoder.encode(
        (text, phonemes, example.speaker))

    character_to_word_slice = character_to_word[alignments[0].text[0]:alignments[-1].text[-1]]
    # Ensure the alignment doesn't cut a word in half.
    if alignments[0].text[0] > 0:
        assert character_to_word[alignments[0].text[0] - 1] != character_to_word_slice[0]
    if alignments[-1].text[-1] < len(character_to_word) - 1:
        assert character_to_word_slice[-1] != character_to_word[alignments[-1].text[-1] + 1]
    expanded_word_vectors = torch.stack([
        torch.zeros(word_vectors.shape[1]) if w < 0 else torch.from_numpy(word_vectors[w])
        for w in character_to_word_slice
    ])

    # Make loudness annotation
    loudness = torch.zeros(num_characters)
    loudness_mask = torch.ones(num_characters)
    for alignment in _random_nonoverlapping_alignments(alignments, max_loudness_annotations):
        slice_ = slice(alignment.text[0], alignment.text[1])
        loudness[slice_] = _get_loudness(audio, sample_rate, alignment, loudness_implementation,
                                         loudness_precision)
        loudness_mask[slice_] = 0.0

    # Make speed annotations
    speed = torch.zeros(num_characters)
    speed_mask = torch.ones(num_characters)
    for alignment in _random_nonoverlapping_alignments(alignments, max_speed_annotations):
        slice_ = slice(alignment.text[0], alignment.text[1])
        speed[slice_] = round(
            (alignment.text[1] - alignment.text[0]) / (alignment.audio[1] - alignment.audio[0]),
            speed_precision)
        speed_mask[slice_] = 0.0

    # TODO: The RMS function is a naive computation of loudness; therefore, it'd likely
    # be more accurate to use our spectrogram for trimming with augmentations like A-weighting.
    # TODO: The RMS function that trim uses mentions that it's likely better to use a
    # spectrogram if it's available:
    # https://librosa.github.io/librosa/generated/librosa.feature.rms.html?highlight=rms#librosa.feature.rms
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
    audio = audio[trim[0]:trim[1]]

    # TODO: Now that `get_signal_to_db_mel_spectrogram` is implemented in PyTorch, we could
    # batch process spectrograms. This would likely be faster. Also, it could be fast to
    # compute spectrograms on-demand.
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
        word_vectors=expanded_word_vectors,
        encoded_phonemes=encoded_phonemes,
        loudness=loudness,
        loudness_mask=loudness_mask,
        speed=speed,
        speed_mask=speed_mask,
        alignments=example.alignments,
        metadata=example.metadata)


class SpectrogramModelExampleBatch(typing.NamedTuple):
    """ Batch of preprocessed `Example` used to training or evaluating the spectrogram model.
    """
    audio_path: typing.List[pathlib.Path]

    audio: typing.List[torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #                  torch.LongTensor [1, batch_size])
    spectrogram: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    spectrogram_mask: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size, frame_channels],
    #                  torch.LongTensor [1, batch_size])
    spectrogram_extended_mask: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_frames, batch_size], torch.LongTensor [1, batch_size])
    stop_token: SequenceBatch[torch.Tensor, torch.Tensor]

    speaker: typing.List[src.datasets.Speaker]

    # SequenceBatch[torch.LongTensor [1, batch_size], torch.LongTensor [1, batch_size])
    encoded_speaker: SequenceBatch[torch.Tensor, torch.Tensor]

    text: typing.List[str]

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    encoded_text: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    encoded_letter_case: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.LongTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    word_vectors: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.LongTensor [num_phonemes, batch_size], torch.LongTensor [1, batch_size])
    encoded_phonemes: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    loudness: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.BoolTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    loudness_mask: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.FloatTensor [num_characters, batch_size],
    #               torch.LongTensor [1, batch_size])
    speed: SequenceBatch[torch.Tensor, torch.Tensor]

    # SequenceBatch[torch.BoolTensor [num_characters, batch_size], torch.LongTensor [1, batch_size])
    speed_mask: SequenceBatch[torch.Tensor, torch.Tensor]

    alignments: typing.List[typing.List[src.datasets.Alignment]]

    metadata: typing.List[typing.Dict[str, typing.Any]]


def batch_spectrogram_examples(
        examples: typing.List[SpectrogramModelExample]) -> SpectrogramModelExampleBatch:
    return SpectrogramModelExampleBatch(
        audio_path=[e.audio_path for e in examples],
        audio=[e.audio for e in examples],
        spectrogram=stack_and_pad_tensors([e.spectrogram for e in examples], dim=1),
        spectrogram_mask=stack_and_pad_tensors([e.spectrogram_mask for e in examples], dim=1),
        spectrogram_extended_mask=stack_and_pad_tensors(
            [e.spectrogram_extended_mask for e in examples], dim=1),
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
        metadata=[e.metadata for e in examples])


def worker_init_fn(worker_id, seed, device_index, digits=16):
    # NOTE: To ensure each worker generates different dataset examples, set a unique seed for
    # each worker.
    # Learn more: https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    seed = hashlib.sha256(str([seed, device_index, worker_id]).encode('utf-8')).hexdigest()
    environment.set_seed(int(seed, 16) % 10**digits)


@contextmanager
def model_context(model: torch.nn.Module, comet_ml: typing.Union[comet_ml.Experiment,
                                                                 comet_ml.ExistingExperiment],
                  name: str, is_train: bool):
    with comet_ml.context_manager(name):
        mode = model.training
        model.train(mode=is_train)
        with torch.set_grad_enabled(mode=is_train):
            yield
        model.train(mode=mode)


def get_rms_level(spectrogram: torch.Tensor,
                  mask: typing.Optional[torch.Tensor] = None,
                  **kwargs) -> float:
    """ Get the RMS level given a spectrogram.

    Args:
        spectrogram (torch.FloatTensor [num_frames, batch_size, frame_channels])
        mask (torch.FloatTensor [num_frames, batch_size])
        **kwargs: Additional key word arguments passed to `framed_rms_from_power_spectrogram`.

    Returns:
        The RMS level in decibels of the spectrogram.
    """
    device = spectrogram.device
    spectrogram = src.audio.db_to_power(spectrogram.transpose(0, 1))
    target_rms = src.audio.framed_rms_from_power_spectrogram(spectrogram, **kwargs)
    mask = torch.ones(*target_rms.shape, device=device) if mask is None else mask.transpose(0, 1)

    # TODO: This conversion from framed RMS to global RMS is not accurate. The original
    # spectrogram is padded such that it's length is some constant multiple (256x) of the signal
    # length. In order to accurately convert a framed RMS to a global RMS, each sample
    # has to appear an equal number of times in the frames. Supposing there is 25% overlap,
    # that means each sample has to appear 4 times. That nessecarly means that the first and
    # last sample needs to be evaluated 3x more times, adding 6 frames to the total number of
    # frames. Adding a constant number of frames, is not compatible with a constant multiple
    # supposing any sequence length must be supported. To fix this, we need to remove the
    # requirement for a constant multiple. The requirement comes from the signal model that
    # upsamples via constant multiples at the moment. We could adjust the signal model so that
    # it upsamples -6 frames then a constant multiple, each time. A change like this
    # would ensure that the first and last frame have 4x overlapping frames to describe
    # the audio sequence, increase performance at the boundary.
    return src.audio.power_to_db((target_rms * mask).pow(2).sum() / (mask.sum())).item()
