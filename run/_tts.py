"""
TODO: Add tests for every function.
"""

import dataclasses
import enum
import functools
import logging
import pathlib
import subprocess
import sys
import threading
import typing
from queue import SimpleQueue
from subprocess import PIPE

import numpy
import spacy
import spacy.language
import spacy.tokens
import torch

from lib.environment import PT_EXTENSION, load
from lib.utils import get_chunks, tqdm_
from run import train
from run._config import CHECKPOINTS_PATH, load_spacy_nlp, normalize_vo_script
from run._models.signal_model import SignalModel, generate_waveform
from run._models.spectrogram_model import (
    Inputs,
    Mode,
    Preds,
    PreprocessedInputs,
    SpectrogramModel,
    norm_respellings,
    preprocess_inputs,
)
from run.data._loader import Session, Span

logger = logging.getLogger(__name__)


def load_checkpoints(
    directory: pathlib.Path,
    root_directory_name: str,
    gcs_path: str,
    sig_model_dir_name: str = "signal_model",
    spec_model_dir_name: str = "spectrogram_model",
    link_template: str = "https://www.comet.ml/api/experiment/redirect?experimentKey={key}",
    **kwargs,
) -> typing.Tuple[
    train.spectrogram_model._worker.Checkpoint, train.signal_model._worker.Checkpoint
]:
    """Load checkpoints from GCP.

    Args:
        directory: Directory to cache the dataset.
        root_directory_name: Name of the directory inside `directory` to store data.
        gcs_path: The base GCS path storing the data.
        sig_model_dir_name: The name of the signal model directory on GCS.
        spec_model_dir_name: The name of the spectrogram model directory on GCS.
        link_template: Template string for formating a Comet experiment link.
        kwargs: Additional key-word arguments passed to `load`.
    """
    logger.info("Loading `%s` checkpoints.", root_directory_name)

    root = (pathlib.Path(directory) / root_directory_name).absolute()
    root.mkdir(exist_ok=True)
    directories = [root / d for d in (spec_model_dir_name, sig_model_dir_name)]

    checkpoints: typing.List[train._utils.Checkpoint] = []
    for directory in directories:
        directory.mkdir(exist_ok=True)
        command = ["gsutil", "cp", "-n"]
        command += [f"{gcs_path}/{directory.name}/*{PT_EXTENSION}", f"{directory}/"]
        subprocess.run(command, check=True)
        files_ = [p for p in directory.iterdir() if p.suffix == PT_EXTENSION]
        assert len(files_) == 1
        checkpoints.append(load(files_[0], **kwargs))
        link = link_template.format(key=checkpoints[-1].comet_experiment_key)
        logger.info("Loaded `%s` checkpoint from experiment %s", directory.name, link)

    spec_chkpt, sig_chkpt = tuple(checkpoints)
    spec_chkpt = typing.cast(train.spectrogram_model._worker.Checkpoint, spec_chkpt)
    sig_chkpt = typing.cast(train.signal_model._worker.Checkpoint, sig_chkpt)
    spec_chkpt.check_invariants()
    sig_chkpt.check_invariants()
    return spec_chkpt, sig_chkpt


class Checkpoints(enum.Enum):
    """
    Catalog of checkpoints uploaded to "wellsaid_labs_checkpoints".

    You can upload a new checkpoint, for example, like so:

        $ gsutil -m cp -r disk/checkpoints/v10_2022_06_15_staging \
                        gs://wellsaid_labs_checkpoints/v10_2022_06_15_staging
    """

    """
    These checkpoints were deployed into staging as version "10.beta.1".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/409
    Spectrogram Model Experiment (Step: 527,553):
    https://www.comet.ml/wellsaid-labs/michael-spectrogram-model-03-2022/669e69f9a8db4dd3aa20386a4b195150
    Signal Model Experiment (Step: 827,151):
    https://www.comet.ml/wellsaid-labs/michael-signal-model-2022-04/a2d2e4b313e7490098ca2f3b4935f6d6
    """

    V10_2022_05_03_STAGING: typing.Final = "v10_2022_05_03_staging"

    """
    These checkpoints were deployed into staging as version "10.beta.2".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/389
    Spectrogram Model Experiment (Step: 885,735):
    https://www.comet.ml/wellsaid-labs/michael-spectrogram-model-03-2022/f43e617f5ab74eddb2eb8239b5fc10f0
    Signal Model Experiment (Step: 1,331,883):
    https://www.comet.ml/wellsaid-labs/michael-signal-model-2022-04/db35cf6b0e07463692af2a90c80724bb
    """

    V10_2022_06_08_STAGING: typing.Final = "v10_2022_06_08_staging"

    """
    These checkpoints were deployed into staging as version "10.beta.3".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/389
    Spectrogram Model Experiment (Step: 1,843,641):
    https://www.comet.ml/wellsaid-labs/michael-spectrogram-model-03-2022/bc51533a6c874938ae0043b6b0e56d59
    Signal Model Experiment (Step: 1,331,883):
    https://www.comet.ml/wellsaid-labs/michael-signal-model-2022-04/db35cf6b0e07463692af2a90c80724bb
    """

    V10_2022_06_15_STAGING: typing.Final = "v10_2022_06_15_staging"


_GCS_PATH = "gs://wellsaid_labs_checkpoints/"
CHECKPOINTS_LOADERS = {
    e: functools.partial(load_checkpoints, CHECKPOINTS_PATH, e.value, _GCS_PATH + e.value)
    for e in Checkpoints
}


@dataclasses.dataclass(frozen=True)
class TTSPackage:
    """A package of Python objects required to run TTS in inference mode.

    Args:
        ...
        spec_model_comet_experiment_key: In order to identify the model origin, the
            comet ml experiment key is required.
        spec_model_step: In order to identify the model origin, the checkpoint step which
            corresponds to the comet ml experiment is required.
        ...
    """

    spec_model: SpectrogramModel
    signal_model: SignalModel
    spec_model_comet_experiment_key: typing.Optional[str] = None
    spec_model_step: typing.Optional[int] = None
    signal_model_comet_experiment_key: typing.Optional[str] = None
    signal_model_step: typing.Optional[int] = None

    def session_vocab(self) -> typing.Set[Session]:
        """Get the sessions these models are familiar with."""
        sesh = set(self.signal_model.session_embed.vocab.keys())
        inter = set(self.spec_model.session_embed.vocab.keys()).intersection(sesh)
        return set(typing.cast(Session, s) for s in inter if isinstance(s, tuple))


def package_tts(
    spectrogram_checkpoint: train.spectrogram_model._worker.Checkpoint,
    signal_checkpoint: train.signal_model._worker.Checkpoint,
):
    """Package together objects required for running TTS inference."""
    return TTSPackage(
        spectrogram_checkpoint.export(),
        signal_checkpoint.export(),
        spec_model_comet_experiment_key=spectrogram_checkpoint.comet_experiment_key,
        spec_model_step=spectrogram_checkpoint.step,
        signal_model_comet_experiment_key=signal_checkpoint.comet_experiment_key,
        signal_model_step=signal_checkpoint.step,
    )


class PublicTextValueError(ValueError):
    pass


class PublicSpeakerValueError(ValueError):
    pass


def process_tts_inputs(
    nlp: spacy.language.Language, package: TTSPackage, script: str, session: Session
) -> typing.Tuple[Inputs, PreprocessedInputs]:
    """Process TTS `script`, `speaker` and `session` for use with the model(s)."""
    normalized = normalize_vo_script(script, session[0].language)
    if len(normalized) == 0:
        raise PublicTextValueError("Text cannot be empty.")

    inputs = Inputs([session], [nlp(norm_respellings(normalized))])
    preprocessed = preprocess_inputs(inputs)

    tokens = typing.cast(typing.List[str], set(preprocessed.tokens[0]))
    excluded = [t for t in tokens if t not in package.spec_model.token_embed.vocab]
    if len(excluded) > 0:
        difference = ", ".join([repr(c)[1:-1] for c in sorted(set(excluded))])
        raise PublicTextValueError("Text cannot contain these characters: %s" % difference)

    if session not in package.session_vocab():
        # NOTE: We do not expose speaker information in the `ValueError` because this error
        # is passed on to the public via the API.
        raise PublicSpeakerValueError("Speaker is not available.")

    return inputs, preprocessed


def text_to_speech(
    package: TTSPackage,
    script: str,
    session: Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end with friendly errors."""
    nlp = load_spacy_nlp(session[0].language)
    inputs, preprocessed_inputs = process_tts_inputs(nlp, package, script, session)
    preds = package.spec_model(inputs=preprocessed_inputs, mode=Mode.INFER)
    splits = preds.frames.split(split_size)
    generator = generate_waveform(package.signal_model, splits, inputs.session)
    wave = typing.cast(torch.Tensor, torch.cat(list(generator), dim=-1))
    return wave.squeeze(0).detach().numpy()


class TTSInputOutput(typing.NamedTuple):
    """Text-to-speech input and output."""

    inputs: Inputs
    spec_model: Preds
    sig_model: numpy.ndarray


def batch_span_to_speech(
    package: TTSPackage, spans: typing.List[Span], **kwargs
) -> typing.List[TTSInputOutput]:
    """
    NOTE: This method doesn't consider `Span` context for TTS generation.
    """
    inputs = [(s.script, s.session) for s in spans]
    return batch_text_to_speech(package, inputs, **kwargs)


def _multilingual_spacy_pipe(
    inputs: typing.List[typing.Tuple[str, Session]]
) -> typing.List[typing.Tuple[spacy.tokens.doc.Doc, Session]]:
    """Efficiently pipe `inputs` through spaCy with batching per language."""
    seshs = [s for _, s in inputs]
    langs = set(s[0].language for s in seshs)
    result: typing.List[typing.Optional[spacy.tokens.doc.Doc]] = [None] * len(inputs)
    for lang in langs:
        nlp = load_spacy_nlp(lang)
        scripts = [(i, s) for i, (s, sesh) in enumerate(inputs) if sesh[0].language is lang]
        docs = nlp.pipe(s for _, s in scripts)
        for (i, _), doc in zip(scripts, docs):
            result[i] = doc
    return list(zip(typing.cast(typing.List[spacy.tokens.doc.Doc], result), seshs))


def batch_text_to_speech(
    package: TTSPackage,
    inputs: typing.List[typing.Tuple[str, Session]],
    batch_size: int = 8,
) -> typing.List[TTSInputOutput]:
    """Run TTS end-to-end quickly with a verbose output."""
    inputs = [(normalize_vo_script(script, sesh[0].language), sesh) for script, sesh in inputs]
    logger.info(f"Processing {len(inputs)} examples with spaCy...")
    inputs_ = list(enumerate(_multilingual_spacy_pipe(inputs)))
    inputs_ = sorted(inputs_, key=lambda i: len(str(i[1][0])))
    results: typing.Dict[int, TTSInputOutput] = {}
    logger.info(f"Processing {len(inputs)} examples with TTS models...")
    for batch in tqdm_(list(get_chunks(inputs_, batch_size))):
        batch_input = Inputs(doc=[i[1][0] for i in batch], session=[i[1][1] for i in batch])
        preds = package.spec_model(inputs=batch_input, mode=Mode.INFER)
        spectrogram = preds.frames.transpose(0, 1)
        signals = package.signal_model(spectrogram, batch_input.session, preds.frames_mask)
        num_samples = preds.num_frames * package.signal_model.upscale_factor
        more_results = {
            j: TTSInputOutput(
                inputs=Inputs(doc=[batch_input.doc[i]], session=[batch_input.session[i]]),
                spec_model=Preds(
                    frames=preds.frames[: preds.num_frames[i], i : i + 1],
                    stop_tokens=preds.stop_tokens[: preds.num_frames[i], i : i + 1],
                    alignments=preds.alignments[
                        : preds.num_frames[i], i : i + 1, : preds.num_tokens[i]
                    ],
                    num_frames=preds.num_frames[i : i + 1],
                    frames_mask=preds.frames_mask[i : i + 1, : preds.num_frames[i]],
                    num_tokens=preds.num_tokens[i : i + 1],
                    tokens_mask=preds.tokens_mask[i : i + 1, : preds.num_tokens[i]],
                    reached_max=preds.reached_max[i : i + 1],
                ),
                sig_model=signals[0][i : i + 1, : num_samples[i]].detach().numpy(),
            )
            for i, (j, _) in zip(range(len(batch)), batch)
        }
        results.update(more_results)
    return [results[i] for i in range(len(inputs))]


def _enqueue(out: typing.IO[bytes], queue: SimpleQueue):
    """Enqueue all lines from a file-like object to `queue`."""
    for line in iter(out.readline, b""):
        queue.put(line)


def _dequeue(queue: SimpleQueue) -> typing.Generator[bytes, None, None]:
    """Dequeue all items from `queue`."""
    while not queue.empty():
        yield queue.get_nowait()


def text_to_speech_ffmpeg_generator(
    package: TTSPackage,
    inputs: Inputs,
    preprocessed_inputs: PreprocessedInputs,
    sample_rate: int,
    logger: logging.Logger = logger,
    input_flags: typing.Tuple[str, ...] = ("-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1"),
    output_flags: typing.Tuple[str, ...] = ("-f", "mp3", "-b:a", "192k"),
) -> typing.Generator[bytes, None, None]:
    """Make a TTS generator.

    This implementation was inspired by:
    https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python

    TODO: Consider adding a retry mechanism if the TTS makes a mistake.
    TODO: Consider adding support for event tracking mechanism, like stackdriver or mixpanel.
    TODO: Consider adding a support for a timeout, in case, the user isn't consuming the data.

    NOTE: `Exception` does not catch `GeneratorExit`.
    https://stackoverflow.com/questions/18982610/difference-between-except-and-except-exception-as-e-in-python

    Returns: A generator that generates an audio file as defined by `output_flags`.
    """

    def get_spectrogram():
        for pred in package.spec_model(preprocessed_inputs, mode=Mode.GENERATE):
            # [num_frames, batch_size, num_frame_channels] →
            # [batch_size, num_frames, num_frame_channels]
            yield pred.frames.transpose(0, 1)

    command = (
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ar", str(sample_rate)]
        + list(input_flags)
        + ["-i", "pipe:"]
    )
    command += list(output_flags) + ["pipe:"]
    pipe = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=sys.stdout.buffer)
    queue: SimpleQueue = SimpleQueue()
    thread = threading.Thread(target=_enqueue, args=(pipe.stdout, queue), daemon=True)

    def close():
        assert pipe.stdin is not None
        pipe.stdin.close()
        pipe.wait()
        thread.join()
        assert pipe.stdout is not None
        pipe.stdout.close()

    try:
        thread.start()
        logger.info("Generating waveform...")
        generator = get_spectrogram()
        for waveform in generate_waveform(package.signal_model, generator, inputs.session):
            assert pipe.stdin is not None
            pipe.stdin.write(waveform.squeeze(0).cpu().numpy().tobytes())
            yield from _dequeue(queue)
        close()
        yield from _dequeue(queue)
        logger.info("Finished waveform generation.")
    except BaseException:
        close()
        logger.exception("Abrupt stop to waveform generation...")
