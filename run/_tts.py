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

import config as cf
import numpy
import spacy
import spacy.language
import spacy.tokens
import torch

from lib.audio import griffin_lim
from lib.environment import PT_EXTENSION, load
from lib.text import XMLType, xml_to_text
from lib.utils import get_chunks, tqdm_
from run import train
from run._config import (
    CHECKPOINTS_PATH,
    load_spacy_nlp,
    normalize_and_verbalize_text,
    normalize_vo_script,
)
from run._models.signal_model import SignalModel, generate_waveform
from run._models.spectrogram_model import (
    Inputs,
    Mode,
    Preds,
    PreprocessedInputs,
    SpectrogramModel,
    preprocess,
)
from run.data._loader import Session, Span
from run.train.spectrogram_model._data import InferBatch, make_batch

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

        $ gsutil -m cp -r disk/checkpoints/v11_2023_03_01_staging_1 \
                        gs://wellsaid_labs_checkpoints/v11_2023_03_01_staging_1
    """

    """
    These checkpoints were deployed into staging as version "11.beta.1".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/468
    Spectrogram Model Experiment (Step: 208,347):
    https://www.comet.com/wellsaid-labs/v11-research-spectrogram/8fd4e8c2ed0946f78b0e82ef7e748c86
    Signal Model Experiment (Step: 546,207):
    https://www.comet.com/wellsaid-labs/v11-research-signal/c212a0b20d124578b3ac77846c3eb934
    """

    V11_2023_01_05_STAGING: typing.Final = "v11_2023_01_05_staging"

    """
    These checkpoints were deployed into staging as version "11.beta.2".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/473
    Spectrogram Model Experiment (Step: 675,720):
    https://www.comet.com/wellsaid-labs/v11-research-spectrogram/597e5ed553e0488e8382959f59e51ae8
    Signal Model Experiment (Step: 625,041):
    https://www.comet.com/wellsaid-labs/v11-research-signal/1be0f2d8cd03423888777b656e4543ad
    """

    V11_2023_01_12_STAGING: typing.Final = "v11_2023_01_12_staging"

    """
    These checkpoints were deployed into staging as version "11.beta.3".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/496
    Spectrogram Model Experiment (Step: 1,334,547):
    https://www.comet.com/wellsaid-labs/v11-research-spectrogram/6e8d8d2d93954c5c9f34f360276cb098
    Signal Model Experiment (Step: 242,133):
    https://www.comet.com/wellsaid-labs/v11-research-signal/3b981f23e6c44e829ff59877e9a7ec95
    """

    V11_2023_03_01_STAGING: typing.Final = "v11_2023_03_01_staging"

    """
    These checkpoints were deployed into staging as version "11.beta.4".

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/496
    Spectrogram Model Experiment (Step: 1,334,547):
    https://www.comet.com/wellsaid-labs/v11-research-spectrogram/6e8d8d2d93954c5c9f34f360276cb098
    Signal Model Experiment (Step: 281,550):
    https://www.comet.com/wellsaid-labs/v11-research-signal/3b981f23e6c44e829ff59877e9a7ec95
    """

    V11_2023_03_01_STAGING_1: typing.Final = "v11_2023_03_01_staging_1"


_GCS_PATH = "gs://wellsaid_labs_checkpoints/"
CHECKPOINTS_LOADERS = {
    e: functools.partial(load_checkpoints, CHECKPOINTS_PATH, e.value, _GCS_PATH + e.value)
    for e in Checkpoints
}


def get_session_vocab(*models: typing.Union[SpectrogramModel, SignalModel]) -> typing.Set[Session]:
    """Given a list of models find sessions that they all support."""
    return set.intersection(*tuple(set(m.session_embed.get_vocab()) for m in models))


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
        return get_session_vocab(self.spec_model, self.signal_model)


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


def _process_tts_inputs(
    nlp: spacy.language.Language,
    session_vocab: typing.Set[Session],
    token_vocab: typing.Set[str],
    script: XMLType,
    session: Session,
) -> typing.Tuple[Inputs, PreprocessedInputs]:
    """Process TTS `script` and `session` for use with the model(s)."""
    normalized = normalize_and_verbalize_text(script, session.spkr.language)
    if len(normalized) == 0:
        raise PublicTextValueError("Text cannot be empty.")

    inputs = Inputs.from_xml(normalized, nlp(xml_to_text(normalized)), session)
    preprocessed = cf.partial(preprocess)(inputs)

    tokens = typing.cast(typing.List[str], set(preprocessed.tokens[0]))
    excluded = [t for t in tokens if t not in token_vocab]
    if len(excluded) > 0:
        difference = ", ".join([repr(c)[1:-1] for c in sorted(set(excluded))])
        raise PublicTextValueError("Text cannot contain these characters: %s" % difference)

    if session not in session_vocab:
        # NOTE: We do not expose speaker information in the `ValueError` because this error
        # is passed on to the public via the API.
        raise PublicSpeakerValueError("Speaker is not available.")

    return inputs, preprocessed


def process_tts_inputs(
    package: TTSPackage,
    nlp: spacy.language.Language,
    script: XMLType,
    session: Session,
) -> typing.Tuple[Inputs, PreprocessedInputs]:
    """Process TTS `script` and `session` for use with the model(s) in `package`."""
    token_vocab = set(package.spec_model.token_embed.vocab.keys())
    session_vocab = package.session_vocab()
    return _process_tts_inputs(nlp, session_vocab, token_vocab, script, session)


def griffin_lim_tts(
    spec_model: SpectrogramModel, script: XMLType, session: Session
) -> numpy.ndarray:
    """Run TTS with griffin-lim."""
    nlp = load_spacy_nlp(session.spkr.language)
    session_vocab = set(spec_model.session_embed.vocab.keys())
    token_vocab = set(spec_model.token_embed.vocab.keys())
    _, preprocessed = _process_tts_inputs(nlp, session_vocab, token_vocab, script, session)
    preds = spec_model(inputs=preprocessed, mode=Mode.INFER)
    return cf.partial(griffin_lim)(preds.frames.squeeze(1).detach().numpy())


def basic_tts(
    package: TTSPackage,
    script: XMLType,
    session: Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end with friendly errors."""
    nlp = load_spacy_nlp(session.spkr.language)
    inputs, preprocessed = process_tts_inputs(package, nlp, script, session)
    preds = package.spec_model(inputs=preprocessed, mode=Mode.INFER)
    splits = preds.frames.transpose(0, 1).split(split_size)
    generator = generate_waveform(package.signal_model, splits, inputs.session)
    wave = typing.cast(torch.Tensor, torch.cat(list(generator), dim=-1))
    return wave.squeeze(0).detach().numpy()


class TTSResult(typing.NamedTuple):
    """The inputs and corresponding predictions from running TTS."""

    inputs: InferBatch
    spec_model: Preds
    sig_model: numpy.ndarray


TTSResults = typing.List[TTSResult]
BatchGen = typing.Generator[typing.Tuple[InferBatch, typing.List[int]], None, None]


def _process_input_batch(
    inputs: typing.List[typing.Tuple[XMLType, Session]]
) -> typing.List[typing.Tuple[XMLType, spacy.tokens.doc.Doc, Session]]:
    """Efficiently process text with spaCy and text normalization.

    TODO: Should this support script verbalization?
    """
    logger.info(f"Processing {len(inputs)} examples with spaCy...")
    seshs = [s for _, s in inputs]
    xmls = [x for x, _ in inputs]
    langs = set(s.spkr.language for s in seshs)
    result: typing.List[typing.Optional[spacy.tokens.doc.Doc]] = [None] * len(inputs)
    for lang in langs:
        nlp = load_spacy_nlp(lang)
        scripts = [(i, s) for i, (s, sesh) in enumerate(inputs) if sesh.spkr.language is lang]
        normalized = (xml_to_text(XMLType(normalize_vo_script(s, lang))) for _, s in scripts)
        docs = nlp.pipe(normalized)
        for (i, _), doc in zip(scripts, docs):
            result[i] = doc
    return list(zip(xmls, typing.cast(typing.List[spacy.tokens.doc.Doc], result), seshs))


def make_batches(
    inputs: typing.List[typing.Tuple[XMLType, Session]], batch_size: int = 8
) -> BatchGen:
    """Generate a sorted batch of inputs for batch TTS generation."""
    processed: typing.List[typing.Tuple[XMLType, spacy.tokens.doc.Doc, Session]]
    processed = _process_input_batch(inputs)
    sorted_ = sorted(enumerate(processed), key=lambda i: len(str(i[1][0])))
    batches = list(get_chunks(sorted_, batch_size))
    for batch in tqdm_(batches):
        xmls, docs = [b[1][0] for b in batch], [b[1][1] for b in batch]
        seshs, indicies = [b[1][2] for b in batch], [i for i, _ in batch]
        batch_input = Inputs.from_xml_batch(xmls, docs, seshs)  # type: ignore
        batch = InferBatch(
            spans=docs,
            xmls=xmls,
            inputs=batch_input,
            processed=cf.partial(preprocess)(batch_input),
        )
        yield batch, indicies


def make_training_batches(spans: typing.List[Span], batch_size: int = 8) -> BatchGen:
    """Generate a sorted batch of inputs for batch TTS generation with similar processing to
    the training pipeline."""
    spans_ = sorted(enumerate(spans), key=lambda s: s[1].audio_length)
    batches = list(get_chunks(spans_, batch_size))
    for batch in tqdm_(batches):
        batch_ = make_batch([s for _, s in batch], add_inputs=True)
        indicies = [i for i, _ in batch]
        assert batch_.inputs is not None
        yield batch_, indicies


def batch_tts(package: TTSPackage, batch_gen: BatchGen) -> TTSResults:
    """Run TTS on batches generated by `batch_gen`"""
    results: typing.Dict[int, TTSResult] = {}
    for batch, indicies in batch_gen:
        device = package.spec_model.output_scalar.device
        batch = batch.apply(lambda t: t.to(device))
        preds = package.spec_model(batch.processed, mode=Mode.INFER)
        spec = preds.frames.transpose(0, 1)
        assert batch.inputs is not None
        signals = package.signal_model(spec, batch.inputs.session, preds.frames_mask)
        num_samples = preds.num_frames * package.signal_model.upscale_factor
        batch, preds = batch.apply(lambda t: t.cpu()), preds.apply(lambda t: t.cpu())
        for i, j in zip(range(len(batch)), indicies):
            key = slice(i, i + 1)
            signal = signals[0][key, : num_samples[i]].cpu().detach().numpy()
            results[j] = TTSResult(inputs=batch[key], spec_model=preds[key], sig_model=signal)
    return [results[i] for i in range(len(results))]


def batch_griffin_lim_tts(
    spec_model: SpectrogramModel, batch_gen: BatchGen, **kwargs
) -> TTSResults:
    """Run TTS with griffin lim on batches generated by `batch_gen`"""
    # TODO: Add an option to add a streamlit progress bar when this is being used. Let's consider
    # moving this functionality to `_streamlit.py` because it's only used with in workbooks.
    results: typing.Dict[int, TTSResult] = {}
    for batch, indicies in batch_gen:
        device = spec_model.output_scalar.device
        # TODO: Rename `apply` to indicate that it's in-place.
        batch = batch.apply(lambda t: t.to(device))
        preds = spec_model(batch.processed, mode=Mode.INFER)
        batch, preds = batch.apply(lambda t: t.cpu()), preds.apply(lambda t: t.cpu())
        for i, j in zip(range(preds.frames.shape[1]), indicies):
            key = slice(i, i + 1)
            pred = preds[key]
            signal = cf.call(griffin_lim, pred.frames[:, 0].cpu().detach().numpy(), **kwargs)
            results[j] = TTSResult(inputs=batch[key], spec_model=pred, sig_model=signal)
    return [results[i] for i in range(len(results))]


def _enqueue(out: typing.IO[bytes], queue: SimpleQueue):
    """Enqueue all lines from a file-like object to `queue`."""
    for line in iter(out.readline, b""):
        queue.put(line)


def _dequeue(queue: SimpleQueue) -> typing.Generator[bytes, None, None]:
    """Dequeue all items from `queue`."""
    while not queue.empty():
        yield queue.get_nowait()


def tts_ffmpeg_generator(
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
