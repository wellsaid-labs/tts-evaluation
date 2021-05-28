"""
TODO: Add tests for every function.
"""

import enum
import functools
import gc
import logging
import pathlib
import subprocess
import sys
import threading
import typing
from queue import SimpleQueue
from subprocess import PIPE

import numpy
import torch
from hparams import HParam, configurable
from spacy.lang.en import English
from third_party import LazyLoader
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.utils import lengths_to_mask

from lib.environment import PT_EXTENSION, load
from lib.signal_model import SignalModel, generate_waveform
from lib.spectrogram_model import Infer, Mode, Params, SpectrogramModel
from lib.text import (
    GRAPHEME_TO_PHONEME_RESTRICTED,
    grapheme_to_phoneme,
    load_en_core_web_md,
    normalize_vo_script,
)
from lib.utils import get_chunks, tqdm_
from run import train
from run._config import CHECKPOINTS_PATH
from run.data._loader import Session, Span, Speaker
from run.train.spectrogram_model._data import DecodedInput, EncodedInput, InputEncoder

if typing.TYPE_CHECKING:  # pragma: no cover
    import spacy.tokens
else:
    spacy = LazyLoader("spacy", globals(), "spacy")

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
    V9_STAGING: typing.Final = "v9_staging"


_GCS_PATH = "gs://wellsaid_labs_checkpoints/"
CHECKPOINTS_LOADERS = {
    e: functools.partial(load_checkpoints, CHECKPOINTS_PATH, e.value, _GCS_PATH + e.value)
    for e in Checkpoints
}


class TTSPackage(typing.NamedTuple):
    """A package of Python objects required to run TTS in inference mode."""

    input_encoder: InputEncoder
    spectrogram_model: SpectrogramModel
    signal_model: SignalModel


def package_tts(
    spectrogram_checkpoint: train.spectrogram_model._worker.Checkpoint,
    signal_checkpoint: train.signal_model._worker.Checkpoint,
):
    """Package together objects required for running TTS inference."""
    return TTSPackage(*spectrogram_checkpoint.export(), signal_checkpoint.export())


class PublicTextValueError(ValueError):
    pass


class PublicSpeakerValueError(ValueError):
    pass


@configurable
def encode_tts_inputs(
    nlp: English,
    input_encoder: InputEncoder,
    script: str,
    speaker: Speaker,
    session: Session,
    seperator: str = HParam(),
) -> EncodedInput:
    """Encode TTS `script`, `speaker` and `session` for use with the model(s) with friendly errors
    for common issues.
    """
    normalized = normalize_vo_script(script)
    if len(normalized) == 0:
        raise PublicTextValueError("Text cannot be empty.")
    for substring in list(GRAPHEME_TO_PHONEME_RESTRICTED) + [seperator]:
        if substring in normalized:
            raise PublicTextValueError(f"Text cannot contain these characters: {substring}")

    phonemes = typing.cast(str, grapheme_to_phoneme(nlp(normalized)))
    if len(phonemes) == 0:
        raise PublicTextValueError(f'Invalid text: "{script}"')

    decoded = DecodedInput(normalized, phonemes, speaker, (speaker, session))
    phoneme_encoder = input_encoder.phoneme_encoder
    try:
        phoneme_encoder.encode(decoded.phonemes)
    except ValueError:
        vocab = set(phoneme_encoder.vocab)
        difference = set(phoneme_encoder.tokenize(decoded.phonemes)).difference(vocab)
        difference = ", ".join([repr(c)[1:-1] for c in sorted(list(difference))])
        raise PublicTextValueError("Text cannot contain these characters: %s" % difference)

    try:
        input_encoder.speaker_encoder.encode(decoded.speaker)
        input_encoder.session_encoder.encode(decoded.session)
    except ValueError:
        # NOTE: We do not expose speaker information in the `ValueError` because this error
        # is passed on to the public via the API.
        raise PublicSpeakerValueError("Speaker is not available.")

    return input_encoder.encode(decoded)


def text_to_speech(
    package: TTSPackage,
    script: str,
    speaker: Speaker,
    session: Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end with friendly errors."""
    nlp = load_en_core_web_md(disable=("parser", "ner"))
    encoded = encode_tts_inputs(nlp, package.input_encoder, script, speaker, session)
    params = Params(tokens=encoded.phonemes, speaker=encoded.speaker, session=encoded.session)
    preds = typing.cast(Infer, package.spectrogram_model(params=params, mode=Mode.INFER))
    splits = preds.frames.split(split_size)
    predicted = list(generate_waveform(package.signal_model, splits))
    predicted = typing.cast(torch.Tensor, torch.cat(predicted, dim=-1))
    return predicted.detach().numpy()


class TTSInputOutput(typing.NamedTuple):
    """Text-to-speech input and output."""

    params: Params
    spec_model: Infer
    sig_model: numpy.ndarray


def batch_span_to_speech(
    package: TTSPackage, spans: typing.List[Span], **kwargs
) -> typing.List[TTSInputOutput]:
    """
    NOTE: This method doesn't consider `Span` context for TTS generation.
    """
    inputs = [(s.script, s.speaker, s.session) for s in spans]
    return batch_text_to_speech(package, inputs, **kwargs)


def batch_text_to_speech(
    package: TTSPackage,
    inputs: typing.List[typing.Tuple[str, Speaker, Session]],
    batch_size: int = 8,
) -> typing.List[TTSInputOutput]:
    """Run TTS end-to-end quickly with a verbose output."""
    nlp = load_en_core_web_md(disable=("parser", "ner"))
    inputs = [(normalize_vo_script(sc), sp, se) for sc, sp, se in inputs]
    docs: typing.List[spacy.tokens.Doc] = list(nlp.pipe([i[0] for i in inputs]))
    phonemes = typing.cast(typing.List[str], grapheme_to_phoneme(docs))
    decoded = [DecodedInput(sc, p, sp, (sp, se)) for (sc, sp, se), p in zip(inputs, phonemes)]
    encoded = [(i, package.input_encoder.encode(d)) for i, d in enumerate(decoded)]
    encoded = sorted(encoded, key=lambda i: i[1].phonemes.numel())
    results: typing.Dict[int, TTSInputOutput] = {}
    for batch in tqdm_(list(get_chunks(encoded, batch_size))):
        tokens = stack_and_pad_tensors([e.phonemes for _, e in batch], dim=1)
        params = Params(
            tokens=tokens.tensor,
            speaker=torch.stack([e.speaker for _, e in batch]).view(1, len(batch)),
            session=torch.stack([e.session for _, e in batch]).view(1, len(batch)),
            num_tokens=tokens.lengths,
        )
        preds = typing.cast(Infer, package.spectrogram_model(params=params, mode=Mode.INFER))
        spectrogram = preds.frames.transpose(0, 1)
        spectrogram_mask = lengths_to_mask(preds.lengths)
        signals = package.signal_model(spectrogram, spectrogram_mask)
        lengths = preds.lengths * package.signal_model.upscale_factor
        more_results = {
            j: TTSInputOutput(
                Params(
                    tokens=batch[i][1].phonemes,
                    speaker=batch[i][1].speaker,
                    session=batch[i][1].session,
                ),
                Infer(
                    frames=preds.frames[: preds.lengths[:, i], i],
                    stop_tokens=preds.stop_tokens[: preds.lengths[:, i], i],
                    alignments=preds.alignments[: preds.lengths[:, i], i, : tokens.lengths[:, i]],
                    lengths=preds.lengths[:, i],
                    reached_max=preds.reached_max[:, i],
                ),
                signals[i][: lengths[:, i]].detach().numpy(),
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


@configurable
def text_to_speech_ffmpeg_generator(
    logger_: logging.Logger,
    package: TTSPackage,
    input: EncodedInput,
    sample_rate: int = HParam(),
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
        params = Params(tokens=input.phonemes, speaker=input.speaker, session=input.session)
        for pred in package.spectrogram_model(params=params, mode=Mode.GENERATE):
            # [num_frames, batch_size (optional), num_frame_channels] →
            # [batch_size (optional), num_frames, num_frame_channels]
            gc.collect()
            yield pred.frames.transpose(0, 1) if pred.frames.dim() == 3 else pred.frames

    command = ["ffmpeg", "-ar", str(sample_rate)] + list(input_flags) + ["-i", "pipe:"]
    command += list(output_flags) + ["pipe:"]
    pipe = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=sys.stdout.buffer)
    queue: SimpleQueue = SimpleQueue()
    thread = threading.Thread(target=_enqueue, args=(pipe.stdout, queue), daemon=True)

    def close():
        pipe.stdin.close()
        pipe.wait()
        thread.join()
        pipe.stdout.close()

    try:
        thread.start()
        logger_.info("Generating waveform...")
        generator = get_spectrogram()
        for waveform in generate_waveform(package.signal_model, generator):
            pipe.stdin.write(waveform.cpu().numpy().tobytes())
            yield from _dequeue(queue)
        close()
        yield from _dequeue(queue)
    except BaseException:
        close()
        logger_.info("Finished generating waveform.")
