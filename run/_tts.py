"""
TODO: Add tests for every function.
"""

import gc
import logging
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

from lib.signal_model import SignalModel, generate_waveform
from lib.spectrogram_model import Infer, Mode, Params, SpectrogramModel
from lib.text import (
    GRAPHEME_TO_PHONEME_RESTRICTED,
    grapheme_to_phoneme,
    load_en_core_web_md,
    normalize_vo_script,
)
from run import train
from run.data._loader import Session, Speaker
from run.train.spectrogram_model._data import DecodedInput, EncodedInput, InputEncoder


class TTSBundle(typing.NamedTuple):
    """The bare minimum required to run a TTS model in inference mode."""

    input_encoder: InputEncoder
    spectrogram_model: SpectrogramModel
    signal_model: SignalModel


def make_tts_bundle(
    spectrogram_checkpoint: train.spectrogram_model._worker.Checkpoint,
    signal_checkpoint: train.signal_model._worker.Checkpoint,
):
    """Make a bundle of objects required for running TTS infernece."""
    return TTSBundle(*spectrogram_checkpoint.export(), signal_checkpoint.export())


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
    input_encoder: InputEncoder,
    spec_model: SpectrogramModel,
    sig_model: SignalModel,
    script: str,
    speaker: Speaker,
    session: Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end.

    TODO: Add an end-to-end function for stream TTS.
    TODO: Add an end-to-end function for batch TTS.
    """
    nlp = load_en_core_web_md(disable=("parser", "ner"))
    encoded = encode_tts_inputs(nlp, input_encoder, script, speaker, session)
    params = Params(tokens=encoded.phonemes, speaker=encoded.speaker, session=encoded.session)
    preds = typing.cast(Infer, spec_model(params=params, mode=Mode.INFER))
    splits = preds.frames.split(split_size)
    predicted = list(generate_waveform(sig_model, splits, encoded.speaker, encoded.session))
    predicted = typing.cast(torch.Tensor, torch.cat(predicted, dim=-1))
    return predicted.detach().numpy()


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
    logger: logging.Logger,
    spec_model: SpectrogramModel,
    sig_model: SignalModel,
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
        for pred in spec_model(params=params, mode=Mode.GENERATE):
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
        logger.info("Generating waveform...")
        generator = get_spectrogram()
        for waveform in generate_waveform(sig_model, generator, input.speaker, input.session):
            pipe.stdin.write(waveform.cpu().numpy().tobytes())
            yield from _dequeue(queue)
        close()
        yield from _dequeue(queue)
    except BaseException:
        close()
        logger.info("Finished generating waveform.")
