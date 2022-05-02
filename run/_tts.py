"""
TODO: Add tests for every function.
"""

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
import torch
from hparams import HParam, configurable
from spacy.lang.en import English
from third_party import LazyLoader
from torchnlp.encoders.text import stack_and_pad_tensors

from lib.environment import PT_EXTENSION, load
from lib.signal_model import SignalModel, generate_waveform
from lib.spectrogram_model import Infer, Mode, Params, SpectrogramModel
from lib.text import grapheme_to_phoneme, load_en_core_web_md
from lib.utils import get_chunks, lengths_to_mask, tqdm_
from run import train
from run._config import CHECKPOINTS_PATH, GRAPHEME_TO_PHONEME_RESTRICTED, normalize_vo_script
from run.data._loader import Language, Session, Span, Speaker
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
    """
    Catalog of checkpoints uploaded to "wellsaid_labs_checkpoints".

    You can upload a new checkpoint, for example, like so:

        $ gsutil -m cp -r disk/checkpoints/2021_7_30_custom_voices \
                        gs://wellsaid_labs_checkpoints/v9_2021_6_30_custom_voices
    """

    """
    These checkpoints were deployed into production as Version 9.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/302
    Spectrogram Model Experiment (Step: 569,580):
    https://www.comet.ml/wellsaid-labs/1-stft-mike-2020-12/f52cc3ca9a394367a13bd06f26d78832
    Signal Model Experiment (Step: 770,733):
    https://www.comet.ml/wellsaid-labs/1-wav-mike-2021-03/0f4a4de9937c445bb7292d2a8f719fe1
    """

    V9: typing.Final = "v9"

    """
    These checkpoints include the Energy Industry Academy and The Explanation Company custom voices.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/334
    Spectrogram Model Experiment (Step: 649,128):
    https://www.comet.ml/wellsaid-labs/1-stft-mike-2020-12/881cea24682e470480786d1b2e20596b
    Signal Model Experiment (Step: 1,030,968):
    https://www.comet.ml/wellsaid-labs/1-wav-mike-2021-03/07a194f3bb99489d83061d3f2331536d
    """

    V9_2021_6_30_CUSTOM_VOICES: typing.Final = "v9_2021_6_30_custom_voices"

    """
    These checkpoints include V9 versions of the following custom voices:
    Energy Industry Academy, Happify, Super HiFi, The Explanation Company, US Pharmacopeia, Veritone

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/356
    Spectrogram Model Experiment (Step: 597,312):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/17289f19e0294d919bad9267cab4d5a0
    Signal Model Experiment (Step: 1,054,080):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/03ca7b7191c84fc7bd6bd348343e3d9e
    """

    V9_2021_8_03_CUSTOM_VOICES: typing.Final = "v9_2021_8_03_custom_voices"

    """
    These checkpoints include the Viacom custom voice.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/357
    Spectrogram Model Experiment (Step: 722,352):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/abf7e103ef824b7ab45bdfb35d07d6b3
    Signal Model Experiment (Step: 722,352):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/b8f3a52f181f4d67b245a85476fe5b0c
    """

    V9_2021_8_09_UPDATE_EIA_TEC_CUSTOM_VOICES: typing.Final = (
        "v9_2021_8_09_update_eia_tec_custom_voices"
    )

    """
    These checkpoints include the Viacom custom voice.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/355
    Spectrogram Model Experiment (Step: 590,423):
    https://www.comet.ml/wellsaid-labs/train-v9-viacom/eb24e3fb70f74f9c9a9490aa96d96f55
    Signal Model Experiment (Step: 734,542):
    https://www.comet.ml/wellsaid-labs/train-v9-viacom/df670689773b48608dd1ebb3dd6d7ea0
    """

    V9_2021_8_05_VIACOM_CUSTOM_VOICE: typing.Final = "v9_2021_8_05_viacom_custom_voice"

    """
    These checkpoints include the Hour One X NBC custom voice.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/358
    Spectrogram Model Experiment (Step: 901,518):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/17289f19e0294d919bad9267cab4d5a0
    Signal Model Experiment (Step: 868,989):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/e016a01e44904fe083401e0bf83eaf36
    """

    V9_2021_8_11_HOUR_ONE_X_NBC_CUSTOM_VOICE: typing.Final = (
        "v9_2021_8_11_hour_one_x_nbc_custom_voice"
    )

    """
    These checkpoints include the UneeQ X ASB [updated] custom voice.
    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 163,722):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/c9b857ba7d2f4cce9545bec429bd52be
    Signal Model Experiment (Step: 944,550):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/a9261ebef9d04f85b069365a049123e6
    """

    V9_2021_10_06_UNEEQ_X_ASB_CUSTOM_VOICE: typing.Final = "v9_2021_10_06_uneeq_x_asb_custom_voice"

    """
    These checkpoints include the StudySync custom voice.
    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 681, 340):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/db8d706b02e14d47be79d9966c57b959
    Signal Model Experiment (Step: 1,191,300):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/fbb56cfd643e416699047c7383eee9cf
    """

    V9_2021_11_04_STUDYSYNC_CUSTOM_VOICE: typing.Final = "v9_2021_11_04_studysync_custom_voice"

    """
    These checkpoints include the Five9 custom voice.
    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 830,467):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/d38734aa992a414db482f36fb5e8a961
    Signal Model Experiment (Step: 1,114,691):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/86c687102eba456da504a11093ee7366
    """

    V9_2021_11_09_FIVENINE_CUSTOM_VOICE: typing.Final = "v9_2021_11_09_fivenine_custom_voice"

    """
    These checkpoints include the StudySync custom voice (version 2).
    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 681, 340):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/db8d706b02e14d47be79d9966c57b959
    Signal Model Experiment (Step: 748,220):
    https://www.comet.ml/wellsaid-labs/v9-custom-voices/7e5aef579ce54539aff1668bfb4a9022
    """

    V9_2021_12_01_STUDYSYNC_CUSTOM_VOICE: typing.Final = "v9_2021_12_01_studysync_custom_voice"

    """
    These checkpoints include the UneeQ X ASB (V3) final custom voice.
    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 861,875):
    https://www.comet.ml/wellsaid-labs/uneeq-asb-experiments/360480297ec9416dbefec838c508c139
    Signal Model Experiment (Step: 796,250):
    https://www.comet.ml/wellsaid-labs/uneeq-asb-experiments/cf02f0011fcb44438be5a363721a991e
    """

    V9_2021_12_16_UNEEQ_X_ASB_CUSTOM_VOICE: typing.Final = "v9_2021_12_16_uneeq_x_asb_custom_voice"

    """
    These checkpoints include the 2021 Q4 Marketplace Expansion voices:
    Steve B, Paul B, Eric S, Marcus G, Chase J, Jude D, Charlie Z, Bella B, Tilda C

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/374
    Spectrogram Model Experiment (Step: 703,927):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/011893b50e4947ba9480cd0bc6d4dd1e
    Signal Model Experiment (Step: 958,209):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/90a55cdfc1174a9d8399e014fcee5fc8
    """

    V9_2021_Q4_MARKETPLACE_EXPANSION: typing.Final = "v9_2021_q4_marketplace_expansion"

    """
    These checkpoints include the 2022 Q1 Marketplace Expansion voices:
    Conversational: Patrick K, Kai M, Nicole L, Ava M, Vanessa N, Wade C, Sofia H
    Narration:      Jodi P, Gia V, Antony A, Raine B, Owen C, Genevieve M, Jarvis H, Theo K, James B
    Promo:          Zach E

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/386
    Spectrogram Model Experiment (Step: 736,392):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/d17cdb86c9314b678468d20921e5f4f2
    Signal Model Experiment (Step: 746,452):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/4d87bd1990c648baa9ba9d9340ca41dc
    """

    V9_2022_Q1_MARKETPLACE_EXPANSION: typing.Final = "v9_2022_q1_marketplace_expansion"


_GCS_PATH = "gs://wellsaid_labs_checkpoints/"
CHECKPOINTS_LOADERS = {
    e: functools.partial(load_checkpoints, CHECKPOINTS_PATH, e.value, _GCS_PATH + e.value)
    for e in Checkpoints
}


class TTSPackage(typing.NamedTuple):
    """A package of Python objects required to run TTS in inference mode.

    Args:
        ...
        spectrogram_model_comet_experiment_key: In order to identify the model origin, the
            comet ml experiment key is required.
        spectrogram_model_step: In order to identify the model origin, the checkpoint step which
            corresponds to the comet ml experiment is required.
        ...
    """

    input_encoder: InputEncoder
    spectrogram_model: SpectrogramModel
    signal_model: SignalModel
    spectrogram_model_comet_experiment_key: typing.Optional[str] = None
    spectrogram_model_step: typing.Optional[int] = None
    signal_model_comet_experiment_key: typing.Optional[str] = None
    signal_model_step: typing.Optional[int] = None


def package_tts(
    spectrogram_checkpoint: train.spectrogram_model._worker.Checkpoint,
    signal_checkpoint: train.signal_model._worker.Checkpoint,
):
    """Package together objects required for running TTS inference."""
    return TTSPackage(
        *spectrogram_checkpoint.export(),
        signal_checkpoint.export(),
        spectrogram_model_comet_experiment_key=spectrogram_checkpoint.comet_experiment_key,
        spectrogram_model_step=spectrogram_checkpoint.step,
        signal_model_comet_experiment_key=signal_checkpoint.comet_experiment_key,
        signal_model_step=signal_checkpoint.step,
    )


class PublicTextValueError(ValueError):
    pass


class PublicSpeakerValueError(ValueError):
    pass


def encode_tts_inputs(
    nlp: English, input_encoder: InputEncoder, script: str, speaker: Speaker, session: Session
) -> EncodedInput:
    """Encode TTS `script`, `speaker` and `session` for use with the model(s) with friendly errors
    for common issues.
    """
    normalized = normalize_vo_script(script, speaker.language)
    if len(normalized) == 0:
        raise PublicTextValueError("Text cannot be empty.")

    if speaker.language == Language.ENGLISH:
        for substring in GRAPHEME_TO_PHONEME_RESTRICTED:
            if substring in normalized:
                raise PublicTextValueError(f"Text cannot contain these characters: {substring}")
        tokens = typing.cast(str, grapheme_to_phoneme(nlp(normalized)))
    else:
        tokens = normalized

    if len(tokens) == 0:
        raise PublicTextValueError(f'Invalid text: "{script}"')

    decoded = DecodedInput(normalized, tokens, speaker, (speaker, session))
    token_encoder = input_encoder.token_encoder
    try:
        token_encoder.encode(decoded.tokens)
    except ValueError:
        vocab = set(token_encoder.vocab)
        difference = set(token_encoder.tokenize(decoded.tokens)).difference(vocab)
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
    params = Params(tokens=encoded.tokens, speaker=encoded.speaker, session=encoded.session)
    preds = typing.cast(Infer, package.spectrogram_model(params=params, mode=Mode.INFER))
    splits = preds.frames.split(split_size)
    predicted = list(
        generate_waveform(package.signal_model, splits, encoded.speaker, encoded.session)
    )
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
    inputs = [(normalize_vo_script(sc, sp.language), sp, se) for sc, sp, se in inputs]

    en_inputs = [(i, t) for i, t in enumerate(inputs) if t[1].language == Language.ENGLISH]
    en_tokens = []
    if len(en_inputs) > 0:
        docs: typing.List[spacy.tokens.Doc] = list(nlp.pipe([i[1][0] for i in en_inputs]))
        en_tokens = typing.cast(typing.List[str], grapheme_to_phoneme(docs))
    decoded = [DecodedInput(sc, sc, sp, (sp, se)) for sc, sp, se in inputs]
    for (i, (script, speaker, session)), tokens in zip(en_inputs, en_tokens):
        decoded[i] = DecodedInput(script, tokens, speaker, (speaker, session))

    encoded = [(i, package.input_encoder.encode(d)) for i, d in enumerate(decoded)]
    encoded = sorted(encoded, key=lambda i: i[1].tokens.numel())

    results: typing.Dict[int, TTSInputOutput] = {}
    for batch in tqdm_(list(get_chunks(encoded, batch_size))):
        tokens = stack_and_pad_tensors([e.tokens for _, e in batch], dim=1)
        params = Params(
            tokens=tokens.tensor,
            speaker=torch.stack([e.speaker for _, e in batch]).view(1, len(batch)),
            session=torch.stack([e.session for _, e in batch]).view(1, len(batch)),
            num_tokens=tokens.lengths,
        )
        preds = typing.cast(Infer, package.spectrogram_model(params=params, mode=Mode.INFER))
        spectrogram = preds.frames.transpose(0, 1)
        spectrogram_mask = lengths_to_mask(preds.lengths)
        signals = package.signal_model(
            spectrogram, params.speaker, params.session, spectrogram_mask
        )
        lengths = preds.lengths * package.signal_model.upscale_factor
        more_results = {
            j: TTSInputOutput(
                Params(
                    tokens=batch[i][1].tokens,
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
    package: TTSPackage,
    input: EncodedInput,
    logger_: logging.Logger = logger,
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
        params = Params(tokens=input.tokens, speaker=input.speaker, session=input.session)
        for pred in package.spectrogram_model(params=params, mode=Mode.GENERATE):
            # [num_frames, batch_size (optional), num_frame_channels] →
            # [batch_size (optional), num_frames, num_frame_channels]
            yield pred.frames.transpose(0, 1) if pred.frames.dim() == 3 else pred.frames

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
        logger_.info("Generating waveform...")
        generator = get_spectrogram()
        for waveform in generate_waveform(
            package.signal_model, generator, input.speaker, input.session
        ):
            assert pipe.stdin is not None
            pipe.stdin.write(waveform.cpu().numpy().tobytes())
            yield from _dequeue(queue)
        close()
        yield from _dequeue(queue)
        logger_.info("Finished waveform generation.")
    except BaseException:
        close()
        logger_.exception("Abrupt stop to waveform generation...")
