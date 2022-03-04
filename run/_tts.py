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
from third_party import LazyLoader

from lib.environment import PT_EXTENSION, load
from lib.utils import get_chunks, tqdm_
from run import train
from run._config import CHECKPOINTS_PATH
from run._lang_config import normalize_vo_script
from run.data._loader import Session, Span, Speaker
from run.train.signal_model._model import SignalModel, generate_waveform
from run.train.spectrogram_model._model import Inputs, Mode, Preds, SpectrogramModel

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

    These checkpoints include the 2021 Q4 Marketplace Expansion voices:
    Steve B., Paul B., Eric S., Marcus G., Chase J., Jude D., Charlie Z., Bella B., Tilda C.

    Pull Request: https://github.com/wellsaid-labs/Text-to-Speech/pull/374
    Spectrogram Model Experiment (Step: 703,927):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/011893b50e4947ba9480cd0bc6d4dd1e
    Signal Model Experiment (Step: 958,209):
    https://www.comet.ml/wellsaid-labs/v9-marketplace-voices/90a55cdfc1174a9d8399e014fcee5fc8
    """

    V9_2021_Q4_MARKETPLACE_EXPANSION: typing.Final = "v9_2021_q4_marketplace_expansion"


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
        spectrogram_checkpoint.export(),
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


# TODO: Remove
encode_tts_inputs = None


def process_tts_inputs(
    script: str, speaker: Speaker, session: Session
) -> typing.Tuple[typing.List[str], Speaker, Session]:
    """Process TTS `script`, `speaker` and `session` for use with the model(s)."""
    normalized = normalize_vo_script(script, speaker.language)
    if len(normalized) == 0:
        raise PublicTextValueError("Text cannot be empty.")

    tokens = list(normalized)

    if len(tokens) == 0:
        raise PublicTextValueError(f'Invalid text: "{script}"')

    return tokens, speaker, session


def make_tts_inputs(
    package: TTSPackage, tokens: typing.List[str], speaker: Speaker, session: Session
) -> Inputs:
    """Create TTS `Inputs` and check compatibility."""
    excluded = [t for t in tokens if t not in package.spectrogram_model.token_vocab]
    if len(excluded) > 0:
        difference = ", ".join([repr(c)[1:-1] for c in sorted(set(excluded))])
        raise PublicTextValueError("Text cannot contain these characters: %s" % difference)

    if (
        speaker not in package.spectrogram_model.speaker_vocab
        or speaker not in package.signal_model.speaker_vocab
        or session not in package.spectrogram_model.session_vocab
        or session not in package.signal_model.session_vocab
    ):
        # NOTE: We do not expose speaker information in the `ValueError` because this error
        # is passed on to the public via the API.
        raise PublicSpeakerValueError("Speaker is not available.")

    return Inputs(speaker=[speaker], session=[session], tokens=[tokens])


def text_to_speech(
    package: TTSPackage,
    script: str,
    speaker: Speaker,
    session: Session,
    split_size: int = 32,
) -> numpy.ndarray:
    """Run TTS end-to-end with friendly errors."""
    tokens, speaker, session = process_tts_inputs(script, speaker, session)
    inputs = make_tts_inputs(package, tokens, speaker, session)
    preds = typing.cast(Preds, package.spectrogram_model(inputs=inputs, mode=Mode.INFER))
    splits = preds.frames.split(split_size)
    generator = generate_waveform(package.signal_model, splits, inputs.speaker, inputs.session)
    predicted = typing.cast(torch.Tensor, torch.cat(list(generator), dim=-1))
    return predicted.squeeze(0).detach().numpy()


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
    inputs = [(s.script, s.speaker, s.session) for s in spans]
    return batch_text_to_speech(package, inputs, **kwargs)


def batch_text_to_speech(
    package: TTSPackage,
    inputs: typing.List[typing.Tuple[str, Speaker, Session]],
    batch_size: int = 8,
) -> typing.List[TTSInputOutput]:
    """Run TTS end-to-end quickly with a verbose output."""
    inputs = [(normalize_vo_script(sc, sp.language), sp, se) for sc, sp, se in inputs]
    inputs_ = [(i, (list(t), sp, sh)) for i, (t, sp, sh) in enumerate(inputs)]
    inputs_ = sorted(inputs_, key=lambda i: len(i[1][0]))
    results: typing.Dict[int, TTSInputOutput] = {}
    for batch in tqdm_(list(get_chunks(inputs_, batch_size))):
        model_inputs = Inputs(
            speaker=[i[1][1] for i in batch],
            session=[i[1][2] for i in batch],
            tokens=[i[1][0] for i in batch],
        )
        preds = typing.cast(Preds, package.spectrogram_model(inputs=model_inputs, mode=Mode.INFER))
        spectrogram = preds.frames.transpose(0, 1)
        signals = package.signal_model(
            spectrogram, model_inputs.speaker, model_inputs.session, preds.frames_mask
        )
        lengths = preds.num_frames * package.signal_model.upscale_factor
        more_results = {
            j: TTSInputOutput(
                Inputs(
                    tokens=[model_inputs.tokens[i]],
                    speaker=[model_inputs.speaker[i]],
                    session=[model_inputs.session[i]],
                ),
                Preds(
                    frames=preds.frames[: preds.num_frames[i], i],
                    stop_tokens=preds.stop_tokens[: preds.num_frames[i], i],
                    alignments=preds.alignments[: preds.num_frames[i], i, : preds.num_tokens[i]],
                    num_frames=preds.num_frames[i],
                    frames_mask=preds.frames_mask[i, : preds.num_frames[i]],
                    num_tokens=preds.num_tokens[i],
                    tokens_mask=preds.tokens_mask[i, : preds.num_tokens[i]],
                    reached_max=preds.reached_max[i],
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
    inputs: Inputs,
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
        for pred in package.spectrogram_model(inputs=inputs, mode=Mode.GENERATE):
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
        logger_.info("Generating waveform...")
        generator = get_spectrogram()
        for waveform in generate_waveform(
            package.signal_model, generator, inputs.speaker, inputs.session
        ):
            assert pipe.stdin is not None
            pipe.stdin.write(waveform.squeeze(0).cpu().numpy().tobytes())
            yield from _dequeue(queue)
        close()
        yield from _dequeue(queue)
        logger_.info("Finished waveform generation.")
    except BaseException:
        close()
        logger_.exception("Abrupt stop to waveform generation...")
