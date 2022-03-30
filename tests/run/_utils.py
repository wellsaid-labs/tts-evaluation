import pathlib
import re
import shutil
import tempfile
import typing
from unittest import mock

import config as cf
import torch

import lib
import run
from run import train
from run._tts import TTSPackage, package_tts
from run.data._loader import structures as struc
from run.data._loader.english import M_AILABS_DATASETS, lj_speech, m_ailabs
from run.data._loader.structures import Alignment, _make_nonalignments
from run.train._utils import save_checkpoint
from tests import _utils
from tests._utils import make_metadata


def make_speaker(
    label: str, style: struc.Style = struc.Style.LIBRI, dialect: struc.Dialect = struc.Dialect.EN_US
):
    return struc.Speaker(label, style, dialect, label, label)


def make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """Make an `Alignment` for testing."""
    return struc.Alignment(script, audio, transcript)


def make_alignments_1d(
    alignments: typing.Sequence[typing.Tuple[int, int]]
) -> typing.List[Alignment]:
    """
    Make a tuple of `Alignment`(s) for testing where `script`, `transcript` and `audio`
    have the same alignments.
    """
    return [make_alignment(a, a, a) for a in alignments]


def script_to_alignments(script: str) -> typing.Tuple[typing.Tuple[int, int]]:
    """Get the indicies of each "word" in `script`.

    Example:
        >>> script_to_alignments("This is a test")
        ((0, 3), (5, 6), (8, 8), (10, 13))
    """
    return tuple([(m.start(), m.end()) for m in re.finditer(r"\S+", script)])


def make_alignments_2d(
    script: str, audio_alignments: typing.Sequence[typing.Tuple[float, float]]
) -> typing.List[Alignment]:
    """
    Make a tuple of `Alignment`(s) for testing where `script` and `transcript`
    have the same alignments.
    """
    script_alignments = script_to_alignments(script)
    return [make_alignment(a, a, b) for a, b in zip(script_alignments, audio_alignments)]


def _max_alignment(
    alignments: typing.Sequence[Alignment], attr: typing.Literal["script", "transcript", "audio"]
):
    return max([getattr(a, attr)[1] for a in alignments])


def make_passage(
    alignments: typing.Optional[typing.Sequence[Alignment]] = None,
    nonalignments: typing.Optional[typing.Sequence[Alignment]] = None,
    speaker: struc.Speaker = make_speaker(""),
    audio_file: lib.audio.AudioMetadata = make_metadata(),
    script: typing.Optional[str] = None,
    transcript: typing.Optional[str] = None,
    speech_segments: typing.Optional[typing.Sequence[struc.Span]] = None,
    **kwargs,
) -> struc.Passage:
    """Make a `Passage` for testing."""
    # Set `alignments`, `script`, and `transcript`
    alignments = [] if alignments is None and script is None else alignments
    if alignments is None and script is not None:
        alignments = [Alignment(a, a, a) for a in script_to_alignments(script)]
    elif alignments is not None and script is None:
        if len(alignments) > 0:
            script = "." * _max_alignment(alignments, "script")
            transcript = "." * _max_alignment(alignments, "transcript")
        else:
            script, transcript = "", ""

    assert script is not None
    assert alignments is not None
    passage = struc.Passage(
        audio_file,
        struc.Session((speaker, str(audio_file))),
        script,
        script if transcript is None else transcript,
        Alignment.stow(alignments),
        **kwargs,
    )

    # Set `index`, `passages`, and `nonalignments`
    object.__setattr__(passage, "index", 0)
    object.__setattr__(passage, "passages", [passage])
    if nonalignments is None:
        nonalignments_ = _make_nonalignments(passage)
    else:
        nonalignments_ = Alignment.stow(nonalignments)
    object.__setattr__(passage, "nonalignments", nonalignments_)

    # Set `speech_segments`
    default_speech_segments = [passage[i] for i in range(len(alignments))]
    speech_segments = default_speech_segments if speech_segments is None else speech_segments
    object.__setattr__(passage, "speech_segments", tuple(speech_segments))

    return passage


def mock_distributed_data_parallel(module, *_, **__):
    # NOTE: `module.module = module` would cause the `named_children` property to error, so
    # instead we set a `property`, learn more:
    # https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    module.__class__.module = property(lambda self: self)
    return module


def make_small_dataset() -> run._utils.Dataset:
    """Create a small dataset for running tests."""
    directory = _utils.TEST_DATA_PATH / "datasets"
    temp_directory = pathlib.Path(tempfile.TemporaryDirectory().name)
    shutil.copytree(directory, temp_directory)
    books = [run.data._loader.english.m_ailabs.DOROTHY_AND_WIZARD_OZ]
    return {
        m_ailabs.JUDY_BIEBER: M_AILABS_DATASETS[m_ailabs.JUDY_BIEBER](temp_directory, books=books),
        lj_speech.LINDA_JOHNSON: lj_speech.lj_speech_dataset(temp_directory),
    }


def make_spec_worker_state(
    comet: train._utils.CometMLExperiment, device: torch.device
) -> train.spectrogram_model._worker._State:
    """Create a spectrogram model worker state for testing."""
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        make_spec_model_state = train.spectrogram_model._worker._State.from_scratch
        return make_spec_model_state(comet, device)


def make_spec_and_sig_worker_state(
    comet: train._utils.CometMLExperiment, device: torch.device
) -> typing.Tuple[
    train.spectrogram_model._worker._State,
    train.signal_model._worker._State,
    tempfile.TemporaryDirectory,
]:
    """Create a spectrogram and signal model worker state for testing.

    NOTE: If `temp_dir` is garbage collected, then the spectrogram checkpoint will be deleted.
    """
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name)
    spec_state = make_spec_worker_state(comet, device)
    checkpoint_path = save_checkpoint(spec_state.to_checkpoint(), temp_dir_path, "ckpt")
    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        sig_state = train.signal_model._worker._State.make(checkpoint_path, comet, device)
    sig_state.spectrogram_model_.allow_unk_on_eval(True)
    return spec_state, sig_state, temp_dir


def make_mock_tts_package() -> typing.Tuple[run._utils.Dataset, TTSPackage]:
    """Create the required components needed for running TTS inference end-to-end.

    TODO: Consider decreasing the size of the spectrogram and signal model for performance.
    """
    run._config.configure()
    comet = train._utils.CometMLExperiment(disabled=True, project_name="project name")
    device = torch.device("cpu")
    dataset = make_small_dataset()
    cf.add(train.spectrogram_model.__main__._make_configuration(dataset, dataset, False))
    cf.add(train.signal_model.__main__._make_configuration(dataset, dataset, False))
    spec_state, sig_state, _ = make_spec_and_sig_worker_state(comet, device)
    return dataset, package_tts(spec_state.to_checkpoint(), sig_state.to_checkpoint())
