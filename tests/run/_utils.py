import pathlib
import shutil
import tempfile
import typing
from unittest import mock

import torch
from hparams import add_config

import lib
import run
from run import train
from run._tts import TTSPackage, package_tts
from run.data._loader import Alignment, Passage, Session, Span, Speaker, make_en_speaker
from run.data._loader.data_structures import _make_nonalignments
from run.data._loader.english import (
    JUDY_BIEBER,
    LINDA_JOHNSON,
    lj_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
)
from run.train._utils import save_checkpoint
from tests import _utils
from tests._utils import make_metadata


def make_alignment(script=(0, 0), transcript=(0, 0), audio=(0.0, 0.0)):
    """Make an `Alignment` for testing."""
    return Alignment(script, audio, transcript)


def make_alignments(
    alignments: typing.Tuple[typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]], ...]
) -> typing.List[Alignment]:
    """
    Make a tuple of `Alignment`(s) for testing where `script` and `transcript`
    have the same alignments.
    """
    return [make_alignment(a, a, b) for a, b in alignments]


def make_passage(
    alignments: typing.List[Alignment] = [],
    nonalignments: typing.Optional[typing.List[Alignment]] = None,
    speaker: Speaker = make_en_speaker(""),
    audio_file: lib.audio.AudioMetadata = make_metadata(),
    script: typing.Optional[str] = None,
    transcript: typing.Optional[str] = None,
    speech_segments: typing.Optional[typing.Tuple[Span, ...]] = None,
    **kwargs,
) -> Passage:
    """Make a `Passage` for testing."""
    alignments_ = Alignment.stow(alignments)
    max_ = lambda attr: max([getattr(a, attr)[1] for a in alignments])
    make_str: typing.Callable[[str], str]
    make_str = lambda attr: "." * max_(attr) if len(alignments) else ""
    script_ = make_str("script") if script is None else script
    transcript_ = make_str("transcript") if transcript is None else transcript
    sesh = Session((speaker, str(audio_file)))
    passage = Passage(audio_file, sesh, script_, transcript_, alignments_, **kwargs)
    object.__setattr__(passage, "index", 0)
    object.__setattr__(passage, "passages", [passage])
    if nonalignments is None:
        nonalignments_ = _make_nonalignments(passage)
    else:
        nonalignments_ = Alignment.stow(nonalignments)
    object.__setattr__(passage, "nonalignments", nonalignments_)
    default_speech_segments = tuple(passage[i] for i in range(len(alignments)))
    speech_segments = default_speech_segments if speech_segments is None else speech_segments
    object.__setattr__(passage, "speech_segments", speech_segments)
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
        JUDY_BIEBER: m_ailabs_en_us_judy_bieber_speech_dataset(temp_directory, books=books),
        LINDA_JOHNSON: lj_speech_dataset(temp_directory),
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
    add_config(train.spectrogram_model.__main__._make_configuration(dataset, dataset, False))
    add_config(train.signal_model.__main__._make_configuration(dataset, dataset, False))
    spec_state, sig_state, _ = make_spec_and_sig_worker_state(comet, device)
    return dataset, package_tts(spec_state.to_checkpoint(), sig_state.to_checkpoint())
