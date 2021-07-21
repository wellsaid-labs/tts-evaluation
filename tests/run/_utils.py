import pathlib
import shutil
import tempfile
import typing
from unittest import mock

import torch
from hparams import add_config

import lib
import run
from lib.utils import Tuple
from run import train
from run._tts import TTSPackage, package_tts
from run.data._loader import Alignment, Session, Span
from run.data._loader.wsl_init__english import (
    JUDY_BIEBER,
    LINDA_JOHNSON,
    lj_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
)
from run.train._utils import save_checkpoint
from tests import _utils
from tests._utils import make_metadata


def make_passage(
    alignments: Tuple[Alignment] = Alignment.stow([]),
    nonalignments: Tuple[Alignment] = Alignment.stow([]),
    speaker: run.data._loader.Speaker = run.data._loader.Speaker(""),
    audio_file: lib.audio.AudioMetadata = make_metadata(),
    script: typing.Optional[str] = None,
    transcript: typing.Optional[str] = None,
    speech_segments: typing.Optional[typing.Tuple[Span, ...]] = None,
    **kwargs,
) -> run.data._loader.Passage:
    """Make a `Passage` for testing."""
    max_ = lambda attr: max([getattr(a, attr)[1] for a in alignments])
    make_str: typing.Callable[[str], str]
    make_str = lambda attr: "." * max_(attr) if len(alignments) else ""
    script_ = make_str("script") if script is None else script
    transcript_ = make_str("transcript") if transcript is None else transcript
    sesh = Session(str(audio_file))
    passage = run.data._loader.Passage(
        audio_file, sesh, speaker, script_, transcript_, alignments, **kwargs
    )
    object.__setattr__(passage, "nonalignments", nonalignments)
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


def make_small_dataset() -> run._config.Dataset:
    """Create a small dataset for running tests."""
    directory = _utils.TEST_DATA_PATH / "datasets"
    temp_directory = pathlib.Path(tempfile.TemporaryDirectory().name)
    shutil.copytree(directory, temp_directory)
    books = [run.data._loader.m_ailabs__english_datasets.DOROTHY_AND_WIZARD_OZ]
    return {
        JUDY_BIEBER: m_ailabs_en_us_judy_bieber_speech_dataset(temp_directory, books=books),
        LINDA_JOHNSON: lj_speech_dataset(temp_directory),
    }


def make_spec_worker_state(
    train_data: run._config.Dataset,
    dev_data: run._config.Dataset,
    comet: train._utils.CometMLExperiment,
    device: torch.device,
) -> train.spectrogram_model._worker._State:
    """Create a spectrogram model worker state for testing."""
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        make_spec_model_state = train.spectrogram_model._worker._State.from_dataset
        return make_spec_model_state(train_data, dev_data, comet, device)


def make_spec_and_sig_worker_state(
    train_data: run._config.Dataset,
    dev_data: run._config.Dataset,
    comet: train._utils.CometMLExperiment,
    device: torch.device,
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
    spec_state = make_spec_worker_state(train_data, dev_data, comet, device)
    checkpoint_path = save_checkpoint(spec_state.to_checkpoint(), temp_dir_path, "ckpt")
    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel
        sig_state = train.signal_model._worker._State.make(checkpoint_path, comet, device)
    return spec_state, sig_state, temp_dir


def make_mock_tts_package() -> typing.Tuple[run._config.Dataset, TTSPackage]:
    """Create the required components needed for running TTS inference end-to-end.

    TODO: Consider decreasing the size of the spectrogram and signal model for performance.
    """
    run._config.configure()
    comet = train._utils.CometMLExperiment(disabled=True, project_name="project name")
    device = torch.device("cpu")
    dataset = make_small_dataset()
    add_config(train.spectrogram_model.__main__._make_configuration(dataset, dataset, False))
    add_config(train.signal_model.__main__._make_configuration(dataset, dataset, False))
    spec_state, sig_state, _ = make_spec_and_sig_worker_state(dataset, dataset, comet, device)
    return dataset, package_tts(spec_state.to_checkpoint(), sig_state.to_checkpoint())
