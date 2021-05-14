import pathlib
import shutil
import subprocess
import tempfile
import typing
import urllib.request
from unittest import mock

import numpy
import pytest
import torch
from hparams import add_config

import lib
import run
from lib.utils import Tuple
from run._tts import TTSBundle, make_tts_bundle
from run.data._loader import (
    JUDY_BIEBER,
    LINDA_JOHNSON,
    Alignment,
    Session,
    Span,
    lj_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
)
from run.train._utils import save_checkpoint
from tests import _utils

TEST_DATA_PATH = lib.environment.ROOT_PATH / "tests" / "_test_data"


def assert_almost_equal(a: torch.Tensor, b: torch.Tensor, **kwargs):
    numpy.testing.assert_almost_equal(a.cpu().detach().numpy(), b.cpu().detach().numpy(), **kwargs)


def first_parameter_url_side_effect(url: str, *_, **__):
    """ `unittest.mock.side_effect` for functions with a first parameter url.  """
    # TODO: Fix failure case if internet does not work
    assert urllib.request.urlopen(url).getcode() == 200
    return None


def assert_uniform_distribution(counter: typing.Counter, **kwargs):
    """ Assert that the counted distribution is uniform. """
    total = sum(counter.values())
    for value in counter.values():
        assert value / total == pytest.approx(1 / len(counter), **kwargs)


def subprocess_run_side_effect(command, *args, _command: str = "", _func=subprocess.run, **kwargs):
    """`subprocess.run.side_effect` that only returns if `_command` not in `command`. """
    if _command not in command:
        return _func(command, *args, **kwargs)


def make_metadata(
    path: pathlib.Path = pathlib.Path("."),
    sample_rate=1,
    num_channels=0,
    encoding=lib.audio.AudioEncoding.PCM_INT_8_BIT,
    bit_rate="",
    precision="",
    num_samples=0,
) -> lib.audio.AudioMetadata:
    """ Make a `AudioMetadata` for testing. """
    return lib.audio.AudioMetadata(
        path=path,
        sample_rate=sample_rate,
        num_channels=num_channels,
        encoding=encoding,
        bit_rate=bit_rate,
        precision=precision,
        num_samples=num_samples,
    )


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
    """Make a `Passage` for testing.

    TODO: Move this from `tests/_utils.py` to `tests/run/_utils.py`, for better locality.
    """
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


def make_unprocessed_passage(
    audio_path=pathlib.Path("."),
    speaker=run.data._loader.Speaker(""),
    script="",
    transcript="",
    alignments=None,
) -> run.data._loader.utils.UnprocessedPassage:
    """Make a `UnprocessedPassage` for testing.

    TODO: Move this from `tests/_utils.py` to `tests/run/_utils.py`, for better locality.
    """
    return run.data._loader.utils.UnprocessedPassage(
        audio_path, speaker, script, transcript, alignments
    )


def get_audio_metadata_side_effect(
    paths: typing.Union[pathlib.Path, typing.List[pathlib.Path]],
    _func=lib.audio.get_audio_metadata,
    **kwargs,
) -> typing.Union[lib.audio.AudioMetadata, typing.List[lib.audio.AudioMetadata]]:
    """`get_audio_metadata.side_effect` that returns placeholder metadata if the path doesn't
    exist."""
    is_list = isinstance(paths, list)
    paths = paths if isinstance(paths, list) else [paths]
    existing = [p for p in paths if p.exists() and p.is_file()]
    metadatas: typing.List[lib.audio.AudioMetadata] = _func(existing, **kwargs)
    metadatas = [metadatas.pop(0) if p.exists() else make_metadata(path=p) for p in paths]
    return metadatas if is_list else metadatas[0]


def maybe_normalize_audio_and_cache_side_effect(metadata: lib.audio.AudioMetadata):
    """`run.data._loader.normalize_audio_and_cache.side_effect` that returns the path without
    normalization."""
    return metadata.path


# NOTE: `unittest.mock.side_effect` for functions with a second parameter url.
second_parameter_url_side_effect = lambda _, *args, **kwargs: first_parameter_url_side_effect(
    *args, **kwargs
)


def mock_distributed_data_parallel(module, *_, **__):
    # NOTE: `module.module = module` would cause the `named_children` property to error, so
    # instead we set a `property`, learn more:
    # https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically
    module.__class__.module = property(lambda self: self)
    return module


def make_mock_tts_bundle() -> typing.Tuple[run._config.Dataset, TTSBundle]:
    """Create the required components needed for running TTS inference end-to-end."""
    run._config.configure()
    comet = run.train._utils.CometMLExperiment(disabled=True, project_name="project name")
    device = torch.device("cpu")

    books = [run.data._loader.m_ailabs.DOROTHY_AND_WIZARD_OZ]
    directory = _utils.TEST_DATA_PATH / "datasets"
    temp_directory = pathlib.Path(tempfile.TemporaryDirectory().name)
    shutil.copytree(directory, temp_directory)
    m_ailabs = m_ailabs_en_us_judy_bieber_speech_dataset(temp_directory, books=books)
    lj_speech = lj_speech_dataset(temp_directory)
    dataset = {JUDY_BIEBER: m_ailabs, LINDA_JOHNSON: lj_speech}

    add_config(run.train.spectrogram_model.__main__._make_configuration(dataset, dataset, False))
    with mock.patch("torch.nn.parallel.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel

        make_spec_model_state = run.train.spectrogram_model._worker._State.from_dataset
        spec_model_state = make_spec_model_state(dataset, dataset, comet, device)
        spec_model_ckpt = spec_model_state.to_checkpoint()
        spec_model_ckpt_path = save_checkpoint(spec_model_ckpt, temp_directory, "ckpt")

    add_config(run.train.signal_model.__main__._make_configuration(dataset, dataset, False))
    with mock.patch("run.train.signal_model._worker.DistributedDataParallel") as module:
        module.side_effect = mock_distributed_data_parallel

        make_sig_model_state = run.train.signal_model._worker._State.make
        sig_model_state = make_sig_model_state(spec_model_ckpt_path, comet, device)

    return dataset, make_tts_bundle(spec_model_ckpt, sig_model_state.to_checkpoint())
