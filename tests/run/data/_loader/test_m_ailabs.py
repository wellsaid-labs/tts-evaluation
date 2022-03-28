import pathlib
import shutil
import tempfile
from unittest import mock

import lib
import run.data._loader
from run.data._loader import Alignment
from run.data._loader.english import JUDY_BIEBER
from run.data._loader.english.m_ailabs import US_DATASET, m_ailabs_en_us_speech_dataset
from tests import _utils
from tests.run.data._loader._utils import maybe_normalize_audio_and_cache_side_effect


@mock.patch("run.data._loader.data_structures._filter_existing_paths")
@mock.patch("run.data._loader.data_structures.get_audio_metadata")
@mock.patch("run.data._loader.data_structures._loader.utils.maybe_normalize_audio_and_cache")
@mock.patch("run.data._loader.data_structures._loader.utils.get_non_speech_segments_and_cache")
@mock.patch("run.data._loader.data_structures.cf.partial")
@mock.patch("urllib.request.urlretrieve")
def test_m_ailabs_speech_dataset(
    mock_urlretrieve,
    mock_config_partial,
    mock_get_non_speech_segments_and_cache,
    mock_normalize_and_cache,
    mock_get_audio_metadata,
    mock_filter_existing_paths,
):
    """Test `run.data._loader.m_ailabs_en_us_speech_dataset` loads the data."""
    mock_filter_existing_paths.side_effect = lambda x: x
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    mock_get_audio_metadata.side_effect = _utils.get_audio_metadata_side_effect
    mock_normalize_and_cache.side_effect = maybe_normalize_audio_and_cache_side_effect
    mock_config_partial.side_effect = _utils.config_partial_side_effect
    mock_get_non_speech_segments_and_cache.side_effect = lambda *_, **__: lib.utils.Timeline([])
    archive = _utils.TEST_DATA_PATH / "datasets" / "M-AILABS" / "en_US.tgz"

    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        (directory / archive.parent.name).mkdir()
        shutil.copy(archive, directory / archive.parent.name / archive.name)
        data = m_ailabs_en_us_speech_dataset(directory=directory)
        assert len(data) == 2046
        assert sum([len(r.script) for r in data]) == 226649
        path = directory / archive.parent.name / "en_US/by_book/female/judy_bieber"
        path = path / "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav"
        assert data[0] == run.data._loader.Passage(
            audio_file=_utils.make_metadata(path),
            session=run.data._loader.Session(
                (JUDY_BIEBER, "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01")
            ),
            script="To My Readers.",
            transcript="To My Readers.",
            alignments=Alignment.stow([Alignment((0, 14), (0.0, 0.0), (0, 14))]),
            other_metadata={
                1: "To My Readers.",
                "book": run.data._loader.m_ailabs.Book(
                    dataset=US_DATASET,
                    speaker=JUDY_BIEBER,
                    title="dorothy_and_wizard_oz",
                ),
            },
        )
