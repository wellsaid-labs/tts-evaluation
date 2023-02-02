import pathlib
import shutil
import tempfile
from unittest import mock

import run.data._loader
from run.data._loader import structures as struc
from run.data._loader.english.m_ailabs import JUDY_BIEBER, Book, m_ailabs_en_us_speech_dataset
from tests import _utils


@mock.patch("urllib.request.urlretrieve")
def test_m_ailabs_speech_dataset(mock_urlretrieve):
    """Test `run.data._loader.m_ailabs_en_us_speech_dataset` loads the data."""
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    archive = _utils.TEST_DATA_PATH / "datasets" / "M-AILABS" / "en_US.tgz"

    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        (directory / archive.parent.name).mkdir()
        shutil.copy(archive, directory / archive.parent.name / archive.name)
        data = m_ailabs_en_us_speech_dataset(directory=directory)
        passages = [p for d in data for p in d]
        assert len(passages) == 2046
        assert sum(len(p.script) for p in passages) == 226649
        path = directory / archive.parent.name / "en_US/by_book/female/judy_bieber"
        assert passages[0] == run.data._loader.structures.UnprocessedPassage(
            audio_path=path / "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav",
            session=run.data._loader.Session(
                JUDY_BIEBER, "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01"
            ),
            script="To My Readers.",
            transcript="To My Readers.",
            alignments=None,
            other_metadata={
                1: "To My Readers.",
                "book": Book(struc.Dialect.EN_US, JUDY_BIEBER, "dorothy_and_wizard_oz"),
            },
        )
