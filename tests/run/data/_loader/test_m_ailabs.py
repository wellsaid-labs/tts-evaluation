import pathlib
import shutil
import tempfile
from unittest import mock

import run.data._loader
from run.data._loader import Alignment
from tests import _utils


@mock.patch("run.data._loader.utils._exists", return_value=True)
@mock.patch("run.data._loader.utils.get_audio_metadata")
@mock.patch("urllib.request.urlretrieve")
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_get_audio_metadata, _):
    """ Test `run.data._loader.m_ailabs_en_us_speech_dataset` loads the data. """
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    mock_get_audio_metadata.side_effect = _utils.get_audio_metadata_side_effect
    archive = _utils.TEST_DATA_PATH / "datasets" / "M-AILABS" / "en_US.tgz"

    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        (directory / archive.parent.name).mkdir()
        shutil.copy(archive, directory / archive.parent.name / archive.name)
        data = run.data._loader.m_ailabs.m_ailabs_en_us_speech_dataset(directory=directory)
        assert len(data) == 2046
        assert sum([len(r.script) for r in data]) == 226649
        path = directory / archive.parent.name / "en_US/by_book/female/judy_bieber"
        path = path / "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav"
        nonalignments_ = [
            Alignment(script=(0, 0), audio=(0.0, 0.0), transcript=(0, 0)),
            Alignment(script=(14, 14), audio=(0.0, 0.0), transcript=(14, 14)),
        ]
        assert data[0] == run.data._loader.Passage(
            audio_file=_utils.make_metadata(path),
            speaker=run.data._loader.JUDY_BIEBER,
            script="To My Readers.",
            transcript="To My Readers.",
            alignments=Alignment.stow([Alignment((0, 14), (0.0, 0.0), (0, 14))]),
            nonalignments=Alignment.stow(nonalignments_),
            other_metadata={
                1: "To My Readers.",
                "book": run.data._loader.m_ailabs.Book(
                    dataset=run.data._loader.m_ailabs.US_DATASET,
                    speaker=run.data._loader.JUDY_BIEBER,
                    title="dorothy_and_wizard_oz",
                ),
            },
        )
