import pathlib
import shutil
import tempfile
from unittest import mock

import lib
from tests import _utils


@mock.patch("lib.audio.get_audio_metadata")
@mock.patch("pathlib.Path.is_file")
@mock.patch("urllib.request.urlretrieve")
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_is_file, mock_get_audio_metadata):
    """ Test `lib.datasets.m_ailabs_en_us_speech_dataset` loads the data. """
    mock_is_file.return_value = True
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    mock_get_audio_metadata.side_effect = _utils.get_audio_metadata_side_effect
    archive = _utils.TEST_DATA_PATH / "datasets" / "M-AILABS" / "en_US.tgz"
    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        (directory / archive.parent.name).mkdir()
        shutil.copy(archive, directory / archive.parent.name / archive.name)
        data = lib.datasets.m_ailabs.m_ailabs_en_us_speech_dataset(directory=directory)
        assert len(data) == 2046
        assert sum([len(r.text) for r in data]) == 226649
        assert data[0] == lib.datasets.Example(
            audio_path=pathlib.Path(
                directory
                / archive.parent.name
                / "en_US/by_book/female/judy_bieber"
                / "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav"
            ),
            speaker=lib.datasets.JUDY_BIEBER,
            alignments=(lib.datasets.Alignment(text=(0, 14), audio=(0.0, 0)),),
            text="To My Readers.",
            metadata={
                "book": lib.datasets.m_ailabs.Book(
                    speaker=lib.datasets.JUDY_BIEBER,
                    title="dorothy_and_wizard_oz",
                )
            },
        )
