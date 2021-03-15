import pathlib
import shutil
import tempfile
from unittest import mock

import lib
from tests import _utils


@mock.patch("lib.datasets.utils._exists", return_value=True)
@mock.patch("lib.datasets.utils.get_audio_metadata")
@mock.patch("urllib.request.urlretrieve")
def test_m_ailabs_speech_dataset(mock_urlretrieve, mock_get_audio_metadata, _):
    """ Test `lib.datasets.m_ailabs_en_us_speech_dataset` loads the data. """
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    mock_get_audio_metadata.side_effect = _utils.get_audio_metadata_side_effect
    archive = _utils.TEST_DATA_PATH / "datasets" / "M-AILABS" / "en_US.tgz"

    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        (directory / archive.parent.name).mkdir()
        shutil.copy(archive, directory / archive.parent.name / archive.name)
        data = lib.datasets.m_ailabs.m_ailabs_en_us_speech_dataset(directory=directory)
        assert len(data) == 2046
        assert sum([len(r.script) for r in data]) == 226649
        path = directory / archive.parent.name / "en_US/by_book/female/judy_bieber"
        path = path / "dorothy_and_wizard_oz/wavs/dorothy_and_wizard_oz_01_f000001.wav"
        alignments = lib.utils.Tuples(
            [lib.datasets.Alignment((0, 14), (0.0, 0.0), (0, 14))],
            lib.datasets.alignment_dtype,
        )
        nonalignments_ = [
            lib.datasets.Alignment(script=(0, 0), audio=(0.0, 0.0), transcript=(0, 0)),
            lib.datasets.Alignment(script=(14, 14), audio=(0.0, 0.0), transcript=(14, 14)),
        ]
        nonalignments = lib.utils.Tuples(nonalignments_, lib.datasets.alignment_dtype)
        assert data[0] == lib.datasets.Passage(
            audio_file=_utils.make_metadata(path),
            speaker=lib.datasets.JUDY_BIEBER,
            script="To My Readers.",
            transcript="To My Readers.",
            alignments=alignments,
            nonalignments=nonalignments,
            other_metadata={
                1: "To My Readers.",
                "book": lib.datasets.m_ailabs.Book(
                    dataset=lib.datasets.m_ailabs.US_DATASET,
                    speaker=lib.datasets.JUDY_BIEBER,
                    title="dorothy_and_wizard_oz",
                ),
            },
        )
