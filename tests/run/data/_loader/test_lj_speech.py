import pathlib
import re
import shutil
import tempfile
from unittest import mock

import lib
import run.data._loader
from tests import _utils

verbalize_test_cases = {
    "LJ044-0055": "five four four Camp Street New",  # Test special case
    "LJ032-0036": "Number two two zero two one three zero four six two",  # Test special case
    # Test time
    "LJ036-0167": "he would have entered the cab at twelve forty-seven or twelve forty-eight p.m.",
    "LJ002-0215": "the Star Chamber in the sixteenth Charles",  # Test Ordinals
    "LJ013-0088": "England notes for one thousand pounds each,",  # Test currency
    "LJ037-0204": "Post Office Box two nine one five,",  # Test PO Box
    "LJ032-0025": "bearing serial number C two seven six six",  # Test Serial
    "LJ028-0257": "five twenty-one Nebuchadnezzar the third",  # Test Year
    "LJ028-0363": "June thirteen, three twenty-three B.C.,",  # Test Year
    "LJ047-0127": "On August twenty-one, nineteen sixty-three, Bureau",  # Test Year
    "LJ016-0090": "them towards Number one, Newgate Street",  # Test Numero
    "LJ037-0252": "Commission Exhibit Number one sixty-two",  # Test Numero
    # Test Number
    "LJ039-0063": "rifle, at a range of one hundred seventy-seven to two hundred sixty-six feet",
    # Test Abbreviations
    "LJ004-0049": "Mister Gurney, Mister Fry, Messrs Forster, and Mister T. F. Buxton",
    "LJ011-0064": "Reverend Mister Springett",  # Test Abbreviations,
    # Test Normalized White Space
    "LJ047-0160": "found it to be four one one Elm Street. End quote.",
    "LJ017-0007": "Henry the eighth a new",  # Test Roman Numbers
    "LJ016-0257": "d'etre",  # Test Remove Accents
    "LJ018-0029": "Muller",  # Test Remove Accents
    "LJ018-0396": "celebre",  # Test Remove Accents
    "LJ020-0106": "three hours'",  # Test Quotation Normalization
    "LJ020-0002": '"sponge,"',  # Test Quotation Normalization
}


@mock.patch("run.data._loader.utils._exists", return_value=True)
@mock.patch("run.data._loader.utils.get_audio_metadata")
@mock.patch("urllib.request.urlretrieve")
def test_lj_speech_dataset(mock_urlretrieve, mock_get_audio_metadata, _):
    """ Test `run.data._loader.lj_speech_dataset` loads and verbalizes the data. """
    mock_urlretrieve.side_effect = _utils.first_parameter_url_side_effect
    mock_get_audio_metadata.side_effect = _utils.get_audio_metadata_side_effect
    archive = _utils.TEST_DATA_PATH / "datasets" / "LJSpeech-1.1.tar.bz2"

    with tempfile.TemporaryDirectory() as path:
        directory = pathlib.Path(path)
        shutil.copy(archive, directory / archive.name)
        data = run.data._loader.lj_speech_dataset(directory=directory)
        assert len(data) == 13100
        assert sum([len(r.script) for r in data]) == 1310332
        alignments = lib.utils.stow(
            [run.data._loader.Alignment((0, 151), (0.0, 0.0), (0, 151))],
            run.data._loader.alignment_dtype,
        )
        nonalignments_ = [
            run.data._loader.Alignment(script=(0, 0), audio=(0.0, 0.0), transcript=(0, 0)),
            run.data._loader.Alignment(script=(151, 151), audio=(0.0, 0.0), transcript=(151, 151)),
        ]
        nonalignments = lib.utils.stow(nonalignments_, run.data._loader.alignment_dtype)
        assert data[0] == run.data._loader.Passage(
            audio_file=_utils.make_metadata(directory / "LJSpeech-1.1/wavs/LJ001-0001.wav"),
            speaker=run.data._loader.LINDA_JOHNSON,
            script=(
                "Printing, in the only sense with which we are at present concerned, differs "
                "from most if not from all the arts and crafts represented in the Exhibition"
            ),
            transcript=(
                "Printing, in the only sense with which we are at present concerned, differs "
                "from most if not from all the arts and crafts represented in the Exhibition"
            ),
            alignments=alignments,
            nonalignments=nonalignments,
            other_metadata={
                2: (  # type: ignore
                    "Printing, in the only sense with which we are at present concerned, differs "
                    "from most if not from all the arts and crafts represented in the Exhibition"
                ),
            },
        )

        # NOTE: Test verbilization via `verbalize_test_cases`.
        _re_filename = re.compile("LJ[0-9]{3}-[0-9]{4}")
        seen = 0
        for row in data:
            basename = row.audio_file.path.name[:10]
            assert _re_filename.match(basename)
            if basename in verbalize_test_cases:
                seen += 1
                assert verbalize_test_cases[basename] in row.script
        assert seen == len(verbalize_test_cases)
