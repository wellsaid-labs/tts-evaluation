from unittest import mock

import lib
import run
from run._config import (
    Cadence,
    DatasetType,
    Label,
    _find_duplicate_passages,
    get_config_label,
    get_dataset_label,
    get_model_label,
    split_dataset,
)
from tests._utils import make_passage


def test_get_dataset_label():
    """ Test `run._config.get_dataset_label` formats a label appropriately. """
    expected = Label("static/dataset/train/test")
    assert get_dataset_label("test", Cadence.STATIC, DatasetType.TRAIN) == expected
    expected = Label("static/dataset/dev/sam_scholl/test")
    result = get_dataset_label("test", Cadence.STATIC, DatasetType.DEV, lib.datasets.SAM_SCHOLL)
    assert result == expected


def test_get_model_label():
    """ Test `run._config.get_model_label` formats a label appropriately. """
    expected = Label("static/model/test")
    assert get_model_label("test", Cadence.STATIC) == expected
    expected = Label("static/model/sam_scholl/test")
    assert get_model_label("test", Cadence.STATIC, lib.datasets.SAM_SCHOLL) == expected


def test_get_config_label():
    """ Test `run._config.get_config_label` formats a label appropriately. """
    expected = Label("static/config/test")
    assert get_config_label("test", Cadence.STATIC) == expected


def test_configure_audio_processing():
    """ Test `run._config.configure_audio_processing` finds and configures modules. """
    run._config.configure_audio_processing()


def test_configure_models():
    """ Test `run._config.configure_models` finds and configures modules. """
    run._config.configure_models()


def test_configure():
    """ Test `run._config.configure` finds and configures modules. """
    run._config.configure()


def test__find_duplicate_passages():
    """ Test `run._config._find_duplicate_passages` finds duplicate passages. """
    dev_scripts = set(["This is a test."])
    duplicates, rest = _find_duplicate_passages(
        dev_scripts,
        [
            make_passage(script="I'm testing this."),
            make_passage(script="This is a test."),
            make_passage(script="This is a test!"),
        ],
        0.9,
    )
    assert [p.script for p in rest] == ["I'm testing this."]
    assert [p.script for p in duplicates] == ["This is a test.", "This is a test!"]


def test__find_duplicate_passages__no_duplicates():
    """ Test `run._config._find_duplicate_passages` handles no duplicates. """
    dev_scripts = set(["This is a test."])
    duplicates, rest = _find_duplicate_passages(
        dev_scripts, [make_passage(script="I'm testing this.")], 0.9
    )
    assert [p.script for p in rest] == ["I'm testing this."]
    assert len(duplicates) == 0


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__deduplication(_):
    """ Test `run._config.split_dataset` handles deduplication accross multiple speakers. """
    speaker_a = lib.datasets.Speaker("a")
    speaker_b = lib.datasets.Speaker("b")
    speaker_c = lib.datasets.Speaker("c")
    speaker_d = lib.datasets.Speaker("d")
    alignments = lib.utils.Tuples(
        [lib.datasets.Alignment((0, 1), (0, 1), (0, 1))], lib.datasets.alignment_dtype
    )
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=alignments
    )
    dev_speakers = set([speaker_a, speaker_b, speaker_c])
    dev_length = 1
    dataset = {
        speaker_a: [
            passage("This is a test!", speaker_a),  # NOTE: Dev set
            passage("I'm testing this.", speaker_a),  # NOTE: Initially, train set
            passage("More training data testing!", speaker_a),  # NOTE: Train set
            passage("Data for testing training.", speaker_a),  # NOTE: Train set
        ],
        speaker_b: [
            passage("This is a test!", speaker_b),  # NOTE: Duplicate `speaker_a` dev passage
            passage("This is a test.", speaker_b),  # NOTE: Duplicate `speaker_a` dev passage
            passage("Completely different test.", speaker_b),  # NOTE: Train set
            passage("Data for testing training.", speaker_a),  # NOTE: Train set
        ],
        speaker_c: [
            passage("I'm testing this!", speaker_c),  # NOTE: Duplicate `speaker_a` train passage
            passage("Another test.", speaker_c),  # NOTE: Train set
        ],
        speaker_d: [
            passage("Outlier test!", speaker_d),  # NOTE: Train set
            passage("This is a test!", speaker_a),  # NOTE: Non-dev set duplicate passage
        ],
    }
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9)
    assert train == {
        speaker_a: [
            passage("More training data testing!", speaker_a),
            passage("Data for testing training.", speaker_a),
        ],
        speaker_b: [
            passage("Completely different test.", speaker_b),
            passage("Data for testing training.", speaker_a),
        ],
        speaker_c: [passage("Another test.", speaker_c)],
        speaker_d: [passage("Outlier test!", speaker_d)],
    }
    assert dev == {
        speaker_a: [
            passage("I'm testing this.", speaker_a),
            passage("This is a test!", speaker_a),
        ],
        speaker_b: [
            passage("This is a test!", speaker_b),
            passage("This is a test.", speaker_b),
        ],
        speaker_c: [passage("I'm testing this!", speaker_c)],
    }
