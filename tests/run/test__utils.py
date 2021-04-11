from unittest import mock

import hparams
import pytest

import run
from run._utils import _find_duplicate_passages, split_dataset
from run.data._loader import Alignment
from tests._utils import TEST_DATA_PATH, make_passage

TEST_DATA_PATH = TEST_DATA_PATH / "audio"
TEST_DATA_LJ = TEST_DATA_PATH / "bit(rate(lj_speech,24000),32).wav"


@pytest.fixture(autouse=True)
def run_around_tests():
    """ Set a basic configuration. """
    run._config.configure()
    yield
    hparams.clear_config()


def test__find_duplicate_passages():
    """ Test `run._config._find_duplicate_passages` finds duplicate passages. """
    dev_scripts = set(["This is a test."])
    passages = [
        make_passage(script="I'm testing this."),
        make_passage(script="This is a test."),
        make_passage(script="This is a test!"),
    ]
    duplicates, rest = _find_duplicate_passages(dev_scripts, passages, 0.9)
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
    speaker_a = run.data._loader.Speaker("a")
    speaker_b = run.data._loader.Speaker("b")
    speaker_c = run.data._loader.Speaker("c")
    speaker_d = run.data._loader.Speaker("d")
    groups = [set([speaker_a, speaker_b, speaker_c, speaker_d])]
    alignments = Alignment.stow([Alignment((0, 1), (0, 1), (0, 1))])
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
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups)
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


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__order(_):
    """ Test `run._config.split_dataset` handles different dictionary orderings. """
    speaker_a = run.data._loader.Speaker("a")
    speaker_b = run.data._loader.Speaker("b")
    groups = [set([speaker_a, speaker_b])]
    alignments = Alignment.stow([Alignment((0, 1), (0, 1), (0, 1))])
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=alignments
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test!", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("I'm testing this.", speaker_b),
        passage("This is a test!", speaker_b),
    ]
    other_dataset = {}
    other_dataset[speaker_b] = dataset[speaker_b]
    other_dataset[speaker_a] = dataset[speaker_a]
    assert list(other_dataset.keys()) != list(dataset.keys())

    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups)
    other_train, other_dev = split_dataset(other_dataset, dev_speakers, dev_length, 0.9, groups)
    assert train == other_train
    assert dev == other_dev


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__groups(_):
    """ Test `run._config.split_dataset` handles independent speakers. """
    speaker_a = run.data._loader.Speaker("a")
    speaker_b = run.data._loader.Speaker("b")
    alignments = Alignment.stow([Alignment((0, 1), (0, 1), (0, 1))])
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=alignments
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test!", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("I'm testing this.", speaker_b),
        passage("This is a test!", speaker_b),
    ]
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("This is a test!", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test!", speaker_a)],
        speaker_b: [passage("I'm testing this.", speaker_b)],
    }
