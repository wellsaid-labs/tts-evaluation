from unittest import mock

import pytest

from run._utils import _find_duplicate_passages, _len_of_dups, split_dataset
from tests._utils import TEST_DATA_PATH
from tests.run._utils import make_alignments_1d, make_passage, make_speaker

TEST_DATA_PATH = TEST_DATA_PATH / "audio"
TEST_DATA_LJ = TEST_DATA_PATH / "bit(rate(lj_speech,24000),32).wav"


def test__find_duplicate_passages():
    """Test `run._utils._find_duplicate_passages` finds duplicate passages."""
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
    """Test `run._utils._find_duplicate_passages` handles no duplicates."""
    dev_scripts = set(["This is a test."])
    duplicates, rest = _find_duplicate_passages(
        dev_scripts, [make_passage(script="I'm testing this.")], 0.9
    )
    assert [p.script for p in rest] == ["I'm testing this."]
    assert len(duplicates) == 0


def test__len_of_dups():
    """Test `run._utils._len_of_dups` handles a basic case."""
    passages = [
        make_passage(script="a"),
        make_passage(script="b"),
        make_passage(script="c"),
        make_passage(script="a"),
    ]
    assert _len_of_dups((0, passages[0]), passages, 1.0) == 2
    assert _len_of_dups((1, passages[1]), passages, 1.0) == 1
    assert _len_of_dups((2, passages[2]), passages, 1.0) == 1
    assert _len_of_dups((3, passages[3]), passages, 1.0) == 1


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__deduplication(_):
    """Test `run._utils.split_dataset` handles deduplication across multiple speakers,
    so that the same passage is not in both dev and train sets.
    """
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    speaker_c = make_speaker("c")
    speaker_d = make_speaker("d")
    groups = [set([speaker_a, speaker_b, speaker_c, speaker_d])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
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
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
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
            passage("This is a test!", speaker_a),
            passage("I'm testing this.", speaker_a),
        ],
        speaker_b: [
            passage("This is a test!", speaker_b),
            passage("This is a test.", speaker_b),
        ],
        speaker_c: [passage("I'm testing this!", speaker_c)],
    }


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__order(_):
    """Test `run._utils.split_dataset` handles different dictionary orderings."""
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
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

    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
    other_train, other_dev = split_dataset(other_dataset, dev_speakers, dev_length, 0.9, groups, 3)
    assert train == other_train
    assert dev == other_dev


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__groups(_):
    """Test `run._utils.split_dataset` handles independent speakers."""
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [{speaker_a}, {speaker_b}]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
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
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("This is a test!", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test!", speaker_a)],
        speaker_b: [passage("I'm testing this.", speaker_b)],
    }


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__speaker_not_in_group(_):
    """Test `run._utils.split_dataset` raises an exception when a speaker is not in any group."""
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("Here is another test.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("A fresh sentence is for testing.", speaker_b),
        passage("Finally there is a last test.", speaker_b),
    ]
    with pytest.raises(AssertionError):
        train, dev = split_dataset(dataset, dev_speakers, dev_length, 1, groups, 3)


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__speaker_in_multiple_groups(_):
    """Test `run._utils.split_dataset` raises an exception when a speaker is in multiple groups."""
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b]), set([speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("Here is another test.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("A fresh sentence is for testing.", speaker_b),
        passage("Finally there is a last test.", speaker_b),
    ]
    with pytest.raises(AssertionError):
        train, dev = split_dataset(dataset, dev_speakers, dev_length, 1, groups, 3)


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__exact_similarity(_):
    """Test `run._utils.split_dataset` does not deduplicate loose matches when `min_sim=1`,
    so near-duplicate passages remain across the dev and train sets.

    0.903226 min_sim: "This is a test." / "This is my test."
    0.914286 min_sim: "I'm testing this." / "I'm testing these."

    Therefore, these two passage pairs are not exact-match duplicates.
    """
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("I'm testing these.", speaker_b),
        passage("This is my test.", speaker_b),
    ]
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 1, groups, 3)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("This is my test.", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test.", speaker_a)],
        speaker_b: [passage("I'm testing these.", speaker_b)],
    }


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__loose_similarity(_):
    """Test `run._utils.split_dataset` deduplicates loose matches when min_sim=0.9,
    so near-duplicate passages are grouped together in either the dev or train set,
    rather than having the model train on passages it already saw in development.

    0.903226 min_sim: "This is a test." / "This is my test."
    0.914286 min_sim: "I'm testing this." / "I'm testing these."

    Therefore, these two passage pairs are considered duplicates in this case, as they're both >0.9.
    """
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("I'm testing these.", speaker_b),
        passage("This is my test.", speaker_b),
    ]
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("I'm testing these.", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test.", speaker_a)],
        speaker_b: [passage("This is my test.", speaker_b)],
    }


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__dev_duplicates_for_dev_speakers(_):
    """Test `run._utils.split_dataset` keeps duplicates for dev speakers in the dev set.

    Since `speaker_a` and `speaker_b` are both dev speakers, keep the duplicate passages of
    "This is a test." and group them within the dev set.
    """
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a, speaker_b])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("This is a test.", speaker_b),
        passage("Here is another test.", speaker_b),
    ]
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("Here is another test.", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test.", speaker_a)],
        speaker_b: [passage("This is a test.", speaker_b)],
    }


@mock.patch("random.shuffle", return_value=None)
def test_split_dataset__discard_duplicates_for_nondev_speaker(_):
    """Test `run._utils.split_dataset` discards duplicates for a non-dev speaker.

    Since only speaker_a is a dev speaker, discard speaker_b's duplicate "This is a test."
    """
    speaker_a = make_speaker("a")
    speaker_b = make_speaker("b")
    groups = [set([speaker_a, speaker_b])]
    passage = lambda script, speaker: make_passage(
        script=script, speaker=speaker, alignments=make_alignments_1d([(0, 1)])
    )
    dev_speakers = set([speaker_a])
    dev_length = 1
    dataset = {}
    dataset[speaker_a] = [
        passage("This is a test.", speaker_a),
        passage("I'm testing this.", speaker_a),
    ]
    dataset[speaker_b] = [
        passage("This is a test.", speaker_b),
        passage("Here is another test.", speaker_b),
    ]
    train, dev = split_dataset(dataset, dev_speakers, dev_length, 0.9, groups, 3)
    assert train == {
        speaker_a: [passage("I'm testing this.", speaker_a)],
        speaker_b: [passage("Here is another test.", speaker_b)],
    }
    assert dev == {
        speaker_a: [passage("This is a test.", speaker_a)],
    }
