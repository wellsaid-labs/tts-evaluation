from collections import Counter
from unittest import mock

from torchnlp.random import fork_rng

import pytest

from src.datasets import Gender
from src.datasets import HILARY_NORIEGA
from src.datasets import Speaker
from src.datasets.utils import dataset_generator
from src.datasets.utils import dataset_loader
from src.datasets.utils import Example
from src.environment import TEST_DATA_PATH
from src.utils import flatten


def test_dataset_generator():
    """ Test if `dataset_generator` given a uniform distribution of alignments samples uniformly.
    """
    dataset = [Example((((0, 1), (0, 1)), ((1, 2), (1, 2)), ((2, 3), (2, 3))))]
    iterator = dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    for alignment in dataset[0].alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(dataset[0].alignments), abs=0.01)


def test_dataset_generator__empty():
    """ Test if `dataset_generator` handles an empty list.
    """
    iterator = dataset_generator([], max_seconds=10)
    assert next(iterator, None) is None


def test_dataset_generator__one():
    """ Test if `dataset_generator` handles a one alignment.
    """
    dataset = [
        Example((((0, 1), (0, 1)),)),
        Example((((0, 10), (0, 10)),)),
        Example((((0, 5), (0, 5)),))
    ]
    iterator = dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


def test_dataset_generator__multiple_scripts():
    """ Test if `dataset_generator` given a uniform distribution of alignments samples uniformly.
    """
    dataset = [
        Example((((0, 1), (0, 1)), ((1, 2), (1, 2)), ((2, 3), (2, 3)))),
        Example((((3, 4), (3, 4)), ((4, 5), (4, 5)), ((5, 6), (5, 6))))
    ]
    iterator = dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


def test_dataset_generator__large_pause():
    """ Test if `dataset_generator` samples spans uniformly despite the large pause.
    """
    dataset = [
        Example((((0, 1), (0, 1)), ((1, 2), (1, 2)), ((2, 3), (2, 3)), ((20, 21), (20, 21)),
                 ((40, 41), (40, 41))))
    ]
    iterator = dataset_generator(dataset, max_seconds=4)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)

    total = sum(counter.values())
    for alignment in dataset[0].alignments:
        assert counter[alignment] / total == (
            pytest.approx(1 / len(dataset[0].alignments), abs=0.02))


def test_dataset_generator__multiple_unequal_scripts__large_max_seconds():
    """ Test if `dataset_generator` given multiple scripts with different sizes and a large span
    it samples uniformly.
    """
    dataset = [
        Example((((0, 1), (0, 1)),)),
        Example((((3, 4), (3, 4)), ((4, 5), (4, 5)), ((5, 6), (5, 6))))
    ]
    iterator = dataset_generator(dataset, max_seconds=1000)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


def test_dataset_generator__unequal_alignment_size__small_span():
    """ Test if `dataset_generator` given unequal alignments, it samples the timeline uniformly.
    """
    dataset = [Example((((0, 1), (0, 1)), ((1, 5), (1, 5)), ((5, 20), (5, 20))))]
    iterator = dataset_generator(dataset, max_seconds=20)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


@mock.patch('src.datasets.utils.subprocess.run', return_value=None)
def test_dataset_loader(_):
    """ Test if `dataset_loader` is able to load a dataset.
    """
    examples = dataset_loader('hilary_noriega', '', HILARY_NORIEGA, max_seconds=5.0)
    assert examples[0] == Example(
        alignments=[[[0, 6], [0.0, 0.6]], [[7, 9], [0.6, 0.8]], [[10, 13], [0.8, 0.8]],
                    [[14, 20], [0.8, 1.4]], [[21, 27], [1.4, 1.8]], [[28, 34], [1.8, 2.5]],
                    [[35, 42], [2.5, 2.6]], [[43, 47], [2.6, 3.3]]],
        text='Author of the danger trail, Philip Steels, etc.',
        audio_path=TEST_DATA_PATH / '_disk/data/hilary_noriega/recordings/Script 1.wav',
        speaker=HILARY_NORIEGA,
        metadata={
            'Index': 0,
            'Source': 'CMU',
            'Title': 'CMU'
        })
