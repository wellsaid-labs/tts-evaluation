from collections import Counter
from unittest import mock

from torchnlp.random import fork_rng

import pytest

from src.datasets import Gender
from src.datasets import Speaker
from src.datasets.utils import _alignment_generator
from src.datasets.utils import _gcs_alignment_dataset_loader
from src.datasets.utils import Example
from src.environment import TEST_DATA_PATH


def test__alignment_generator__empty():
    """ Test if `_alignment_generator` handles an empty list.
    """
    iterator = _alignment_generator([], max_seconds=10)
    assert next(iterator, None) is None


def test__alignment_generator__one():
    """ Test if `_alignment_generator` handles a one item list.
    """
    example = Example(
        alignments=(((0, 1), (0, 1)),), script='abc', audio_path=None, speaker=None, metadata={})
    iterator = _alignment_generator([example], max_seconds=10)
    for i in range(100):
        assert len(next(iterator).alignments) == 1


def test__alignment_generator():
    """ Test if `_alignment_generator` given a uniform distribution of alignments samples uniformly.
    """
    dataset = [
        Example(
            alignments=(((0, 1), (0, 1)), ((1, 2), (1, 2)), ((2, 3), (2, 3))),
            script='abc',
            audio_path=None,
            speaker=None,
            metadata={})
    ]
    iterator = _alignment_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    for alignment in dataset[0].alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(dataset[0].alignments), abs=0.01)


@mock.patch('src.datasets.utils.subprocess.run', return_value=None)
def test__gcs_alignment_dataset_loader(_):
    """ Test if `_gcs_alignment_dataset_loader` is able to load a dataset.
    """
    speaker = Speaker('Hilary Noriega', Gender.FEMALE)
    with fork_rng(seed=123):
        train, dev = _gcs_alignment_dataset_loader(
            'hilary_noriega', speaker, [6.0], max_seconds=5.0)
        example = next(train)
        assert example == Example(
            alignments=[[[48, 53], [11.6, 11.8]], [[54, 60], [11.8, 12.4]]],
            script='For the twentieth time that evening the two men shook hands.',
            audio_path=TEST_DATA_PATH / '_disk/data/hilary_noriega/recordings/Script 1.wav',
            speaker=speaker,
            metadata={
                'Index': 2,
                'Source': 'CMU',
                'Title': 'CMU'
            })
        example = next(dev)
        assert example == Example(
            alignments=[[[35, 42], [2.5, 2.6]]],
            script='Author of the danger trail, Philip Steels, etc.',
            audio_path=TEST_DATA_PATH / '_disk/data/hilary_noriega/recordings/Script 1.wav',
            speaker=speaker,
            metadata={
                'Index': 0,
                'Source': 'CMU',
                'Title': 'CMU'
            })
