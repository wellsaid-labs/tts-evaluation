from collections import Counter
from unittest import mock

import pathlib

import pytest

from tests import _utils

import lib


def _get_example(alignments=(),
                 audio_path=pathlib.Path('.'),
                 speaker=lib.datasets.Speaker(''),
                 text='',
                 metadata={}):
    """ Get `lib.datasets.Example` for testing. """
    alignments = tuple([lib.datasets.Alignment(a, a) for a in alignments])
    return lib.datasets.Example(audio_path, speaker, alignments, text, metadata)


def test_dataset_generator():
    """ Test `lib.datasets.dataset_generator` samples uniformly given a uniform distribution of
    alignments.
    """
    dataset = [_get_example(((0, 1), (1, 2), (2, 3)))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    for alignment in dataset[0].alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(dataset[0].alignments), abs=0.01)


def test_dataset_generator__empty():
    """ Test `lib.datasets.dataset_generator` handles an empty list. """
    iterator = lib.datasets.dataset_generator([], max_seconds=10)
    assert next(iterator, None) is None


def test_dataset_generator__singular():
    """ Test `lib.datasets.dataset_generator` handles multiple examples with a singular alignment
    of varying lengths. """
    dataset = [_get_example(((0, 1),)), _get_example(((0, 10),)), _get_example(((0, 5),))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = lib.utils.flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


def test_dataset_generator__multiple_multiple():
    """ Test `lib.datasets.dataset_generator` handles multiple scripts with a uniform alignment
    distribution. """
    dataset = [_get_example(((0, 1), (1, 2), (2, 3))), _get_example(((3, 4), (4, 5), (5, 6)))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=10)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = lib.utils.flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


def test_dataset_generator__pause():
    """ Test `lib.datasets.dataset_generator` samples uniformly despite a large pause. """
    dataset = [_get_example(((0, 1), (1, 2), (2, 3), (20, 21), (40, 41)))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=4)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)

    total = sum(counter.values())
    for alignment in dataset[0].alignments:
        assert counter[alignment] / total == (
            pytest.approx(1 / len(dataset[0].alignments), abs=0.02))


def test_dataset_generator__multiple_unequal_examples__large_max_seconds():
    """ Test `lib.datasets.dataset_generator` samples uniformly despite unequal example sizes,
    and large max seconds. """
    dataset = [_get_example(((0, 1),)), _get_example(((3, 4), (4, 5), (5, 6)))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=1000000)

    alignments_counter = Counter()
    spans_counter = Counter()
    num_examples = 10000
    for i in range(num_examples):
        example = next(iterator)
        alignments_counter.update(example.alignments)
        spans_counter[example.alignments] += 1

    total_alignments = sum(alignments_counter.values())
    unique_alignments = lib.utils.flatten([d.alignments for d in dataset])

    for alignment in unique_alignments:
        assert alignments_counter[alignment] / total_alignments == pytest.approx(
            1 / len(unique_alignments), abs=0.01)

    for example in dataset:
        assert spans_counter[example.alignments] / num_examples == pytest.approx(
            1 / len(dataset), abs=0.01)


def test_dataset_generator__unequal_alignment_sizes():
    """ Test `lib.datasets.dataset_generator` samples uniformly despite unequal alignment sizes. """
    dataset = [_get_example(((0, 1), (1, 5), (5, 20)))]
    iterator = lib.datasets.dataset_generator(dataset, max_seconds=20)
    counter = Counter()
    for i in range(10000):
        counter.update(next(iterator).alignments)
    total = sum(counter.values())
    alignments = lib.utils.flatten([d.alignments for d in dataset])
    for alignment in alignments:
        assert counter[alignment] / total == pytest.approx(1 / len(alignments), abs=0.01)


@mock.patch('lib.datasets.utils.subprocess.run', return_value=None)
def test_dataset_loader(_):
    """ Test `lib.datasets.dataset_loader` loads a dataset. """
    examples = lib.datasets.dataset_loader(
        'hilary_noriega',
        '',
        lib.datasets.HILARY_NORIEGA,
        _utils.TEST_DATA_PATH / '_disk' / 'data',
        max_seconds=5.0)
    alignments = tuple([
        lib.datasets.Alignment(tuple(a[0]), tuple(a[1]))
        for a in [[[0, 6], [0.0, 0.6]], [[7, 9], [0.6, 0.8]], [[10, 13], [0.8, 0.8]],
                  [[14, 20], [0.8, 1.4]], [[21, 27], [1.4, 1.8]], [[28, 34], [1.8, 2.5]],
                  [[35, 42], [2.5, 2.6]], [[43, 47], [2.6, 3.3]]]
    ])
    assert examples[0] == lib.datasets.Example(
        alignments=alignments,
        text='Author of the danger trail, Philip Steels, etc.',
        audio_path=_utils.TEST_DATA_PATH / '_disk/data/hilary_noriega/recordings/Script 1.wav',
        speaker=lib.datasets.HILARY_NORIEGA,
        metadata={
            'Index': 0,
            'Source': 'CMU',
            'Title': 'CMU'
        })


# NOTE: `lib.datasets.precut_dataset_loader` is tested in `lj_speech` as part of loading the
# dataset.
