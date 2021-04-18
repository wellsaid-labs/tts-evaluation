import math
import typing

import numpy
import pytest
import torch
import torch.nn
from torchnlp.random import fork_rng

import lib
from lib.utils import Timeline, TimelineMap, pad_tensor
from tests._utils import assert_almost_equal


def test_round_():
    """Test `lib.utils.round_` handles basic cases."""
    assert lib.utils.round_(0.3, 1) == 0
    assert lib.utils.round_(0.4, 0.25) == 0.5
    assert lib.utils.round_(1, 4) == 0
    assert lib.utils.round_(3, 4) == 4


def test_random_sample():
    """Test `lib.utils.random_sample` handles the basic case, an empty list, and a large
    `sample_size`."""
    with fork_rng(1234):
        assert lib.utils.random_sample([1, 2, 3, 4], 0) == []
        assert lib.utils.random_sample([1, 2, 3, 4], 2) == [4, 1]
        assert lib.utils.random_sample([1, 2, 3, 4], 5) == [1, 4, 3, 2]


def test_mean():
    """ Test `lib.utils.mean` handles empty and non-empty iterables. """
    assert lib.utils.mean([1, 2, 3]) == 2
    assert lib.utils.mean(range(3)) == 1
    assert numpy.isnan(lib.utils.mean([]))  # type: ignore


def test_get_chunks():
    assert list(lib.utils.get_chunks([1, 2, 3, 4], 3)) == [[1, 2, 3], [4]]


def test_get_weighted_std():
    """Test `lib.utils.get_weighted_std` on a basic case, and respects weighting.

    NOTE: 0.50 is equal to the population standard deviation for 1, 2
    NOTE: 0.81649658093 is equal to the population standard deviation for 1, 2, 3
    """
    tensor = torch.tensor(
        [[[0.3333333, 0.3333333, 0.3333334], [0, 0.5, 0.5]], [[0, 0.5, 0.5], [0, 0.5, 0.5]]]
    )
    assert_almost_equal(
        lib.utils.get_weighted_std(tensor, dim=2),
        torch.tensor([[0.8164966106414795, 0.50], [0.50, 0.50]]),
    )


def test_get_weighted_std__one_data_point():
    """Test `lib.utils.get_weighted_std` computes the correct standard deviation for one data
    point."""
    assert lib.utils.get_weighted_std(torch.tensor([0, 1, 0]), dim=0) == torch.zeros(1)


def test_get_weighted_std__zero_elements():
    """Test `lib.utils.get_weighted_std` handles zero elements."""
    assert lib.utils.get_weighted_std(torch.empty(1024, 0, 1024), dim=2).shape == (1024, 0)


def test_get_weighted_std__bias():
    """Test `lib.utils.get_weighted_std` computes the correct standard deviation.

    NOTE: Equal to the population standard deviation for 1, 2, 3, 4
    """
    standard_deviation = lib.utils.get_weighted_std(torch.tensor([0.25, 0.25, 0.25, 0.25]), dim=0)
    assert standard_deviation.item() == pytest.approx(1.1180339887499)


def test_get_weighted_std__error():
    """ Test `lib.utils.get_weighted_std` errors if the distribution is not normalized. """
    with pytest.raises(AssertionError):
        lib.utils.get_weighted_std(torch.tensor([0, 0.25, 0.25, 0.25]), dim=0)


def test_flatten():
    assert lib.utils.flatten([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]
    assert lib.utils.flatten([[1, [[2]], [[[3]]]], [["4"], {5: 5}]]) == [1, 2, 3, "4", {5: 5}]
    assert lib.utils.flatten([[1], [2, 3], [4, [5, [6, [7, [8]]]]]]) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert lib.utils.flatten([[[[]]], [], [[]], [[], []]]) == []


def test_flatten_2d():
    assert lib.utils.flatten_2d([[1, 2], [3, 4], [5]]) == [1, 2, 3, 4, 5]


def test_flatten_3d():
    assert lib.utils.flatten_3d([[[1], [2]], [[3], [4]], [[5]]]) == [1, 2, 3, 4, 5]


class MockModel(torch.nn.Module):
    # REFERENCE: http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)


def test_flatten_parameters():
    """ Test that `lib.utils.flatten_parameters` executes. """
    lib.utils.flatten_parameters(MockModel())
    lib.utils.flatten_parameters(torch.nn.LSTM(10, 10))


def test_identity():
    assert lib.utils.identity(2) == 2


def test_split():
    """ Test `lib.utils.split` splits once. """
    assert list(lib.utils.split([1, 2, 3, 4, 5], [4])) == [[1, 2], [3, 4, 5]]


def test_split__exact():
    """ Test `lib.utils.split` splits exactly on `[1, 2]`. """
    assert list(lib.utils.split([1, 2, 3], [3])) == [[1, 2], [3]]


def test_split__zero():
    """ Test `lib.utils.split` handles a zero split. """
    assert list(lib.utils.split([1, 2, 3], [0])) == [[], [1, 2, 3]]


def test_split__empty_split():
    """ Test `lib.utils.split` returns empty splits if threshold is not met. """
    assert list(lib.utils.split([3], [2])) == [[], [3]]
    assert list(lib.utils.split([1, 2, 3, 4, 5], [8, 3])) == [[1, 2, 3], [], [4, 5]]


def test_split__infinity():
    """ Test `lib.utils.split` handles infinite thresholds and overflow. """
    expected = [[1, 2, 3], [4, 5], []]
    assert list(lib.utils.split([1, 2, 3, 4, 5], [8, float("inf"), 3])) == expected


def test_split__no_thresholds():
    """ Test `lib.utils.split` handles no thresholds. """
    assert list(lib.utils.split([1, 2, 3, 4, 5], [])) == [[1, 2, 3, 4, 5]]


def test_log_runtime():
    """ Test `lib.utils.log_runtime` executes. """

    @lib.utils.log_runtime
    def _helper():
        pass

    _helper()


def test_log_runtime__type_hints__documentation():
    """ Test if `lib.utils.log_runtime` passes along type hints and documentation. """

    @lib.utils.log_runtime
    def _helper(arg: str):
        """ Docs """
        return arg

    assert typing.get_type_hints(_helper)["arg"] == str
    assert _helper.__doc__ == " Docs "


def test_sort_together():
    assert lib.utils.sort_together(["a", "b", "c"], [2, 3, 1]) == ["c", "a", "b"]


def test_pool():
    with lib.utils.Pool() as pool:
        assert pool.map(
            typing.cast(typing.Callable[[int], int], lib.utils.identity), [1, 2, 3]
        ) == [1, 2, 3]


def test_pad_tensor():
    """ Test `pad_tensor` for various `dim`. """
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=0).shape == (5, 4, 5)
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=-1).shape == (3, 4, 7)
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=1).shape == (3, 6, 5)


def test_pad_tensor__kwargs():
    """ Test `pad_tensor` `kwargs` are passed along. """
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=1, value=1.0).sum() == 2 * 3 * 5


def test_trim_tensors():
    """ Test `lib.utils.trim_tensors` trims a 1-d tensor. """
    a, b = lib.utils.trim_tensors(torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3]), dim=0)
    assert torch.equal(a, torch.tensor([2, 3]))
    assert torch.equal(b, torch.tensor([2, 3]))


def test_trim_tensors__3d():
    """ Test `lib.utils.trim_tensors` trims a 3-d tensor. """
    a, b = lib.utils.trim_tensors(torch.zeros(2, 4, 2), torch.zeros(2, 2, 2), dim=1)
    assert a.shape == (2, 2, 2)
    assert b.shape == (2, 2, 2)


def test_trim_tensors__uneven():
    """ Test `lib.utils.trim_tensors` raises if it needs to trim unevenly. """
    with pytest.raises(AssertionError):
        lib.utils.trim_tensors(torch.tensor([1, 2, 3]), torch.tensor([2, 3]), dim=0)


def test_lstm():
    """ Test `lib.utils.LSTM` and `torch.nn.LSTM` return the same output, given a hidden state. """
    input_ = torch.randn(5, 3, 10)
    hidden_state = (torch.randn(4, 3, 20), torch.randn(4, 3, 20))

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=True)
    output, updated_hidden_state = rnn(input_, hidden_state)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=True)
    other_output, other_updated_hidden_state = other_rnn(input_, hidden_state)

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm__hidden_state():
    """ Test `lib.utils.LSTM` uses the initial hidden state correctly. """
    input_ = torch.randn(5, 1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=True)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=True)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.initial_hidden_state, other_rnn.initial_cell_state)
    )

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm__batch_first():
    """ Test if `lib.utils.LSTM` works with the `batch_first` parameter. """
    input_ = torch.randn(1, 3, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=True, batch_first=True)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=True, batch_first=True)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.initial_hidden_state, other_rnn.initial_cell_state)
    )

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm__mono():
    """ Test if `lib.utils.LSTM` works with the `bidirectional` parameter. """
    input_ = torch.randn(5, 1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=False)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=False)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.initial_hidden_state, other_rnn.initial_cell_state)
    )

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm_cell():
    """Test `lib.utils.LSTMCell` and `torch.nn.LSTM` return the same output, given a
    hidden state."""
    input_ = torch.randn(3, 10)
    hidden_state = (torch.randn(3, 20), torch.randn(3, 20))

    with fork_rng(seed=123):
        rnn = torch.nn.LSTMCell(10, 20)
    updated_hidden_state = rnn(input_, hidden_state)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTMCell(10, 20)
    other_updated_hidden_state = other_rnn(input_, hidden_state)

    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm_cell__hidden_state():
    """ Test `lib.utils.LSTMCell` uses the initial hidden state correctly. """
    input_ = torch.randn(1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTMCell(10, 20)
    other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTMCell(10, 20)
    updated_hidden_state = rnn(
        input_, (other_rnn.initial_hidden_state, other_rnn.initial_cell_state)
    )

    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_clamp():
    """ Test `lib.utils.clamp` with basic cases. """
    assert lib.utils.clamp(3, min_=1, max_=2) == 2
    assert lib.utils.clamp(2, min_=1, max_=2) == 2
    assert lib.utils.clamp(1, min_=1, max_=2) == 1
    assert lib.utils.clamp(0, min_=1, max_=2) == 1


def test_clamp__infinity():
    """ Test `lib.utils.clamp` with infinity. """
    assert lib.utils.clamp(3, min_=1, max_=math.inf) == 3
    assert lib.utils.clamp(3, min_=-math.inf, max_=2) == 2
    assert lib.utils.clamp(0, min_=1, max_=math.inf) == 1
    assert lib.utils.clamp(0, min_=-math.inf, max_=2) == 0


def test_call_once():
    """ Test `lib.utils.call_once` only executes callable once with the same arguments. """
    count = 0

    def add_(a, b=0):
        nonlocal count
        count += 1
        return a + b

    assert lib.utils.call_once(add_, 0) == 0  # type: ignore
    assert count == 1
    assert lib.utils.call_once(add_, 0) == 0  # type: ignore
    assert count == 1
    assert lib.utils.call_once(add_, 0, 0) == 0  # type: ignore
    assert count == 2


def test_mapped_iterator():
    """ Test `MappedIterator` returns iterator items. """
    map = lib.utils.MappedIterator(iter(range(3)))
    assert map[0] == 0
    assert map[1] == 1
    assert map[2] == 2


def test_mapped_iterator__out_of_order():
    """ Test `MappedIterator` returns iterator items out of order. """
    map = lib.utils.MappedIterator(iter(range(3)))
    assert map[1] == 1

    with pytest.raises(AssertionError):
        assert map[0] == 0

    assert map[2] == 2


def test__tuple():
    """ Test `Tuple` can store and retrieve `tuple`s. """
    dtype = numpy.dtype([("f0", str, 1), ("f1", numpy.float32)])
    dtype = numpy.dtype([("f0", numpy.int32), ("f1", dtype)])
    data = [(1, ("a", 1.0)), (2, ("b", 2.0)), (3, ("c", 3.0))]
    items = lib.utils._Tuple(data, dtype=dtype)
    assert items[0] == data[0]
    assert len(items) == 3
    assert data[1] in items
    assert data[0] not in items[1:]
    assert all(d == t for d, t in zip(data, items))
    assert all(d == t for d, t in zip(data[1:2], items[1:2]))
    assert items[:] == items
    assert hash(items) == hash(items[:])
    assert str(items) == "((1, ('a', 1.0)), (2, ('b', 2.0)), (3, ('c', 3.0)))"
    assert repr(items) == "((1, ('a', 1.0)), (2, ('b', 2.0)), (3, ('c', 3.0)))"


class MockNamedTuple(typing.NamedTuple):
    string: str
    tuple: typing.Tuple[float, int]
    default: int = 1


def test__tuple__named():
    """ Test `Tuple` can store and retrieve `NamedTuple`s. """
    dtype = numpy.dtype([("f0", numpy.float32), ("f1", numpy.int32)])
    dtype = numpy.dtype([("string", str, 1), ("tuple", dtype), ("default", numpy.int32)])
    data = [
        MockNamedTuple("a", (1.0, 1)),
        MockNamedTuple("b", (2.0, 2), 2),
        MockNamedTuple("c", (3.0, 3), 3),
    ]
    items = lib.utils._Tuple(data, dtype=dtype)
    assert items[0] == data[0]
    assert items[0].tuple == data[0].tuple
    assert items[0].tuple[0] == 1.0
    assert type(items[0].tuple[0]) is float
    assert len(items) == 3
    assert data[1] in items
    assert all(d == t for d, t in zip(data, items))
    assert all(d == t for d, t in zip(data[1:2], items[1:2]))


def test__tuple__empty():
    """ Test `Tuple` can store no data. """
    items = lib.utils._Tuple([])

    with pytest.raises(IndexError):
        items[0]

    assert items[0:0] == items
    assert "test" not in items
    assert list(items) == []


def test_corrected_random_choice():
    """ Test `lib.utils.corrected_random_choice` handles basic cases. """
    distribution = {i: 0.0 for i in range(10)}
    for _ in range(10000):
        choice = lib.utils.corrected_random_choice(distribution)
        # NOTE: Every time we sample `choice`, we add `choice` creating non-uniformity.
        # `corrected_random_choice` should correct for this non-uniformity.
        distribution[choice] += choice + 1

    total = sum(distribution.values())
    for value in distribution.values():
        assert value / total == pytest.approx(1 / len(distribution), abs=0.01)


def test_corrected_random_choice__non_uniform():
    """ Test `lib.utils.corrected_random_choice` handles non-uniform distribution. """
    distribution = {i: 0.0 for i in range(10)}
    expected = {i: 1 / (i + 1) for i in range(10)}
    for _ in range(10000):
        choice = lib.utils.corrected_random_choice(distribution, expected)
        # NOTE: Every time we sample `choice`, we add `choice` creating non-uniformity.
        # `corrected_random_choice` should correct for this non-uniformity.
        distribution[choice] += choice + 1

    total = sum(distribution.values())
    total_expected = sum(expected.values())
    for value, expected in zip(distribution.values(), expected.values()):
        assert value / total == pytest.approx(expected / total_expected, abs=0.01)


def test_timeline():
    """ Test `Timeline` handles basic cases. """
    intervals = [(0, 1), (0.5, 1), (1, 2)]
    timeline = Timeline(intervals, dtype=numpy.float64)
    assert timeline.intervals() == intervals
    for i in range(len(intervals)):
        assert intervals[i][0] == timeline.start(i)
        assert intervals[i][1] == timeline.stop(i)
    assert timeline._intervals[0].data.contiguous and timeline._intervals[0].dtype == numpy.float64
    assert timeline._intervals[1].data.contiguous and timeline._intervals[1].dtype == numpy.float64
    assert timeline[0.5].tolist() == [[0, 1], [0.5, 1]]
    assert [intervals[i] for i in timeline.indicies(0.5)] == [(0, 1), (0.5, 1)]
    assert timeline.intervals(0.5) == [(0, 1), (0.5, 1)]
    assert timeline[0.5:1.5].tolist() == [[0, 1], [0.5, 1], [1, 2]]
    assert timeline[6:10].tolist() == []
    assert timeline.num_intervals() == 3


def test_timeline__zero():
    """ Test `Timeline` handles zero intervals. """
    timeline = Timeline([], dtype=numpy.float64)
    assert timeline.intervals() == []
    with pytest.raises(IndexError):
        timeline.start(0)
        timeline.stop(0)
    assert timeline[0.5].tolist() == []
    assert list(timeline.indicies(0.5)) == []
    assert timeline.intervals(0.5) == []
    assert timeline[0.5:1.5].tolist() == []
    assert timeline.num_intervals() == 0


def test_timeline_map():
    """ Test `TimelineMap` handles basic cases. """
    intervals = [
        ((3, 4), "a"),  # NOTE: Out of order
        ((0, 1), "a"),
        ((0.5, 1), "b"),  # NOTE: Overlapping
        ((1, 2), "c"),  # NOTE: Independent
    ]
    timeline: TimelineMap[str] = TimelineMap(intervals)
    assert timeline[0.5] == ("a", "b")
    assert timeline[0:1] == ("a", "b", "c")
    assert timeline[-0.5:0.5] == ("a", "b")
    assert timeline[0.5:1.5] == ("a", "b", "c")
    assert timeline[1.5:2.5] == ("c",)
    assert timeline[0:4] == ("a", "b", "c", "a")
    assert timeline[6:10] == tuple()


def test_triplets():
    """ Test `triplets`. """
    assert list(lib.utils.triplets(["a", "b", "c"])) == [
        (None, "a", "b"),
        ("a", "b", "c"),
        ("b", "c", None),
    ]
