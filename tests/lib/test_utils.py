import math
import pathlib
import tempfile
import typing
from collections import defaultdict

import numpy
import pytest
import torch
import torch.distributed
import torch.nn
from torchnlp.random import fork_rng

import lib
from lib.utils import Timeline, TimelineMap, lengths_to_mask, offset_slices, pad_tensor
from tests._utils import assert_almost_equal


def test_round_():
    """Test `lib.utils.round_` handles basic cases."""
    assert lib.utils.round_(0.3, 1) == 0
    assert lib.utils.round_(0.4, 0.25) == 0.5
    assert lib.utils.round_(1, 4) == 0
    assert lib.utils.round_(3, 4) == 4
    # NOTE: Without additional rounding, this regressed to: 1.1500000000000001.
    assert lib.utils.round_(1.17, 0.05) == 1.15
    # NOTE: `round` should handle `inf` and `nan` like `round`.
    assert lib.utils.round_(math.inf, 0.1) == round(math.inf, 1)
    assert math.isnan(lib.utils.round_(math.nan, 0.1)) == math.isnan(round(math.nan, 1))


def test_random_sample():
    """Test `lib.utils.random_sample` handles the basic case, an empty list, and a large
    `sample_size`."""
    with fork_rng(1234):
        assert lib.utils.random_sample([1, 2, 3, 4], 0) == []
        assert lib.utils.random_sample([1, 2, 3, 4], 2) == [4, 1]
        assert lib.utils.random_sample([1, 2, 3, 4], 5) == [1, 4, 3, 2]


def test_random_nonoverlapping_intervals():
    """Test `lib.utils.random_nonoverlapping_intervals` handles the basic case(s)."""
    assert lib.utils.random_nonoverlapping_intervals(2, 1) == ((0, 1),)
    assert lib.utils.random_nonoverlapping_intervals(2, 0) == tuple()


def _get_distribution(**kwargs):
    with fork_rng(1234):
        total_intervals = 0
        total_interval_length = 0
        no_intervals = 0
        distribution = defaultdict(int)
        num_bounds = 30
        buckets = [0] * (num_bounds - 1)
        passes = 1000
        for _ in range(passes):
            intervals = lib.utils.random_nonoverlapping_intervals(num_bounds, **kwargs)
            no_intervals += len(intervals) == 0
            total_intervals += len(intervals)
            total_interval_length += sum(b - a for a, b in intervals)
            for a, b in intervals:
                distribution[b - a] += 1
                assert b - a > 0
                for i in range(a, b):
                    buckets[i] += 1
    return buckets, distribution, no_intervals, passes, total_interval_length, total_intervals


def test_random_nonoverlapping_intervals__distribution():
    """Test `lib.utils.random_nonoverlapping_intervals` has the correct distribution."""
    (
        buckets,
        distribution,
        no_intervals,
        passes,
        total_interval_length,
        total_intervals,
    ) = _get_distribution(avg_intervals=3)
    assert no_intervals == 27  # NOTE: Only 3% of the time there are no annotations
    # NOTE: Like expected, there are 3 annotations on average
    assert total_intervals / passes == 2.957
    assert total_interval_length / passes == 9.637
    # NOTE: Around 86% of our annotations would be 1 to 5 units long. There is certainly a
    # bias toward shorter segments.
    assert distribution == {
        1: 1526,
        2: 498,
        3: 257,
        4: 155,
        5: 109,
        6: 61,
        7: 58,
        8: 50,
        9: 30,
        10: 29,
        11: 17,
        12: 23,
        13: 16,
        14: 12,
        15: 8,
        16: 12,
        17: 13,
        18: 8,
        19: 9,
        20: 3,
        21: 5,
        22: 2,
        23: 1,
        24: 5,
        25: 7,
        26: 2,
        27: 3,
        28: 2,
        29: 36,
    }
    # NOTE: There is an equal probability that each bucket of data is found inside an interval.
    assert buckets == [
        318,
        330,
        332,
        332,
        333,
        347,
        319,
        333,
        327,
        337,
        345,
        341,
        336,
        327,
        331,
        325,
        324,
        319,
        337,
        344,
        323,
        324,
        340,
        353,
        335,
        331,
        330,
        329,
        335,
    ]


def test_mean():
    """Test `lib.utils.mean` handles empty and non-empty iterables."""
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
    expected = torch.tensor([[0.8164966106414795, 0.50], [0.50, 0.50]])
    assert_almost_equal(lib.utils.get_weighted_std(tensor, dim=2), expected)


def test_get_weighted_std__one_data_point():
    """Test `lib.utils.get_weighted_std` computes the correct standard deviation for one data
    point."""
    assert lib.utils.get_weighted_std(torch.tensor([0, 1, 0]), dim=0) == torch.zeros(1)


def test_get_weighted_std__no_data_points():
    """Test `lib.utils.get_weighted_std` returns NaN if zeros are passed."""
    assert torch.isnan(lib.utils.get_weighted_std(torch.tensor([0, 0, 0]), dim=0))


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
    """Test `lib.utils.get_weighted_std` errors if the distribution is not normalized."""
    with pytest.raises(AssertionError):
        lib.utils.get_weighted_std(torch.tensor([0, 0.25, 0.25, 0.25]), dim=0, strict=True)


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
    """Test that `lib.utils.flatten_parameters` executes."""
    lib.utils.flatten_parameters(MockModel())
    lib.utils.flatten_parameters(torch.nn.LSTM(10, 10))


def test_identity():
    assert lib.utils.identity(2) == 2


def test_split():
    """Test `lib.utils.split` splits once."""
    assert list(lib.utils.split([1, 2, 3, 4, 5], [4])) == [[1, 2], [3, 4, 5]]


def test_split__exact():
    """Test `lib.utils.split` splits exactly on `[1, 2]`."""
    assert list(lib.utils.split([1, 2, 3], [3])) == [[1, 2], [3]]


def test_split__zero():
    """Test `lib.utils.split` handles a zero split."""
    assert list(lib.utils.split([1, 2, 3], [0])) == [[], [1, 2, 3]]


def test_split__empty_split():
    """Test `lib.utils.split` returns empty splits if threshold is not met."""
    assert list(lib.utils.split([3], [2])) == [[], [3]]
    assert list(lib.utils.split([1, 2, 3, 4, 5], [8, 3])) == [[1, 2, 3], [], [4, 5]]


def test_split__infinity():
    """Test `lib.utils.split` handles infinite thresholds and overflow."""
    expected = [[1, 2, 3], [4, 5], []]
    assert list(lib.utils.split([1, 2, 3, 4, 5], [8, float("inf"), 3])) == expected


def test_split__multiple_infinities():
    """Test `lib.utils.split` handles multiple infinite thresholds and overflow."""
    expected = [[1, 2, 3], [4, 5], [], []]
    assert list(lib.utils.split([1, 2, 3, 4, 5], [8, float("inf"), float("inf"), 3])) == expected


def test_split__no_thresholds():
    """Test `lib.utils.split` handles no thresholds."""
    assert list(lib.utils.split([1, 2, 3, 4, 5], [])) == [[1, 2, 3, 4, 5]]


def test_log_runtime():
    """Test `lib.utils.log_runtime` executes."""

    @lib.utils.log_runtime
    def _helper():
        pass

    _helper()


def test_log_runtime__type_hints__documentation():
    """Test if `lib.utils.log_runtime` passes along type hints and documentation."""

    @lib.utils.log_runtime
    def _helper(arg: str):
        """Docs"""
        return arg

    assert typing.get_type_hints(_helper)["arg"] == str
    assert _helper.__doc__ == "Docs"


def test_disk_cache():
    """Test is `lib.utils.disk_cache` caches the return values regardless of the arguments."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name) / "cache.pickle"
    assert not temp_dir_path.exists()
    wrapped = lib.utils.disk_cache(temp_dir_path)(lib.utils.identity)
    assert wrapped(1) == 1
    assert temp_dir_path.exists()
    assert wrapped(3) == 1


def test_disk_cache__clear_cache():
    """Test is `lib.utils.disk_cache` can clear cache."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = pathlib.Path(temp_dir.name) / "cache.pickle"
    assert not temp_dir_path.exists()
    wrapped = lib.utils.disk_cache(temp_dir_path)(lib.utils.identity)
    assert wrapped(1) == 1
    assert temp_dir_path.exists()
    wrapped.clear_cache()
    assert not temp_dir_path.exists()
    assert wrapped(3) == 3


def test_sort_together():
    assert lib.utils.sort_together(["a", "b", "c"], [2, 3, 1]) == ["c", "a", "b"]


def test_pool():
    with lib.utils.Pool() as pool:
        assert pool.map(
            typing.cast(typing.Callable[[int], int], lib.utils.identity), [1, 2, 3]
        ) == [1, 2, 3]


def test_pad_tensor():
    """Test `pad_tensor` for various `dim`."""
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=0).shape == (5, 4, 5)
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=-1).shape == (3, 4, 7)
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=1).shape == (3, 6, 5)


def test_pad_tensor__kwargs():
    """Test `pad_tensor` `kwargs` are passed along."""
    assert pad_tensor(torch.zeros(3, 4, 5), pad=(1, 1), dim=1, value=1.0).sum() == 2 * 3 * 5


def test_trim_tensors():
    """Test `lib.utils.trim_tensors` trims a 1-d tensor."""
    a, b = lib.utils.trim_tensors(torch.tensor([1, 2, 3, 4]), torch.tensor([2, 3]), dim=0)
    assert torch.equal(a, torch.tensor([2, 3]))
    assert torch.equal(b, torch.tensor([2, 3]))


def test_trim_tensors__3d():
    """Test `lib.utils.trim_tensors` trims a 3-d tensor."""
    a, b = lib.utils.trim_tensors(torch.zeros(2, 4, 2), torch.zeros(2, 2, 2), dim=1)
    assert a.shape == (2, 2, 2)
    assert b.shape == (2, 2, 2)


def test_trim_tensors__uneven():
    """Test `lib.utils.trim_tensors` raises if it needs to trim unevenly."""
    with pytest.raises(AssertionError):
        lib.utils.trim_tensors(torch.tensor([1, 2, 3]), torch.tensor([2, 3]), dim=0)


def test_lstm():
    """Test `lib.utils.LSTM` and `torch.nn.LSTM` return the same output, given a hidden state."""
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
    """Test `lib.utils.LSTM` uses the initial hidden state correctly."""
    input_ = torch.randn(5, 1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=True)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=True)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.init_hidden_state, other_rnn.init_cell_state)
    )

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm__batch_first():
    """Test if `lib.utils.LSTM` works with the `batch_first` parameter."""
    input_ = torch.randn(1, 3, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=True, batch_first=True)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=True, batch_first=True)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.init_hidden_state, other_rnn.init_cell_state)
    )

    assert_almost_equal(output, other_output)
    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_lstm__mono():
    """Test if `lib.utils.LSTM` works with the `bidirectional` parameter."""
    input_ = torch.randn(5, 1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTM(10, 20, 2, bidirectional=False)
    other_output, other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTM(10, 20, 2, bidirectional=False)
    output, updated_hidden_state = rnn(
        input_, (other_rnn.init_hidden_state, other_rnn.init_cell_state)
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
    """Test `lib.utils.LSTMCell` uses the initial hidden state correctly."""
    input_ = torch.randn(1, 10)

    with fork_rng(seed=123):
        other_rnn = lib.utils.LSTMCell(10, 20)
    other_updated_hidden_state = other_rnn(input_)

    with fork_rng(seed=123):
        rnn = torch.nn.LSTMCell(10, 20)
    updated_hidden_state = rnn(input_, (other_rnn.init_hidden_state, other_rnn.init_cell_state))

    assert_almost_equal(updated_hidden_state[0], other_updated_hidden_state[0])
    assert_almost_equal(updated_hidden_state[1], other_updated_hidden_state[1])


def test_clamp():
    """Test `lib.utils.clamp` with basic cases."""
    assert lib.utils.clamp(3, min_=1, max_=2) == 2
    assert lib.utils.clamp(2, min_=1, max_=2) == 2
    assert lib.utils.clamp(1, min_=1, max_=2) == 1
    assert lib.utils.clamp(0, min_=1, max_=2) == 1


def test_clamp__infinity():
    """Test `lib.utils.clamp` with infinity."""
    assert lib.utils.clamp(3, min_=1, max_=math.inf) == 3
    assert lib.utils.clamp(3, min_=-math.inf, max_=2) == 2
    assert lib.utils.clamp(0, min_=1, max_=math.inf) == 1
    assert lib.utils.clamp(0, min_=-math.inf, max_=2) == 0


def test_call_once():
    """Test `lib.utils.call_once` only executes callable once with the same arguments."""
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
    """Test `MappedIterator` returns iterator items."""
    map = lib.utils.MappedIterator(iter(range(3)))
    assert map[0] == 0
    assert map[1] == 1
    assert map[2] == 2


def test_mapped_iterator__out_of_order():
    """Test `MappedIterator` returns iterator items out of order."""
    map = lib.utils.MappedIterator(iter(range(3)))
    assert map[1] == 1

    with pytest.raises(AssertionError):
        assert map[0] == 0

    assert map[2] == 2


def test__tuple():
    """Test `Tuple` can store and retrieve `tuple`s."""
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
    """Test `Tuple` can store and retrieve `NamedTuple`s."""
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
    """Test `Tuple` can store no data."""
    items = lib.utils._Tuple([])

    with pytest.raises(IndexError):
        items[0]

    assert items[0:0] == items
    assert "test" not in items
    assert list(items) == []


def test_corrected_random_choice():
    """Test `lib.utils.corrected_random_choice` handles basic cases."""
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
    """Test `lib.utils.corrected_random_choice` handles non-uniform distribution."""
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
    """Test `Timeline` handles basic cases."""
    intervals = [(0, 1), (0.5, 1), (1, 2)]
    timeline = Timeline(intervals, dtype=numpy.float64)
    assert timeline.intervals() == intervals
    for i in range(len(intervals)):
        assert intervals[i][0] == timeline.start(i)
        assert intervals[i][1] == timeline.stop(i)
    assert timeline._intervals[0].data.contiguous and timeline._intervals[0].dtype == numpy.float64
    assert timeline._intervals[1].data.contiguous and timeline._intervals[1].dtype == numpy.float64
    assert timeline[0.5].tolist() == [[0, 1], [0.5, 1]]
    assert [intervals[i] for i in timeline.indices(0.5)] == [(0, 1), (0.5, 1)]
    assert timeline.intervals(0.5) == [(0, 1), (0.5, 1)]
    assert timeline[0.5:1.5].tolist() == [[0, 1], [0.5, 1], [1, 2]]
    assert timeline[6:10].tolist() == []
    assert timeline.num_intervals() == 3


def test_timeline__zero():
    """Test `Timeline` handles zero intervals."""
    timeline = Timeline([], dtype=numpy.float64)
    assert timeline.intervals() == []
    with pytest.raises(IndexError):
        timeline.start(0)
        timeline.stop(0)
    assert timeline[0.5].tolist() == []
    assert list(timeline.indices(0.5)) == []
    assert timeline.intervals(0.5) == []
    assert timeline[0.5:1.5].tolist() == []
    assert timeline.num_intervals() == 0


def test_timeline_map():
    """Test `TimelineMap` handles basic cases."""
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
    """Test `triplets`."""
    assert list(lib.utils.triplets(["a", "b", "c"])) == [
        (None, "a", "b"),
        ("a", "b", "c"),
        ("b", "c", None),
    ]


def test_lengths_to_mask():
    """Test `lengths_to_mask` with a variety of shapes."""
    # Test tensors with various shapes
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    assert torch.equal(lengths_to_mask([1, 2, 3]), expected)
    assert torch.equal(lengths_to_mask(torch.tensor([1, 2, 3])), expected)
    assert torch.equal(lengths_to_mask(torch.tensor([[1, 2, 3]])), expected)

    # Test scalars with various shapes
    assert torch.equal(lengths_to_mask(1), torch.tensor([[True]]))
    assert torch.equal(lengths_to_mask(torch.tensor(1)), torch.tensor([[True]]))
    assert torch.equal(lengths_to_mask(torch.tensor([1])), torch.tensor([[True]]))

    # Test empty tensors with various shapes
    assert torch.equal(lengths_to_mask([]), torch.empty(0, 0, dtype=torch.bool))
    assert torch.equal(lengths_to_mask(torch.tensor([])), torch.empty(0, 0, dtype=torch.bool))
    assert torch.equal(lengths_to_mask(torch.tensor([[]])), torch.empty(0, 0, dtype=torch.bool))


def test_offset_slices():
    """Test `offset_slices` updates `slices` based on a list of updates."""
    assert offset_slices([slice(1, 3)], []) == [slice(1, 3)]
    assert offset_slices([slice(1, 3)], [(slice(0, 1), 0)]) == [slice(0, 2)]
    assert offset_slices([slice(1, 3)], [(slice(1, 2), 0)]) == [slice(1, 2)]
    assert offset_slices([slice(1, 3)], [(slice(1, 3), 0)]) == [slice(1, 1)]
    assert offset_slices([slice(1, 3)], [(slice(1, 3), 5)]) == [slice(1, 6)]
    assert offset_slices([slice(1, 3)], [(slice(2, 3), 0)]) == [slice(1, 2)]
    assert offset_slices([slice(1, 3)], [(slice(3, 4), 0)]) == [slice(1, 3)]

    expected = [slice(0, 1), slice(1, 2), slice(3, 4)]
    assert offset_slices([slice(0, 1), slice(1, 3), slice(4, 5)], [(slice(1, 2), 0)]) == expected
    updates = [(slice(0, 1), 0), (slice(1, 2), 0)]
    assert offset_slices([slice(0, 1), slice(1, 2)], updates) == [slice(0, 0), slice(0, 0)]
    updates = [(slice(0, 1), 2), (slice(1, 2), 2)]
    assert offset_slices([slice(0, 1), slice(1, 2)], updates) == [slice(0, 2), slice(2, 4)]


def test_arange():
    """Test `lib.utils.arange` is similar to `range`."""
    assert list(range(0, 10, 1)) == list(lib.utils.arange(0, 10, 1))
    assert list(range(0, -10, -1)) == list(lib.utils.arange(0, -10, -1))
    assert list(range(0, -10, 1)) == list(lib.utils.arange(0, -10, 1))
    assert list(range(0, 10, -1)) == list(lib.utils.arange(0, 10, -1))
    result = [round(i, 1) for i in lib.utils.arange(0, 1, 0.1)]
    assert result == [round(i / 10, 1) for i in range(0, 10, 1)]
    result = [round(i, 1) for i in lib.utils.arange(0, -1, -0.1)]
    assert result == [round(i / 10, 1) for i in range(0, -10, -1)]


def test_zip_strict():
    """Test `lib.utils.zip_strict` is similar to `zip`."""
    assert list(lib.utils.zip_strict((1, 2, 3), (1, 2, 3))) == [(1, 1), (2, 2), (3, 3)]
    with pytest.raises(AssertionError):
        assert list(lib.utils.zip_strict((1, 2, 3), (1, 2))) == [(1, 1), (2, 2), (3, 3)]
    with pytest.raises(AssertionError):
        assert list(lib.utils.zip_strict((1, 2), (1, 2, 3))) == [(1, 1), (2, 2), (3, 3)]


def test_slice_seq():
    """Test `lib.utils.slice_seq` on basic cases."""
    slices = [(slice(0, 1), 1.0), (slice(3, 5), 3.0)]
    result = lib.utils.slice_seq(slices, 5)
    expected = torch.tensor([1.0, 0.0, 0.0, 3.0, 3.0])
    assert torch.equal(result, expected)

    with pytest.raises(AssertionError):  # Error on overlap
        lib.utils.slice_seq([(slice(0, 1), 1.0), (slice(0, 1), 3.0)], 5)

    with pytest.raises(AssertionError):  # Error on funky step size
        lib.utils.slice_seq([(slice(0, 1, 2), 1.0)], 5)

    with pytest.raises(AssertionError):  # Error on if not sorted
        lib.utils.slice_seq([(slice(2, 3), 1.0), (slice(1, 2), 1.0)], 5)

    with pytest.raises(AssertionError):  # Error length too small
        lib.utils.slice_seq([(slice(0, 7), 1.0)], 5)
