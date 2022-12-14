import lib
from run.review.dataset.dashboard._utils import _ngrams, _signal_to_db_rms_level


def test__ngrams():
    """Test `_ngrams` against basic cases."""
    _get_ngrams = lambda l, n: [l[s] for s in _ngrams(l, n)]
    assert _get_ngrams([1, 2, 3, 4, 5, 6], n=1) == [[1], [2], [3], [4], [5], [6]]
    assert _get_ngrams([1, 2, 3, 4, 5, 6], n=3) == [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]]


def test__signal_to_db_rms_level():
    """Test `_signal_to_db_rms_level` against basic cases."""
    assert _signal_to_db_rms_level(lib.audio.full_scale_sine_wave()) == -3
