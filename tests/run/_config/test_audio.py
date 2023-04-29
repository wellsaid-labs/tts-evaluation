import pytest

from run._config import audio


def test__norm_anno_len():
    """Test `audio._norm_anno_len` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_len(0) == -1.0
    assert audio._norm_anno_len(40.5) == 0.0
    assert audio._norm_anno_len(101) == 1.0
    assert audio._norm_anno_len(81) == 1.0


def test__norm_sesh_tempo():
    """Test `audio._norm_sesh_tempo` normalizes input to approx -1 to 1."""
    assert audio._norm_sesh_tempo(0.7) == -1.5000000000000002
    assert audio._norm_sesh_tempo(1.0) == 0.0
    assert audio._norm_sesh_tempo(1.2) == 0.9999999999999998


def test__norm_anno_tempo():
    """Test `audio._norm_anno_tempo` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_tempo(0.8, 1.0) == (-0.9999999999999998, 1.25)
    assert audio._norm_anno_tempo(1.0, 1.0) == (0.0, 0.0)
    assert audio._norm_anno_tempo(1.5, 1.0) == (2.5, -1.6666666666666667)
    abs_tempo, sesh_tempo = 0.8, 0.9
    rel_tempo = abs_tempo / sesh_tempo
    expected = (rel_tempo, (1 / abs_tempo) / (1 / sesh_tempo))
    assert audio._norm_anno_tempo(rel_tempo, sesh_tempo, avg_val=0, compression=1) == expected
    for i in range(1, 20):
        abs_tempo = i / 10
        a, b = audio._norm_anno_tempo(abs_tempo / sesh_tempo, sesh_tempo, avg_val=0, compression=1)
        assert a * b == pytest.approx(1)


def test__norm_sesh_loudness():
    """Test `audio._norm_sesh_loudness` normalizes input to approx -1 to 1."""
    assert audio._norm_sesh_loudness(-15) == 0.8
    assert audio._norm_sesh_loudness(-23) == 0.0
    assert audio._norm_sesh_loudness(-35) == -1.2


def test__norm_anno_loudness():
    """Test `audio._norm_anno_loudness` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_loudness(3, -23) == (0.3,)
    assert audio._norm_anno_loudness(0, -23) == (0.0,)
    assert audio._norm_anno_loudness(-4, -23) == (-0.4,)
