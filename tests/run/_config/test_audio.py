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
