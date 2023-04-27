from run._config import audio


def test__norm_anno_len():
    """Test `audio._norm_anno_len` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_len(0) == -1.0
    assert audio._norm_anno_len(40.5) == 0.0
    assert audio._norm_anno_len(101) == 1.0
    assert audio._norm_anno_len(81) == 1.0


def test__norm_anno_rel_tempo():
    """Test `audio._norm_anno_rel_tempo` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_rel_tempo(0.3) == -0.7
    assert audio._norm_anno_rel_tempo(1.0) == 0.0
    assert audio._norm_anno_rel_tempo(3.0) == 2.0


def test__norm_sesh_tempo():
    """Test `audio._norm_sesh_tempo` normalizes input to approx -1 to 1."""
    assert audio._norm_sesh_tempo(0.8) == -0.9999999999999998
    assert audio._norm_sesh_tempo(1.0) == 0.0
    assert audio._norm_sesh_tempo(1.2) == 0.9999999999999998


def test__norm_anno_abs_tempo():
    """Test `audio._norm_anno_abs_tempo` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_rel_tempo(0.3, 1.0) == -0.7
    assert audio._norm_anno_rel_tempo(1.0, 1.0) == 0.0
    assert audio._norm_anno_rel_tempo(3.0, 1.0) == 2.0


def test__norm_anno_rel_loudness():
    """Test `audio._norm_anno_rel_loudness` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_rel_loudness(-55) == -1.1
    assert audio._norm_anno_rel_loudness(0) == 0.0
    assert audio._norm_anno_rel_loudness(10) == 0.2


def test__norm_sesh_loudness():
    """Test `audio._norm_sesh_loudness` normalizes input to approx -1 to 1."""
    assert audio._norm_sesh_loudness(-18) == 1.0
    assert audio._norm_sesh_loudness(-23) == 0.0
    assert audio._norm_sesh_loudness(-33) == -2.0


def test__norm_anno_abs_loudness():
    """Test `audio._norm_anno_abs_loudness` normalizes input to approx -1 to 1."""
    assert audio._norm_anno_rel_loudness(-55) == -1.1
    assert audio._norm_anno_rel_loudness(0) == 0.0
    assert audio._norm_anno_rel_loudness(10) == 0.2
