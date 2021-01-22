from lib.datasets.utils import Alignment


def test_alignment():
    """ Test that `Alignment` behaves like a conventional `NamedTuple`. """
    item = Alignment((0, 1), (1.0, 2.0), (2, 3))

    assert item.script == (0, 1)
    assert item.audio == (1.0, 2.0)
    assert item.transcript == (2, 3)
    assert list(item) == [(0, 1), (1.0, 2.0), (2, 3)]
    assert item == Alignment(item.script, item.audio, item.transcript)
    assert item == item._replace()
    assert hash(item) == hash(item._replace())

    assert Alignment._fields == ("script", "audio", "transcript")

    other = item._replace(script=(0, 0))
    assert other.script == (0, 0)
    assert other.audio == (1.0, 2.0)
    assert other.transcript == (2, 3)
