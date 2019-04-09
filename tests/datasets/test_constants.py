from src.datasets import Speaker


def test_speaker_length():
    assert len(Speaker) > 0


def test_speaker__smoke_screen():
    int(Speaker.JUDY_BIEBER)
    hash(Speaker.JUDY_BIEBER)
    repr(Speaker.JUDY_BIEBER)


def test_speaker_eq():
    assert Speaker.JUDY_BIEBER == Speaker.JUDY_BIEBER
