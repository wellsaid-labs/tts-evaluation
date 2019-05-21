from src.datasets import Speaker
from src.datasets import Gender


def test_speaker_eq():
    assert Speaker('Judy Bieber', Gender.FEMALE) == Speaker('Judy Bieber', Gender.FEMALE)


def test_speaker_neq():
    assert Speaker('Judy Bieber', Gender.MALE) != Speaker('Judy Bieber', Gender.FEMALE)
    assert Speaker('Judy', Gender.FEMALE) != Speaker('Judy Bieber', Gender.FEMALE)
    assert Speaker('Judy Bieber', Gender.FEMALE) != Speaker('judy bieber', Gender.FEMALE)
