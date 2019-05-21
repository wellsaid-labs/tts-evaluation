import pytest

from src.datasets import Speaker
from src.datasets import Gender
from src.spectrogram_model import InputEncoder


def test_input_encoder():
    encoder = InputEncoder(
        ['a', 'b', 'c'],
        [Speaker('Judy Bieber', Gender.FEMALE),
         Speaker('Mary Ann', Gender.FEMALE)])
    input_ = ('a', Speaker('Judy Bieber', Gender.FEMALE))
    encoded = encoder.batch_encode([input_])[0]
    assert encoder.decode(encoded) == input_


def test_input_encoder__reversible():
    encoder = InputEncoder(
        ['a', 'b', 'c'],
        [Speaker('Judy Bieber', Gender.FEMALE),
         Speaker('Mary Ann', Gender.FEMALE)])

    with pytest.raises(ValueError):  # Text is not reversible
        encoder.encode(('d', Speaker('Judy Bieber', Gender.FEMALE)))

    with pytest.raises(ValueError):  # Speaker is not reversible
        encoder.encode(('a', Speaker('Hilary Noriega', Gender.FEMALE)))
