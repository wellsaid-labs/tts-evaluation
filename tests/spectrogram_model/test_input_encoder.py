import pytest

from src.datasets import Speaker
from src.spectrogram_model import InputEncoder


def test_input_encoder():
    encoder = InputEncoder(['a', 'b', 'c'], [Speaker.JUDY_BIEBER, Speaker.MARY_ANN])
    input_ = ('a', Speaker.JUDY_BIEBER)
    encoded = encoder.batch_encode([input_])[0]
    assert encoder.decode(encoded) == input_


def test_input_encoder__reversible():
    encoder = InputEncoder(['a', 'b', 'c'], [Speaker.JUDY_BIEBER, Speaker.MARY_ANN])

    with pytest.raises(ValueError):  # Text is not reversible
        encoder.encode(('d', Speaker.JUDY_BIEBER))

    with pytest.raises(ValueError):  # Speaker is not reversible
        encoder.encode(('a', Speaker.HILARY_NORIEGA))
