import pytest

from src.datasets import HILARY_NORIEGA
from src.datasets import JUDY_BIEBER
from src.datasets import MARY_ANN
from src.spectrogram_model import InputEncoder


def test_input_encoder():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])
    input_ = ('a', JUDY_BIEBER)
    encoded = encoder.batch_encode([input_])[0]
    assert encoder.decode(encoded) == input_


def test_input_encoder__reversible():
    encoder = InputEncoder(['a', 'b', 'c'], [JUDY_BIEBER, MARY_ANN])

    with pytest.raises(ValueError):  # Text is not reversible
        encoder.encode(('d', JUDY_BIEBER))

    with pytest.raises(ValueError):  # Speaker is not reversible
        encoder.encode(('a', HILARY_NORIEGA))
