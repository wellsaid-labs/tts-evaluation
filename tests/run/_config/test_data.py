import hparams
import pytest

import run
from run._config.data import _include_span
from run.data import _loader
from tests.run._utils import make_alignments_1d, make_alignments_2d, make_passage


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    hparams.clear_config()


def test__include_span():
    """Test `_include_span` handles on a basic cases."""
    assert _include_span(make_passage(script="This")[:])

    # Exclude script
    assert not _include_span(make_passage(script="This1")[:])
    assert not _include_span(make_passage(script="Th>s")[:])
    assert not _include_span(make_passage(script="Th/s")[:])

    # Exclude audio alignments
    alignments = make_alignments_2d("ThisThisThis", ((0, 0.3),))
    assert not _include_span(make_passage(script="ThisThisThis", alignments=alignments)[:])
    alignments = make_alignments_2d("This", ((0, 0.15),))
    assert not _include_span(make_passage(script="This", alignments=alignments)[:])

    # Exclude script due to context
    assert _include_span(make_passage(script="This is test")[1:-1])
    assert not _include_span(make_passage(script="Thi1 is test")[1:-1])

    # Exclude due to mistranscription / nonalignments
    alignments = make_alignments_1d(((0, 1), (4, 5)))
    span = make_passage(script="A B C", alignments=alignments)[:]
    assert _loader.has_a_mistranscription(span)
    assert not _include_span(span)
