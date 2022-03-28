import pathlib

import config as cf
import pytest

from run._config.data import _include_passage, _include_span, configure
from run.data import _loader
from tests.run._utils import make_alignments_1d, make_alignments_2d, make_passage


@pytest.fixture(autouse=True)
def run_around_test():
    configure()
    yield
    cf.purge()


def test__include_passage():
    # Exclude upper case scripts
    assert not _include_passage(make_passage(script="THIS IS"), pathlib.Path())
    # Include single word scripts
    assert _include_passage(make_passage(script="THIS"), pathlib.Path())


def test__include_span():
    """Test `_include_span` handles basic cases."""
    assert _include_span(make_passage(script="This")[:])

    # Exclude invalid scripts
    assert not _include_span(make_passage(script="This1")[:])
    assert not _include_span(make_passage(script="Th>s")[:])
    assert not _include_span(make_passage(script="Th/s")[:])

    # Exclude empty audio alignments
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

    # Include if there is no mistranscription
    alignments = make_alignments_1d(((0, 1), (4, 5)))
    span = make_passage(script="A   C", alignments=alignments)[:]
    assert not _loader.has_a_mistranscription(span)
    assert _include_span(span)
