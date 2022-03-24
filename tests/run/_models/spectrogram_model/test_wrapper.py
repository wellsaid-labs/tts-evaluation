import config as cf
import pytest
import spacy.vocab
import torch

import lib
import run
from run._models.spectrogram_model.wrapper import (
    InputsWrapper,
    _Casing,
    preprocess_inputs,
    preprocess_spans,
)
from tests.run._utils import make_passage


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()


def test_preprocess_spans():
    """Test that `preprocess_spans` handles a basic input."""
    nlp = lib.text.load_en_core_web_md()
    script = "In 1968 the U.S. Army"
    doc = nlp(script)
    passage = make_passage(script=script)
    processed = preprocess_spans([passage[:-1]])
    assert processed.tokens == [list(script.lower())]
    assert processed.seq_metadata == [(passage.speaker, passage.session)]
    casing = [
        _Casing.UPPER,  # I
        _Casing.LOWER,  # n
        _Casing.NO_CASING,
        _Casing.NO_CASING,  # 1
        _Casing.NO_CASING,  # 9
        _Casing.NO_CASING,  # 6
        _Casing.NO_CASING,  # 8
        _Casing.NO_CASING,
        _Casing.LOWER,  # t
        _Casing.LOWER,  # h
        _Casing.LOWER,  # e
        _Casing.NO_CASING,
        _Casing.UPPER,  # u
        _Casing.NO_CASING,  # .
        _Casing.UPPER,  # s
        _Casing.NO_CASING,  # .
        _Casing.NO_CASING,
        _Casing.UPPER,  # A
        _Casing.LOWER,  # r
        _Casing.LOWER,  # m
        _Casing.LOWER,  # y
    ]
    assert processed.token_metadata == [[(c,) for c in casing]]
    vocab: spacy.vocab.Vocab = nlp.vocab
    embeddings = [
        torch.from_numpy(vocab["in"].vector).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, nlp.meta["vectors"]["width"]),
        torch.from_numpy(vocab["1968"].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, nlp.meta["vectors"]["width"]),
        torch.from_numpy(vocab["the"].vector).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, nlp.meta["vectors"]["width"]),
        torch.from_numpy(vocab["u.s."].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, nlp.meta["vectors"]["width"]),
        torch.from_numpy(vocab["army"].vector).unsqueeze(0).repeat(4, 1),
    ]
    token_embeddings = torch.cat(embeddings)
    assert len(processed.token_embeddings) == 1
    assert torch.allclose(processed.token_embeddings[0], token_embeddings)
    assert processed.slices == [slice(0, len(doc.text) - 5)]


def test_preprocess_inputs_and_spans():
    """Test that `preprocess_spans` and `preprocess_inputs` function similarly."""
    nlp = lib.text.load_en_core_web_md()
    script = "In 1968 the U.S. Army"
    passage = make_passage(script=script)
    pre_span = preprocess_spans([passage[1:-1]])
    pre_doc = preprocess_inputs(
        InputsWrapper(session=[passage.session], doc=[nlp(passage[1:-1].script)])
    )
    assert pre_doc.seq_metadata == pre_span.seq_metadata
    assert pre_doc.tokens[0] == pre_span.tokens[0][3:-5]
    assert pre_doc.token_metadata[0] == pre_span.token_metadata[0][3:-5]
    assert torch.allclose(pre_doc.token_embeddings[0], pre_span.token_embeddings[0][3:-5])
    doc_slice, span_slice = pre_doc.slices[0], pre_span.slices[0]
    assert doc_slice.stop - doc_slice.start == span_slice.stop - span_slice.start
