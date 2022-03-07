import spacy.vocab
import torch

import lib
from run.data._loader import Session
from run.data._loader.english import JUDY_BIEBER
from run.train.spectrogram_model._model import Inputs, _Casing, _preprocess_inputs


def test__preprocess_inputs():
    """Test that `_preprocess_inputs` handles a basic input."""
    nlp = lib.text.load_en_core_web_md()
    doc = nlp("In 1968 the U.S. Army")
    sesh = Session((JUDY_BIEBER, "sesh"))
    inputs = Inputs(speaker=[sesh[0]], session=[sesh], spans=[doc[:-1]])
    processed = _preprocess_inputs(inputs, 3)
    assert processed.tokens == [list("in 1968 the u.s. army")]
    assert processed.seq_metadata == [(sesh[0], sesh)]
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


def test__preprocess_inputs__doc():
    """Test that `_preprocess_inputs` handles a `Doc`, similarly to a `Span`."""
    nlp = lib.text.load_en_core_web_md()
    doc = nlp("In 1968 the U.S. Army")
    sesh = Session((JUDY_BIEBER, "sesh"))
    inputs = Inputs(speaker=[sesh[0]], session=[sesh], spans=[doc[:-1]])
    pre_span = _preprocess_inputs(inputs, 3)
    pre_doc = _preprocess_inputs(inputs._replace(spans=[inputs.spans[0].as_doc()]), 3)
    assert pre_doc.seq_metadata == pre_span.seq_metadata
    assert pre_doc.tokens[0] == pre_span.tokens[0][:-5]
    assert pre_doc.token_metadata[0] == pre_span.token_metadata[0][:-5]
    assert torch.allclose(pre_doc.token_embeddings[0], pre_span.token_embeddings[0][:-5])
    assert pre_doc.slices == pre_span.slices
