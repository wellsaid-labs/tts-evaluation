import config as cf
import pytest
import spacy.vocab
import torch

import run
from run._config import load_spacy_nlp
from run._models.spectrogram_model.inputs import (
    Casing,
    Context,
    InputsWrapper,
    Pronunciation,
    RespellingError,
    _preprocess,
    preprocess_inputs,
    preprocess_spans,
)
from run.data._loader.structures import Language
from tests.run._utils import make_passage, make_session


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    yield
    cf.purge()


def test__preprocess():
    """Test that `_preprocess` handles a basic input."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "In 1968 the U.S. Army", make_session()
    doc = nlp(script)
    processed = _preprocess([(sesh, doc, doc[:-1])])
    assert processed.reconstruct_text(0) == script
    assert processed.tokens == [list(script.lower())]
    assert processed.seq_metadata[0] == [sesh[0].label]
    assert processed.seq_metadata[1] == [sesh]
    assert script[processed.slices[0]] == str(doc[:-1])
    casing = [
        (Pronunciation.NORMAL, Casing.UPPER),  # I
        (Pronunciation.NORMAL, Casing.LOWER),  # n
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.NO_CASING),  # 1
        (Pronunciation.NORMAL, Casing.NO_CASING),  # 9
        (Pronunciation.NORMAL, Casing.NO_CASING),  # 6
        (Pronunciation.NORMAL, Casing.NO_CASING),  # 8
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.LOWER),  # t
        (Pronunciation.NORMAL, Casing.LOWER),  # h
        (Pronunciation.NORMAL, Casing.LOWER),  # e
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.UPPER),  # u
        (Pronunciation.NORMAL, Casing.NO_CASING),  # .
        (Pronunciation.NORMAL, Casing.UPPER),  # s
        (Pronunciation.NORMAL, Casing.NO_CASING),  # .
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.UPPER),  # A
        (Pronunciation.NORMAL, Casing.LOWER),  # r
        (Pronunciation.NORMAL, Casing.LOWER),  # m
        (Pronunciation.NORMAL, Casing.LOWER),  # y
    ]
    assert processed.token_metadata[0] == [casing]
    context = [
        Context.SCRIPT,  # I
        Context.SCRIPT,  # n
        Context.SCRIPT,
        Context.SCRIPT,  # 1
        Context.SCRIPT,  # 9
        Context.SCRIPT,  # 6
        Context.SCRIPT,  # 8
        Context.SCRIPT,
        Context.SCRIPT,  # t
        Context.SCRIPT,  # h
        Context.SCRIPT,  # e
        Context.SCRIPT,
        Context.SCRIPT,  # u
        Context.SCRIPT,  # .
        Context.SCRIPT,  # s
        Context.SCRIPT,  # .
        Context.CONTEXT,
        Context.CONTEXT,  # A
        Context.CONTEXT,  # r
        Context.CONTEXT,  # m
        Context.CONTEXT,  # y
    ]
    assert processed.token_metadata[1] == [context]
    vocab: spacy.vocab.Vocab = nlp.vocab
    length = nlp.meta["vectors"]["width"]
    embeddings = [
        torch.from_numpy(vocab["in"].vector).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, length),
        torch.from_numpy(vocab["1968"].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, length),
        torch.from_numpy(vocab["the"].vector).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, length),
        torch.from_numpy(vocab["u.s."].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, length),
        torch.from_numpy(vocab["army"].vector).unsqueeze(0).repeat(4, 1),
    ]
    token_embeddings = torch.cat(embeddings)
    assert len(processed.token_embeddings) == 1
    assert torch.allclose(processed.token_embeddings[0][:, :length], token_embeddings)


def test__preprocess_respelling():
    """Test that `_preprocess` handles apostrophes, dashes, initialisms and existing respellings."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "Don't |\\PEE\\puhl\\| from EDGE catch-the-flu?", make_session()
    doc = nlp(script)
    processed = _preprocess([(sesh, doc, doc[1:-1])], respell_prob=1.0)
    assert processed.reconstruct_text(0) == "Don't PEE\\puhl from EDGE KACH-the-FLOO?"
    casing = [
        (Pronunciation.NORMAL, Casing.UPPER),  # D
        (Pronunciation.NORMAL, Casing.LOWER),  # o
        (Pronunciation.NORMAL, Casing.LOWER),  # n
        (Pronunciation.NORMAL, Casing.NO_CASING),  # '
        (Pronunciation.NORMAL, Casing.LOWER),  # t
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.RESPELLING, Casing.UPPER),  # P
        (Pronunciation.RESPELLING, Casing.UPPER),  # E
        (Pronunciation.RESPELLING, Casing.UPPER),  # E
        (Pronunciation.RESPELLING, Casing.NO_CASING),  # \
        (Pronunciation.RESPELLING, Casing.LOWER),  # p
        (Pronunciation.RESPELLING, Casing.LOWER),  # u
        (Pronunciation.RESPELLING, Casing.LOWER),  # h
        (Pronunciation.RESPELLING, Casing.LOWER),  # l
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.LOWER),  # f
        (Pronunciation.NORMAL, Casing.LOWER),  # r
        (Pronunciation.NORMAL, Casing.LOWER),  # o
        (Pronunciation.NORMAL, Casing.LOWER),  # m
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.UPPER),  # E
        (Pronunciation.NORMAL, Casing.UPPER),  # D
        (Pronunciation.NORMAL, Casing.UPPER),  # G
        (Pronunciation.NORMAL, Casing.UPPER),  # E
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.RESPELLING, Casing.UPPER),  # K
        (Pronunciation.RESPELLING, Casing.UPPER),  # A
        (Pronunciation.RESPELLING, Casing.UPPER),  # C
        (Pronunciation.RESPELLING, Casing.UPPER),  # H
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.NORMAL, Casing.LOWER),  # t
        (Pronunciation.NORMAL, Casing.LOWER),  # h
        (Pronunciation.NORMAL, Casing.LOWER),  # e
        (Pronunciation.NORMAL, Casing.NO_CASING),
        (Pronunciation.RESPELLING, Casing.UPPER),  # F
        (Pronunciation.RESPELLING, Casing.UPPER),  # L
        (Pronunciation.RESPELLING, Casing.UPPER),  # O
        (Pronunciation.RESPELLING, Casing.UPPER),  # O
        (Pronunciation.NORMAL, Casing.NO_CASING),  # ?
    ]
    assert processed.token_metadata[0] == [casing]
    tokens = "n't pee\\puhl from edge kach-the-floo"
    assert "".join(processed.tokens[0][processed.slices[0]]) == tokens  # type: ignore
    context = [
        Context.CONTEXT,  # D
        Context.CONTEXT,  # o
        Context.SCRIPT,  # n
        Context.SCRIPT,  # '
        Context.SCRIPT,  # t
        Context.SCRIPT,
        Context.SCRIPT,  # P
        Context.SCRIPT,  # E
        Context.SCRIPT,  # E
        Context.SCRIPT,  # \
        Context.SCRIPT,  # p
        Context.SCRIPT,  # u
        Context.SCRIPT,  # h
        Context.SCRIPT,  # l
        Context.SCRIPT,
        Context.SCRIPT,  # f
        Context.SCRIPT,  # r
        Context.SCRIPT,  # o
        Context.SCRIPT,  # m
        Context.SCRIPT,
        Context.SCRIPT,  # E
        Context.SCRIPT,  # D
        Context.SCRIPT,  # G
        Context.SCRIPT,  # E
        Context.SCRIPT,
        Context.SCRIPT,  # K
        Context.SCRIPT,  # A
        Context.SCRIPT,  # C
        Context.SCRIPT,  # H
        Context.SCRIPT,
        Context.SCRIPT,  # t
        Context.SCRIPT,  # h
        Context.SCRIPT,  # e
        Context.SCRIPT,
        Context.SCRIPT,  # F
        Context.SCRIPT,  # L
        Context.SCRIPT,  # O
        Context.SCRIPT,  # O
        Context.CONTEXT,  # ?
    ]
    assert processed.token_metadata[1] == [context]
    assert processed.token_embeddings[0][-5:-1].sum().item() == 0.0  # FLOO
    assert processed.token_embeddings[0][:3].sum().item() != 0.0  # n't


def test__preprocess_zero_length():
    """Test that `_preprocess` handles a zero length script."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "", make_session()
    doc = nlp(script)
    processed = _preprocess([(sesh, doc, doc)], respell_prob=1.0)
    assert processed.reconstruct_text(0) == ""
    assert "".join(processed.tokens[0][processed.slices[0]]) == ""  # type: ignore
    assert processed.token_metadata[0] == [[]]
    assert processed.token_metadata[1] == [[]]
    assert processed.slices[0] == slice(0, 0)
    assert processed.token_embeddings[0].shape == (0, 0)


def test__preprocess_schwa():
    """Test that `_preprocess` handles the special character schwa."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "motorcycle", make_session()
    doc = nlp(script)
    processed = _preprocess([(sesh, doc, doc)], respell_prob=1.0)
    assert processed.reconstruct_text(0) == "MOH\\tur\\sy\\kuhl"


def test__preprocess_invalid_respelling():
    """Test that `_preprocess` handles errors if respelling is invalid."""
    nlp = load_spacy_nlp(Language.ENGLISH)

    with pytest.raises(RespellingError):
        doc = nlp("|\\\\|")  # Zero length
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\\\MOH\\|")  # Invalid prefix
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\MOH\\\\|")  # Invalid suffix
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\MOH tər\\|")  # Invalid character
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\MOH5tər\\|")  # Invalid character
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\MOH|tər\\|")  # Invalid character
        _preprocess([(make_session(), doc, doc)])

    with pytest.raises(RespellingError):
        doc = nlp("|\\Moh\\|")  # Mix capitalization
        _preprocess([(make_session(), doc, doc)])


def test_preprocess_inputs_and_spans():
    """Test that `preprocess_spans` and `preprocess_inputs` function similarly."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script = "In 1968 the U.S. Army"
    passage = make_passage(script=script)
    pre_span = preprocess_spans([passage[1:-1]])
    inputs = InputsWrapper(session=[passage.session], doc=[nlp(passage[1:-1].script)])
    pre_doc = preprocess_inputs(inputs)
    assert pre_doc.seq_metadata == pre_span.seq_metadata
    assert pre_doc.tokens[0] == pre_span.tokens[0][3:-5]
    assert pre_doc.token_metadata == [[s[3:-5] for s in m] for m in pre_span.token_metadata]
    length = nlp.meta["vectors"]["width"]
    assert torch.allclose(
        pre_doc.token_embeddings[0][:, :length],
        pre_span.token_embeddings[0][3:-5][:, :length],
    )
    doc_slice, span_slice = pre_doc.slices[0], pre_span.slices[0]
    assert doc_slice.stop - doc_slice.start == span_slice.stop - span_slice.start
