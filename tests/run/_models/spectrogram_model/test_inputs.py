import typing

import config as cf
import pytest
import spacy.vocab
import torch
from thinc.types import FloatsXd

import run
from lib.text import load_en_english, xml_to_text
from run._config import load_spacy_nlp
from run._models.spectrogram_model.inputs import (
    Casing,
    Context,
    InputsWrapper,
    Pronun,
    PublicValueError,
    XMLType,
    _embed_annotations,
    _get_case,
    _Schema,
    preprocess,
)
from run.data._loader.structures import Language
from tests._utils import assert_almost_equal
from tests.run._utils import make_session


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    config = {
        run._models.spectrogram_model.inputs.InputsWrapper.check_invariants: cf.Args(
            min_loudness=-100, max_loudness=0, min_tempo=0.025, max_tempo=1
        ),
    }
    cf.add(config, overwrite=True)
    yield
    cf.purge()


def test_inputs_wrapper():
    """Test `InputsWrapper` with no annotations."""
    nlp = load_en_english()
    script = "this is a test"
    span = nlp(script)
    input_ = InputsWrapper(
        session=[make_session()],
        span=[span],
        context=[span],
        loudness=[[]],
        tempo=[[]],
        respellings=[{}],
    )
    assert len(input_) == 1
    assert input_.get(0) == input_
    assert input_.to_xml(0) == f"<{_Schema.SPEAK} {_Schema._VALUE}='-1'>{script}</{_Schema.SPEAK}>"


def test_inputs_wrapper__from_xml():
    """Test `InputsWrapper.from_xml` with no annotations."""
    nlp = load_en_english()
    script = "this is a test"
    span = nlp(script)
    result = InputsWrapper.from_xml(XMLType(script), span, make_session())
    expected = InputsWrapper(
        session=[make_session()],
        span=[span],
        context=[span],
        loudness=[[]],
        tempo=[[]],
        respellings=[{}],
    )
    assert result == expected


def test_inputs_wrapper__from_xml_batch():
    """Test `InputsWrapper.from_xml_batch` processes a batch correctly."""
    nlp = load_en_english()
    script = "this is a test"
    doc = nlp(script)
    xml = XMLType(script)
    sesh = make_session()
    result = InputsWrapper.from_xml_batch([xml, xml], [doc, doc], [sesh, sesh])
    expected = InputsWrapper(
        session=[make_session()],
        span=[doc],
        context=[doc],
        loudness=[[]],
        tempo=[[]],
        respellings=[{}],
    )
    assert result.get(0) == expected
    assert result.get(1) == expected


def test_inputs_wrapper__from_xml__annotated():
    """Test `InputsWrapper.from_xml` and `InputsWrapper.to_xml` with annotations."""
    nlp = load_en_english()
    script = "this is a test"
    xml = f"<{_Schema.LOUDNESS} {_Schema._VALUE}='-20'>Over the river and "
    xml += f"<{_Schema.TEMPO} {_Schema._VALUE}='0.04'>through the "
    xml += f"<{_Schema.RESPELL} {_Schema._VALUE}='wuuds'>woods</{_Schema.RESPELL}>"
    xml += f"</{_Schema.TEMPO}>.</{_Schema.LOUDNESS}>"
    xml = XMLType(xml)
    script = xml_to_text(xml)
    doc = nlp(script)
    result = InputsWrapper.from_xml(xml, doc, make_session())
    tempo = slice(len("Over the river and "), len("Over the river and through the woods"))
    expected = InputsWrapper(
        session=[make_session()],
        span=[doc],
        context=[doc],
        loudness=[[(slice(0, len("Over the river and through the woods.")), -20)]],
        tempo=[[(tempo, 0.04)]],
        respellings=[{doc[-2]: "wuuds"}],
    )
    assert result == expected
    assert expected.to_xml(0) == f"<{_Schema.SPEAK} {_Schema._VALUE}='-1'>{xml}</{_Schema.SPEAK}>"


def check_annotation(anno: _Schema, text: str, respelling: str, prefix: str = "", suffix: str = ""):
    nlp = load_en_english()
    xml = XMLType(f'{prefix}<{anno} {_Schema._VALUE}="{respelling}">{text}</{anno}>{suffix}')
    InputsWrapper.from_xml(xml, nlp(prefix + text + suffix), make_session())


def test__inputs_wrapper__from_xml__token_annotations():
    """Test `InputsWrapper.from_xml` validates respellings."""
    check_annotation(_Schema.RESPELL, "scientific", "SY-uhn-TIH-fihk")  # Valid

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "")  # No value

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "", "SY-uhn-TIH-fihk")  # No text

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "", "SY-uhn-TIH-fihk", "scientific")  # No text

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "-SY-uhn-TIH-fihk")  # Invalid prefix

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "SY-uhn-TIH-fihk-")  # Invalid suffix

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "SY uhn TIH fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "SY1uhn2TIH3fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "SY|uhn|TIH|fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific", "Sy-uhn-TIH-fihk")  # Mix capitalization

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scientific-process", "SY-uhn-TIH-fihk")  # Multiple words

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "entific", "SY-uhn-TIH-fihk", "sci")  # Prefix

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.RESPELL, "scienti", "SY-uhn-TIH-fihk", suffix="fic")  # Suffix


def test__inputs_wrapper__from_xml__span_annotations():
    """Test `InputsWrapper.from_xml` validates span annotations."""
    check_annotation(_Schema.LOUDNESS, "scientific process", "-20")  # Valid
    check_annotation(_Schema.TEMPO, "scientific process", "0.04")  # Valid

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "scientific process", "")  # No value

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "", "-20")  # No text

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "", "-20", "scientific", "process")  # No text

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "scientific process", "NA")  # No number

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "scientific process", "-1000")  # Too small

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "scientific process", "1000")  # Too big

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "entific process", "-20", "sci")  # Prefix

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.LOUDNESS, "scientific pro", "-20", suffix="cess")  # Suffix

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.TEMPO, "scientific process", "0")  # Too small

    with pytest.raises(PublicValueError):
        check_annotation(_Schema.TEMPO, "scientific process", "1000")  # Too big


def test_inputs_wrapper__to_xml__context():
    """Test `InputsWrapper.to_xml` with context."""
    nlp = load_en_english()
    script = "this is a test"
    doc = nlp(script)
    span = doc[1:-1]
    xml = f"<{_Schema.LOUDNESS} {_Schema._VALUE}='-20'>{str(span)}</{_Schema.LOUDNESS}>"
    xml = XMLType(xml)
    input_ = InputsWrapper(
        session=[make_session()],
        span=[span],
        context=[doc],
        loudness=[[(slice(0, len(xml)), -20)]],
        tempo=[[]],
        respellings=[{}],
    )
    expected = f"this <{_Schema.SPEAK} {_Schema._VALUE}='-1'>{xml}</{_Schema.SPEAK}> test"
    assert input_.to_xml(0, include_context=True) == expected


def test__preprocess():
    """Test that `_preprocess` handles a basic input."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "In 1968 the U.S. Army", make_session()
    doc = nlp(script)
    input_ = InputsWrapper.from_xml(XMLType(doc[:-1].text), doc[:-1], sesh, doc)
    processed = preprocess(input_, {}, {})
    assert processed.tokens == [list(script.lower())]
    assert processed.seq_metadata[0] == [sesh[0].label]
    assert processed.seq_metadata[1] == [sesh]
    assert script[processed.slices[0]] == str(doc[:-1])
    casing = [
        (Pronun.NORMAL, Casing.UPPER),  # I
        (Pronun.NORMAL, Casing.LOWER),  # n
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.NO_CASING),  # 1
        (Pronun.NORMAL, Casing.NO_CASING),  # 9
        (Pronun.NORMAL, Casing.NO_CASING),  # 6
        (Pronun.NORMAL, Casing.NO_CASING),  # 8
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.LOWER),  # t
        (Pronun.NORMAL, Casing.LOWER),  # h
        (Pronun.NORMAL, Casing.LOWER),  # e
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.UPPER),  # u
        (Pronun.NORMAL, Casing.NO_CASING),  # .
        (Pronun.NORMAL, Casing.UPPER),  # s
        (Pronun.NORMAL, Casing.NO_CASING),  # .
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.UPPER),  # A
        (Pronun.NORMAL, Casing.LOWER),  # r
        (Pronun.NORMAL, Casing.LOWER),  # m
        (Pronun.NORMAL, Casing.LOWER),  # y
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
    word_embedding_length = nlp.meta["vectors"]["width"]
    # NOTE: The word embeddings are case sensitive.
    word_embeddings = [
        torch.from_numpy(vocab["In"].vector).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, word_embedding_length),
        torch.from_numpy(vocab["1968"].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, word_embedding_length),
        torch.from_numpy(vocab["the"].vector).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, word_embedding_length),
        torch.from_numpy(vocab["U.S."].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, word_embedding_length),
        torch.from_numpy(vocab["Army"].vector).unsqueeze(0).repeat(4, 1),
    ]
    contextual_embedding_length = typing.cast(FloatsXd, doc[0].tensor).shape[0]
    contextual_embeddings = [
        torch.from_numpy(doc[0].tensor).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, contextual_embedding_length),
        torch.from_numpy(doc[1].tensor).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, contextual_embedding_length),
        torch.from_numpy(doc[2].tensor).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, contextual_embedding_length),
        torch.from_numpy(doc[3].tensor).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, contextual_embedding_length),
        torch.from_numpy(doc[4].tensor).unsqueeze(0).repeat(4, 1),
    ]
    stack = (
        torch.cat(word_embeddings),
        torch.cat(contextual_embeddings),
        torch.zeros(len(script), 4),
    )
    token_embeddings = torch.cat(stack, dim=1)
    assert len(processed.token_embeddings) == 1
    assert torch.allclose(processed.token_embeddings[0], token_embeddings)


def test__get_case():
    """Test `_get_case` on basic cases."""
    assert _get_case("A") == Casing.UPPER
    assert _get_case("a") == Casing.LOWER
    assert _get_case("1") == Casing.NO_CASING
    with pytest.raises(AssertionError):
        _get_case("")


def test__embed_annotations():
    """Test `_embed_annotations` on basic cases."""
    annotations = [(slice(0, 2), 20), (slice(3, 4), -10), (slice(6, 8), 0.99)]

    embedding = _embed_annotations(9, annotations)
    expected = torch.tensor([[20, 20, 0, -10, 0, 0, 0.99, 0.99, 0], [1, 1, 0, -1, 0, 0, 1, 1, 0]])
    assert_almost_equal(embedding, expected.transpose(0, 1))

    embedding = _embed_annotations(9, annotations, 1, 1, 10)
    expected = [[0.1, 2.1, 2.1, 0.1, -0.9, 0.1, 0.1, 0.199, 0.199], [0, 1, 1, 0, -1, 0, 0, 1, 1]]
    assert_almost_equal(embedding, torch.tensor(expected).transpose(0, 1))


def test__preprocess_respelling():
    """Test that `_preprocess` handles apostrophes, dashes, initialisms and existing respellings."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    xml = "n't <respell value='PEE-puhl'>people</respell> from EDGE "
    xml += "<respell value='KACH'>catch</respell>-the-<respell value='FLOO'>flu</respell>"
    xml = XMLType(xml)
    script = "Don't people from EDGE catch-the-flu?"
    sesh = make_session()
    doc = nlp(script)
    input_ = InputsWrapper.from_xml(xml, doc[1:-1], sesh, doc)
    processed = preprocess(input_, {}, {})
    casing = [
        (Pronun.NORMAL, Casing.UPPER),  # D
        (Pronun.NORMAL, Casing.LOWER),  # o
        (Pronun.NORMAL, Casing.LOWER),  # n
        (Pronun.NORMAL, Casing.NO_CASING),  # '
        (Pronun.NORMAL, Casing.LOWER),  # t
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.RESPELLING, Casing.UPPER),  # P
        (Pronun.RESPELLING, Casing.UPPER),  # E
        (Pronun.RESPELLING, Casing.UPPER),  # E
        (Pronun.RESPELLING, Casing.NO_CASING),  # -
        (Pronun.RESPELLING, Casing.LOWER),  # p
        (Pronun.RESPELLING, Casing.LOWER),  # u
        (Pronun.RESPELLING, Casing.LOWER),  # h
        (Pronun.RESPELLING, Casing.LOWER),  # l
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.LOWER),  # f
        (Pronun.NORMAL, Casing.LOWER),  # r
        (Pronun.NORMAL, Casing.LOWER),  # o
        (Pronun.NORMAL, Casing.LOWER),  # m
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.UPPER),  # E
        (Pronun.NORMAL, Casing.UPPER),  # D
        (Pronun.NORMAL, Casing.UPPER),  # G
        (Pronun.NORMAL, Casing.UPPER),  # E
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.RESPELLING, Casing.UPPER),  # K
        (Pronun.RESPELLING, Casing.UPPER),  # A
        (Pronun.RESPELLING, Casing.UPPER),  # C
        (Pronun.RESPELLING, Casing.UPPER),  # H
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.NORMAL, Casing.LOWER),  # t
        (Pronun.NORMAL, Casing.LOWER),  # h
        (Pronun.NORMAL, Casing.LOWER),  # e
        (Pronun.NORMAL, Casing.NO_CASING),
        (Pronun.RESPELLING, Casing.UPPER),  # F
        (Pronun.RESPELLING, Casing.UPPER),  # L
        (Pronun.RESPELLING, Casing.UPPER),  # O
        (Pronun.RESPELLING, Casing.UPPER),  # O
        (Pronun.NORMAL, Casing.NO_CASING),  # ?
    ]
    # TODO: We need to test if there is a space after span before context.
    assert processed.token_metadata[0] == [casing]
    tokens = "n't pee-puhl from edge kach-the-floo"
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
        Context.SCRIPT,  # -
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
    expected = torch.from_numpy(nlp.vocab["flu"].vector)  # FLOO
    for idx in range(-5, -1):
        assert_almost_equal(processed.token_embeddings[0][idx][: expected.shape[0]], expected)
    expected = torch.from_numpy(nlp.vocab["n't"].vector)
    for idx in range(2, 4):
        assert_almost_equal(processed.token_embeddings[0][idx][: expected.shape[0]], expected)
