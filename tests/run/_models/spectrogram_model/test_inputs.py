import typing

import config as cf
import pytest
import spacy.vocab
import torch
from spacy import language
from thinc.types import FloatsXd

import run
from lib.text import load_en_english, text_to_xml, xml_to_text
from lib.utils import identity
from run._config import load_spacy_nlp
from run._models.spectrogram_model.inputs import (
    Casing,
    Context,
    Inputs,
    InputsWrapper,
    Pronun,
    PublicValueError,
    SpanDoc,
    XMLType,
    _get_case,
)
from run._models.spectrogram_model.inputs import _Schema as Sch
from run._models.spectrogram_model.inputs import preprocess
from run.data._loader.structures import Language, Session
from tests._utils import assert_almost_equal
from tests.run._utils import make_session


@pytest.fixture(autouse=True)
def run_around_tests():
    """Set a basic configuration."""
    run._config.configure()
    config = {
        run._models.spectrogram_model.inputs.InputsWrapper.check_invariants: cf.Args(
            min_loudness=-100, max_loudness=0, min_tempo=0, max_tempo=10
        ),
    }
    cf.add(config, overwrite=True)
    yield
    cf.purge()


def test_inputs_wrapper():
    """Test `InputsWrapper` with no annotations on basic functionality."""
    script = "this is a test"
    span = load_en_english()(script)
    inpt = InputsWrapper([make_session()], [span], [span], [[]], [[]], [{}])
    assert len(inpt) == 1
    assert inpt[0] == inpt
    assert inpt.to_xml(0) == f"<{Sch.SPEAK} {Sch._VALUE}='-1'>{script}</{Sch.SPEAK}>"


def test_inputs_wrapper__from_xml():
    """Test `InputsWrapper.from_xml` with no annotations initializes correctly."""
    script = "this is a test"
    span = load_en_english()(script)
    result = InputsWrapper.from_xml(XMLType(script), span, make_session())
    expected = InputsWrapper([make_session()], [span], [span], [[]], [[]], [{}])
    assert result == expected


def test_inputs_wrapper__from_xml__annotatations():
    """Test `InputsWrapper.from_xml` and `InputsWrapper.to_xml` with various annotations."""
    xml = (
        f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>Over the river and <{Sch.TEMPO} {Sch._VALUE}='0.04'>"
        f"through the <{Sch.RESPELL} {Sch._VALUE}='wuuds'>woods</{Sch.RESPELL}></{Sch.TEMPO}>."
        f"</{Sch.LOUDNESS}>"
    )
    xml = XMLType(xml)
    script = xml_to_text(xml)
    doc = load_en_english()(script)
    result = InputsWrapper.from_xml(xml, doc, make_session())
    tempo = [(slice(len("Over the river and "), len("Over the river and through the woods")), 0.04)]
    loudness = [(slice(0, len("Over the river and through the woods.")), -20.0)]
    respell = {doc[-2]: "wuuds"}
    expected = InputsWrapper([make_session()], [doc], [doc], [loudness], [tempo], [respell])
    assert result == expected
    assert expected.to_xml(0) == f"<{Sch.SPEAK} {Sch._VALUE}='-1'>{xml}</{Sch.SPEAK}>"


def test_inputs_wrapper__from_xml__nested_slice_anno_error():
    """Test `InputsWrapper.from_xml` does not accept nested slice annotations, like loudness."""
    start = f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>"
    end = f"</{Sch.LOUDNESS}>"
    xml = XMLType(f"{start}this {start}is{end} a test{end}")
    script = xml_to_text(xml)
    with pytest.raises(PublicValueError, match=r".*loudness annotations cannot not be nested*"):
        InputsWrapper.from_xml(xml, load_en_english()(script), make_session())


def test_inputs_wrapper__from_xml__trim_spaces_error():
    """Test `InputsWrapper.from_xml` does not accept untrimmed spaces."""
    start = f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>"
    end = f"</{Sch.LOUDNESS}>"

    xml = XMLType(f" {start}this is a test{end} ")
    script = xml_to_text(xml)
    with pytest.raises(PublicValueError, match=r".*must be stripped of white spaces*"):
        InputsWrapper.from_xml(xml, load_en_english()(script), make_session())

    xml = XMLType(f"{start} this is a test {end}")
    script = xml_to_text(xml)
    with pytest.raises(PublicValueError, match=r".*must be stripped of white spaces*"):
        InputsWrapper.from_xml(xml, load_en_english()(script), make_session())


def test_inputs_wrapper__from_xml__nested_token_anno_error():
    """Test `InputsWrapper.from_xml` does not accept nested token annotations, like respell."""
    xml = f"<{Sch.RESPELL} {Sch._VALUE}='wuuds'>woods</{Sch.RESPELL}>"
    xml = XMLType(f"<{Sch.RESPELL} {Sch._VALUE}='wuuds'>{xml}</{Sch.RESPELL}>")
    script = xml_to_text(xml)
    with pytest.raises(PublicValueError, match=r".*XML is invalid.*"):
        InputsWrapper.from_xml(xml, load_en_english()(script), make_session())


def test_inputs_wrapper__from_xml_batch():
    """Test `InputsWrapper.from_xml_batch` processes a basic batch."""
    batch_size = 5
    docs, xmls, seshs = [], [], []
    for i in range(batch_size):
        script = f"[{i}] this is a test"
        docs.append(load_en_english()(script))
        xmls.append(XMLType(script))
        seshs.append(make_session())
    batch = InputsWrapper.from_xml_batch(xmls, docs, seshs)
    # NOTE: Please refer to this typing issue: https://github.com/python/mypy/issues/9737
    for i, result in enumerate(batch):  # type: ignore
        assert result == InputsWrapper([seshs[i]], [docs[i]], [docs[i]], [[]], [[]], [{}])


def test_inputs_wrapper__from_xml__escaped_chars():
    """Test `InputsWrapper.from_xml` and `InputsWrapper.to_xml` handles escaped html characters."""
    script = "Over the <<river>> and through the woods."
    xml = XMLType(f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>{text_to_xml(script)}</{Sch.LOUDNESS}>")
    doc = load_en_english()(script)
    result = InputsWrapper.from_xml(xml, doc, make_session())
    loudness = [(slice(0, len(script)), -20.0)]
    expected = InputsWrapper([make_session()], [doc], [doc], [loudness], [[]], [{}])
    assert result == expected
    assert result.to_xml(0) == f"<{Sch.SPEAK} {Sch._VALUE}='-1'>{xml}</{Sch.SPEAK}>"


def _check_anno(anno: Sch, text: str, val: str, prefix: str = "", suffix: str = ""):
    """Check how `anno` with `text` and `val` is processed with optional context."""
    nlp = load_en_english()
    xml = XMLType(f'{prefix}<{anno} {Sch._VALUE}="{val}">{text}</{anno}>{suffix}')
    InputsWrapper.from_xml(xml, nlp(prefix + text + suffix), make_session())


def test__inputs_wrapper__from_xml__token_annotations():
    """Test `InputsWrapper.from_xml` validates respellings."""
    _check_anno(Sch.RESPELL, "scientific", "SY-uhn-TIH-fihk")
    _check_anno(Sch.RESPELL, "S.C.I.E.N.T.I.F.I.C", "SY-uhn-TIH-fihk")

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "don't", "dont")  # SpaCy considers this two tokens

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "")  # No value

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "", "SY-uhn-TIH-fihk")  # No text

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "", "SY-uhn-TIH-fihk", "scientific")  # No text

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "-SY-uhn-TIH-fihk")  # Invalid prefix

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "SY-uhn-TIH-fihk-")  # Invalid suffix

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "SY uhn TIH fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "SY1uhn2TIH3fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "SY|uhn|TIH|fihk")  # Invalid character

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific", "Sy-uhn-TIH-fihk")  # Mix capitalization

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scientific-process", "SY-uhn-TIH-fihk")  # Multiple words

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "entific", "SY-uhn-TIH-fihk", "sci")  # Prefix

    with pytest.raises(PublicValueError):
        _check_anno(Sch.RESPELL, "scienti", "SY-uhn-TIH-fihk", suffix="fic")  # Suffix


def test__inputs_wrapper__from_xml__span_annotations():
    """Test `InputsWrapper.from_xml` validates span annotations."""
    _check_anno(Sch.LOUDNESS, "scientific process", "-20.0")
    _check_anno(Sch.TEMPO, "scientific process", "0.04")

    # NOTE: Punctuation and spacing may be included in annotation.
    _check_anno(Sch.LOUDNESS, " scientific process ", "-20.0", "Yes", "No")
    _check_anno(Sch.LOUDNESS, ", scientific process.", "-20.0")

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "scientific process", "")  # No value

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "", "-20.0")  # No text

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "", "-20.0", "scientific", "process")  # No text

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "scientific process", "NA")  # No number

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "scientific process", "-1000")  # Too small

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "scientific process", "1000")  # Too big

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "entific process", "-20.0", "sci")  # Prefix

    with pytest.raises(PublicValueError):
        _check_anno(Sch.LOUDNESS, "scientific pro", "-20.0", suffix="cess")  # Suffix

    with pytest.raises(PublicValueError):
        _check_anno(Sch.TEMPO, "scientific process", "-1000")  # Too small

    with pytest.raises(PublicValueError):
        _check_anno(Sch.TEMPO, "scientific process", "1000")  # Too big


def test__inputs_wrapper__from_xml__overlapping_span_anno():
    """Test `InputsWrapper.from_xml` and `InputsWrapper.to_xml` handles annotations that start
    and end at the same index."""
    xml = XMLType(
        f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>This</{Sch.LOUDNESS}>"
        f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'> is a</{Sch.LOUDNESS}>"
    )
    script = xml_to_text(xml)
    inp = InputsWrapper.from_xml(xml, load_en_english()(script), make_session())
    assert inp.to_xml(0) == f"<{Sch.SPEAK} {Sch._VALUE}='-1'>{xml}</{Sch.SPEAK}>"


def test_inputs_wrapper__to_xml__context():
    """Test `InputsWrapper.to_xml` handles context."""
    script = "this is a test"
    doc = load_en_english()(script)
    span = doc[1:-1]
    xml = XMLType(f"<{Sch.LOUDNESS} {Sch._VALUE}='-20.0'>{str(span)}</{Sch.LOUDNESS}>")
    loudness = [(slice(0, len(span.text)), -20.0)]
    inp = InputsWrapper([make_session()], [span], [doc], [loudness], [[]], [{}])
    expected = f"this <{Sch.SPEAK} {Sch._VALUE}='-1'>{xml}</{Sch.SPEAK}> test"
    assert inp.to_xml(0, include_context=True) == expected


def test__get_case():
    """Test `_get_case` on various cases."""
    assert _get_case("A") == Casing.UPPER
    assert _get_case("a") == Casing.LOWER
    assert _get_case("1") == Casing.NO_CASING
    with pytest.raises(AssertionError):
        _get_case("")


def _check_processed(
    nlp: language.Language, processed: Inputs, script: str, doc: SpanDoc, sesh: Session
):
    """Helper function for `test_preprocess`."""
    assert processed.tokens == [list(script.lower())]
    assert processed.seq_meta[0][:2] == [sesh.spkr.label, sesh]
    assert processed.seq_meta_transposed[0] == [sesh.spkr.label]
    assert processed.seq_meta_transposed[1] == [sesh]
    assert len(processed.seq_meta_transposed) == 5
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
    context = [
        Context.SCRIPT_START,  # I
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
        Context.SCRIPT_STOP,  # .
        Context.CONTEXT,
        Context.CONTEXT,  # A
        Context.CONTEXT,  # r
        Context.CONTEXT,  # m
        Context.CONTEXT,  # y
    ]
    assert processed.token_meta[0] == [casing, context]
    assert processed.token_meta_transposed[0] == [casing]
    assert processed.token_meta_transposed[1] == [context]
    assert len(processed.token_meta_transposed) == 2
    seq_vector = torch.tensor([sesh.loudness, sesh.tempo])
    assert torch.equal(processed.seq_vectors, seq_vector.view(1, 2))
    assert processed.token_vector_idx == dict(
        loudness_vector=slice(0, 2),
        loudness_mask=slice(2, 3),
        tempo_vector=slice(3, 5),
        tempo_mask=slice(5, 6),
        word_vector=slice(6, 402),
    )
    vocab: spacy.vocab.Vocab = nlp.vocab
    word_vector_len = nlp.meta["vectors"]["width"]
    # NOTE: The word embeddings are case sensitive.
    word_vectors = [
        torch.from_numpy(vocab["In"].vector).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, word_vector_len),
        torch.from_numpy(vocab["1968"].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, word_vector_len),
        torch.from_numpy(vocab["the"].vector).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, word_vector_len),
        torch.from_numpy(vocab["U.S."].vector).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, word_vector_len),
        torch.from_numpy(vocab["Army"].vector).unsqueeze(0).repeat(4, 1),
    ]
    contextual_word_vector_len = typing.cast(FloatsXd, doc[0].tensor).shape[0]
    contextual_embeds = [
        torch.from_numpy(doc[0].tensor).unsqueeze(0).repeat(2, 1),
        torch.zeros(1, contextual_word_vector_len),
        torch.from_numpy(doc[1].tensor).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, contextual_word_vector_len),
        torch.from_numpy(doc[2].tensor).unsqueeze(0).repeat(3, 1),
        torch.zeros(1, contextual_word_vector_len),
        torch.from_numpy(doc[3].tensor).unsqueeze(0).repeat(4, 1),
        torch.zeros(1, contextual_word_vector_len),
        torch.from_numpy(doc[4].tensor).unsqueeze(0).repeat(4, 1),
    ]
    word_vectors = torch.cat((torch.cat(contextual_embeds), torch.cat(word_vectors)), dim=1)
    assert torch.allclose(processed.get_token_vec("word_vector"), word_vectors)
    _word_vectors = processed.get_token_vec("word_vector", size=400)[:, :, :-4]
    assert torch.allclose(_word_vectors, word_vectors)
    expected = torch.zeros(len(script)).view(1, -1, 1)
    assert torch.equal(processed.get_token_vec("tempo_mask"), expected)
    assert torch.equal(processed.get_token_vec("loudness_mask"), expected)
    expected = torch.ones(len(script)).view(1, -1, 1)
    assert processed.slices[0] == slice(0, len(doc[:-1].text))
    assert processed.max_audio_len[0] == len(doc[:-1].text)
    assert torch.equal(processed.num_tokens[0], torch.tensor(len(doc.text)))
    assert torch.equal(processed.num_sliced_tokens[0], torch.tensor(len(doc[:-1].text)))
    expected = torch.ones(len(doc[:-1].text), dtype=torch.bool)
    assert torch.equal(processed.sliced_tokens_mask[0], expected)


def test_preprocess():
    """Test that `preprocess` handles basic input with no annotations."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    script, sesh = "In 1968 the U.S. Army", make_session()
    doc = nlp(script + " Phone")[:-1]
    inp = InputsWrapper.from_xml(XMLType(doc[:-1].text), doc[:-1], sesh, doc)
    processed = preprocess(inp, lambda t: len(t), identity, identity, identity, identity)
    _check_processed(nlp, processed, script, doc, sesh)
    _check_processed(nlp, processed[0], script, doc, sesh)
    assert len(processed) == 1


def test_preprocess__respelling():
    """Test that `preprocess` handles apostrophes, dashes, initialisms and existing respellings."""
    nlp = load_spacy_nlp(Language.ENGLISH)
    xml = XMLType(
        "<respell value='doh-NT'>don't</respell> <respell value='PEE-puhl'>people</respell> from "
        "EDGE <respell value='KACH'>catch</respell>-the-<respell value='FLOO'>flu</respell>"
    )
    script = "Why don't people from EDGE catch-the-flu?"
    doc = nlp(script)
    inp = InputsWrapper.from_xml(xml, doc[1:-1], make_session(), doc)
    processed = preprocess(inp, lambda t: len(t), identity, identity, identity, identity)
    casing = [
        (Pronun.NORMAL, Casing.UPPER),  # W
        (Pronun.NORMAL, Casing.LOWER),  # h
        (Pronun.NORMAL, Casing.LOWER),  # y
        (Pronun.NORMAL, Casing.NO_CASING),  #
        (Pronun.RESPELLING, Casing.LOWER),  # d
        (Pronun.RESPELLING, Casing.LOWER),  # o
        (Pronun.RESPELLING, Casing.LOWER),  # h
        (Pronun.RESPELLING, Casing.NO_CASING),  # -
        (Pronun.RESPELLING, Casing.UPPER),  # N
        (Pronun.RESPELLING, Casing.UPPER),  # T
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
    context = [
        Context.CONTEXT,  # W
        Context.CONTEXT,  # h
        Context.CONTEXT,  # y
        Context.CONTEXT,  #
        Context.SCRIPT_START,  # d
        Context.SCRIPT,  # o
        Context.SCRIPT,  # h
        Context.SCRIPT,  # -
        Context.SCRIPT,  # N
        Context.SCRIPT,  # T
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
        Context.SCRIPT_STOP,  # O
        Context.CONTEXT,  # ?
    ]
    assert processed.token_meta[0] == [casing, context]

    # NOTE: Double check that respellings were replaced.
    tokens = "doh-nt pee-puhl from edge kach-the-floo"
    assert "".join(processed.tokens[0][processed.slices[0]]) == tokens  # type: ignore

    # NOTE: Double check respellings have the correct word vectors.
    word_vectors = processed.get_token_vec("word_vector")[0]
    expected = torch.from_numpy(nlp.vocab["flu"].vector)
    for idx in range(-len("FLOO?"), -len("?")):
        assert_almost_equal(word_vectors[idx][-expected.shape[0] :], expected)

    # NOTE: Double check regular words have not been affected.
    expected = torch.from_numpy(nlp.vocab["from"].vector)
    for idx in range(len("why doh-NT pee-puhl "), len("why doh-NT pee-puhl from")):
        assert_almost_equal(word_vectors[idx][-expected.shape[0] :], expected)

    # TODO: `from_xml` modifies the underlying `doc` object in instances like this. It shouldn't
    # do that...
    assert doc[1].tensor is not None and sum(doc[1].tensor) != 0
    expected = torch.from_numpy(doc[1].vector)
    for idx in range(len("why "), len("why doh-NT")):
        assert_almost_equal(word_vectors[idx][-expected.shape[0] :], expected)


def test_preprocess__slice_anno():
    """Test that `preprocess` handles annotations along with context and duplicate respellings."""
    xml = XMLType(
        f"<{Sch.LOUDNESS} {Sch._VALUE}='-49.0'>not <{Sch.RESPELL} {Sch._VALUE}='uh-BOWT'>about"
        f"</{Sch.RESPELL}> <{Sch.RESPELL} {Sch._VALUE}='WHAT'>about</{Sch.RESPELL}>"
        f"<{Sch.TEMPO} {Sch._VALUE}='5.0'> </{Sch.TEMPO}>I gain</{Sch.LOUDNESS}>"
    )
    sesh = make_session()
    script = "analysis, not about about I gain and"
    doc = load_spacy_nlp(Language.ENGLISH)(script)
    inp = InputsWrapper.from_xml(xml, doc[2:-1], sesh, doc)
    processed = preprocess(
        inp,
        get_max_audio_len=lambda t: len(t),
        norm_anno_len=lambda f: f * 2,
        norm_anno_loudness=lambda f: f * 3,
        norm_sesh_loudness=lambda f: f * 4,
        norm_tempo=lambda f: f * 5,
    )

    # Loudness vectors
    l_anno_len = len("not uh-BOWT WHAT I gain")
    l_prefix = [0] * len("analysis, ")
    l_suffix = [0] * len(" and")
    result = [
        l_prefix + [-49.0 * 3] * l_anno_len + l_suffix,
        l_prefix + [len("not about about I gain") * 2] * l_anno_len + l_suffix,
    ]
    result = torch.tensor(result, dtype=torch.float)
    assert_almost_equal(processed.get_token_vec("loudness_vector")[0].T, result)

    l_mask = [l_prefix + [1] * l_anno_len + l_suffix]
    result = torch.tensor(l_mask, dtype=torch.float)
    assert_almost_equal(processed.get_token_vec("loudness_mask")[0].T, result)

    # Tempo vectors
    t_anno_len = len(" ")
    t_prefix = [0] * len("analysis, not uh-BOWT WHAT")
    t_suffix = [0] * len("I gain and")
    result = [
        t_prefix + [5 * 5] * t_anno_len + t_suffix,
        t_prefix + [t_anno_len * 2] * t_anno_len + t_suffix,
    ]
    result = torch.tensor(result, dtype=torch.float)
    assert_almost_equal(processed.get_token_vec("tempo_vector")[0].T, result)

    t_mask = [t_prefix + [1] * t_anno_len + t_suffix]
    result = torch.tensor(t_mask, dtype=torch.float)
    assert_almost_equal(processed.get_token_vec("tempo_mask")[0].T, result)
