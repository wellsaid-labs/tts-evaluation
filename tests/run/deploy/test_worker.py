import logging
from functools import partial

import config as cf
import pytest

import lib
from run._models.spectrogram_model.inputs import Token
from run.data._loader import Language, Session
from run.data._loader.english.m_ailabs import JUDY_BIEBER
from run.deploy.worker import FlaskException, validate_and_unpack
from tests.run._utils import make_mock_tts_package

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True, scope="module")
def run_around_tests():
    """Set a basic configuration."""
    yield
    cf.purge()


def test_flask_exception():
    """Test `FlaskException` `to_dict` produces the correct dictionary."""
    exception = FlaskException("This is a test", 404, code="NOT_FOUND", payload={"blah": "hi"})
    assert exception.to_dict() == {"blah": "hi", "code": "NOT_FOUND", "message": "This is a test"}


def test_validate_and_unpack():
    """Test `validate_and_unpack` handles all sorts of arguments."""
    sesh = Session((JUDY_BIEBER, "sesh"))
    script = "This is a exposé. ABC."
    _, package = make_mock_tts_package()

    # TODO: Refactor this, so, it's a bit easier to create a `package` with a preset vocabulary.
    # Or the vocabulary is based off a dataset which can be used, also? That handles special
    # characters, also.
    package.spec_model.token_embed.update_tokens(list(script.lower()) + [Token.delim])
    package.spec_model.speaker_embed.update_tokens([sesh[0].label])
    package.spec_model.session_embed.update_tokens([sesh])
    package.spec_model.dialect_embed.update_tokens([sesh[0].dialect])
    package.spec_model.style_embed.update_tokens([sesh[0].style])
    package.spec_model.language_embed.update_tokens([sesh[0].language])
    package.signal_model.speaker_embed.update_tokens([sesh[0].label])
    package.signal_model.session_embed.update_tokens([sesh])

    language_to_spacy = {Language.ENGLISH: lib.text.load_en_core_web_sm()}
    speaker_id = 0
    speaker_id_to_session = {speaker_id: sesh}
    args = {"speaker_id": speaker_id, "text": script}
    validate_ = partial(
        validate_and_unpack,
        tts=package,
        language_to_spacy=language_to_spacy,
        speaker_id_to_session=speaker_id_to_session,
    )

    # TODO: Ensure the right message is printed.

    with pytest.raises(FlaskException):
        validate_({})  # type: ignore

    with pytest.raises(FlaskException):  # `text` argument missing
        validate_({"speaker_id": 0})  # type: ignore

    with pytest.raises(FlaskException):  # `speaker_id` argument missing
        validate_({"text": script})  # type: ignore

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_({**args, "speaker_id": "blah"})  # type: ignore

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_({**args, "speaker_id": 2.1})  # type: ignore

    with pytest.raises(FlaskException):  # `text` must be smaller than `max_chars`
        request_args = {**args, "text": "a" * 20000}
        validate_(request_args, max_chars=1000)  # type: ignore

    with pytest.raises(FlaskException):  # `text` must be a string
        validate_({**args, "text": 1.0})  # type: ignore

    with pytest.raises(FlaskException):  # `text` must be not empty
        validate_({**args, "text": ""})  # type: ignore

    with pytest.raises(FlaskException):  # `speaker_id` must be in `speaker_id_to_session`
        validate_({**args, "speaker_id": 2 ** 31})  # type: ignore

    with pytest.raises(FlaskException):  # `speaker_id` must be positive
        validate_({**args, "speaker_id": -1})  # type: ignore

    with pytest.raises(FlaskException, match=r".*cannot contain these characters: r, v, w*"):
        # NOTE: Should contain graphemes that are not already in mock `package.spec_model`.
        validate_({**args, "text": "wherever"})  # type: ignore

    # `text` gets normalized and `speaker` is dereferenced
    request_args = {**args, "text": "exposé😁 ::a-B::"}
    encoded = validate_(request_args)  # type: ignore
    # NOTE: The emoji is removed because there is no unicode equivilent.
    # TODO: Should this special notation be configured?
    assert str(encoded[0].doc[0]) == "exposé |\\a\\B\\|"
