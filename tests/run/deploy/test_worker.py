import logging
from functools import partial

import hparams
import pytest

import lib
import run
from lib.text import _line_grapheme_to_phoneme
from run.data._loader import Session
from run.data._loader.english import JUDY_BIEBER
from run.deploy.worker import FlaskException, validate_and_unpack
from run.train.spectrogram_model._data import InputEncoder

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def run_around_tests():
    config = {
        lib.text.grapheme_to_phoneme: hparams.HParams(separator=run._config.PHONEME_SEPARATOR),
        InputEncoder.__init__: hparams.HParams(token_separator=run._config.PHONEME_SEPARATOR),
    }
    hparams.add_config(config)
    yield
    hparams.clear_config()


def test_flask_exception():
    """Test `FlaskException` `to_dict` produces the correct dictionary."""
    exception = FlaskException("This is a test", 404, code="NOT_FOUND", payload={"blah": "hi"})
    assert exception.to_dict() == {"blah": "hi", "code": "NOT_FOUND", "message": "This is a test"}


def test_validate_and_unpack():
    """Test `validate_and_unpack` handles all sorts of arguments."""
    speaker = JUDY_BIEBER
    session = Session("sesh")
    script = "This is a test. ABC."
    phonemes = _line_grapheme_to_phoneme([script], separator=run._config.PHONEME_SEPARATOR)[0]
    input_encoder = InputEncoder([script], [phonemes], [speaker], [(speaker, session)])
    speaker_id = input_encoder.speaker_encoder.token_to_index[speaker]
    speaker_id_to_speaker = {0: (speaker, session)}
    args = {"speaker_id": speaker_id, "text": script, "api_key": "abc"}
    api_keys = ["abc"]
    nlp = lib.text.load_en_core_web_sm(disable=("parser", "ner"))
    validate_ = partial(validate_and_unpack, nlp=nlp, input_encoder=input_encoder)
    with pytest.raises(FlaskException):
        validate_({})

    with pytest.raises(FlaskException):  # API key must be a string
        validate_({**args, "api_key": 1}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key length must be larger than the minimum
        validate_({**args, "api_key": "ab"}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key length must be smaller than the maximum
        validate_({**args, "api_key": "abcd"}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key must be in `api_keys`
        validate_({**args, "api_key": "cba"}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` argument missing
        validate_({"api_key": "abc", "speaker_id": 0}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` argument missing
        validate_({"api_key": "abc", "text": script}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_({**args, "speaker_id": "blah"}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_({**args, "speaker_id": 2.1}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` must be smaller than `max_chars`
        request_args = {**args, "text": "a" * 20000}
        validate_(request_args, api_keys=api_keys, max_chars=1000)

    with pytest.raises(FlaskException):  # `text` must be a string
        validate_({**args, "text": 1.0}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` must be not empty
        validate_({**args, "text": ""}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be in `input_encoder`
        validate_({**args, "speaker_id": 2 ** 31}, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be positive
        validate_({**args, "speaker_id": -1}, api_keys=api_keys)

    with pytest.raises(FlaskException, match=r".*cannot contain these characters: i, j,*"):
        # NOTE: "wɛɹɹˈɛvɚ kˌoːɹzənjˈuːski" should contain phonemes that are not already in
        # mock `input_encoder`.
        validate_({**args, "text": "wherever korzeniewski"}, api_keys=api_keys)

    # `text` gets normalized and `speaker` is dereferenced
    request_args = {**args, "text": "á😁"}
    encoded = validate_(
        request_args, api_keys=api_keys, speaker_id_to_speaker=speaker_id_to_speaker
    )
    decoded = input_encoder.decode(encoded)
    # NOTE: The emoji is removed because there is no unicode equivilent.
    assert decoded.graphemes == "a"
    assert decoded.speaker == speaker
    assert decoded.session == (speaker, session)
