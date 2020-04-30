import pytest
import torch

from src.service.worker import FlaskException
from src.service.worker import load_checkpoints
from src.service.worker import stream_text_to_speech_synthesis
from src.service.worker import validate_and_unpack
from tests._utils import get_tts_mocks
from torchnlp.random import fork_rng


def test_flask_exception():
    exception = FlaskException('This is a test', 404, code='NOT_FOUND', payload={'blah': 'hi'})
    assert exception.to_dict() == {'blah': 'hi', 'code': 'NOT_FOUND', 'message': 'This is a test'}


def test_load_checkpoints():
    mocks = get_tts_mocks()
    signal_model, spectrogram_model, input_encoder = load_checkpoints(
        mocks['spectrogram_model_checkpoint'].path, mocks['signal_model_checkpoint'].path)
    assert type(mocks['signal_model']) == type(signal_model)
    assert type(mocks['spectrogram_model']) == type(spectrogram_model)
    assert type(mocks['input_encoder']) == type(input_encoder)


def test_stream_text_to_speech_synthesis():
    with fork_rng(seed=123):
        mocks = get_tts_mocks()
        example = mocks['dev_dataset'][0]
        generator, file_size = stream_text_to_speech_synthesis(mocks['signal_model'].eval(),
                                                               mocks['spectrogram_model'].eval(),
                                                               example.text, example.speaker)
        assert len(b''.join([s for s in generator()])) == 1082


def test_validate_and_unpack():
    mocks = get_tts_mocks()
    input_encoder = mocks['input_encoder']
    example = mocks['dev_dataset'][0]
    speaker = example.speaker
    speaker_id = input_encoder.speaker_encoder.token_to_index[example.speaker]
    text = example.text
    speaker_id_to_speaker = {
        i: t for i, t in enumerate(input_encoder.speaker_encoder.index_to_token)
    }
    args = {'speaker_id': speaker_id, 'text': text, 'api_key': 'abc'}
    api_keys = ['abc']

    with pytest.raises(FlaskException):
        validate_and_unpack({}, input_encoder)

    with pytest.raises(FlaskException):  # API key must be a string
        validate_and_unpack({**args, 'api_key': 1}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key length must be larger than the minimum
        validate_and_unpack({**args, 'api_key': 'ab'}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key length must be smaller than the maximum
        validate_and_unpack({**args, 'api_key': 'abcd'}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # API key must be in `api_keys`
        validate_and_unpack({**args, 'api_key': 'cba'}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` argument missing
        validate_and_unpack({'api_key': 'abc', 'speaker_id': 0}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` argument missing
        validate_and_unpack({'api_key': 'abc', 'text': text}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_and_unpack({**args, 'speaker_id': 'blah'}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        validate_and_unpack({**args, 'speaker_id': 2.1}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` must be smaller than `max_characters`
        request_args = {**args, 'text': 'a' * 20000}
        validate_and_unpack(request_args, input_encoder, api_keys=api_keys, max_characters=1000)

    with pytest.raises(FlaskException):  # `text` must be a string
        validate_and_unpack({**args, 'text': 1.0}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `text` must be not empty
        validate_and_unpack({**args, 'text': ''}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be in `input_encoder`
        validate_and_unpack({**args, 'speaker_id': 2**31}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException):  # `speaker_id` must be positive
        validate_and_unpack({**args, 'speaker_id': -1}, input_encoder, api_keys=api_keys)

    with pytest.raises(FlaskException, match=r".*cannot contain these characters: i, o,*"):
        # NOTE: "wɛɹɹˈɛvɚ kˌoːɹzənjˈuːski" should contain phonemes that are not already in
        # mock `input_encoder`.
        validate_and_unpack(
            {
                **args, 'text': 'wherever korzeniewski'
            },
            input_encoder,
            api_keys=api_keys,
        )

    # `text` gets normalized and `speaker` is dereferenced
    request_args = {**args, 'text': 'é😁'}
    result_text, result_speaker = validate_and_unpack(
        request_args, input_encoder, api_keys=api_keys, speaker_id_to_speaker=speaker_id_to_speaker)
    assert torch.is_tensor(result_text)
    assert result_text.shape[0] == 2
    # NOTE: The emoji is removed because there is no unicode equivilent.
    assert input_encoder.decode(
        (result_text, result_speaker)) == ('ˈ' + input_encoder.delimiter + 'iː', speaker)
    assert torch.is_tensor(result_speaker)
