import pytest

from src.service.worker import FlaskException
from src.service.worker import load_checkpoints
from src.service.worker import stream_text_to_speech_synthesis
from src.service.worker import validate_and_unpack
from tests._utils import get_tts_mocks


def test_load_checkpoints():
    mocks = get_tts_mocks()
    signal_model, spectrogram_model, input_encoder = load_checkpoints(
        mocks['spectrogram_model_checkpoint'].path, mocks['signal_model_checkpoint'].path)
    assert type(mocks['signal_model'].to_inferrer()) == type(signal_model)
    assert type(mocks['spectrogram_model']) == type(spectrogram_model)
    assert type(mocks['input_encoder']) == type(input_encoder)


def test_stream_text_to_speech_synthesis():
    mocks = get_tts_mocks()
    example = mocks['dev_dataset'][0]
    generator, file_size = stream_text_to_speech_synthesis(mocks['signal_model'].to_inferrer(),
                                                           mocks['spectrogram_model'].eval(),
                                                           mocks['input_encoder'], example.text,
                                                           example.speaker)
    file_contents = b''.join([s for s in generator()])
    assert len(file_contents) == file_size


def test_validate_and_unpack():
    mocks = get_tts_mocks()
    input_encoder = mocks['input_encoder']
    example = mocks['dev_dataset'][0]
    speaker = example.speaker
    speaker_id = input_encoder.speaker_encoder.stoi[example.speaker]
    text = example.text
    speaker_id_to_speaker_id = {i: i for i in input_encoder.speaker_encoder.stoi.values()}

    with pytest.raises(FlaskException):
        validate_and_unpack({}, input_encoder)

    with pytest.raises(FlaskException):  # API key must be a string
        validate_and_unpack({'api_key': 1}, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # API key length must be larger than the minimum
        validate_and_unpack({'api_key': 'ab'}, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # API key length must be smaller than the maximum
        validate_and_unpack({'api_key': 'abcd'}, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # API key must be in `api_keys`
        validate_and_unpack({'api_key': 'cba'}, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `text` argument missing
        request_args = {'api_key': 'abc', 'speaker_id': 0}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `speaker_id` argument missing
        request_args = {'api_key': 'abc', 'text': text}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        request_args = {'api_key': 'abc', 'speaker_id': 'blah', 'text': text}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `speaker_id` must be an integer
        request_args = {'api_key': 'abc', 'speaker_id': 2.1, 'text': text}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `text` must be smaller than `max_characters`
        request_args = {'api_key': 'abc', 'speaker_id': speaker_id, 'text': 'a' * 20000}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'], max_characters=1000)

    with pytest.raises(FlaskException):  # `text` must be a string
        request_args = {'api_key': 'abc', 'speaker_id': speaker_id, 'text': 1.0}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `text` must be not empty
        request_args = {'api_key': 'abc', 'speaker_id': speaker_id, 'text': ''}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `speaker_id` must be in `input_encoder`
        request_args = {'api_key': 'abc', 'speaker_id': 2**31, 'text': text}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    with pytest.raises(FlaskException):  # `speaker_id` must be positive
        request_args = {'api_key': 'abc', 'speaker_id': -1, 'text': text}
        validate_and_unpack(request_args, input_encoder, api_keys=['abc'])

    # `text` gets normalized and `speaker` is dereferenced
    request_args = {'api_key': 'abc', 'speaker_id': speaker_id, 'text': 'é😁'}
    result_text, result_speaker = validate_and_unpack(
        request_args,
        input_encoder,
        api_keys=['abc'],
        speaker_id_to_speaker_id=speaker_id_to_speaker_id)
    assert result_text == 'e'  # NOTE: The emoji is removed because there is no unicode equivilent.
    assert result_speaker == speaker
