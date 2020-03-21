""" Run a web service with the spectrogram and signal models.

During designing this API, we considered the requirements of a website.

LEARNINGS:
- `HTMLAudioElement` requires it's `src` property to be set with a url; Otherwise, one must use
  `AudioContext.createMediaStreamSource()` (Learn more:
  https://developers.google.com/web/fundamentals/media/mse/basics) which does not support hardly any
  media types. Critically, it does not support WAV files (Learn more:
  https://github.com/w3c/media-source/issues/55). However, it does support FMP4 and MP3 files. Those
  formats are difficult to transcribe to, from fragmented raw PCM.
- Web Audio API supports play back of `AudioBuffer`s. However, it does not have support abstractions
  of `AudioBuffer` lists. Creating the abstraction would require significant effort similar to:
  https://github.com/samirkumardas/pcm-player
- Another option would be use a player built on top of flash; however, the options for flash
  audio players today, is small.
- `HTMLAudioElement` does not buffer metadata until multiple seconds of the audio data have been
  submitted; therefore, until the `loadeddata` there are no affordances for the client.

With this ecosystem, the only simple solution is to stream WAV files directly.

CONS:
- No additional context can be returned to the client. For example, this prevents us from sending
  the model hidden state used by the client to restart generation.
- The request for the stream must be a GET request. This prevents us, for example, from sending a
  Spectrogram used to condition the speech synthesis.

The cons in summary are that the client cannot manage there own state due to the immaturity of the
web audio api; therefore, the server must manage it via some database.

Example:
      $ PYTHONPATH=. YOUR_SPEECH_API_KEY=123 python -m src.service.worker
"""
from functools import lru_cache

import os
import sys
import unidecode

from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from flask import send_file
from flask import send_from_directory

import torch

from src.audio import build_wav_header
from src.environment import set_basic_logging_config
from src.hparams import set_hparams
from src.service.worker_config import SIGNAL_MODEL_CHECKPOINT_PATH
from src.service.worker_config import SPEAKER_ID_TO_SPEAKER
from src.service.worker_config import SPECTROGRAM_MODEL_CHECKPOINT_PATH
from src.signal_model import generate_waveform
from src.utils import Checkpoint

# NOTE: Flask documentation requests that logging is configured before `app` is created.
set_basic_logging_config()

app = Flask(__name__)
DEVICE = torch.device('cpu')
NO_CACHE_HEADERS = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
}
API_KEY_SUFFIX = '_SPEECH_API_KEY'
API_KEYS = set([v for k, v in os.environ.items() if API_KEY_SUFFIX in k])


@lru_cache()
def load_checkpoints(spectrogram_model_checkpoint_path=SPECTROGRAM_MODEL_CHECKPOINT_PATH,
                     signal_model_checkpoint_path=SIGNAL_MODEL_CHECKPOINT_PATH):
    """
    Args:
        spectrogram_model_checkpoint_path (str)
        signal_model_checkpoint_path (str)

    Returns:
        signal_model (torch.nn.Module)
        spectrogram_model (torch.nn.Module)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
    """
    if 'NUM_CPU_THREADS' in os.environ:
        torch.set_num_threads(int(os.environ['NUM_CPU_THREADS']))

    app.logger.info('PyTorch version: %s', torch.__version__)
    app.logger.info('Found MKL: %s', torch.backends.mkl.is_available())
    app.logger.info('Threads: %s', torch.get_num_threads())

    set_hparams()

    spectrogram_model_checkpoint = Checkpoint.from_path(
        spectrogram_model_checkpoint_path, device=DEVICE)
    signal_model_checkpoint = Checkpoint.from_path(signal_model_checkpoint_path, device=DEVICE)

    spectrogram_model = spectrogram_model_checkpoint.model
    input_encoder = spectrogram_model_checkpoint.input_encoder
    app.logger.info('Loading speakers: %s', input_encoder.speaker_encoder.vocab)
    signal_model = signal_model_checkpoint.model

    return signal_model, spectrogram_model.eval(), input_encoder


class FlaskException(Exception):
    """
    Inspired by http://flask.pocoo.org/docs/1.0/patterns/apierrors/

    Args:
        message (str)
        status_code (int)
        payload (dict): Additional context to send.
    """

    def __init__(self, message, status_code=400, code='BAD_REQUEST', payload=None):
        super().__init__(self, message)

        self.message = message
        self.status_code = status_code
        self.payload = payload
        self.code = code

    def to_dict(self):
        response = dict(self.payload or ())
        response['message'] = self.message
        response['code'] = self.code
        app.logger.info('Responding with warning: %s', self.message)
        return response


@app.errorhandler(FlaskException)
def handle_invalid_usage(error):  # Register an error response
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


def stream_text_to_speech_synthesis(signal_model, spectrogram_model, input_encoder, text, speaker):
    """ Helper function for starting a speech synthesis stream.

    Args:
        signal_model (torch.nn.Module)
        spectrogram_model (torch.nn.Module)
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        text (str)
        speaker (src.datasets.Speaker)

    Returns:
        (callable): Callable that returns a generator incrementally returning a WAV file.
        (int): Number of bytes to be returned in total by the generator.
    """
    # Compute spectrogram
    text, speaker = input_encoder.encode((text, speaker))

    app.logger.info('Generating spectrogram...')
    with torch.no_grad():
        _, spectrogram, _, _, _, is_max_frames = spectrogram_model(text, speaker, use_tqdm=True)
    app.logger.info('Generated spectrogram of shape %s for text of shape %s.', spectrogram.shape,
                    text.shape)

    if is_max_frames:
        # NOTE: Status code 508 is "The server detected an infinite loop while processing the
        # request". Learn more here: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
        raise FlaskException('Failed to render, try again.', status_code=508, code='RENDER_FAILED')

    # TODO: If a loud sound is created, cut off the stream or consider rerendering.
    # TODO: Consider logging various events to stackdriver, to keep track.

    app.logger.info('Generating waveform header...')
    upscale_factor = signal_model.upscale_factor
    wav_header, wav_file_size = build_wav_header(upscale_factor * spectrogram.shape[0])

    def response():
        """ Generator incrementally generating a WAV file.
        """
        try:
            assert sys.byteorder == 'little', 'Ensure byte order is of little-endian format.'
            yield wav_header
            app.logger.info('Generating waveform...')
            for waveform in generate_waveform(signal_model, spectrogram):
                waveform = waveform.numpy()
                app.logger.info('Waveform shape %s', waveform.shape)
                yield waveform.tostring()
            app.logger.info('Finished generating waveform.')
        # NOTE: Flask may abort this generator if the underlying request aborts.
        except Exception as error:
            app.logger.warning('Finished generating waveform with an exception.', exc_info=True)
            raise error

    return response, wav_file_size


def validate_and_unpack(request_args,
                        input_encoder,
                        max_characters=10000,
                        api_keys=API_KEYS,
                        speaker_id_to_speaker=SPEAKER_ID_TO_SPEAKER):
    """ Validate and unpack the request object.

    Args:
        args (dict) {
          speaker_id (int or str)
          text (str)
          api_key (str)
        }
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        max_characters (int, optional)
        api_keys (list of str, optional)
        speaker_id_to_speaker (dict, optional)

    Returns:
        speaker (src.datasets.Speaker)
        text (str)
        api_key (str)
    """
    if 'api_key' not in request_args:
        raise FlaskException('API key was not provided.', status_code=401, code='MISSING_ARGUMENT')

    # TODO: Consider using the authorization header instead of a parameter ``api_key``.
    api_key = request_args.get('api_key')
    min_api_key_length = min([len(key) for key in api_keys])
    max_api_key_length = min([len(key) for key in api_keys])

    if not (isinstance(api_key, str) and len(api_key) >= min_api_key_length and
            len(api_key) <= max_api_key_length):
        raise FlaskException(
            'API key must be a string between %d and %d characters.' %
            (min_api_key_length, max_api_key_length),
            status_code=401,
            code='INVALID_API_KEY')

    if api_key not in api_keys:
        raise FlaskException('API key is not valid.', status_code=401, code='INVALID_API_KEY')

    if not ('speaker_id' in request_args and 'text' in request_args):
        raise FlaskException(
            'Must call with keys `speaker_id` and `text`.', code='MISSING_ARGUMENT')

    speaker_id = request_args.get('speaker_id')
    text = request_args.get('text')

    if not isinstance(speaker_id, (str, int)):
        raise FlaskException(
            'Speaker ID must be either an integer or string.', code='INVALID_SPEAKER_ID')

    if isinstance(speaker_id, str) and not speaker_id.isdigit():
        raise FlaskException(
            'Speaker ID string must only consist of the symbols 0 - 9.', code='INVALID_SPEAKER_ID')

    speaker_id = int(speaker_id)

    if not (isinstance(text, str) and len(text) < max_characters and len(text) > 0):
        raise FlaskException(
            'Text must be a string under %d characters and more than 0 characters.' %
            max_characters,
            code='INVALID_TEXT_LENGTH_EXCEEDED')

    if not (speaker_id <= max(speaker_id_to_speaker.keys()) and
            speaker_id >= min(speaker_id_to_speaker.keys())):
        raise FlaskException(
            'Speaker ID must be an integer between %d and %d.' %
            (min(speaker_id_to_speaker.keys()), max(speaker_id_to_speaker.keys())),
            code='INVALID_SPEAKER_ID')

    # NOTE: Normalize text similar to the normalization during dataset creation.
    text = unidecode.unidecode(text)
    input_encoder.text_encoder.enforce_reversible = False
    preprocessed_text = input_encoder._preprocess(text)
    processed_text = input_encoder.text_encoder.decode(
        input_encoder.text_encoder.encode(preprocessed_text))
    if processed_text != preprocessed_text:
        improper_characters = set(preprocessed_text).difference(
            set(input_encoder.text_encoder.vocab))
        improper_characters = ', '.join(sorted(list(improper_characters)))
        raise FlaskException(
            'Text cannot contain these characters: %s' % improper_characters, code='INVALID_TEXT')

    return text, speaker_id_to_speaker[speaker_id]


@app.route('/healthy', methods=['GET'])
def healthy():
    load_checkpoints()  # Healthy iff ``load_checkpoints`` succeeds and this route succeeds.
    return 'ok'


# NOTE: `/api/speech_synthesis/v1/` was added for backward compatibility.


@app.route('/api/speech_synthesis/v1/text_to_speech/input_validated', methods=['GET', 'POST'])
@app.route('/api/text_to_speech/input_validated', methods=['GET', 'POST'])
def get_input_validated():
    """ Validate the input to our text-to-speech endpoint before making a stream request.

    NOTE: The API splits the validation responsibility from the streaming responsibility. During
    streaming, we are unable to access any error codes generated by the validation script.
    NOTE: The API supports both GET and POST requests. GET and POST requests have different
    tradeoffs, GET allows for streaming with <audio> elements while POST allows more than 2000
    characters of data to be passed.

    Args:
        speaker_id (str)
        text (str)
        api_key (str): Security token sent on behalf of client to ensure authenticity of the
            request.

    Returns:
        Response with status 200 if the arguments are valid; Otherwise, returning a
        `FlaskException`.
    """
    request_args = request.get_json() if request.method == 'POST' else request.args
    input_encoder = load_checkpoints()[2]
    validate_and_unpack(request_args, input_encoder)
    return jsonify({'message': 'OK'})


@app.route('/api/speech_synthesis/v1/text_to_speech/stream', methods=['GET', 'POST'])
@app.route('/api/text_to_speech/stream', methods=['GET', 'POST'])
def get_stream():
    """ Get speech given `text` and `speaker`.

    Args:
        speaker_id (str)
        text (str)
        api_key (str): Security token sent on behalf of client to ensure authenticity of the
            request.

    Returns:
        `audio/wav` streamed in chunks given that the arguments are valid.
    """
    request_args = request.get_json() if request.method == 'POST' else request.args
    signal_model, spectrogram_model, input_encoder = load_checkpoints()
    text, speaker = validate_and_unpack(request_args, input_encoder)
    response, content_length = stream_text_to_speech_synthesis(signal_model, spectrogram_model,
                                                               input_encoder, text, speaker)
    headers = NO_CACHE_HEADERS.copy()
    headers['Content-Length'] = content_length
    return Response(response(), headers=headers, mimetype='audio/wav')


@app.route('/')
def index():
    return send_file('public/index.html')


@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('public', path)


if __name__ == "__main__":
    load_checkpoints()  # Cache checkpoints on worker start.
    app.run(host='0.0.0.0', port=8000, debug=True)
