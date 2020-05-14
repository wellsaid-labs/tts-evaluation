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

TODO: Apply `exponential_moving_parameter_average` before running locally, for best performance.

The cons in summary are that the client cannot manage there own state due to the immaturity of the
web audio api; therefore, the server must manage it via some database.

Example (Flask):

      $ PYTHONPATH=. YOUR_SPEECH_API_KEY=123 python -m src.service.worker

Example (Gunicorn):

      $ YOUR_SPEECH_API_KEY=123 gunicorn src.service.worker:app --timeout=3600 --env='GUNICORN=1'
"""
from queue import SimpleQueue

import gc
import os
import subprocess
import sys
import threading
import warnings

from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from flask import send_file
from flask import send_from_directory
from hparams import configurable
from hparams import HParam

import en_core_web_sm
import torch

from src.environment import set_basic_logging_config
from src.hparams import set_hparams
from src.service.worker_config import SIGNAL_MODEL_CHECKPOINT_PATH
from src.service.worker_config import SPEAKER_ID_TO_SPEAKER
from src.service.worker_config import SPECTROGRAM_MODEL_CHECKPOINT_PATH
from src.signal_model import generate_waveform
from src.spectrogram_model.input_encoder import InvalidSpeakerValueError
from src.spectrogram_model.input_encoder import InvalidTextValueError
from src.utils import Checkpoint
from src.utils import get_functions_with_disk_cache

if 'NUM_CPU_THREADS' in os.environ:
    torch.set_num_threads(int(os.environ['NUM_CPU_THREADS']))

# NOTE: Flask documentation requests that logging is configured before `app` is created.
set_basic_logging_config()

app = Flask(__name__)

app.logger.info('PyTorch version: %s', torch.__version__)
app.logger.info('Found MKL: %s', torch.backends.mkl.is_available())
app.logger.info('Threads: %s', torch.get_num_threads())

DEVICE = torch.device('cpu')
API_KEY_SUFFIX = '_SPEECH_API_KEY'
API_KEYS = set([v for k, v in os.environ.items() if API_KEY_SUFFIX in k])
SIGNAL_MODEL = None
SPECTROGRAM_MODEL = None
INPUT_ENCODER = None
SPACY = None


class FlaskException(Exception):
    """
    Inspired by http://flask.pocoo.org/docs/1.0/patterns/apierrors/

    Args:
        message (str)
        status_code (int): An HTTP response status codes.
        code (str): A string code.
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


@app.before_first_request
def before_first_request():
    # NOTE: Remove this warning after this is fixed...
    # https://github.com/PetrochukM/HParams/issues/6
    warnings.filterwarnings(
        'ignore',
        module=r'.*hparams',
        message=r'.*The decorator was not executed immediately before*')
    # NOTE: Ensure that our cache doesn't grow while the server is running.
    for function in get_functions_with_disk_cache():
        function.use_disk_cache(False)


def _enqueue(out, queue):
    """ Enqueue all lines from a file-like object to `queue`.

    Args:
        out (file-like object)
        queue (Queue)
    """
    for line in iter(out.readline, b''):
        queue.put(line)


def _dequeue(queue):
    """ Dequeue all items from `queue`.

    Args:
        queue (Queue)
    """
    while not queue.empty():
        yield queue.get_nowait()


@configurable
def stream_text_to_speech_synthesis(text,
                                    speaker,
                                    signal_model,
                                    spectrogram_model,
                                    sample_rate=HParam()):
    """ Helper function for starting a speech synthesis stream.

    TODO: If a loud sound is created, cut off the stream or consider rerendering.
    TODO: Consider logging various events to stackdriver, to keep track.

    Args:
        text (str)
        speaker (src.datasets.Speaker)
        signal_model (torch.nn.Module)
        spectrogram_model (torch.nn.Module)
        sample_rate (int)

    Returns:
        (callable): Callable that returns a generator incrementally returning a WAV file.
        (int): Number of bytes to be returned in total by the generator.
    """

    def get_spectrogram():
        for item in spectrogram_model(text, speaker, is_generator=True):
            # [num_frames, batch_size (optional), frame_channels] →
            # [batch_size (optional), num_frames, frame_channels]
            gc.collect()
            yield item[1].transpose(0, 1) if item[1].dim() == 3 else item[1]

    # TODO: Add a timeout in case the client is keeping the connection alive and not consuming
    # any data.
    def response():
        # NOTE: Inspired by:
        # https://stackoverflow.com/questions/375427/non-blocking-read-on-a-subprocess-pipe-in-python
        with torch.no_grad():
            try:
                command = (
                    'ffmpeg -f f32le -acodec pcm_f32le -ar %d -ac 1 -i pipe: -f mp3 -b:a 192k pipe:'
                    % sample_rate).split()
                pipe = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=sys.stdout.buffer)
                queue = SimpleQueue()
                thread = threading.Thread(target=_enqueue, args=(pipe.stdout, queue), daemon=True)
                thread.start()
                app.logger.info('Generating waveform...')
                for waveform in generate_waveform(signal_model, get_spectrogram()):
                    pipe.stdin.write(waveform.cpu().numpy().tobytes())
                    yield from _dequeue(queue)
                pipe.stdin.close()
                pipe.wait()
                thread.join()
                pipe.stdout.close()
                yield from _dequeue(queue)
                app.logger.info('Finished generating waveform.')
            # NOTE: `Exception` does not catch `GeneratorExit`.
            # https://stackoverflow.com/questions/18982610/difference-between-except-and-except-exception-as-e-in-python
            except:
                pipe.stdin.close()
                pipe.wait()
                thread.join()
                pipe.stdout.close()
                app.logger.info('Aborted waveform generation.')
                raise

    return response


def validate_and_unpack(request_args,
                        input_encoder,
                        max_characters=100000,
                        api_keys=API_KEYS,
                        speaker_id_to_speaker=SPEAKER_ID_TO_SPEAKER,
                        **kwargs):
    """ Validate and unpack the request object.

    Args:
        request_args (dict) {
          speaker_id (int or str)
          text (str)
          api_key (str)
        }
        input_encoder (src.spectrogram_model.InputEncoder): Spectrogram model input encoder.
        max_characters (int, optional)
        api_keys (list of str, optional)
        speaker_id_to_speaker (dict, optional)
        **kwargs: Key-word arguments passed to `input_encoder.encode`.

    Returns:
        text (str)
        speaker (src.datasets.Speaker)
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

    speaker = speaker_id_to_speaker[speaker_id]

    gc.collect()

    try:
        app.logger.info('Encoding text: %s', text)
        text, speaker = input_encoder.encode((text, speaker), **kwargs)
    except InvalidSpeakerValueError as error:
        raise FlaskException(str(error), code='INVALID_SPEAKER_ID')
    except InvalidTextValueError as error:
        raise FlaskException(str(error), code='INVALID_TEXT')

    return text, speaker


@app.route('/healthy', methods=['GET'])
def healthy():
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
    validate_and_unpack(request_args, INPUT_ENCODER, get_spacy_model=lambda: SPACY)
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
        `audio/mpeg` streamed in chunks given that the arguments are valid.
    """
    request_args = request.get_json() if request.method == 'POST' else request.args
    text, speaker = validate_and_unpack(request_args, INPUT_ENCODER, get_spacy_model=lambda: SPACY)
    return Response(
        stream_text_to_speech_synthesis(text, speaker, SIGNAL_MODEL, SPECTROGRAM_MODEL)(),
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        },
        mimetype='audio/mpeg')


@app.route('/')
def index():
    return send_file('public/index.html')


@app.route('/<path:path>')
def send_static(path):
    return send_from_directory('public', path)


if __name__ == "__main__" or 'GUNICORN' in os.environ:
    set_hparams()

    # NOTE: These models are cached globally to enable sharing between processes, learn more:
    # https://github.com/benoitc/gunicorn/issues/2007
    spectrogram_model_checkpoint = Checkpoint.from_path(SPECTROGRAM_MODEL_CHECKPOINT_PATH, DEVICE)
    SPECTROGRAM_MODEL = spectrogram_model_checkpoint.model.eval()
    INPUT_ENCODER = spectrogram_model_checkpoint.input_encoder
    app.logger.info('Loaded speakers: %s', INPUT_ENCODER.speaker_encoder.vocab)

    signal_model_checkpoint = Checkpoint.from_path(SIGNAL_MODEL_CHECKPOINT_PATH, DEVICE)
    SIGNAL_MODEL = signal_model_checkpoint.model.eval()

    SPACY = en_core_web_sm.load(disable=['parser', 'ner'])
    app.logger.info('Loaded spaCy.')

    # NOTE: In order to support copy-on-write, we freeze all the objects tracked by `gc`, learn
    # more:
    # https://docs.python.org/3/library/gc.html#gc.freeze
    # https://instagram-engineering.com/copy-on-write-friendly-python-garbage-collection-ad6ed5233ddf
    # https://github.com/benoitc/gunicorn/issues/1640
    gc.freeze()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
