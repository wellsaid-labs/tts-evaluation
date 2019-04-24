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
  the WaveRNN hidden state used by the client to restart generation.
- The request for the stream must be a GET request. This prevents us, for example, from sending a
  Spectrogram used to condition the speech synthesis.

The cons in summary are that the client cannot manage there own state due to the immaturity of the
web audio api; therefore, the server must manage it via some database.

Example:
      $ export PYTHONPATH=.; python3 -m src.service.serve;

Example:
      $ gunicorn src.service.serve:app;

TODO: Write tests for this module.
"""
from functools import lru_cache

import logging
import os
import pathlib
import sys

from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import request
from flask import Response

import torch

from src.audio import build_wav_header
from src.audio import combine_signal
# NOTE: `src.datasets.constants` to not import all `src.datasets` dependencies
from src.datasets.constants import Speaker
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import set_basic_logging_config

app = Flask(__name__)
logger = logging.getLogger(__name__)
set_basic_logging_config()
load_dotenv()
DEVICE = torch.device('cpu')
NO_CACHE_HEADERS = {
    'Cache-Control': 'no-cache, no-store, must-revalidate',
    'Pragma': 'no-cache',
    'Expires': '0'
}
API_KEY_SUFFIX = '_SPEECH_API_KEY'
API_KEYS = set([v for k, v in os.environ.items() if API_KEY_SUFFIX in k])

# TODO: Upload the models to a bucket online, so that they can be downloaded anywhere at anytime.
SPECTROGRAM_MODEL_CHECKPOINT_PATH = pathlib.Path(
    'experiments/spectrogram_model/jan_12/00:51:22/checkpoints/1548398998/step_213156.pt')
SIGNAL_MODEL_CHECKPOINT_PATH = pathlib.Path(
    'experiments/signal_model/jan_11/20:51:46/checkpoints/1549309174/step_3775828.pt')

# TODO: Factor out these changes into individual branches; enabling me to move forward with
# training more voices and running more experiments.


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
    assert spectrogram_model_checkpoint_path.is_file(
    ), 'Spectrogram model checkpoint cannot be found.'
    assert signal_model_checkpoint_path.is_file(), 'Signal model checkpoint cannot be found.'

    set_hparams()

    spectrogram_model = Checkpoint.from_path(spectrogram_model_checkpoint_path, device=DEVICE)
    spectrogram_model, input_encoder = (spectrogram_model.model, spectrogram_model.input_encoder)

    signal_model = Checkpoint.from_path(signal_model_checkpoint_path, device=DEVICE)
    signal_model = signal_model.model.to_inferrer()
    return signal_model, spectrogram_model, input_encoder


class InvalidUsage(Exception):
    """
    Inspired by http://flask.pocoo.org/docs/1.0/patterns/apierrors/

    Args:
        message (str)
        status_code (int)
        payload (dict): Additional context to send.
    """

    def __init__(self, message, status_code=400, payload=None):
        Exception.__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        response = dict(self.payload or ())
        response['message'] = self.message
        logger.info('Responding with warning: %s', self.message)
        return response


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    # Register an error response
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.before_first_request
def setup():
    # Set based off the resources dedicated to this worker in `master.js`
    torch.set_num_threads(4)

    logger.info('PyTorch version: %s', torch.__version__)
    logger.info('Found MKL: %s', torch.backends.mkl.is_available())
    logger.info('Threads: %s', torch.get_num_threads())


def _stream_text_to_speech_synthesis(text, speaker, stop_threshold=None, split_size=20):
    """ Helper function for starting a speech synthesis stream.

    Args:
        text (str)
        speaker (src.datasets.Speaker)
        stop_threshold (float, optional): Probability to stop predicting frames.
        split_size (int): Number of frames to synthesize at a time.

    Returns:
        (callable): Callable that returns a generator incrementally returning a WAV file.
        (int): Number of bytes to be returned in total by the generator.
    """
    logger.info('Requested stream conditioned on: "%s", "%s" and "%s".', speaker, stop_threshold,
                text)
    signal_model, spectrogram_model, input_encoder = load_checkpoints()

    # Compute spectrogram
    text, speaker = input_encoder.encode((text, speaker))
    kwargs = {}
    if isinstance(stop_threshold, float):
        kwargs['stop_threshold'] = stop_threshold

    # TODO: Replace with ``signal_model.conditional_features_upsample.scale_factor`` for newer
    # checkpoints
    scale_factor = (
        signal_model.conditional_features_upsample.pre_net[-1].net[-1].out_channels *
        signal_model.conditional_features_upsample.upsample_repeat)
    # TODO: Replace with ``signal_model.conditional_features_upsample.padding`` for newer
    # checkpoints
    padding = signal_model.conditional_features_upsample.min_padding
    half_padding = int(padding / 2)

    logger.info('Generating spectrogram.')

    with torch.no_grad():
        spectrogram = spectrogram_model(text, speaker, use_tqdm=True, **kwargs)[1]

    # TODO: If ``spectrogram`` reaches the ``max_frames_per_token``, return a 'failed to render'
    # error.
    # TODO: If a loud sound is created, cut off the stream or consider rerendering.
    # TODO: Consider logging various events to stackdriver, to keep track.

    logger.info('Generated spectrogram of shape %s.', spectrogram.shape)

    num_frames = spectrogram.shape[0]  # [num_frames, num_channels]
    num_samples = scale_factor * num_frames
    spectrogram = torch.nn.functional.pad(spectrogram, (0, 0, half_padding, half_padding))
    wav_header, wav_file_size = build_wav_header(num_samples)

    def response():
        """ Generator incrementally generating a WAV file.
        """
        assert sys.byteorder == 'little', 'Ensure byte order is of little-endian format.'
        yield wav_header
        hidden_state = None
        for start_frame in list(range(half_padding, num_frames + half_padding, split_size)):
            # Get padded split
            end_frame = start_frame + split_size
            padded_start_frame = start_frame - half_padding
            padded_end_frame = end_frame + half_padding
            split = spectrogram[padded_start_frame:padded_end_frame]

            with torch.no_grad():
                coarse, fine, hidden_state = signal_model(split, hidden_state, pad=False)

            waveform = combine_signal(coarse, fine, return_int=True).numpy()
            logger.info('Waveform shape %s', waveform.shape)
            yield waveform.tostring()

        logger.info('Finished generating waveform.')

    return response, wav_file_size


def _validate_and_unpack(args, max_characters=1000, num_api_key_characters=32):
    """ Validate and unpack the request object.

    Args:
        args (dict) {
          speaker_id (int or str)
          text (str)
          api_key (str)
          stop_threshold (float or None)
        }
        max_characters (int)
        num_api_key_characters (int)

    Returns:
        speaker (src.datasets.Speaker)
        text (str)
        api_key (str)
        stop_threshold (float or None)
    """
    if 'api_key' not in args:
        raise InvalidUsage('API key was not provided.', status_code=401)

    # TODO: Consider using the authorization header instead of a parameter ``api_key``.
    api_key = args.get('api_key')

    if not (isinstance(api_key, str) and len(api_key) == num_api_key_characters):
        raise InvalidUsage(
            'API key must be a string with %d characters.' % num_api_key_characters,
            status_code=401)

    if api_key not in API_KEYS:
        raise InvalidUsage('API key is not valid.', status_code=401)

    if not ('speaker_id' in args and 'text' in args):
        raise InvalidUsage('Must call with keys `speaker_id` and `text`.')

    speaker_id = args.get('speaker_id')
    text = args.get('text')
    stop_threshold = args.get('stop_threshold', None)

    if not isinstance(speaker_id, (str, int)):
        raise InvalidUsage('Speaker ID must be either an integer or string.')

    if isinstance(speaker_id, str) and not speaker_id.isdigit():
        raise InvalidUsage('Speaker ID string must only consist of the symbols 0 - 9.')

    speaker_id = int(speaker_id)

    # TODO: Check that ``speaker_id`` is in ``input_encoder.speaker_encoder.vocab``
    if not (isinstance(speaker_id, int) and speaker_id < len(Speaker) and speaker_id >= 0):
        raise InvalidUsage('Speaker ID must be an integer between %d and %d.' % (0, len(Speaker)))

    if not (isinstance(text, str) and len(text) < max_characters and len(text) > 0):
        # TODO: The error string should suggest the text must be none-empty.
        raise InvalidUsage('Text must be a string under %d characters' % max_characters)

    input_encoder = load_checkpoints()[2]
    processed_text = input_encoder.text_encoder.decode(input_encoder.text_encoder.encode(text))
    if processed_text != text:
        improper_characters = set(text).difference(set(processed_text))
        improper_characters = ', '.join(sorted(list(improper_characters)))
        raise InvalidUsage('Text cannot contain these characters: %s' % improper_characters)

    if not (isinstance(stop_threshold, float) or stop_threshold is None):
        raise InvalidUsage('Stop threshold must be a float.')

    if isinstance(stop_threshold, float):
        stop_threshold = round(stop_threshold, 2)

    speaker = getattr(Speaker, str(speaker_id))
    return speaker, text, stop_threshold


# NOTE: The route `/api/speech_synthesis/v1/` standads for "speech synthesis api v1"


@app.route('/healthy', methods=['GET'])
def healthy():
    # Healthy iff ``load_checkpoints`` succeeds and this route succeeds.
    load_checkpoints()

    return 'ok'


# TODO: Remove the `speech_synthesis` namespace, it's not nessecary.


@app.route('/api/speech_synthesis/v1/text_to_speech/input_validated', methods=['GET', 'POST'])
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
        stop_threshold (float, optional)

    Returns:
        Response with status 200 if the arguments are valid; Otherwise, returning a `InvalidUsage`.
    """
    if request.method == 'POST':
        args = request.get_json()
    else:
        args = request.args

    _validate_and_unpack(args)
    return jsonify({'message': 'OK'})


# TODO: Remove `/stream` namespace it's not a helpful segmentation for right now.


@app.route('/api/speech_synthesis/v1/text_to_speech/stream', methods=['GET', 'POST'])
def get_stream():
    """ Get speech given `text`, `speaker`, and `stop_threshold`.

    Args:
        speaker_id (str)
        text (str)
        api_key (str): Security token sent on behalf of client to ensure authenticity of the
            request.
        stop_threshold (float, optional)

    Returns:
        `audio/wav` streamed in chunks given that the arguments are valid.
    """
    if request.method == 'POST':
        # NOTE: There is an edge case when both `speaker_id="3"` and
        # text="Welcome to Dan’s pizza and shoe repair." breaks the below line. To reproduce, you
        # must use `node` oddly as well. The error was not reproducible on PostMan. Ditto with
        # `speaker_id` must be a `string` and the text must be exact; otherwise, the error
        # was not reproducible.
        args = request.get_json()
    else:
        args = request.args

    speaker, text, stop_threshold = _validate_and_unpack(args)
    response, content_length = _stream_text_to_speech_synthesis(
        text=text, speaker=speaker, stop_threshold=stop_threshold)
    headers = NO_CACHE_HEADERS.copy()
    headers['Content-Length'] = content_length
    return Response(response(), headers=headers, mimetype='audio/wav')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
