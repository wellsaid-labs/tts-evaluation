""" Run a web service with the spectrogram and signal models.

During designing this API, there was multiple days of consideration of usage.

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

TODO: Write tests for serve.py
"""
try:
    # NOTE: Comet needs to be imported before torch
    import comet_ml  # noqa: F401
except ImportError:
    pass

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

assert SPECTROGRAM_MODEL_CHECKPOINT_PATH.is_file(), 'Spectrogram model checkpoint cannot be found.'
assert SIGNAL_MODEL_CHECKPOINT_PATH.is_file(), 'Signal model checkpoint cannot be found.'


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
        text_encoder (torchnlp.TextEncoder)
        speaker_encoder (torchnlp.TextEncoder)
    """
    set_hparams()

    spectrogram_model = Checkpoint.from_path(spectrogram_model_checkpoint_path, device=DEVICE)
    spectrogram_model, text_encoder, speaker_encoder = (spectrogram_model.model,
                                                        spectrogram_model.text_encoder,
                                                        spectrogram_model.speaker_encoder)

    signal_model = Checkpoint.from_path(signal_model_checkpoint_path, device=DEVICE)
    signal_model = signal_model.model.to_inferrer()
    return signal_model, spectrogram_model, text_encoder, speaker_encoder


# Cache the models
load_checkpoints()


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
    signal_model, spectrogram_model, text_encoder, speaker_encoder = load_checkpoints()

    # Compute spectrogram
    text = text_encoder.encode(text)
    speaker = speaker_encoder.encode(speaker)
    kwargs = {}
    if isinstance(stop_threshold, float):
        kwargs['stop_threshold'] = stop_threshold

    # TODO: Replace with `signal_model.conditional_features_upsample.scale_factor` for newer
    # checkpoints
    scale_factor = (
        signal_model.conditional_features_upsample.pre_net[-1].net[-1].out_channels *
        signal_model.conditional_features_upsample.upsample_repeat)
    # TODO: Replace with `signal_model.conditional_features_upsample.padding` for newer checkpoints
    padding = signal_model.conditional_features_upsample.min_padding
    half_padding = int(padding / 2)

    logger.info('Generating spectrogram.')

    with torch.no_grad():
        spectrogram = spectrogram_model(text, speaker, use_tqdm=True, **kwargs)[1]

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

    if not (isinstance(speaker_id, int) and speaker_id < len(Speaker) and speaker_id >= 0):
        raise InvalidUsage('Speaker ID must be an integer between %d and %d.' % (0, len(Speaker)))

    if not (isinstance(text, str) and len(text) < max_characters and len(text) > 0):
        raise InvalidUsage('Text must be a string under %d characters' % max_characters)

    text_encoder = load_checkpoints()[2]
    processed_text = text_encoder.decode(text_encoder.encode(text))
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
    return 'ok'


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
