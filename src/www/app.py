""" Run a webservice with the spectrogram and signal models

Example:

      $ export PYTHONPATH=.;
      $ sudo python3 -m src.www.app; # ``sudo`` is required to run on port 80 as a webservice
"""
from functools import lru_cache
from pathlib import Path

import logging
import os
import re
import uuid

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

import librosa

from src.audio import combine_signal
from src.audio import griffin_lim
from src.datasets import Speaker
from src.hparams import log_config
from src.hparams import set_hparams
from src.utils import Checkpoint
from src.utils import evaluate

# GLOBAL MEMORY
set_hparams()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

samples_folder = Path(os.getcwd()) / 'src/www/static/samples'


@lru_cache()
def get_spectrogram_model_checkpoint():
    spectrogram_model_checkpoint_path = ('experiments/feature_model/09_24/'
                                         'normalized__encoder_norm/checkpoints/'
                                         '1537985655/step_91097.pt')
    return Checkpoint.from_path(spectrogram_model_checkpoint_path)


@lru_cache()
def get_signal_model_checkpoint():
    signal_model_checkpoint_path = ('experiments/signal_model/09_28/'
                                    'feature_model_normalized__encoder_norm/'
                                    'checkpoints/1539187496/step_9015794.pt')
    return Checkpoint.from_path(signal_model_checkpoint_path)


def cache_models():
    """ Cache spectrogram and signal models """
    get_spectrogram_model_checkpoint()
    get_signal_model_checkpoint()


# ERROR HANDLERS
# INSPIRED BY: http://flask.pocoo.org/docs/1.0/patterns/apierrors/


class GenericException(Exception):
    """ New exception that can take a proper human readable message, a status code for the error
    and some optional payload to give more context for the error.
    """
    status_code = 400  # default status code for an exception

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        result = dict(self.payload or ())
        result['message'] = self.message
        return result


@app.errorhandler(GenericException)
def handle_generic_exception(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/')
def home():
    return send_file('index.html')


@app.route('/demo')
def demo():
    return send_file('demo.html')


@app.route('/samples/<filename>')
def get_sample(filename, default_sample_filename='voiceover.wav'):
    """ Returns the ``filename`` audio.

    TODO: Test this functionality

    Args:
        filename (str)
        default_sample_filename (str)
    """
    # Ensure that a adversary has provided a valid filename
    assert re.match('^[A-Za-z0-9_-]*$', filename)
    as_attachment = request.args.get('attachment', None)
    attachment_filename = default_sample_filename if as_attachment else None
    path_to_file = samples_folder / filename
    assert path_to_file.is_file(), 'Unable to find %s file' % str(path_to_file)
    return send_file(
        str(path_to_file),
        conditional=True,
        as_attachment=as_attachment,
        attachment_filename=attachment_filename)


def _synthesize(text, speaker, is_high_fidelity):
    """ Synthesize audio given ``text``, returning the audio filename.
    """
    log_config()
    spectrogram_model_checkpoint = get_spectrogram_model_checkpoint()
    signal_model_checkpoint = get_signal_model_checkpoint()

    text_encoder = spectrogram_model_checkpoint.text_encoder
    encoded_text = text_encoder.encode(text)

    if text_encoder.decode(encoded_text) != text:
        raise GenericException('Text has improper characters.')

    speaker = getattr(Speaker, str(speaker))  # Get speaker by ID or name
    encoded_speaker = spectrogram_model_checkpoint.speaker_encoder.encode(speaker)

    with evaluate(spectrogram_model_checkpoint.model, signal_model_checkpoint.model):
        # predicted_frames [num_frames, batch_size, frame_channels]
        predicted_frames = spectrogram_model_checkpoint.model.infer(
            tokens=encoded_text, speaker=encoded_speaker)[1]

        if is_high_fidelity:
            # [num_frames, batch_size, frame_channels] → [batch_size, num_frames, frame_channels]
            predicted_frames = predicted_frames.transpose(0, 1)
            predicted_coarse, predicted_fine, _ = signal_model_checkpoint.model.infer(
                predicted_frames)
            waveform = combine_signal(predicted_coarse, predicted_fine).numpy()
        else:
            waveform = griffin_lim(predicted_frames[:, 0].numpy())

    # TODO: Fix this unique_id, it could cause a duplicate
    unique_id = str(uuid.uuid4())
    filename = samples_folder / '{}{}.wav'.format('generated_audio_', unique_id)
    librosa.output.write_wav(str(filename), waveform)

    return filename


@app.route('/synthesize', methods=['POST'])
def synthesize():
    """ Synthesize the requested text as speech.

    TODO: Add the ability to pick from multiple speakers in the JS
    TODO: Fix the 60 second timeout issue
    """
    request_data = request.get_json()
    logger.info('Got request %s', request_data)
    filename = _synthesize(
        text=request_data['text'],
        speaker=request_data['speaker'],
        is_high_fidelity=request_data['isHighFidelity'])
    return jsonify({'filename': str(filename.name)})


if __name__ == "__main__":
    from torch import multiprocessing

    app.run(host='0.0.0.0', port=8000, processes=multiprocessing.cpu_count(), threaded=False)
