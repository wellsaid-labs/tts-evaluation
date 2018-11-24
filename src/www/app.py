""" Run a webservice with the spectrogram and signal models

Example:

      $ export PYTHONPATH=.;
      $ sudo python3 -m src.www.app; # ``sudo`` is required to run on port 80 as a webservice
"""
from pathlib import Path

import argparse
import logging
import os
import re
import torch
import uuid

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

import librosa

from src.audio import griffin_lim
from src.hparams import set_hparams
from src.hparams import log_config
from src.utils import combine_signal
from src.utils import Checkpoint
from src.datasets import Speaker

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
samples_folder = Path(os.getcwd()) / 'src/www/static/samples'
spectrogram_model_checkpoint = None
signal_model_checkpoint = None

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


@app.route('/demo')
def index():
    return send_file('index.html')


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

    text_encoder = spectrogram_model_checkpoint.text_encoder
    encoded_text = text_encoder.encode(text)

    speaker = getattr(Speaker, str(speaker))  # Get speaker by ID or name
    speaker_encoder = spectrogram_model_checkpoint.speaker_encoder
    encoded_speaker = speaker_encoder.encode(speaker)

    if text_encoder.decode(encoded_text) != text:
        raise GenericException('Text has improper characters.')

    with torch.no_grad():
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--signal_model', type=str, required=True, help='Signal model checkpoint to serve.')
    parser.add_argument(
        '--spectrogram_model',
        type=str,
        required=True,
        help='Spectrogram model checkpoint to serve.')
    cli_args = parser.parse_args()

    # Global memory
    set_hparams()
    spectrogram_model_checkpoint = Checkpoint.from_path(cli_args.spectrogram_model)
    spectrogram_model_checkpoint.model.eval()
    signal_model_checkpoint = Checkpoint.from_path(cli_args.signal_model)
    signal_model_checkpoint.model.eval()

    app.run(host='0.0.0.0', port=80, processes=8, threaded=False)
