from pathlib import Path

import os
import random
import requests
import string
import logging
import torch

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

from src.audio import griffin_lim
from src.bin.feature_model._utils import set_hparams as set_feature_model_hparams
from src.bin.signal_model._utils import set_hparams as set_signal_model_hparams
from src.utils import load_checkpoint

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
torch.set_grad_enabled(False)

SAMPLES_FOLDER = Path('static/samples')
DEFAULT_SAMPLE_FILENAME = 'voiceover.mp3'

def load_feature_model(path):
    """ Load the feature model from a ``Path`` or ``str`` path argument """
    set_feature_model_hparams()
    checkpoint = load_checkpoint(path, torch.device('cuda'))
    logger.info('Loaded feature model at step %d', checkpoint['step'])
    return checkpoint['text_encoder'], checkpoint['model'].eval()

def load_signal_model(path):
    """ Load the signal model from a ``Path`` or ``str`` path argument """
    set_signal_model_hparams()
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    logger.info('Loaded signal model at step %d', checkpoint['step'])
    return checkpoint['model'].eval()

EXPERIMENT_ROOT = Path('/home/michaelp/WellSaid-Labs-Text-To-Speech/experiments/')
FEATURE_MODEL, TEXT_ENCODER = load_feature_model(EXPERIMENT_ROOT /
                'feature_model/09_16/post_net_no_dropout/' /
                'checkpoints/1538001059/step_289899.pt')
SIGNAL_MODEL = load_signal_model(EXPERIMENT_ROOT /
                'signal_model/09_28/feature_model_post_net_no_dropout/' /
                'checkpoints/1538601423/step_2217350.pt')

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


# FILE ROUTES


@app.route('/', methods=['POST', 'GET'])
def index(filename=None):
    return send_file('index.html')


@app.route('/samples/<filename>', methods=['GET'])
def get_sample(filename):
    as_attachment = request.args.get('attachment', None)
    attachment_filename = DEFAULT_SAMPLE_FILENAME if as_attachment else None
    path_to_file = os.path.join(SAMPLES_FOLDER, filename)
    return send_file(
        path_to_file,
        conditional=True,
        as_attachment=as_attachment,
        attachment_filename=attachment_filename)


@app.route('/synthesize', methods=['POST'])
def synthesize():
    request_data = request.get_json()
    speaker = request_data['speaker'].lower()
    style = request_data['style'].lower()
    text = request_data['text'].lower()

    if TEXT_ENCODER.decode(TEXT_ENCODER.encode(text)) != text:
        raise ValueError('Text has improper characters.')

    encoded = TEXT_ENCODER.encode(text)
    encoded = encoded.unsqueeze(1).to(torch.device('cuda'))
    # predicted_frames [num_frames, batch_size, frame_channels]
    predicted_frames = feature_model(tokens=encoded)[1]
    waveform = griffin_lim(predicted_frames[:, 0].cpu().numpy())

    # TODO: Fix this unique_id, it could cause a duplicate
    unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    filename = SAMPLES_FOLDER / '{}{}.mp3'.format('generated_audio_', unique_id)
    librosa.output.write_wav(str(filename), waveform)

    return jsonify({'filename': str(filename)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
