"""
Example:

      $ export PYTHONPATH=.
      $ python3 -m src.www.app
"""
from pathlib import Path

import logging
import torch
import uuid

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

import librosa

from src.audio import griffin_lim
from src.bin.feature_model._utils import set_hparams as set_feature_model_hparams
from src.bin.signal_model._utils import set_hparams as set_signal_model_hparams
from src.utils import combine_signal
from src.utils import load_checkpoint

app = Flask(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SAMPLES_FOLDER = Path('src/www/static/samples')
DEFAULT_SAMPLE_FILENAME = 'voiceover.wav'


def load_feature_model(path):
    """ Load the feature model from a ``Path`` or ``str`` path argument """
    set_feature_model_hparams()
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    logger.info('Loaded feature model at step %d', checkpoint['step'])
    return checkpoint['model'].eval(), checkpoint['text_encoder']


def load_signal_model(path):
    """ Load the signal model from a ``Path`` or ``str`` path argument """
    set_signal_model_hparams()
    checkpoint = load_checkpoint(path, torch.device('cpu'))
    logger.info('Loaded signal model at step %d', checkpoint['step'])
    return checkpoint['model'].eval()


EXPERIMENT_ROOT = Path('/home/michaelp/WellSaid-Labs-Text-To-Speech/experiments/')
FEATURE_MODEL, TEXT_ENCODER = load_feature_model(
    EXPERIMENT_ROOT / 'feature_model/09_16/post_net_no_dropout/' /
    'checkpoints/1538001059/step_289899.pt')
SIGNAL_MODEL = load_signal_model(
    EXPERIMENT_ROOT / 'signal_model/09_28/feature_model_post_net_no_dropout/' /
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
    path_to_file = SAMPLES_FOLDER / filename
    return send_file(
        path_to_file,
        conditional=True,
        as_attachment=as_attachment,
        attachment_filename=attachment_filename)


# TODO: We do not need multiple GPUs since WaveRNN works best on CPU
@app.route('/synthesize', methods=['POST'])
def synthesize():
    request_data = request.get_json()
    # TODO: Add the ability to pick from multiple speakers
    text = request_data['text'].lower()
    is_high_fidelity = request_data['isHighFidelity']
    logger.info('Got request %s', request_data)

    if TEXT_ENCODER.decode(TEXT_ENCODER.encode(text)) != text:
        raise ValueError('Text has improper characters.')

    with torch.set_grad_enabled(False):
        encoded = TEXT_ENCODER.encode(text)
        encoded = encoded.unsqueeze(1)
        # predicted_frames [num_frames, batch_size, frame_channels]
        predicted_frames = FEATURE_MODEL(tokens=encoded)[1]

        if is_high_fidelity:
            # [num_frames, batch_size, frame_channels] → [num_frames, frame_channels]
            predicted_frames = predicted_frames.squeeze(1)
            # TODO: The padding should be handled by the model
            padded_predicted_frames = torch.nn.functional.pad(predicted_frames, (0, 0, 5, 5))
            padded_predicted_frames = padded_predicted_frames.unsqueeze(0)
            # [batch_size, signal_length]
            predicted_coarse, predicted_fine, _ = SIGNAL_MODEL.infer(padded_predicted_frames)

            predicted_coarse = predicted_coarse.squeeze(0)
            predicted_fine = predicted_fine.squeeze(0)
            waveform = combine_signal(predicted_coarse, predicted_fine).numpy()
        else:
            waveform = griffin_lim(predicted_frames[:, 0].numpy())

    # TODO: Fix this unique_id, it could cause a duplicate
    unique_id = str(uuid.uuid4())
    filename = SAMPLES_FOLDER / '{}{}.wav'.format('generated_audio_', unique_id)
    librosa.output.write_wav(str(filename), waveform)

    return jsonify({'filename': str(filename.name)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, processes=4, threaded=False)
