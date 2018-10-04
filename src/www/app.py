import os
import random
import requests
import string

from flask import Flask
from flask import jsonify
from flask import request
from flask import send_file

app = Flask(__name__)

API_ENDPOINT = 'https://www.voicery.com/api/generate'
ALIASES = {'alicia': 'nicole', 'hilary': 'emily', 'liam': 'steven'}
SAMPLES_FOLDER = 'static/samples'
DEFAULT_SAMPLE_FILENAME = 'voiceover.mp3'

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

    speaker = ALIASES[speaker]

    try:
        # TODO: make identifier truly unique (hash of current time)
        unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        filename = '{}{}.mp3'.format('generated_audio_', unique_id)
        file_path = os.path.join(SAMPLES_FOLDER, filename)

        response = requests.post(
            API_ENDPOINT, data={
                'text': text,
                'speaker': speaker,
                'style': style
            })
        response.encoding = 'audio/mp3'

        with open(file_path, 'wb') as file_:
            file_.write(response.content)

        return jsonify({'filename': filename})
    except Exception as e:
        error_name = type(e).__name__
        raise ValueError('An %s occured!' % error_name)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
