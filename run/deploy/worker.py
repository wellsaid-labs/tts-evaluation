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

Example (Flask):

      $ CHECKPOINTS=""  # Example: v9
      $ python -m run.deploy.package_tts $CHECKPOINTS
      $ PYTHONPATH=. YOUR_SPEECH_API_KEY=123 python -m run.deploy.worker

Example (Gunicorn):

      $ CHECKPOINTS=""
      $ python -m run.deploy.package_tts $CHECKPOINTS
      $ YOUR_SPEECH_API_KEY=123 gunicorn run.deploy.worker:app --timeout=3600 --env='GUNICORN=1'
"""
import gc
import os
import typing
import warnings

import en_core_web_sm
import torch
import torch.backends.mkl
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from spacy.lang.en import English

from lib.environment import load, set_basic_logging_config
from run._config import TTS_PACKAGE_PATH, configure
from run._tts import (
    PublicSpeakerValueError,
    PublicTextValueError,
    TTSPackage,
    encode_tts_inputs,
    text_to_speech_ffmpeg_generator,
)
from run.data import _loader
from run.data._loader import Session, Speaker
from run.train.spectrogram_model._data import EncodedInput, InputEncoder

if "NUM_CPU_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["NUM_CPU_THREADS"]))

if __name__ == "__main__":
    # NOTE: Incase this module is imported, don't run `set_basic_logging_config`.
    # NOTE: Flask documentation requests that logging is configured before `app` is created.
    set_basic_logging_config()

app = Flask(__name__)

DEVICE = torch.device("cpu")
API_KEY_SUFFIX = "_SPEECH_API_KEY"
MAX_CHARS = 10000
API_KEYS: typing.Set[str] = set([v for k, v in os.environ.items() if API_KEY_SUFFIX in k])
TTS_PACKAGE: TTSPackage
SPACY: English
# NOTE: The keys need to stay the same for backwards compatibility.
SPEAKER_ID_TO_SPEAKER: typing.Dict[int, typing.Tuple[Speaker, Session]] = {
    0: (
        _loader.wsl_init__english.JUDY_BIEBER,
        Session(Session("emerald_city_of_oz/wavs/emerald_city_of_oz_06")),
    ),
    1: (_loader.wsl_init__english.MARY_ANN, Session("northandsouth/wavs/northandsouth_09")),
    2: (_loader.wsl_init__english.LINDA_JOHNSON, Session("LJ003")),
    3: (_loader.wsl_init__english.HILARY_NORIEGA, Session("script_3.wav")),
    4: (_loader.wsl_init__english.BETH_CAMERON, Session("7.wav")),
    5: (_loader.wsl_init__english.BETH_CAMERON__CUSTOM, Session("sukutdental_021819.wav")),
    6: (_loader.wsl_init__english.LINDA_JOHNSON, Session("LJ003")),
    7: (_loader.wsl_init__english.SAM_SCHOLL, Session("102-107.wav")),
    8: (_loader.wsl_init__english.ADRIENNE_WALKER_HELLER, Session("14.wav")),
    9: (_loader.wsl_init__english.FRANK_BONACQUISTI, Session("copy_of_wsl-_script_022-027.wav")),
    10: (_loader.wsl_init__english.SUSAN_MURPHY, Session("76-81.wav")),
    11: (_loader.wsl_init__english.HEATHER_DOE, Session("heather_4-21_a.wav")),
    12: (_loader.wsl_init__english.ALICIA_HARRIS, Session("well_said_script_16-21.wav")),
    13: (_loader.wsl_init__english.GEORGE_DRAKE_JR, Session("copy_of_drake_jr-script_46-51.wav")),
    14: (
        _loader.wsl_init__english.MEGAN_SINCLAIR,
        Session("copy_of_wsl_-_megansinclairscript40-45.wav"),
    ),
    15: (
        _loader.wsl_init__english.ELISE_RANDALL,
        Session("wsl_elise_randall_enthusiastic_script-16.wav"),
    ),
    16: (
        _loader.wsl_init__english.HANUMAN_WELCH,
        Session("wsl_hanuman_welch_enthusiastic_script-7.wav"),
    ),
    17: (
        _loader.wsl_init__english.JACK_RUTKOWSKI,
        Session("wsl_jackrutkowski_enthusiastic_script_24.wav"),
    ),
    18: (_loader.wsl_init__english.MARK_ATHERLAY, Session("wsl_markatherlay_diphone_script-4.wav")),
    19: (
        _loader.wsl_init__english.STEVEN_WAHLBERG,
        Session("WSL_StevenWahlberg_DIPHONE_Script-6.wav"),
    ),
    20: (
        _loader.wsl_init__english.ADRIENNE_WALKER_HELLER__PROMO,
        Session("promo_script_3_walker.wav"),
    ),
    21: (
        _loader.wsl_init__english.DAMON_PAPADOPOULOS__PROMO,
        Session("promo_script_2_papadopoulos.wav"),
    ),
    22: (_loader.wsl_init__english.DANA_HURLEY__PROMO, Session("promo_script_8_hurley.wav")),
    23: (_loader.wsl_init__english.ED_LACOMB__PROMO, Session("promo_script_1_la_comb.wav")),
    24: (_loader.wsl_init__english.LINSAY_ROUSSEAU__PROMO, Session("promo_script_1_rousseau.wav")),
    25: (_loader.wsl_init__english.MARI_MONGE__PROMO, Session("promo_script_1_monge.wav")),
    26: (_loader.wsl_init__english.SAM_SCHOLL__PROMO, Session("promo_script_3_scholl.wav")),
    27: (
        _loader.wsl_init__english.JOHN_HUNERLACH__NARRATION,
        Session("johnhunerlach_enthusiastic_21.wav"),
    ),
    28: (_loader.wsl_init__english.JOHN_HUNERLACH__RADIO, Session("johnhunerlach_diphone_1.wav")),
    29: (
        _loader.wsl_init__english.OTIS_JIRY__STORY,
        Session("otis-jiry_the_happening_at_crossroads.wav"),
    ),
    30: (_loader.wsl_init__english.SAM_SCHOLL__MANUAL_POST, Session("70-75.wav")),
    31: (
        _loader.wsl_init__english.ALICIA_HARRIS__MANUAL_POST,
        Session("copy_of_well_said_script_40-45-processed.wav"),
    ),
    32: (
        _loader.wsl_init__english.JACK_RUTKOWSKI__MANUAL_POST,
        Session("wsl_jackrutkowski_enthusiastic_script_27-processed.wav"),
    ),
    # NOTE: Custom voice IDs are random numbers larger than 10,000...
    # TODO: Retrain some of these voices, and reconfigure them.
    11541: (_loader.wsl_init__english.LINCOLN__CUSTOM, Session("")),
    13268907: (_loader.wsl_init__english.JOSIE__CUSTOM, Session("")),
    95313811: (_loader.wsl_init__english.JOSIE__CUSTOM__MANUAL_POST, Session("")),
    78252076: (Speaker(""), Session("")),  # TODO: Add Veritone Custom Voice
    70695443: (Speaker(""), Session("")),  # TODO: Add Super Hi-Fi Custom Voice
    64197676: (Speaker(""), Session("")),  # TODO: Add USP Custom Voice
    41935205: (Speaker(""), Session("")),  # TODO: Add Happify Custom Voice
    42400423: (
        _loader.wsl_init__english.THE_EXPLANATION_COMPANY__CUSTOM_VOICE,
        Session("is_it_possible_to_become_invisible.wav"),
    ),
    61137774: (
        _loader.wsl_init__english.ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE,
        Session("sample_script_2.wav"),
    ),
}


class FlaskException(Exception):
    """
    Inspired by http://flask.pocoo.org/docs/1.0/patterns/apierrors/

    Args:
        message
        status_code: An HTTP response status codes.
        code: A string code.
        payload: Additional context to send.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "BAD_REQUEST",
        payload: typing.Optional[typing.Dict] = None,
    ):
        super().__init__(self, message)
        self.message = message
        self.status_code = status_code
        self.payload = payload
        self.code = code

    def to_dict(self):
        response = dict(self.payload or ())
        response["message"] = self.message
        response["code"] = self.code
        app.logger.info("Responding with warning: %s", self.message)
        return response


@app.errorhandler(FlaskException)
def handle_invalid_usage(error: FlaskException):
    """Response for a `FlaskException`."""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.before_first_request
def before_first_request():
    # NOTE: Remove this warning after this is fixed...
    # https://github.com/PetrochukM/HParams/issues/6
    warnings.filterwarnings(
        "ignore",
        module=r".*hparams",
        message=r".*The decorator was not executed immediately before*",
    )


class RequestArgs(typing.TypedDict):
    speaker_id: int
    text: str
    api_key: str


def validate_and_unpack(
    request_args: RequestArgs,
    input_encoder: InputEncoder,
    nlp: English,
    max_chars: int = MAX_CHARS,
    api_keys: typing.Set[str] = API_KEYS,
    speaker_id_to_speaker: typing.Dict[int, typing.Tuple[Speaker, Session]] = SPEAKER_ID_TO_SPEAKER,
) -> EncodedInput:
    """Validate and unpack the request object.

    TODO: Consider using the authorization header instead of a parameter `api_key`.
    """
    if "api_key" not in request_args:
        raise FlaskException("API key was not provided.", status_code=401, code="MISSING_ARGUMENT")

    api_key = request_args.get("api_key")
    min_api_key_len = min(len(key) for key in api_keys)
    max_api_key_len = max(len(key) for key in api_keys)

    if not (
        isinstance(api_key, str)
        and (len(api_key) >= min_api_key_len and len(api_key) <= max_api_key_len)
    ):
        message = "API key must be a string between"
        message += f" {min_api_key_len} and {max_api_key_len} characters."
        raise FlaskException(message, status_code=401, code="INVALID_API_KEY")

    if api_key not in api_keys:
        raise FlaskException("API key is not valid.", status_code=401, code="INVALID_API_KEY")

    if not ("speaker_id" in request_args and "text" in request_args):
        message = "Must call with keys `speaker_id` and `text`."
        raise FlaskException(message, code="MISSING_ARGUMENT")

    speaker_id = request_args.get("speaker_id")
    text = request_args.get("text")

    if not isinstance(speaker_id, (str, int)):
        message = "Speaker ID must be either an integer or string."
        raise FlaskException(message, code="INVALID_SPEAKER_ID")

    if isinstance(speaker_id, str) and not speaker_id.isdigit():
        message = "Speaker ID string must only consist of the symbols 0 - 9."
        raise FlaskException(message, code="INVALID_SPEAKER_ID")

    speaker_id = int(speaker_id)

    if not (isinstance(text, str) and len(text) < max_chars and len(text) > 0):
        message = f"Text must be a string under {max_chars} characters and more than 0 characters."
        raise FlaskException(message, code="INVALID_TEXT_LENGTH_EXCEEDED")

    min_speaker_id = min(speaker_id_to_speaker.keys())
    max_speaker_id = max(speaker_id_to_speaker.keys())

    if not (
        (speaker_id >= min_speaker_id and speaker_id <= max_speaker_id)
        and speaker_id in speaker_id_to_speaker
    ):
        raise FlaskException("Speaker ID is invalid.", code="INVALID_SPEAKER_ID")

    speaker, session = speaker_id_to_speaker[speaker_id]

    gc.collect()

    try:
        return encode_tts_inputs(nlp, input_encoder, text, speaker, session)
    except PublicSpeakerValueError as error:
        app.logger.exception("Invalid speaker: %r", text)
        raise FlaskException(str(error), code="INVALID_SPEAKER_ID")
    except PublicTextValueError as error:
        app.logger.exception("Invalid text: %r", text)
        raise FlaskException(str(error), code="INVALID_TEXT")


@app.route("/healthy", methods=["GET"])
def healthy():
    return "ok"


# NOTE: `/api/speech_synthesis/v1/` was added for backward compatibility.


@app.route("/api/speech_synthesis/v1/text_to_speech/input_validated", methods=["GET", "POST"])
@app.route("/api/text_to_speech/input_validated", methods=["GET", "POST"])
def get_input_validated():
    """Validate the input to our text-to-speech endpoint before making a stream request.

    NOTE: The API splits the validation responsibility from the streaming responsibility. During
    streaming, we are unable to access any error codes generated by the validation script.
    NOTE: The API supports both GET and POST requests. GET and POST requests have different
    tradeoffs, GET allows for streaming with <audio> elements while POST allows more than 2000
    characters of data to be passed.

    Returns: Response with status 200 if the arguments are valid; Otherwise, returning a
        `FlaskException`.
    """
    request_args = request.get_json() if request.method == "POST" else request.args
    request_args = typing.cast(RequestArgs, request_args)
    validate_and_unpack(request_args, TTS_PACKAGE.input_encoder, SPACY)
    return jsonify({"message": "OK"})


@app.route("/api/speech_synthesis/v1/text_to_speech/stream", methods=["GET", "POST"])
@app.route("/api/text_to_speech/stream", methods=["GET", "POST"])
def get_stream():
    """Get speech given `text` and `speaker`.

    NOTE: Consider the scenario where the requester isn't consuming the stream quickly, the
    worker would need to wait for the requester.

    Returns: `audio/mpeg` streamed in chunks given that the arguments are valid.
    """
    request_args = request.get_json() if request.method == "POST" else request.args
    request_args = typing.cast(RequestArgs, request_args)
    input = validate_and_unpack(request_args, TTS_PACKAGE.input_encoder, SPACY)
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    output_flags = ("-f", "mp3", "-b:a", "192k")
    generator = text_to_speech_ffmpeg_generator(
        TTS_PACKAGE, input, app.logger, output_flags=output_flags
    )
    return Response(generator, headers=headers, mimetype="audio/mpeg")


@app.route("/")
def index():
    return send_file("public/index.html")


@app.route("/<path:path>")
def send_static(path):
    return send_from_directory("public", path)


if __name__ == "__main__" or "GUNICORN" in os.environ:
    app.logger.info("PyTorch version: %s", torch.__version__)
    app.logger.info("Found MKL: %s", torch.backends.mkl.is_available())
    app.logger.info("Threads: %s", torch.get_num_threads())

    configure()

    # NOTE: These models are cached globally to enable sharing between processes, learn more:
    # https://github.com/benoitc/gunicorn/issues/2007
    TTS_PACKAGE = typing.cast(TTSPackage, load(TTS_PACKAGE_PATH, DEVICE))
    app.logger.info("Loaded speakers: %s", TTS_PACKAGE.input_encoder.speaker_encoder.vocab)

    for (speaker, session) in SPEAKER_ID_TO_SPEAKER.values():
        if speaker in TTS_PACKAGE.input_encoder.speaker_encoder.token_to_index:
            message = "Speaker recording session not found."
            lookup = TTS_PACKAGE.input_encoder.session_encoder.token_to_index
            assert (speaker, session) in lookup, message

    SPACY = en_core_web_sm.load(disable=("parser", "ner"))
    app.logger.info("Loaded spaCy.")

    # NOTE: In order to support copy-on-write, we freeze all the objects tracked by `gc`, learn
    # more:
    # https://docs.python.org/3/library/gc.html#gc.freeze
    # https://instagram-engineering.com/copy-on-write-friendly-python-garbage-collection-ad6ed5233ddf
    # https://github.com/benoitc/gunicorn/issues/1640
    # TODO: Measure shared memory to check if it's working well.
    gc.freeze()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
