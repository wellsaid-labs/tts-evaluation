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
      $ PYTHONPATH=. python -m run.deploy.worker

Example (Gunicorn):

      $ CHECKPOINTS=""
      $ python -m run.deploy.package_tts $CHECKPOINTS
      $ gunicorn run.deploy.worker:app --timeout=3600 --env='GUNICORN=1'
"""
import gc
import os
import typing

import config as cf
import en_core_web_sm
import torch
import torch.backends.mkl
from flask import Flask, jsonify, request
from flask.wrappers import Response
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
from run.data._loader import Session, Speaker, english
from run.train.spectrogram_model._data import EncodedInput, InputEncoder

if "NUM_CPU_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["NUM_CPU_THREADS"]))

if __name__ == "__main__":
    # NOTE: Incase this module is imported, don't run `set_basic_logging_config`.
    # NOTE: Flask documentation requests that logging is configured before `app` is created.
    set_basic_logging_config()

app = Flask(__name__)

DEVICE = torch.device("cpu")
MAX_CHARS = 10000
TTS_PACKAGE: TTSPackage
SPACY: English
# NOTE: The keys need to stay the same for backwards compatibility.
_SPEAKER_ID_TO_SPEAKER: typing.Dict[int, typing.Tuple[Speaker, str]] = {
    0: (english.m_ailabs.JUDY_BIEBER, "emerald_city_of_oz/wavs/emerald_city_of_oz_06"),
    1: (english.m_ailabs.MARY_ANN, "northandsouth/wavs/northandsouth_09"),
    2: (english.lj_speech.LINDA_JOHNSON, "LJ003"),
    3: (english.wsl.ALANA_B, "script_3.wav"),
    4: (english.wsl.RAMONA_J, "7.wav"),
    5: (english.wsl.RAMONA_J__CUSTOM, "sukutdental_021819.wav"),
    6: (english.lj_speech.LINDA_JOHNSON, "LJ003"),
    7: (english.wsl.WADE_C, "102-107.wav"),
    8: (english.wsl.SOFIA_H, "14.wav"),
    9: (english.wsl.DAVID_D, "copy_of_wsl-_script_022-027.wav"),
    10: (english.wsl.VANESSA_N, "76-81.wav"),
    11: (english.wsl.ISABEL_V, "heather_4-21_a.wav"),
    12: (english.wsl.AVA_M, "well_said_script_16-21.wav"),
    13: (english.wsl.JEREMY_G, "copy_of_drake_jr-script_46-51.wav"),
    14: (english.wsl.NICOLE_L, "copy_of_wsl_-_megansinclairscript40-45.wav"),
    15: (english.wsl.PAIGE_L, "wsl_elise_randall_enthusiastic_script-16.wav"),
    16: (english.wsl.TOBIN_A, "wsl_hanuman_welch_enthusiastic_script-7.wav"),
    17: (english.wsl.KAI_M, "wsl_jackrutkowski_enthusiastic_script_24.wav"),
    18: (english.wsl.TRISTAN_F, "wsl_markatherlay_diphone_script-4.wav"),
    19: (english.wsl.PATRICK_K, "WSL_StevenWahlberg_DIPHONE_Script-6.wav"),
    20: (english.wsl.SOFIA_H__PROMO, "promo_script_3_walker.wav"),
    21: (english.wsl.DAMIAN_P__PROMO, "promo_script_2_papadopoulos.wav"),
    22: (english.wsl.JODI_P__PROMO, "promo_script_8_hurley.wav"),
    23: (english.wsl.LEE_M__PROMO, "promo_script_1_la_comb.wav"),
    24: (english.wsl.SELENE_R__PROMO, "promo_script_1_rousseau.wav"),
    25: (english.wsl.MARI_MONGE__PROMO, "promo_script_1_monge.wav"),
    26: (english.wsl.WADE_C__PROMO, "promo_script_3_scholl.wav"),
    27: (english.wsl.JOE_F__NARRATION, "johnhunerlach_enthusiastic_21.wav"),
    28: (english.wsl.JOE_F__RADIO, "johnhunerlach_diphone_1.wav"),
    29: (english.wsl.GARRY_J__STORY, "otis-jiry_the_happening_at_crossroads.wav"),
    30: (english.wsl.WADE_C__MANUAL_POST, "70-75.wav"),
    31: (english.wsl.AVA_M__MANUAL_POST, "copy_of_well_said_script_40-45-processed.wav"),
    32: (
        english.wsl.KAI_M__MANUAL_POST,
        "wsl_jackrutkowski_enthusiastic_script_27-processed.wav",
    ),
    33: (english.wsl.JUDE_D__EN_GB, "enthusiastic_script_5_davis.wav"),
    34: (english.wsl.ERIC_S__EN_IE__PROMO, "promo_script_7_diamond.wav"),
    35: (
        english.wsl.CHASE_J__PROMO,
        "promo_script_5_daniels.wav",
    ),  # Test in staging due to low quality
    36: (
        english.wsl.DAN_FURCA__PROMO,
        "furca_audio_part3.wav",
    ),  # Test in staging due to low quality
    37: (english.wsl.STEVE_B__PROMO, "promo_script_1_cupit_02.wav"),
    38: (english.wsl.BELLA_B__PROMO, "promo_script_5_tugman.wav"),
    39: (english.wsl.TILDA_C__PROMO, "promo_script_6_mckell.wav"),
    40: (
        english.wsl.SHARON_GAULD_ALEXANDER__PROMO,
        "promo_script_5_alexander.wav",
    ),  # Do not release till paid
    41: (english.wsl.PAUL_B__PROMO, "promo_script_9_williams.wav"),
    42: (english.wsl.SOFIA_H__CONVO, "conversational_script_5_walker.wav"),
    43: (english.wsl.AVA_M__CONVO, "conversational_script_6_harris.wav"),
    44: (english.wsl.KAI_M__CONVO, "conversational_script_3_rutkowski.wav"),
    45: (english.wsl.NICOLE_L__CONVO, "conversational_script_1_sinclair.wav"),
    46: (english.wsl.WADE_C__CONVO, "conversational_script_2_scholl.wav"),
    47: (english.wsl.PATRICK_K__CONVO, "conversational_script_3_wahlberg.wav"),
    48: (english.wsl.VANESSA_N__CONVO, "conversational_script_4_murphy.wav"),
    49: (english.wsl.GIA_V, "narration_script_5_ruiz.wav"),
    50: (english.wsl.ANTONY_A, "narration_script_3_marrero.wav"),
    51: (english.wsl.JODI_P, "narration_script_2_hurley.wav"),
    52: (english.wsl.RAINE_B, "narration_script_5_black.wav"),
    53: (english.wsl.OWEN_C, "narration_script_5_white.wav"),
    54: (english.wsl.ZACH_E, "narration_script_5_jones.wav"),
    55: (english.wsl.GENEVIEVE_M, "narration_script_2_reppert.wav"),
    56: (english.wsl.JARVIS_H, "narration_script_5_hillknight.wav"),
    57: (english.wsl.THEO_K, "narration_script_8_kohnke.wav"),
    58: (english.wsl.JAMES_B, "newman_final_page_13.wav"),
    # NOTE: Custom voice IDs are random numbers larger than 10,000...
    # TODO: Retrain some of these voices, and reconfigure them.
    11541: (english.wsl_archive.LINCOLN__CUSTOM, ""),
    13268907: (english.wsl_archive.JOSIE__CUSTOM, ""),
    95313811: (english.wsl_archive.JOSIE__CUSTOM__MANUAL_POST, ""),
    50794582: (english.wsl.UNEEQ__ASB_CUSTOM_VOICE, "script-20-enthusiastic.wav"),
    50794583: (english.wsl.UNEEQ__ASB_CUSTOM_VOICE_COMBINED, "script-28-enthusiastic.wav"),
    78252076: (english.wsl.VERITONE__CUSTOM_VOICE, ""),
    70695443: (english.wsl.SUPER_HI_FI__CUSTOM_VOICE, "promo_script_5_superhifi.wav"),
    64197676: (english.wsl.US_PHARMACOPEIA__CUSTOM_VOICE, "enthusiastic_script-22.wav"),
    41935205: (
        english.wsl.HAPPIFY__CUSTOM_VOICE,
        "anna_long_emotional_clusters_1st_half_clean.wav",
    ),
    42400423: (
        english.wsl.THE_EXPLANATION_COMPANY__CUSTOM_VOICE,
        "is_it_possible_to_become_invisible.wav",
    ),
    61137774: (english.wsl.ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE, "sample_script_2.wav"),
    30610881: (english.wsl.VIACOM__CUSTOM_VOICE, "kelsey_speech_synthesis_section1.wav"),
    50481197: (english.wsl.HOUR_ONE_NBC__BB_CUSTOM_VOICE, "hour_one_nbc_dataset_5.wav"),
    77552139: (english.wsl.STUDY_SYNC__CUSTOM_VOICE, "fernandes_audio_5.wav"),
    25502195: (english.wsl.FIVE_NINE__CUSTOM_VOICE, "wsl_five9_audio_3.wav"),
}
SPEAKER_ID_TO_SPEAKER = {k: (spk, sesh) for k, (spk, sesh) in _SPEAKER_ID_TO_SPEAKER.items()}


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


class RequestArgs(typing.TypedDict):
    speaker_id: int
    text: str


def validate_and_unpack(
    request_args: RequestArgs,
    input_encoder: InputEncoder,
    nlp: English,
    max_chars: int = MAX_CHARS,
    speaker_id_to_speaker: typing.Dict[int, typing.Tuple[Speaker, Session]] = SPEAKER_ID_TO_SPEAKER,
) -> EncodedInput:
    """Validate and unpack the request object."""

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
        TTS_PACKAGE, input, **cf.get(), logger=app.logger, output_flags=output_flags
    )
    return Response(generator, headers=headers, mimetype="audio/mpeg")


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
