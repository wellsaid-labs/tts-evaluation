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
import pprint
import typing

import config as cf
import spacy
import torch
import torch.backends.mkl
from flask import Flask, jsonify, request
from flask.wrappers import Response
from spacy.lang.en import English

from lib.environment import load, set_basic_logging_config
from run._config import TTS_PACKAGE_PATH, configure, load_spacy_nlp
from run._models.spectrogram_model import Inputs, PreprocessedInputs, RespellingError
from run._tts import (
    PublicSpeakerValueError,
    PublicTextValueError,
    TTSPackage,
    process_tts_inputs,
    text_to_speech_ffmpeg_generator,
)
from run.data._loader import Language, Session, Speaker, english

if "NUM_CPU_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["NUM_CPU_THREADS"]))

if __name__ == "__main__":
    # NOTE: Incase this module is imported, don't run `set_basic_logging_config`.
    # NOTE: Flask documentation requests that logging is configured before `app` is created.
    set_basic_logging_config()

app = Flask(__name__)
pprinter = pprint.PrettyPrinter(indent=2)

DEVICE = torch.device("cpu")
MAX_CHARS = 10000
TTS_PACKAGE: TTSPackage
LANGUAGE_TO_SPACY: typing.Dict[Language, spacy.language.Language]
SPACY: English
# NOTE: The keys need to stay the same for backwards compatibility.
_SESSIONS = [
    (english.m_ailabs.JUDY_BIEBER, ""),
    (english.m_ailabs.MARY_ANN, ""),
    (english.lj_speech.LINDA_JOHNSON, ""),
    (english.wsl.ALANA_B, "script_3"),
    (english.wsl.RAMONA_J, "7"),
    (english.wsl.RAMONA_J__CUSTOM, "sukutdental_021819"),
    (english.lj_speech.LINDA_JOHNSON, ""),
    (english.wsl.WADE_C, "102-107"),
    (english.wsl.SOFIA_H, "14"),
    (english.wsl.DAVID_D, ""),
    (english.wsl.VANESSA_N, "76-81"),
    (english.wsl.ISABEL_V, "heather_4-21_a"),
    (english.wsl.AVA_M, "well_said_script_16-21"),
    (english.wsl.JEREMY_G, "copy_of_drake_jr-script_46-51"),
    (english.wsl.NICOLE_L, "copy_of_wsl_-_megansinclairscript40-45"),
    (english.wsl.PAIGE_L, "wsl_elise_randall_enthusiastic_script-16"),
    (english.wsl.TOBIN_A, "wsl_hanuman_welch_enthusiastic_script-7"),
    (english.wsl.KAI_M, "wsl_jackrutkowski_enthusiastic_script_24"),
    (english.wsl.TRISTAN_F, "wsl_markatherlay_diphone_script-4"),
    (english.wsl.PATRICK_K, "WSL_StevenWahlberg_DIPHONE_Script-6"),
    (english.wsl.SOFIA_H__PROMO, "promo_script_3_walker"),
    (english.wsl.DAMIAN_P__PROMO, "promo_script_2_papadopoulos"),
    (english.wsl.JODI_P__PROMO, "promo_script_8_hurley"),
    (english.wsl.LEE_M__PROMO, "promo_script_1_la_comb"),
    (english.wsl.SELENE_R__PROMO, "promo_script_1_rousseau"),
    (english.wsl.MARI_MONGE__PROMO, "promo_script_1_monge"),
    (english.wsl.WADE_C__PROMO, "promo_script_3_scholl"),
    (english.wsl.JOE_F__NARRATION, "johnhunerlach_enthusiastic_21"),
    (english.wsl.JOE_F__RADIO, "johnhunerlach_diphone_1"),
    (english.wsl.GARRY_J__STORY, "otis-jiry_the_happening_at_crossroads"),
    (english.wsl.WADE_C__MANUAL_POST, "70-75"),
    (english.wsl.AVA_M__MANUAL_POST, "copy_of_well_said_script_40-45-processed"),
    (english.wsl.KAI_M__MANUAL_POST, "wsl_jackrutkowski_enthusiastic_script_27-processed"),
    (english.wsl.JUDE_D__EN_GB, "enthusiastic_script_5_davis"),
    (english.wsl.ERIC_S__EN_IE__PROMO, "promo_script_7_diamond"),
    # Test in staging due to low quality # TODO: Remove comment?
    (english.wsl.CHASE_J__PROMO, "promo_script_5_daniels"),
    # Test in staging due to low quality # TODO: Remove comment?
    (english.wsl.DAN_FURCA__PROMO, "furca_audio_part3"),
    (english.wsl.STEVE_B__PROMO, "promo_script_1_cupit_02"),
    (english.wsl.BELLA_B__PROMO, "promo_script_5_tugman"),
    (english.wsl.TILDA_C__PROMO, "promo_script_6_mckell"),
    # Do not release till paid # TODO: Remove comment?
    (english.wsl.CHARLIE_Z__PROMO, "promo_script_5_alexander"),
    (english.wsl.PAUL_B__PROMO, "promo_script_9_williams"),
    (english.wsl.SOFIA_H__CONVO, "conversational_script_5_walker"),
    (english.wsl.AVA_M__CONVO, "conversational_script_6_harris"),
    (english.wsl.KAI_M__CONVO, "conversational_script_3_rutkowski"),
    (english.wsl.NICOLE_L__CONVO, "conversational_script_1_sinclair"),
    (english.wsl.WADE_C__CONVO, "conversational_script_2_scholl"),
    (english.wsl.PATRICK_K__CONVO, "conversational_script_3_wahlberg"),
    (english.wsl.VANESSA_N__CONVO, "conversational_script_4_murphy"),
    (english.wsl.GIA_V, "narration_script_5_ruiz"),
    (english.wsl.ANTONY_A, "narration_script_3_marrero"),
    (english.wsl.JODI_P, "narration_script_2_hurley"),
    (english.wsl.RAINE_B, "narration_script_5_black"),
    (english.wsl.OWEN_C, "narration_script_5_white"),
    (english.wsl.ZACH_E, "narration_script_5_jones"),
    (english.wsl.GENEVIEVE_M, "narration_script_2_reppert"),
    (english.wsl.JARVIS_H, "narration_script_5_hillknight"),
    (english.wsl.THEO_K, "narration_script_8_kohnke"),
    (english.wsl.JAMES_B, "newman_final_page_13"),
    (english.wsl.TERRA_G, "narration_script_1_parrish"),
    (english.wsl.PHILIP_J, "anderson_narration_script-rx_loud_01"),
    (english.wsl.MARCUS_G, "furca_audio_part1"),
    (english.wsl.JORDAN_T, "narration_script_1_whiteside_processed"),
    (english.wsl.FIONA_H, "hughes_narration_script_1"),
    (english.wsl.ROXY_T, "topping_narration_script_1processed"),
    (english.wsl.DONNA_W, "brookhyser_narration_script_1"),
    (english.wsl.GREG_G, "lloyd_narration_script_1"),
    (english.wsl.ZOEY_O, "helen_marion-rowe_script_1_processed"),
    (english.wsl.KARI_N, "noble_narration_script_1"),
    (english.wsl.DIARMID_C, "cherry_narration_script_1"),
    (english.wsl.ELIZABETH_U, "naration_script_6_stringer"),
    (english.wsl.ALAN_T, "narration_script_1_frazer"),
    (english.wsl.AVA_M__PROMO, "promo_script_1_harris"),
    (english.wsl.TOBIN_A__PROMO, "promo_script_1_welch_processed"),
    (english.wsl.TOBIN_A__CONVO, "conversational_script_1_welch_processed"),
]
_SPEAKER_ID_TO_SESSION: typing.Dict[int, typing.Tuple[Speaker, str]] = {
    **{i: s for i, s in enumerate(_SESSIONS)},
    # NOTE: As a weak security measure, we assign random large numbers to custom voices, so
    # that they are hard to discover by querying the API.
    11541: (english.wsl_archive.LINCOLN__CUSTOM, ""),
    13268907: (english.wsl_archive.JOSIE__CUSTOM, ""),
    95313811: (english.wsl_archive.JOSIE__CUSTOM__MANUAL_POST, ""),
    50794582: (english.wsl.UNEEQ__ASB_CUSTOM_VOICE, "script-20-enthusiastic"),
    50794583: (english.wsl.UNEEQ__ASB_CUSTOM_VOICE_COMBINED, "script-28-enthusiastic"),
    78252076: (english.wsl.VERITONE__CUSTOM_VOICE, ""),
    70695443: (english.wsl.SUPER_HI_FI__CUSTOM_VOICE, "promo_script_5_superhifi"),
    64197676: (english.wsl.US_PHARMACOPEIA__CUSTOM_VOICE, "enthusiastic_script-22"),
    41935205: (english.wsl.HAPPIFY__CUSTOM_VOICE, "anna_long_emotional_clusters_1st_half_clean"),
    42400423: (
        english.wsl.THE_EXPLANATION_COMPANY__CUSTOM_VOICE,
        "is_it_possible_to_become_invisible",
    ),
    61137774: (english.wsl.ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE, "sample_script_2"),
    30610881: (english.wsl.VIACOM__CUSTOM_VOICE, "kelsey_speech_synthesis_section1"),
    50481197: (english.wsl.HOUR_ONE_NBC__BB_CUSTOM_VOICE, "hour_one_nbc_dataset_5"),
    77552139: (english.wsl.STUDY_SYNC__CUSTOM_VOICE, "fernandes_audio_5"),
    25502195: (english.wsl.FIVE_NINE__CUSTOM_VOICE, "wsl_five9_audio_3"),
}
SPEAKER_ID_TO_SESSION: typing.Dict[int, Session]
SPEAKER_ID_TO_SESSION = {k: Session(args) for k, args in _SPEAKER_ID_TO_SESSION.items()}


class FlaskException(Exception):
    """
    Inspired by http://flask.pocoo.org/docs/1.0/patterns/apierrors/

    TODO: Create an `enum` with all the error codes

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
    tts: TTSPackage,
    language_to_spacy: typing.Dict[Language, spacy.language.Language],
    max_chars: int = MAX_CHARS,
    speaker_id_to_session: typing.Dict[int, Session] = SPEAKER_ID_TO_SESSION,
) -> typing.Tuple[Inputs, PreprocessedInputs]:
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

    min_speaker_id = min(speaker_id_to_session.keys())
    max_speaker_id = max(speaker_id_to_session.keys())

    if not (
        (speaker_id >= min_speaker_id and speaker_id <= max_speaker_id)
        and speaker_id in speaker_id_to_session
    ):
        raise FlaskException("Speaker ID is invalid.", code="INVALID_SPEAKER_ID")

    session = speaker_id_to_session[speaker_id]

    gc.collect()

    try:
        return process_tts_inputs(language_to_spacy[session[0].language], tts, text, session)
    except PublicSpeakerValueError as error:
        app.logger.exception("Invalid speaker: %r", text)
        raise FlaskException(str(error), code="INVALID_SPEAKER_ID")
    except PublicTextValueError as error:
        app.logger.exception("Invalid text: %r", text)
        raise FlaskException(str(error), code="INVALID_TEXT")
    except RespellingError:
        raise FlaskException(
            "Please format your respelling correctly (https://help.wellsaidlabs.com/pronunciation)",
            code="INVALID_TEXT"
        )
    except BaseException:
        app.logger.exception("Invalid text: %r", text)
        raise FlaskException("Unknown error.", code="UNKNOWN_ERROR")


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
    validate_and_unpack(request_args, TTS_PACKAGE, LANGUAGE_TO_SPACY)
    return jsonify({"message": "OK"})


@app.route("/api/text_to_speech/stream", methods=["GET", "POST"])
def get_stream():
    """Get speech given `text` and `speaker`.

    NOTE: Consider the scenario where the requester isn't consuming the stream quickly, the
    worker would need to wait for the requester.

    Usage:
        http://192.168.50.19:8000/api/text_to_speech/stream?speaker_id=46&text="Hello there"

    Returns: `audio/mpeg` streamed in chunks given that the arguments are valid.
    """
    request_args = request.get_json() if request.method == "POST" else request.args
    request_args = typing.cast(RequestArgs, request_args)
    input = validate_and_unpack(request_args, TTS_PACKAGE, LANGUAGE_TO_SPACY)
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    output_flags = ("-f", "mp3", "-b:a", "192k")
    generator = text_to_speech_ffmpeg_generator(
        TTS_PACKAGE, *input, **cf.get(), logger=app.logger, output_flags=output_flags
    )
    return Response(generator, headers=headers, mimetype="audio/mpeg")


if __name__ == "__main__" or "GUNICORN" in os.environ:
    app.logger.info("PyTorch version: %s", torch.__version__)
    app.logger.info("Found MKL: %s", torch.backends.mkl.is_available())
    app.logger.info("Threads: %s", torch.get_num_threads())
    app.logger.info("Speaker Ids: %s", pprinter.pformat(SPEAKER_ID_TO_SESSION))

    configure()

    # NOTE: These models are cached globally to enable sharing between processes, learn more:
    # https://github.com/benoitc/gunicorn/issues/2007
    TTS_PACKAGE = typing.cast(TTSPackage, load(TTS_PACKAGE_PATH, DEVICE))

    vocab = set(TTS_PACKAGE.session_vocab())
    app.logger.info("Loaded speakers: %s", set(s for s, _ in vocab))

    for session in SPEAKER_ID_TO_SESSION.values():
        if session not in vocab:
            app.logger.warning(f"Session not found in model vocab: {session}")
            avail_sessions = [s[1] for s in vocab if s[0] == session[0]]
            if len(avail_sessions) > 0:
                app.logger.warning(f"Sessions available: {avail_sessions}")

    languages = set(s[0].language for s in vocab)
    LANGUAGE_TO_SPACY = {l: load_spacy_nlp(l) for l in languages}
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
