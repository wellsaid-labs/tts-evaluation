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

      $ CHECKPOINTS="v11_2023_03_01_staging"  # Example: v9
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
import torch.version
from flask import Flask, jsonify, request
from flask.wrappers import Response
from spacy.lang.en import English

from lib.environment import load, set_basic_logging_config
from lib.text import XMLType
from run._config import TTS_PACKAGE_PATH, configure, load_spacy_nlp
from run._models.spectrogram_model import Inputs, PreprocessedInputs, PublicValueError
from run._tts import (
    PublicSpeakerValueError,
    PublicTextValueError,
    TTSPackage,
    process_tts_inputs,
    tts_ffmpeg_generator,
)
from run.data._loader import Language, Session, Speaker, english

if "NUM_CPU_THREADS" in os.environ:
    torch.set_num_threads(int(os.environ["NUM_CPU_THREADS"]))

if __name__ == "__main__" or os.environ.get("DEBUG") == "1":
    # NOTE: Incase this module is imported, don't run `set_basic_logging_config`.
    # NOTE: Flask documentation requests that logging is configured before `app` is created.
    set_basic_logging_config()

app = Flask(__name__)
pprinter = pprint.PrettyPrinter(indent=2)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_CHARS = 10000
TTS_PACKAGE: TTSPackage
LANGUAGE_TO_SPACY: typing.Dict[Language, spacy.language.Language]
SPACY: English
EN = english.wsl
TEC = EN.THE_EXPLANATION_COMPANY__CUSTOM_VOICE
HAPPIFY = EN.HAPPIFY__CUSTOM_VOICE

# NOTE: The keys need to stay the same for backwards compatibility.
_SPKR_ID_TO_SESH = {
    # NOTE: These 3 are open-source voices that didn't consent to be on our platform.
    # 0: (english.m_ailabs.JUDY_BIEBER, ""),
    # 1: (english.m_ailabs.MARY_ANN, ""),
    # 2: (english.lj_speech.LINDA_JOHNSON, ""),
    3: (EN.ALANA_B, "script_3", -25, 1.15, 1.35),
    4: (EN.RAMONA_J, "7", -25, 1.25, 1.2),
    5: (EN.RAMONA_J__CUSTOM, "sukutdental_021819", -23, 1.3, 1.25),
    # NOTE: This is open-source voice that didn't consent to be on our platform.
    # NOTE: This speaker was released twice on accident with different ids, so it's in this list
    # twice.
    # 6: (english.lj_speech.LINDA_JOHNSON, ""),
    # NOTE: There is a new preprocessed version of Wade that has been included.
    # 7: (EN.WADE_C, ""),
    8: (EN.SOFIA_H, "sofia_h__narration_14", -23, 1.15, 1.1),
    # NOTE: David asked for his voice to be removed from the platform.
    # 9: (EN.DAVID_D, ""),
    10: (EN.VANESSA_N, "76-81", -23, 1.2, 1.25),
    11: (EN.ISABEL_V, "isabel_v__narration_004-021-a", -33, 1.2, 1.15),
    # NOTE: There is a new preprocessed version of Ava that has been included.
    # 12: (EN.AVA_M, ""),
    13: (EN.JEREMY_G, "copy_of_drake_jr-script_46-51", -26, 1.05, 1.1),
    14: (EN.NICOLE_L, "copy_of_wsl_-_megansinclairscript40-45", -21, 1.15, 1.15),
    15: (EN.PAIGE_L, "wsl_elise_randall_enthusiastic_script-16", -23, 0.9, 0.9),
    16: (EN.TOBIN_A, "wsl_hanuman_welch_enthusiastic_script-7", -21, 0.9, 0.9),
    # NOTE: There is a new preprocessed version of Kai that has been included.
    # 17: (EN.KAI_M, ""),
    18: (EN.TRISTAN_F, "wsl_markatherlay_diphone_script-4", -22, 1, 1.1),
    19: (EN.PATRICK_K, "WSL_StevenWahlberg_DIPHONE_Script-6", -23, 1, 1.1),
    20: (EN.SOFIA_H__PROMO, "sofia_h__promo_3", -21, 1, 1.0),
    21: (EN.DAMIAN_P__PROMO, "damian_p__promo_2", -25, 1.15, 1.1),
    22: (EN.JODI_P__PROMO, "promo_script_8_hurley", -23, 1.05, 1.05),
    23: (EN.LEE_M__PROMO, "lee_m__promo_1", -19, 1.1, 1.05),
    24: (EN.SELENE_R__PROMO, "promo_script_1_rousseau", -24, 0.95, 0.95),
    25: (EN.MARI_MONGE__PROMO, "promo_script_1_monge", -23, 1.1, 1.1),
    26: (EN.WADE_C__PROMO, "wade_c__promo_3", -21, 1.05, 1.05),
    27: (EN.JOE_F__NARRATION, "johnhunerlach_enthusiastic_21", -24, 1.2, 1.1),
    28: (EN.JOE_F__RADIO, "johnhunerlach_diphone_1", -19, 1.15, 1.1),
    29: (EN.GARRY_J__STORY, "otis-jiry_the_happening_at_crossroads", -21, 1.2, 1.3),
    30: (EN.WADE_C__NARRATION, "wade_c__narration_70-75", -24, 1, 1.0),
    31: (EN.AVA_M__NARRATION, "ava_m__narration_40-45", -24, 1, 1.0),
    32: (EN.KAI_M__MANUAL_POST, "wsl_jackrutkowski_enthusiastic_script_27-processed", -22, 1.05, 1),
    33: (EN.JUDE_D__EN_GB, "enthusiastic_script_5_davis", -23, 1.15, 1.15),
    34: (EN.ERIC_S__EN_IE__PROMO, "promo_script_7_diamond", -22, 1.05, 1.05),
    35: (EN.CHASE_J, "promo_script_5_daniels", -21, 0.95, 1.0),
    36: (EN.DAN_FURCA__PROMO, "furca_audio_part3", -20, 1.15, 1.2),
    37: (EN.STEVE_B__PROMO, "promo_script_1_cupit_02", -25, 1.15, 1.15),
    38: (EN.BELLA_B__PROMO, "promo_script_5_tugman", -20, 1.15, 1.15),
    39: (EN.TILDA_C__PROMO, "promo_script_6_mckell", -24, 1.05, 1.1),
    40: (EN.CHARLIE_Z__PROMO, "promo_script_5_alexander", -23, 1, 1.0),
    41: (EN.PAUL_B__PROMO, "promo_script_9_williams", -19, 1.05, 1.05),
    42: (EN.SOFIA_H__CONVO, "sofia_h__convo_5", -22, 1.15, 1.15),
    43: (EN.AVA_M__CONVO, "ava_m__convo_6", -25, 0.95, 0.95),
    44: (EN.KAI_M__CONVO, "conversational_script_3_rutkowski", -24, 0.9, 0.9),
    45: (EN.NICOLE_L__CONVO, "conversational_script_1_sinclair", -23, 1.05, 1.05),
    46: (EN.WADE_C__CONVO, "wade_c__convo_2", -18, 1.05, 1.05),
    47: (EN.PATRICK_K__CONVO, "conversational_script_3_wahlberg", -23, 1.1, 1.05),
    48: (EN.VANESSA_N__CONVO, "conversational_script_4_murphy", -20, 1.2, 1.2),
    49: (EN.GIA_V, "narration_script_5_ruiz", -20, 1, 1.05),
    50: (EN.ANTONY_A, "narration_script_3_marrero", -23, 1.15, 1.15),
    51: (EN.JODI_P, "narration_script_2_hurley", -21, 1.1, 1.15),
    52: (EN.RAINE_B, "raine_b__narration_5", -22, 1.1, 1.1),
    53: (EN.OWEN_C, "narration_script_5_white", -24, 1.15, 1.1),
    54: (EN.ZACH_E__PROMO, "narration_script_5_jones", -21, 1.05, 1.05),
    55: (EN.GENEVIEVE_M, "narration_script_2_reppert", -20, 1.1, 1.05),
    56: (EN.JARVIS_H, "narration_script_5_hillknight", -21, 1.25, 1.25),
    57: (EN.THEO_K, "narration_script_8_kohnke", -27, 1.2, 1.15),
    58: (EN.JAMES_B, "james_b__narration_13", -22, 1.25, 1.25),
    59: (EN.TERRA_G, "terra_g__narration_1", -21, 1.1, 1.25),
    60: (EN.PHILIP_J, "anderson_narration_script-rx_loud_01", -20, 0.95, 1.0),
    61: (EN.MARCUS_G, "furca_audio_part1", -21, 1.2, 1.2),
    62: (EN.JORDAN_T, "narration_script_1_whiteside_processed", -21, 1.2, 1.15),
    63: (EN.FIONA_H, "hughes_narration_script_1", -21, 1.1, 1.1),
    64: (EN.ROXY_T, "topping_narration_script_1processed", -21, 1.15, 1.15),
    65: (EN.DONNA_W, "brookhyser_narration_script_1", -21, 1.2, 1.15),
    66: (EN.GREG_G, "greg_g__narration_1", -21, 1.05, 1.05),
    67: (EN.ZOEY_O, "helen_marion-rowe_script_1_processed", -21, 1.3, 1.4),
    68: (EN.KARI_N, "noble_narration_script_1", -21, 1.2, 1.15),
    69: (EN.DIARMID_C, "diarmid_c__narration_1", -21, 1.3, 1.3),
    70: (EN.ELIZABETH_U, "naration_script_6_stringer", -21, 1.1, 1.2),
    71: (EN.ALAN_T, "narration_script_1_frazer", -23, 1.2, 1.2),
    72: (EN.AVA_M__PROMO, "ava_m__promo_1", -23, 0.95, 1.0),
    73: (EN.TOBIN_A__PROMO, "tobin_a__promo_2", -23, 0.85, 0.8),
    74: (EN.TOBIN_A__CONVO, "conversational_script_1_welch_processed", -23, 0.9, 0.85),
    # TODO: Update these after v11 training.
    75: (EN.BEN_D, "daniel_barnett_narration_script-09-processed", -20, 1.0, 1.0),
    76: (EN.MICHAEL_V, "forsgren_narration_script-02-processed", -20, 1.0, 1.0),
    77: (EN.GRAY_L, "platis_narration_script-08-processed", -20, 1.0, 1.0),
    78: (EN.PAULA_R, "paula_narration_script-06-processed", -20, 1.0, 1.0),
    79: (EN.BELLA_B, "tugman_narration_script-05-processed", -20, 1.0, 1.0),
    80: (EN.MARCUS_G__CONVO, "marcus_g_conversational-03-processed", -20, 1.0, 1.0),
}
_SPKR_ID_TO_SESH: typing.Dict[int, typing.Tuple[Speaker, str, float, float, float]] = {
    **_SPKR_ID_TO_SESH,
    # NOTE: As a weak security measure, we assign random large numbers to custom voices, so
    # that they are hard to discover by querying the API. This was actually somewhat helpful
    # when we had to momentarily turn off our permissions verification during an outage.
    # NOTE: We have deprecated some of these custom voices. We didn't comment them out so that
    # we don't accidently reuse their ids.
    # 11541: (EN.archive.LINCOLN__CUSTOM, ""),
    # 13268907: (EN.archive.JOSIE__CUSTOM, ""),
    # 95313811: (EN.archive.JOSIE__CUSTOM__MANUAL_POST, ""),
    # 50794582: (EN.UNEEQ__ASB_CUSTOM_VOICE, "script-20-enthusiastic"),
    # 50794583: (EN.UNEEQ__ASB_CUSTOM_VOICE_COMBINED, "script-28-enthusiastic"),
    # 78252076: (EN.VERITONE__CUSTOM_VOICE, ""),
    70695443: (EN.SUPER_HI_FI__CUSTOM_VOICE, "promo_script_5_superhifi", -17, 0.95, 0.95),
    64197676: (EN.US_PHARMACOPEIA__CUSTOM_VOICE, "enthusiastic_script-22", -29, 1.4, 1.4),
    41935205: (HAPPIFY, "anna_long_emotional_clusters_1st_half_clean", -21, 1.25, 1.2),
    42400423: (TEC, "is_it_possible_to_become_invisible", -24, 1, 1),
    61137774: (EN.ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE, "sample_script_1", -18, 1.15, 1.2),
    # 30610881: (EN.VIACOM__CUSTOM_VOICE, "kelsey_speech_synthesis_section1"),
    # 50481197: (EN.HOUR_ONE_NBC__BB_CUSTOM_VOICE, "hour_one_nbc_dataset_5"),
    # 77552139: (EN.STUDY_SYNC__CUSTOM_VOICE, "fernandes_audio_5"),
    25502195: (EN.FIVE_NINE__CUSTOM_VOICE, "wsl_five9_audio_3", -22, 1, 1.05),
    # 81186157: (german.wsl.FIVE9_CUSTOM_VOICE__DE_DE, "janina_five9_script8"),
    # 29363869: (spanish.wsl.FIVE_NINE__CUSTOM_VOICE__ES_CO, "five9_spanish_script_8"),
    # 34957054: (portuguese.wsl.FIVE_NINE__CUSTOM_VOICE__PT_BR, "five9_portuguese_script_3"),
    45105608: (EN.SELECTQUOTE__CUSTOM_VOICE, "SelectQuote_Script2", -19, 1.15, 1.15),
}
SPKR_ID_TO_SESH: typing.Dict[int, Session]
SPKR_ID_TO_SESH = {k: Session(*args) for k, args in _SPKR_ID_TO_SESH.items()}


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
    spkr_id_to_sesh: typing.Dict[int, Session] = SPKR_ID_TO_SESH,
) -> typing.Tuple[Inputs, PreprocessedInputs]:
    """Validate and unpack the request object."""

    if not ("speaker_id" in request_args and "text" in request_args):
        message = "Must call with keys `speaker_id` and `text`."
        raise FlaskException(message, code="MISSING_ARGUMENT")

    speaker_id = request_args.get("speaker_id")
    xml = XMLType(request_args.get("text"))

    if not isinstance(speaker_id, (str, int)):
        message = "Speaker ID must be either an integer or string."
        raise FlaskException(message, code="INVALID_SPEAKER_ID")

    if isinstance(speaker_id, str) and not speaker_id.isdigit():
        message = "Speaker ID string must only consist of the symbols 0 - 9."
        raise FlaskException(message, code="INVALID_SPEAKER_ID")

    speaker_id = int(speaker_id)

    if not (isinstance(xml, str) and len(xml) < max_chars and len(xml) > 0):
        message = f"Text must be a string under {max_chars} characters and more than 0 characters."
        raise FlaskException(message, code="INVALID_TEXT_LENGTH_EXCEEDED")

    min_speaker_id = min(spkr_id_to_sesh.keys())
    max_speaker_id = max(spkr_id_to_sesh.keys())

    if not (
        (speaker_id >= min_speaker_id and speaker_id <= max_speaker_id)
        and speaker_id in spkr_id_to_sesh
    ):
        raise FlaskException("Speaker ID is invalid.", code="INVALID_SPEAKER_ID")

    session = spkr_id_to_sesh[speaker_id]

    gc.collect()

    try:
        return process_tts_inputs(
            tts, language_to_spacy[session.spkr.language], xml, session, device=DEVICE
        )
    except PublicSpeakerValueError as error:
        app.logger.exception("Invalid speaker: %r", xml)
        raise FlaskException(str(error), code="INVALID_SPEAKER_ID")
    except PublicTextValueError as error:
        app.logger.exception("Invalid text: %r", xml)
        raise FlaskException(str(error), code="INVALID_TEXT")
    except PublicValueError as error:
        app.logger.exception("Invalid xml: %r", xml)
        raise FlaskException(str(error), code="INVALID_XML")
    except BaseException:
        app.logger.exception("Unknown error text: %r", xml)
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
    TODO: Create an end point that accepts XML rather than JSON.

    Usage:
        http://127.0.0.1:8000/api/text_to_speech/stream?speaker_id=46&text="Hello there"

    Returns: `audio/mpeg` streamed in chunks given that the arguments are valid.
    """
    request_args = request.get_json() if request.method == "POST" else request.args
    request_args = typing.cast(RequestArgs, request_args)
    input = validate_and_unpack(request_args, TTS_PACKAGE, LANGUAGE_TO_SPACY)
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "X-Text-Length": len(request_args.get("text")),
    }
    output_flags = ("-f", "mp3", "-b:a", "192k")
    generator = tts_ffmpeg_generator(
        TTS_PACKAGE, *input, **cf.get(), logger=app.logger, output_flags=output_flags
    )
    return Response(generator, headers=headers, mimetype="audio/mpeg")


if __name__ == "__main__" or "GUNICORN" in os.environ:
    app.logger.info("Device: %s", DEVICE)
    app.logger.info("PyTorch version: %s", torch.__version__)
    app.logger.info("PyTorch CUDA version: %s", torch.version.cuda)
    app.logger.info("Found MKL: %s", torch.backends.mkl.is_available())
    app.logger.info("Threads: %s", torch.get_num_threads())
    app.logger.info("Speaker Ids: %s", pprinter.pformat(SPKR_ID_TO_SESH))

    configure()

    # NOTE: These models are cached globally to enable sharing between processes, learn more:
    # https://github.com/benoitc/gunicorn/issues/2007
    TTS_PACKAGE = typing.cast(TTSPackage, load(TTS_PACKAGE_PATH, DEVICE))

    vocab = set(TTS_PACKAGE.session_vocab())
    app.logger.info("Loaded speakers: %s", "\n".join(list(set(str(s.spkr) for s in vocab))))

    for id, other_sesh in SPKR_ID_TO_SESH.items():
        for sesh in vocab:
            if (
                other_sesh.label == sesh.label
                and other_sesh.spkr.label == sesh.spkr.label
                and sesh != other_sesh
            ):
                app.logger.warning(f"Model session is different: {other_sesh} → {sesh}")
                SPKR_ID_TO_SESH[id] = sesh

    for session in SPKR_ID_TO_SESH.values():
        if session not in vocab:
            if not any(session.spkr is sesh.spkr for sesh in vocab):
                app.logger.warning(f"Speaker not found in model vocab: {session.spkr}")
            else:
                app.logger.warning(f"Session not found in model vocab: {session}")
                avail_sessions = [s.label for s in vocab if s.spkr == session.spkr]
                if len(avail_sessions) > 0:
                    app.logger.warning(f"Sessions available: {avail_sessions}")

    languages = set(s.spkr.language for s in vocab)
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
