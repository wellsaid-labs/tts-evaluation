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

      $ CHECKPOINTS="v11_2023_04_24_staging"  # Example: v9
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
KAI_M__MP = EN.KAI_M__MANUAL_POST
GEN_M_CONVO = EN.GENEVIEVE_M__CONVO

# NOTE: The keys need to stay the same for backwards compatibility.
_MARKETPLACE = {
    # NOTE: These 3 are open-source voices that didn't consent to be on our platform.
    # 0: (english.m_ailabs.JUDY_BIEBER, ""),
    # 1: (english.m_ailabs.MARY_ANN, ""),
    # 2: (english.lj_speech.LINDA_JOHNSON, ""),
    3: (EN.ALANA_B, "script_3", -25.403, 0.857, 0.749),
    4: (EN.RAMONA_J, "7", -25.106, 0.806, 0.818),
    5: (EN.RAMONA_J__CUSTOM, "sukutdental_021819", -22.658, 0.779, 0.798),
    # NOTE: This is open-source voice that didn't consent to be on our platform.
    # NOTE: This speaker was released twice on accident with different ids, so it's in this list
    # twice.
    # 6: (english.lj_speech.LINDA_JOHNSON, ""),
    # NOTE: There is a new preprocessed version of Wade that has been included.
    # 7: (EN.WADE_C, ""),
    8: (EN.SOFIA_H, "14", -22.559, 0.881, 0.92),
    # NOTE: David asked for his voice to be removed from the platform.
    # 9: (EN.DAVID_D, ""),
    10: (EN.VANESSA_N, "76-81", -22.563, 0.83, 0.802),
    11: (EN.ISABEL_V, "heather_4-21_a", -33.268, 0.835, 0.852),
    # NOTE: There is a new preprocessed version of Ava that has been included.
    # 12: (EN.AVA_M, ""),
    13: (EN.JEREMY_G, "copy_of_drake_jr-script_46-51", -25.589, 0.933, 0.906),
    14: (EN.NICOLE_L, "copy_of_wsl_-_megansinclairscript40-45", -21.243, 0.863, 0.88),
    15: (EN.PAIGE_L, "wsl_elise_randall_enthusiastic_script-16", -22.622, 1.103, 1.095),
    16: (EN.TOBIN_A, "wsl_hanuman_welch_enthusiastic_script-7", -21.322, 1.082, 1.123),
    # NOTE: There is a new preprocessed version of Kai that has been included.
    # 17: (EN.KAI_M, ""),
    18: (EN.TRISTAN_F, "wsl_markatherlay_diphone_script-4", -21.991, 1.004, 0.926),
    19: (EN.PATRICK_K, "WSL_StevenWahlberg_DIPHONE_Script-6", -22.686, 1.004, 0.93),
    20: (EN.SOFIA_H__PROMO, "promo_script_3_walker", -20.589, 0.996, 0.99),
    21: (EN.DAMIAN_P__PROMO, "promo_script_2_papadopoulos", -24.651, 0.888, 0.911),
    22: (EN.JODI_P__PROMO, "promo_script_8_hurley", -23.02, 0.972, 0.955),
    23: (EN.LEE_M__PROMO, "promo_script_1_la_comb", -19.306, 0.903, 0.938),
    24: (EN.SELENE_R__PROMO, "promo_script_1_rousseau", -23.704, 1.06, 1.068),
    25: (EN.MARI_MONGE__PROMO, "promo_script_1_monge", -22.736, 0.907, 0.907),
    26: (EN.WADE_C__PROMO, "promo_script_3_scholl", -20.94, 0.942, 0.941),
    27: (EN.JOE_F__NARRATION, "johnhunerlach_enthusiastic_21", -23.93, 0.841, 0.89),
    28: (EN.JOE_F__RADIO, "johnhunerlach_diphone_1", -19.325, 0.885, 0.914),
    29: (EN.GARRY_J__STORY, "otis-jiry_the_happening_at_crossroads", -21.067, 0.824, 0.781),
    30: (EN.WADE_C__MANUAL_POST, "70-75", -24.348, 0.99, 0.993),
    31: (EN.AVA_M__MANUAL_POST, "copy_of_well_said_script_40-45-processed", -23.934, 0.983, 0.98),
    32: (KAI_M__MP, "wsl_jackrutkowski_enthusiastic_script_27-processed", -21.885, 0.936, 0.979),
    33: (EN.JUDE_D__EN_GB, "enthusiastic_script_5_davis", -23.063, 0.854, 0.854),
    34: (EN.ERIC_S__EN_IE__PROMO, "promo_script_7_diamond", -21.925, 0.966, 0.953),
    35: (EN.CHASE_J__PROMO, "promo_script_5_daniels", -20.986, 1.048, 1.019),
    36: (EN.DAN_FURCA__PROMO, "furca_audio_part3", -20.418, 0.882, 0.837),
    37: (EN.STEVE_B__PROMO, "promo_script_1_cupit_02", -24.992, 0.851, 0.86),
    38: (EN.BELLA_B__PROMO, "promo_script_5_tugman", -20.06, 0.854, 0.872),
    39: (EN.TILDA_C__PROMO, "promo_script_6_mckell", -24.004, 0.947, 0.925),
    40: (EN.CHARLIE_Z__PROMO, "promo_script_5_alexander", -22.789, 1.007, 0.99),
    41: (EN.PAUL_B__PROMO, "promo_script_9_williams", -18.963, 0.971, 0.972),
    42: (EN.SOFIA_H__CONVO, "conversational_script_5_walker", -22.384, 0.885, 0.873),
    43: (EN.AVA_M__CONVO, "conversational_script_6_harris", -24.512, 1.052, 1.036),
    44: (EN.KAI_M__CONVO, "conversational_script_3_rutkowski", -24.467, 1.128, 1.111),
    45: (EN.NICOLE_L__CONVO, "conversational_script_1_sinclair", -23.253, 0.941, 0.942),
    46: (EN.WADE_C__CONVO, "conversational_script_2_scholl", -18.215, 0.956, 0.951),
    47: (EN.PATRICK_K__CONVO, "conversational_script_3_wahlberg", -22.596, 0.928, 0.933),
    48: (EN.VANESSA_N__CONVO, "conversational_script_4_murphy", -20.189, 0.834, 0.818),
    49: (EN.GIA_V, "narration_script_5_ruiz", -19.656, 0.997, 0.94),
    50: (EN.ANTONY_A, "narration_script_3_marrero", -22.557, 0.859, 0.866),
    51: (EN.JODI_P, "narration_script_2_hurley", -21.025, 0.89, 0.869),
    52: (EN.RAINE_B, "narration_script_5_black", -21.801, 0.91, 0.913),
    53: (EN.OWEN_C, "narration_script_5_white", -23.874, 0.884, 0.9),
    54: (EN.ZACH_E, "narration_script_5_jones", -21.42, 0.97, 0.957),
    55: (EN.GENEVIEVE_M, "narration_script_2_reppert", -20.056, 0.919, 0.933),
    56: (EN.JARVIS_H, "narration_script_5_hillknight", -21.163, 0.797, 0.788),
    57: (EN.THEO_K, "narration_script_8_kohnke", -26.634, 0.834, 0.86),
    58: (EN.JAMES_B, "newman_final_page_13", -21.501, 0.796, 0.802),
    59: (EN.TERRA_G, "narration_script_1_parrish", -21.077, 0.9, 0.788),
    60: (EN.PHILIP_J, "anderson_narration_script-rx_loud_01", -20.389, 1.061, 1.017),
    # NOTE: v11 didn't have `furca_audio_part1` available, so this was updated to
    # `furca_audio_part2`.
    61: (EN.MARCUS_G, "furca_audio_part2", -21.035, 0.801, 0.837),
    62: (EN.JORDAN_T, "narration_script_1_whiteside_processed", -21.034, 0.843, 0.86),
    63: (EN.FIONA_H, "hughes_narration_script_1", -21.064, 0.902, 0.908),
    64: (EN.ROXY_T, "topping_narration_script_1processed", -21.035, 0.859, 0.871),
    65: (EN.DONNA_W, "brookhyser_narration_script_1", -21.054, 0.85, 0.888),
    66: (EN.GREG_G, "lloyd_narration_script_1", -20.803, 0.954, 0.972),
    67: (EN.ZOEY_O, "helen_marion-rowe_script_1_processed", -21.039, 0.776, 0.709),
    68: (EN.KARI_N, "noble_narration_script_1", -21.092, 0.841, 0.857),
    69: (EN.DIARMID_C, "cherry_narration_script_1", -21.002, 0.759, 0.781),
    70: (EN.ELIZABETH_U, "naration_script_6_stringer", -21.094, 0.891, 0.847),
    71: (EN.ALAN_T, "narration_script_1_frazer", -22.843, 0.842, 0.819),
    72: (EN.AVA_M__PROMO, "promo_script_1_harris", -23.441, 1.032, 1.016),
    73: (EN.TOBIN_A__PROMO, "promo_script_2_welch_processed", -22.761, 1.176, 1.221),
    74: (EN.TOBIN_A__CONVO, "conversational_script_1_welch_processed", -23.154, 1.123, 1.158),
    75: (EN.BEN_D, "daniel_barnett_narration_script-09-processed", -21.029, 0.995, 1.018),
    76: (EN.MICHAEL_V, "forsgren_narration_script-02-processed", -21.052, 0.936, 0.883),
    77: (EN.GRAY_L, "platis_narration_script-08-processed", -21.072, 0.956, 0.913),
    78: (EN.PAULA_R, "paula_narration_script-06-processed", -21.04, 1.045, 1.021),
    79: (EN.BELLA_B, "tugman_narration_script-05-processed", -21.034, 0.84, 0.836),
    80: (EN.MARCUS_G__CONVO, "marcus_g_conversational-03-processed", -22.55, 0.995, 1.009),
    81: (EN.JORDAN_T__CONVO, "jordan_conversational_script_3_processed", -21.026, 1.04, 1.048),
    82: (EN.JODI_P__CONVO, "jodi_conversational_script_4_processed", -21.047, 0.925, 0.915),
    83: (EN.DIARMID_C__PROMO, "diarmid_promotional_script_1_processed", -21.063, 0.852, 0.843),
    84: (EN.JARVIS_H__CONVO, "jarvis_conversational_script_3_processed", -21.038, 0.91, 0.876),
    85: (EN.JARVIS_H__PROMO, "jarvis_promotional_script_1_processed", -21.045, 0.969, 0.95),
    86: (EN.GIA_V__CONVO, "gia_conversational_script_2_processed", -20.594, 1.107, 1.059),
    87: (EN.GIA_V__PROMO, "gia_promotional_script_2_processed", -21.029, 1.121, 1.153),
    88: (EN.OWEN_C__CONVO, "owen_conversational_script_2_processed", -21.069, 0.892, 0.933),
    89: (EN.OWEN_C__PROMO, "owen_promotional_script_2_processed", -21.056, 1.065, 0.982),
    90: (EN.PHILIP_J__CONVO, "philip_conversational_script_2_processed", -21.085, 1.027, 1.05),
    91: (GEN_M_CONVO, "genevieve_conversational_script_2_processed", -21.005, 1.137, 1.16),
    92: (EN.ANTONY_A__CONVO, "antony_conversational_script_1_processed", -21.034, 1.0, 0.986),
    93: (EN.BELLA_B__CONVO, "bella_conversational_script_1_processed", -21.082, 0.874, 0.891),
    94: (EN.ERIC_S, "eric_narration_script_1_processed", -21.044, 0.947, 0.974),
    95: (EN.GREG_G__CONVO, "greg_conversational_script_2_processed", -21.056, 1.018, 1.005),
    96: (EN.JENSEN_X, "jensen_narration_script_1_processed", -21.012, 1.019, 0.946),
    97: (EN.JACK_C, "wellsaid_script2_processed", -21.049, 0.939, 0.968),
    98: (EN.OLIVER_S, "oliver_narration_script2_processed", -21.085, 0.829, 0.829),
    99: (EN.HANNAH_A, "hannah_narration_script_1_processed", -21.064, 0.961, 1.003),
    100: (EN.LORENZO_D, "lorenzo_narration_script_1_processed", -21.036, 0.895, 0.881),
    101: (EN.LULU_G, "lulu_narration_script_1_processed", -21.022, 1.054, 1.068),
    102: (EN.ABBI_D, "abbi_narration_script_2_processed", -21.065, 0.829, 0.809),
    103: (EN.FIONA_H_IE, "fiona_narration_script_2_processed", -21.078, 0.929, 0.884),
}
_SPKR_ID_TO_SESH: typing.Dict[int, typing.Tuple[Speaker, str, float, float, float]] = {
    **_MARKETPLACE,
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
    70695443: (EN.SUPER_HI_FI__CUSTOM_VOICE, "promo_script_5_superhifi", -16.996, 1.04, 1.032),
    64197676: (EN.US_PHARMACOPEIA__CUSTOM_VOICE, "enthusiastic_script-22", -28.758, 0.717, 0.705),
    41935205: (HAPPIFY, "anna_long_emotional_clusters_1st_half_clean", -20.992, 0.808, 0.845),
    42400423: (TEC, "is_it_possible_to_become_invisible", -23.564, 1.018, 0.991),
    61137774: (
        EN.ENERGY_INDUSTRY_ACADEMY__CUSTOM_VOICE,
        "sample_script_1",
        -23.403,
        0.876,
        0.839,
    ),
    # 30610881: (EN.VIACOM__CUSTOM_VOICE, "kelsey_speech_synthesis_section1"),
    # 50481197: (EN.HOUR_ONE_NBC__BB_CUSTOM_VOICE, "hour_one_nbc_dataset_5"),
    # 77552139: (EN.STUDY_SYNC__CUSTOM_VOICE, "fernandes_audio_5"),
    # NOTE: v11 didn't have `wsl_five9_audio_3` available, so this was updated to
    # `wsl_five9_audio_1`.
    25502195: (EN.FIVE_NINE__CUSTOM_VOICE, "wsl_five9_audio_1", -20.753, 0.889, 0.93),
    # 81186157: (german.wsl.FIVE9_CUSTOM_VOICE__DE_DE, "janina_five9_script8"),
    # 29363869: (spanish.wsl.FIVE_NINE__CUSTOM_VOICE__ES_CO, "five9_spanish_script_8"),
    # 34957054: (portuguese.wsl.FIVE_NINE__CUSTOM_VOICE__PT_BR, "five9_portuguese_script_3"),
    45105608: (EN.SELECTQUOTE__CUSTOM_VOICE, "SelectQuote_Script2", -19.363, 0.886, 0.883),
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
                app.logger.warning(
                    f"Model session is different: {other_sesh} → {sesh} "
                    f'`"{sesh.label}", {sesh.loudness}, {sesh.tempo}, {sesh.spkr_tempo}),`'
                )
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
