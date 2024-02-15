"""Function to send requests to WSL's API
"""

import tempfile

import numpy as np
import requests
import resampy
import soundfile as sf
from dotenv import dotenv_values
from pymcd import Calculate_MCD
from scipy.io import wavfile


from dataclasses import dataclass

api_keys = dotenv_values(".env")
kong_staging = api_keys["KONG_STAGING"]
TTS_API_ENDPOINT = (
    "https://staging.tts.wellsaidlabs.com/api/text_to_speech/stream"
)
SAMPLE_RATE = 24000


@dataclass
class AudioData:
    """Class to store audio data"""
    text: str
    wav_data: np.ndarray
    speaker: str
    respell_text: str
    respell_wav_data: np.ndarray
    mcd_value: float = None
    mcd_dtw_value: float = None
    mcd_dtw_sl_value: float = None
    model_version: str = None


@dataclass
class APIInput:
    text: str
    respell_xml: str
    respell_colon: str
    speaker_id: int
    speaker: str
    model_version: str

    def __post_init__(self):
        self.headers = {
            "X-Api-Key": kong_staging,
            "Accept-Version": self.model_version,
            "Content-Type": "application/json",
        }


mcd_dtw_sl = Calculate_MCD(MCD_mode="dtw_sl")
mcd_plain = Calculate_MCD(MCD_mode="plain")
mcd_dtw = Calculate_MCD(MCD_mode="dtw")


def get_mcd(audio_data):
    """This version of MCD expects audio files, so we write the waveform to a
    file using scipy.wavefile. It could be improved by forking the repo and
    augmenting it to accept raw waveforms"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as exp_audio:
        wavfile.write(exp_audio.name, SAMPLE_RATE, audio_data.wav_data)
        with tempfile.NamedTemporaryFile(suffix=".wav") as ref_audio:
            wavfile.write(
                ref_audio.name, SAMPLE_RATE, audio_data.respell_wav_data
            )

            audio_data.mcd_dtw_value = mcd_dtw.calculate_mcd(
                ref_audio.name, exp_audio.name
            )
            audio_data.mcd_dtw_sl_value = mcd_dtw_sl.calculate_mcd(
                ref_audio.name, exp_audio.name
            )
    return audio_data


def response_to_wav(response, speaker, bandpass_filter=False):
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav") as fp:
            fp.write(response.content)
            fp.seek(0)
            wav_data, source_sr = sf.read(fp.name)
            wav_data = resampy.resample(wav_data, source_sr, SAMPLE_RATE)
            wav_data = np.trim_zeros(wav_data)
            return {
                "wav_data": wav_data,
            }
    except Exception as exc:
        print(exc)
        return f"{speaker}: {exc}"


def query_tts_api(
    task: APIInput, retry_number=0
):
    """Send speaker_id and word to TTS API and convert the resulting file to
    np.ndarray. kwargs are used here to allow for easy multiprocessing
    Args:
        task (APIInput): The task containing all information needed for the API call
    Returns:
        (dict): Dictionary with the inputs as well as the wav_data array
    """
    word_req = {
        "speaker_id": task.speaker_id,
        "text": task.text,
        "consumerId": "id",
        "consumerSource": "source",
    }

    if retry_number <= 2:
        try:
            word_response = requests.post(
                TTS_API_ENDPOINT,
                headers=task.headers,
                json=word_req,
            )
            word_data = response_to_wav(word_response, task.speaker)

            if task.model_version in ["v10", "v11", "v11-1"]:
                respell_req = {
                    "speaker_id": task.speaker_id,
                    "text": (
                        task.respell_colon
                        if task.model_version == "v10"
                        else task.respell_xml
                    ),
                    "consumerId": "id",
                    "consumerSource": "source",
                }
                respell_response = requests.post(
                    TTS_API_ENDPOINT,
                    headers=task.headers,
                    json=respell_req,
                )
                respell_data = response_to_wav(respell_response, task.speaker)
            else:
                respell_data = {}

            if isinstance(respell_data, dict) and isinstance(word_data, dict):
                _return = AudioData(
                    text=task.text,
                    speaker=task.speaker,
                    respell_text=(
                        task.respell_colon
                        if task.model_version == "v10"
                        else task.respell_xml
                    ),
                    respell_wav_data=respell_data.get("wav_data"),
                    # respell_spectrogram=respell_data.get("spectrogram"),
                    wav_data=word_data["wav_data"],
                    # frequencies=word_data["frequencies"],
                    # times=word_data["times"],
                    # spectrogram=word_data["spectrogram"],
                    model_version=task.model_version,
                )
                return _return
            else:
                return "exception"
        except:
            retry_number += 1
            query_tts_api(task, retry_number=retry_number)
    else:
        return "max retries reached"
