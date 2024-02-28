import tempfile
import time
from dataclasses import dataclass, asdict

import numpy as np
import requests
import resampy
import soundfile as sf
from dotenv import dotenv_values
from package_utils.audio import SAMPLE_RATE
from package_utils.environment import logger

api_keys = dotenv_values(".env")
kong_staging = api_keys["KONG_STAGING"]
TTS_API_ENDPOINT = (
    "https://staging.tts.wellsaidlabs.com/api/text_to_speech/stream"
)
np.set_printoptions(threshold=np.inf)


@dataclass
class APITransaction:
    """Class to hold request to and response from WSL's API.
    Attributes:
        text: The text to render speech from
        speaker_id: The WSL speaker ID
        speaker: The name of the speaker+style, e.g. Wade_C__NARRATION
        endpoint: The API endpoint to query
        wav_data: The API audio response converted to np.ndarray
    """

    text: str
    speaker_id: int
    speaker: str
    model_version: str
    endpoint: str = TTS_API_ENDPOINT
    wav_data: np.ndarray = None

    def __post_init__(self):
        self.headers = {
            "X-Api-Key": kong_staging,
            "Accept-Version": self.model_version,
            "Content-Type": "application/json",
        }

    def as_dict(self):
        """Return a dictionary version of this class. Do not include headers as
        they contain sensitive information.
        """
        unwanted_attrs = {"headers"}
        api_transaction_dict = {
            k: v for k, v in asdict(self).items() if k not in unwanted_attrs
        }
        return api_transaction_dict


def query_wsl_api(
    task: APITransaction, attempt_number=1, max_attempts=3
) -> APITransaction:
    """Function to query WSL's API and convert the resulting response to
    np.ndarray. This is kept outside the class definition to simplify
    multiprocessing.
    Args:
        task: The inputs to the API.
        attempt_number: The current attempt out of max_attempts.
        max_attempts: The number of times to attempt an API call before
            returning the original request.
    Returns:
        task: The same input object with the wav_data attribute populated.
    Raises:
        ValueError if the requested speaker isn't available in the requested
            model version.
    """
    if attempt_number > max_attempts:
        logger.error(
            "Max retries reached. Returning original APITransaction object"
        )
        return task

    json_data = {
        "speaker_id": task.speaker_id,
        "text": task.text,
        "consumerId": "id",
        "consumerSource": "source",
    }
    response = requests.post(
        task.endpoint, headers=task.headers, json=json_data
    )

    if response.status_code != 200 and response.status_code != 400:
        logger.warning(
            f"Received status code {response.status_code}. "
            f"Attempt {attempt_number}/{max_attempts}"
        )
        attempt_number += 1
        time.sleep(1)
        query_wsl_api(task, attempt_number)
    elif (
        response.status_code == 400
        and b"INVALID_SPEAKER_ID" in response.content
    ):
        logger.error(
            f"Invalid speaker id: {task.speaker_id} for model version: "
            f"{task.model_version}"
        )
        raise ValueError
    task.wav_data = response_to_wav(response)
    return task


def response_to_wav(response: requests.Response) -> np.ndarray:
    """Convert the response from the API to np.ndarray by writing it to a
    temporary file.
    Args:
        response: The response from the API
    Returns:
        wav_data: The response contents converted to np.ndarray
    Raises:
        RuntimeError if the response cannot be converted to np.ndarray
    """
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav") as fp:
        fp.write(response.content)
        fp.seek(0)
        wav_data, source_sr = sf.read(fp.name)
        wav_data = resampy.resample(wav_data, source_sr, SAMPLE_RATE)
        wav_data = np.trim_zeros(wav_data)
    if isinstance(wav_data, np.ndarray):
        return wav_data

    else:
        logger.error(f"API response: {response} could not be converted to wav")
        raise RuntimeError
