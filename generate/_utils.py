import json
import random
import tempfile
import time
import typing
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import requests
import resampy
import soundfile as sf
import yaml
from dotenv import dotenv_values
from google.cloud import storage

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


DATASET_TYPES = {"absolute", "comparative"}


@dataclass
class DatasetConfig:
    """Class to hold dataset configurations.
    Attributes:
        model_versions: WSL model versions as they are named in kubernetes
        gcs_bucket: The bucket in which to store results
        gcs_path: The path within `gcs_bucket`
        speakers: List of (speaker, speaker_id) tuples
        clips_per_text: The number of times to generate each text
        dataset_type: Comparative for datasets meant to be compared to one
            another, absolute for datasets to be evaluated in and of themselves
        custom_texts: List of custom inputs to render audio with
        predefined_texts: List of existing texts, found in _test_cases.py
    """

    gcs_path: str
    speakers: typing.List[tuple]
    clips_per_text: int
    dataset_type: str
    model_versions: typing.List[str] = None
    gcs_bucket: str = "tts_evaluations"
    task_limit: int = None
    custom_texts: typing.List[str] = None
    predefined_texts: typing.List[str] = None
    combined_texts: typing.List[str] = None

    def __post_init__(self):
        if self.dataset_type not in DATASET_TYPES:
            logger.error(f"Dataset type must be one of {DATASET_TYPES}")
            raise AttributeError
        combined_texts = pd.concat(
            [
                pd.read_csv(i, usecols=[0], names=["text"])
                for i in self.predefined_texts
            ]
        )
        if self.custom_texts:
            for i in self.custom_texts:
                combined_texts.loc[-1] = i
        self.combined_texts = combined_texts.text.tolist()

    @classmethod
    def from_file(cls, conf_path):
        supported = {"json", "yaml", "yml"}
        if not any(conf_path.endswith(s) for s in supported):
            logger.error(
                f"Unsupported filetype in config file {conf_path}. "
                f"Use one of these file extensions: {supported}"
            )
            raise ValueError
        with open(conf_path) as ifp:
            if conf_path.endswith("json"):
                d = json.loads(ifp.read())
            elif conf_path.endswith("yaml") or conf_path.endswith("yml"):
                d = yaml.safe_load(ifp.read())
        return cls(**d)

    def as_dict(self):
        return asdict(self)

    def to_file(self, extension, path):
        supported = {"json", "yaml", "yml"}
        if not any(extension.endswith(s) for s in supported):
            logger.error(
                f"{extension} is not supported. Use one of {supported}"
            )
            raise ValueError
        d = self.as_dict()

        with open(path, "w") as ofp:
            if extension == "json":
                ofp.write(json.dumps(d))
            elif extension in {"yaml", "yml"}:
                ofp.write(yaml.dump(d))

    def get_api_transactions(self) -> typing.List[APITransaction]:
        tasks = [
            APITransaction(
                text=text,
                speaker_id=speaker_id,
                speaker=speaker,
                model_version=model_version,
            )
            for model_version in self.model_versions
            for speaker, speaker_id in self.speakers
            for text in self.combined_texts
        ] * self.clips_per_text
        if self.task_limit > 0:
            tasks = random.sample(tasks, self.task_limit)
        return tasks


@dataclass
class AudioDataset:
    """Class to hold dataset config and corresponding audio
    Attributes:
        config: The DatasetConfig that was used to generate audio
        audio: List of APITransaction objects
    """

    config: DatasetConfig
    audio: typing.List[APITransaction]

    def _audio_as_parquet(self) -> bytes:
        """Save the list of APITransaction objects as a pandas dataframe and
        return the corresponding bytes in parquet format
        Args:
            self: The class itself. Specifically, the list of APITransaction
                objects comprising self.audio
        Returns:
            parquet_bytes: The dataframe converted to parquet, expressed in
                bytes
        """
        df = pd.DataFrame([a.as_dict() for a in self.audio])
        for col in [
            "model_version",
            "text",
            "speaker",
            "speaker_id",
            "endpoint",
        ]:
            df[f"{col}"] = df[f"{col}"].astype("category")
        parquet_bytes = df.to_parquet()
        return parquet_bytes

    def upload_blob_from_memory(self) -> None:
        """Save an object in memory to Google Cloud Storage"""
        bucket_name = self.config.gcs_bucket
        destination_blob_name = self.config.gcs_path
        contents = self._audio_as_parquet()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(contents)
        logger.info(f"{destination_blob_name} uploaded to {bucket_name}.")
