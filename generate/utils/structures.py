import json
import random
import typing
from dataclasses import asdict, dataclass

import pandas as pd
import yaml
from generate.utils.api import APITransaction
from google.cloud import storage
from package_utils.environment import logger

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
