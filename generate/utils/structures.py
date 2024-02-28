from dataclasses import dataclass, asdict
from generate.utils.api import APITransaction
from google.cloud import storage
import typing
import json
import yaml
import pandas as pd
from package_utils.environment import logger
from lib.test_cases import TestCases

DATASET_TYPES = {"absolute", "comparative"}
test_cases = TestCases()


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

    model_versions: typing.List[str]
    gcs_bucket: str
    gcs_path: str
    speakers: typing.List[str]
    clips_per_text: int
    dataset_type: str
    custom_texts: typing.List[str] = None
    predefined_texts: typing.List[str] = None
    combined_texts: typing.List[str] = None

    def __post_init__(self):
        if self.dataset_type not in DATASET_TYPES:
            logger.error(f"Dataset type must be one of {DATASET_TYPES}")
            raise AttributeError
        combined_texts = self.custom_texts
        for text_source in self.predefined_texts:
            list_of_texts = getattr(test_cases, text_source)
            if not list_of_texts:
                logger.error(
                    f"Predefined text source {text_source} could not be found"
                )
                raise ValueError
            combined_texts.extend(list_of_texts)
        self.combined_texts = combined_texts

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
