from dataclasses import dataclass, asdict
from generate._utils.api import APITransaction
from google.cloud import storage
import typing
import json
import logging
import pandas as pd
logger = logging.getLogger(__name__)

DATASET_TYPES = {"absolute", "comparative"}


@dataclass
class DatasetConfig:
    model_versions: str
    texts: typing.List
    gcs_bucket: str
    gcs_path: str
    speakers: typing.List
    clips_per_text: int
    dataset_type: str

    def __post_init__(self):
        assert self.dataset_type in DATASET_TYPES

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as ifp:
            d = json.loads(ifp.read())
            logger.info(d)
        return cls(**d)

    def as_dict(self):
        return asdict(self)


@dataclass
class AudioDataset:
    config: DatasetConfig
    audio: typing.List[APITransaction]

    def _audio_as_parquet(self):
        return pd.DataFrame([a.as_dict() for a in self.audio]).to_parquet()

    def upload_blob_from_memory(self):
        """Uploads a file to Google Cloud Storage"""
        bucket_name = self.config.gcs_bucket
        destination_blob_name = self.config.gcs_path
        contents = self._audio_as_parquet()
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_string(contents)

        print(
            f"{destination_blob_name} uploaded to {bucket_name}."
        )
