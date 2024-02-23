from dataclasses import dataclass, asdict
from generate._utils.api import APITransaction
import typing
import json
import logging
logger = logging.getLogger(__name__)

DATASET_TYPES = {"absolute", "comparative"}


@dataclass
class DatasetConfig:
    model_versions: str
    texts: typing.List
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
