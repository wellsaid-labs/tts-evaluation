from pathlib import Path

from lib.datasets.lj_speech import LINDA_JOHNSON, lj_speech_dataset
from lib.datasets.m_ailabs import (
    ELIZABETH_KLETT,
    ELLIOT_MILLER,
    JUDY_BIEBER,
    MARY_ANN,
    m_ailabs_en_uk_elizabeth_klett_speech_dataset,
    m_ailabs_en_us_elliot_miller_speech_dataset,
    m_ailabs_en_us_judy_bieber_speech_dataset,
    m_ailabs_en_us_mary_ann_speech_dataset,
)
from lib.datasets.utils import (
    Alignment,
    Example,
    Speaker,
    dataset_generator,
    dataset_loader,
    precut_dataset_loader,
)

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.

HILARY_NORIEGA = Speaker("Hilary Noriega")
ALICIA_HARRIS = Speaker("Alicia Harris")
MARK_ATHERLAY = Speaker("Mark Atherlay")
SAM_SCHOLL = Speaker("Sam Scholl")


def hilary_noriega_speech_dataset(*args, speaker: Speaker = HILARY_NORIEGA, **kwargs):
    return _dataset_loader(*args, speaker=speaker, **kwargs)  # type: ignore


WSL_GCS_PATH = "gs://wellsaid_labs_datasets/"


def _dataset_loader(directory: Path, speaker: Speaker, gcs_path: str = WSL_GCS_PATH, **kwargs):
    label = HILARY_NORIEGA.name.lower().replace(" ", "_")
    return dataset_loader(directory, label, gcs_path + label, speaker, **kwargs)


__all__ = [
    "Speaker",
    "Example",
    "Alignment",
    "dataset_generator",
    "dataset_loader",
    "precut_dataset_loader",
    "lj_speech_dataset",
    "m_ailabs_en_us_judy_bieber_speech_dataset",
    "m_ailabs_en_us_mary_ann_speech_dataset",
    "m_ailabs_en_us_elliot_miller_speech_dataset",
    "m_ailabs_en_uk_elizabeth_klett_speech_dataset",
    "ALICIA_HARRIS",
    "ELIZABETH_KLETT",
    "ELLIOT_MILLER",
    "JUDY_BIEBER",
    "LINDA_JOHNSON",
    "MARK_ATHERLAY",
    "MARY_ANN",
    "SAM_SCHOLL",
]
