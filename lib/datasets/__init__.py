import typing
from pathlib import Path

from lib.datasets import m_ailabs
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
    Span,
    Speaker,
    conventional_dataset_loader,
    dataset_loader,
    span_generator,
)

# TODO: Consider not using the actors realnames in the codebase in an effort to protect their
# privacy.
# TODO: Consider updating M-AILABS and LJSpeech to Google Storage, so that we can download
# and upload them faster. It'll also give us protection, if the datasets are deleted.

HILARY_NORIEGA = Speaker("Hilary Noriega")
ALICIA_HARRIS = Speaker("Alicia Harris")
MARK_ATHERLAY = Speaker("Mark Atherlay")
SAM_SCHOLL = Speaker("Sam Scholl")
DataLoader = typing.Callable[[Path], typing.List[Example]]


def hilary_noriega_speech_dataset(*args, **kwargs) -> typing.List[Example]:
    kwargs.update({"speaker": HILARY_NORIEGA})
    return _dataset_loader(*args, **kwargs)


WSL_GCS_PATH = "gs://wellsaid_labs_datasets"


def _dataset_loader(
    directory: Path, speaker: Speaker, gcs_path: str = WSL_GCS_PATH, **kwargs
) -> typing.List[Example]:
    label = speaker.name.lower().replace(" ", "_")
    return dataset_loader(directory, label, f"{gcs_path}/{label}/processed", speaker, **kwargs)


__all__ = [
    "Speaker",
    "Example",
    "Span",
    "DataLoader",
    "Alignment",
    "span_generator",
    "dataset_loader",
    "conventional_dataset_loader",
    "lj_speech_dataset",
    "m_ailabs",
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
