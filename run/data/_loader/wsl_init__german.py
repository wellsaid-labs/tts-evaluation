import typing

from run.data._loader.data_structures import Speaker
from run.data._loader.m_ailabs__german_datasets import (
    ANGELA_MERKEL,
    EVA_K,
    KARLSSON,
    RAMONA_DEININGER,
    REBECCA_BRAUNERT_PLUNKETT,
    m_ailabs_de_de_angela_merkel_speech_dataset,
    m_ailabs_de_de_eva_k_speech_dataset,
    m_ailabs_de_de_karlsson_speech_dataset,
    m_ailabs_de_de_ramona_deininger_speech_dataset,
    m_ailabs_de_de_rebecca_braunert_plunkett_speech_dataset,
)
from run.data._loader.utils import DataLoader
from run.data._loader.wsl_datasets__international import WSL_DATASETS__GERMAN

DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS__GERMAN)
DATASETS[ANGELA_MERKEL] = m_ailabs_de_de_angela_merkel_speech_dataset
DATASETS[EVA_K] = m_ailabs_de_de_eva_k_speech_dataset
DATASETS[RAMONA_DEININGER] = m_ailabs_de_de_ramona_deininger_speech_dataset
DATASETS[REBECCA_BRAUNERT_PLUNKETT] = m_ailabs_de_de_rebecca_braunert_plunkett_speech_dataset
DATASETS[KARLSSON] = m_ailabs_de_de_karlsson_speech_dataset

__all__ = [
    "Speaker",
    "ANGELA_MERKEL",
    "EVA_K",
    "RAMONA_DEININGER",
    "REBECCA_BRAUNERT_PLUNKETT",
    "KARLSSON",
    "m_ailabs_de_de_angela_merkel_speech_dataset",
    "m_ailabs_de_de_eva_k_speech_dataset",
    "m_ailabs_de_de_ramona_deininger_speech_dataset",
    "m_ailabs_de_de_rebecca_braunert_plunkett_speech_dataset",
    "m_ailabs_de_de_karlsson_speech_dataset",
    "DataLoader",
    "DATASETS",
]
