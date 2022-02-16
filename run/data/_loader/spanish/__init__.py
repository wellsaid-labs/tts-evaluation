import typing

from run.data._loader.data_structures import Speaker
from run.data._loader.spanish import m_ailabs, wsl
from run.data._loader.spanish.m_ailabs import (
    KAREN_SAVAGE,
    TUX,
    VICTOR_VILLARRAZA,
    m_ailabs_es_es_karen_savage_speech_dataset,
    m_ailabs_es_es_tux_speech_dataset,
    m_ailabs_es_es_victor_v_speech_dataset,
)
from run.data._loader.spanish.wsl import FIVE_NINE__CUSTOM_VOICE__ES_CO, WSL_DATASETS
from run.data._loader.utils import DataLoader

DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS)
DATASETS[KAREN_SAVAGE] = m_ailabs_es_es_karen_savage_speech_dataset
DATASETS[VICTOR_VILLARRAZA] = m_ailabs_es_es_victor_v_speech_dataset
DATASETS[TUX] = m_ailabs_es_es_tux_speech_dataset

__all__ = [
    "m_ailabs",
    "wsl",
    "KAREN_SAVAGE",
    "VICTOR_VILLARRAZA",
    "TUX",
    "m_ailabs_es_es_karen_savage_speech_dataset",
    "m_ailabs_es_es_victor_v_speech_dataset",
    "m_ailabs_es_es_tux_speech_dataset",
    "DATASETS",
    "FIVE_NINE__CUSTOM_VOICE__ES_CO",
    "WSL_DATASETS",
]
