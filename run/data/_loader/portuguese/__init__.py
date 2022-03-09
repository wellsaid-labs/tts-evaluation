import typing

from run.data._loader.data_structures import Speaker
from run.data._loader.portuguese import librivox, wsl
from run.data._loader.portuguese.librivox import (
    LIBRIVOX_DATASETS,
    RND__LIBRIVOX__FELIPE_PT,
    RND__LIBRIVOX__LENI_PT,
    RND__LIBRIVOX__MIRAMONTES_PT,
    RND__LIBRIVOX__SANDRALUNA_PT,
)
from run.data._loader.portuguese.wsl import FIVE_NINE__CUSTOM_VOICE__PT_BR, WSL_DATASETS
from run.data._loader.utils import DataLoader

DATASETS = typing.cast(typing.Dict[Speaker, DataLoader], WSL_DATASETS)
DATASETS.update(typing.cast(typing.Dict[Speaker, DataLoader], LIBRIVOX_DATASETS))

__all__ = [
    "librivox",
    "wsl",
    "RND__LIBRIVOX__FELIPE_PT",
    "RND__LIBRIVOX__LENI_PT",
    "RND__LIBRIVOX__MIRAMONTES_PT",
    "RND__LIBRIVOX__SANDRALUNA_PT",
    "DATASETS",
    "LIBRIVOX_DATASETS",
    "FIVE_NINE__CUSTOM_VOICE__PT_BR",
    "WSL_DATASETS",
]
