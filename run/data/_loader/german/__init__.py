from run.data._loader.german import m_ailabs, wsl
from run.data._loader.german.m_ailabs import M_AILABS_DATASETS
from run.data._loader.german.wsl import WSL_DATASETS
from run.data._loader.utils import DataLoaders

DATASETS: DataLoaders = {**WSL_DATASETS, **M_AILABS_DATASETS, **DATASETS}

__all__ = ["m_ailabs", "wsl", "DATASETS", "M_AILABS_DATASETS", "WSL_DATASETS"]
