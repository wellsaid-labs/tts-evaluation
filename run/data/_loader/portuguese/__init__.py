from run.data._loader.portuguese import librivox, wsl
from run.data._loader.portuguese.librivox import LIBRIVOX_DATASETS
from run.data._loader.portuguese.wsl import WSL_DATASETS
from run.data._loader.utils import DataLoaders

DATASETS: DataLoaders = {**WSL_DATASETS, **LIBRIVOX_DATASETS}

__all__ = ["librivox", "wsl", "DATASETS", "LIBRIVOX_DATASETS", "WSL_DATASETS"]
