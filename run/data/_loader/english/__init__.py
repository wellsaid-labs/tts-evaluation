from run.data._loader.english import dictionary, lj_speech, m_ailabs, wsl, wsl_archive
from run.data._loader.english.m_ailabs import M_AILABS_DATASETS
from run.data._loader.english.wsl import WSL_DATASETS
from run.data._loader.utils import DataLoaders

# TODO: Consider updating M-AILABS and LJSpeech to Google Storage, so that we can download
# and upload them faster. It'll also give us protection, if the datasets are deleted.

DATASETS: DataLoaders = {**WSL_DATASETS, **M_AILABS_DATASETS}
DATASETS[lj_speech.LINDA_JOHNSON] = lj_speech.lj_speech_dataset
DATASETS[dictionary.GCP_SPEAKER] = dictionary.dictionary_dataset

__all__ = ["m_ailabs", "wsl", "wsl_archive", "WSL_DATASETS", "DATASETS"]
