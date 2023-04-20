import config as cf

import lib
import run

TTS_DISK_CACHE_NAME = ".tts_cache"  # NOTE: Hidden directory stored in other directories for caching
DISK_PATH = lib.environment.ROOT_PATH / "disk"
DATA_PATH = DISK_PATH / "data"
EXPERIMENTS_PATH = DISK_PATH / "experiments"
CHECKPOINTS_PATH = DISK_PATH / "checkpoints"
TEMP_PATH = DISK_PATH / "temp"
SAMPLES_PATH = DISK_PATH / "samples"
# NOTE: For production, store an inference version of signal and spectrogram model.
TTS_PACKAGE_PATH = DISK_PATH / "tts_package.pt"
SIGNAL_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "signal_model"
SPECTROGRAM_MODEL_EXPERIMENTS_PATH = EXPERIMENTS_PATH / "spectrogram_model"
DATASET_CACHE_PATH = TEMP_PATH / "dataset.pickle"
REMOTE_ROOT_PATH = "/opt/wellsaid-labs/Text-to-Speech/"

RANDOM_SEED = 1212212


def configure(overwrite: bool = False):
    """Make disk file structure."""
    for directory in [
        DISK_PATH,
        DATA_PATH,
        EXPERIMENTS_PATH,
        CHECKPOINTS_PATH,
        TEMP_PATH,
        SAMPLES_PATH,
        SIGNAL_MODEL_EXPERIMENTS_PATH,
        SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    ]:
        directory.mkdir(exist_ok=True)

    config = {
        run._utils.get_unprocessed_dataset: cf.Args(path=DATA_PATH),
        run.data._loader.utils._cache_path: cf.Args(cache_dir=TTS_DISK_CACHE_NAME),
        run.data._loader.structures._process_sessions: cf.Args(cache_dir=TTS_DISK_CACHE_NAME),
    }
    cf.add(config, overwrite=overwrite)
