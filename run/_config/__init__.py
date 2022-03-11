from run._config.all import configure
from run._config.audio import FRAME_HOP, NUM_FRAME_CHANNELS
from run._config.data import DATASETS, DEFAULT_SCRIPT, DEV_SPEAKERS
from run._config.environment import (
    CHECKPOINTS_PATH,
    DATA_PATH,
    DATASET_CACHE_PATH,
    RANDOM_SEED,
    SAMPLES_PATH,
    SIGNAL_MODEL_EXPERIMENTS_PATH,
    SPECTROGRAM_MODEL_EXPERIMENTS_PATH,
    TEMP_PATH,
    TTS_PACKAGE_PATH,
)
from run._config.labels import (
    Cadence,
    DatasetType,
    Device,
    GetLabel,
    Label,
    get_config_label,
    get_dataset_label,
    get_environment_label,
    get_model_label,
    get_signal_model_label,
    get_timer_label,
)
from run._config.lang import (
    STT_CONFIGS,
    LanguageCode,
    is_normalized_vo_script,
    is_sound_alike,
    is_voiced,
    normalize_vo_script,
)

# TODO: Reduce the usage of globals, and use configuration if possible.

__all__ = [
    "configure",
    "FRAME_HOP",
    "NUM_FRAME_CHANNELS",
    "DATASETS",
    "DEFAULT_SCRIPT",
    "DEV_SPEAKERS",
    "CHECKPOINTS_PATH",
    "DATA_PATH",
    "DATASET_CACHE_PATH",
    "RANDOM_SEED",
    "SAMPLES_PATH",
    "SIGNAL_MODEL_EXPERIMENTS_PATH",
    "SPECTROGRAM_MODEL_EXPERIMENTS_PATH",
    "TEMP_PATH",
    "TTS_PACKAGE_PATH",
    "Cadence",
    "DatasetType",
    "Device",
    "GetLabel",
    "Label",
    "get_config_label",
    "get_dataset_label",
    "get_environment_label",
    "get_model_label",
    "get_signal_model_label",
    "get_timer_label",
    "STT_CONFIGS",
    "LanguageCode",
    "is_normalized_vo_script",
    "is_sound_alike",
    "is_voiced",
    "normalize_vo_script",
]
